#!/usr/bin/env python3
"""
Script to load .npy FRB data, dedisperse it, and create a publication-quality plot.
The plot shows a dynamic spectrum with time series below and frequency spectrum on the right.

Usage:
    python plot_frb_npy.py <filename.npy> --dm <DM> [options]
"""

import sys
import os
import argparse
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

try:
    from astropy.io import fits as astrofits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

# Set larger font sizes for better readability
plt.rcParams.update({
    'font.family': 'serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'mathtext.default': 'regular',
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})


def dedisperse(data, dm, freqs, tsamp):
    """
    Dedisperse the dynamic spectrum.

    Parameters:
    -----------
    data : 2D array
        Dynamic spectrum (time x frequency)
    dm : float
        Dispersion measure in pc/cm^3
    freqs : 1D array
        Frequency array in MHz
    tsamp : float
        Sampling time in seconds

    Returns:
    --------
    dedispersed : 2D array
        Dedispersed dynamic spectrum
    """
    k_dm = 4.148808  # GHz^2 ms pc^-1 cm^3

    # Delays relative to highest frequency
    delays = k_dm * dm * (freqs**-2 - freqs.max()**-2) / 1000.0

    # Convert delays to sample shifts
    shifts = np.round(delays / tsamp).astype(int)

    dedispersed = np.zeros_like(data)
    for i, shift in enumerate(shifts):
        dedispersed[:, i] = np.roll(data[:, i], -shift)

    return dedispersed


def downsample_2d(data, freq_factor, time_factor):
    """
    Downsample 2D array by averaging over freq_factor x time_factor blocks
    """
    nfreq, ntime = data.shape

    # Trim to be divisible by downsampling factors
    nfreq_trim = (nfreq // freq_factor) * freq_factor
    ntime_trim = (ntime // time_factor) * time_factor

    data_trim = data[:nfreq_trim, :ntime_trim]

    data_down = data_trim.reshape(nfreq_trim // freq_factor, freq_factor,
                                   ntime_trim // time_factor, time_factor)
    data_down = data_down.mean(axis=(1, 3))

    return data_down, nfreq_trim, ntime_trim


def downsample_1d(arr, factor):
    """
    Downsample 1D array by averaging over factor elements
    """
    n = len(arr)
    n_trim = (n // factor) * factor
    arr_trim = arr[:n_trim]
    arr_down = arr_trim.reshape(n_trim // factor, factor).mean(axis=1)
    return arr_down, n_trim


def find_companion_fits(npy_path):
    """
    Look for a PSRFITS file accompanying a .npy waterfall.
    Tries the following candidates in order:
      1. <same_dir>/<same_stem>.fits
      2. <same_dir>/<stem_without_trailing_-NNNN>.fits
    Returns the path of the first match found, or None.
    """
    dirpath = os.path.dirname(os.path.abspath(npy_path))
    stem = os.path.splitext(os.path.basename(npy_path))[0]

    candidates = [os.path.join(dirpath, stem + '.fits')]

    # strip trailing -NNNN (e.g. FRB20201124_0036-0012 -> FRB20201124_0036)
    import re
    short_stem = re.sub(r'-\d+$', '', stem)
    if short_stem != stem:
        candidates.append(os.path.join(dirpath, short_stem + '.fits'))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def read_fits_metadata(fits_path):
    """
    Read frequency range and time resolution from a PSRFITS file.
    Returns a dict with keys: fmin, fmax, tsamp, nchan_fits
    or raises an exception if the file cannot be parsed.
    """
    with astrofits.open(fits_path, memmap=True) as f:
        primary = f['PRIMARY'].header
        subint  = f['SUBINT'].header

        obsfreq  = float(primary['OBSFREQ'])   # MHz, band centre
        obsbw    = float(primary['OBSBW'])     # MHz, total BW (may be negative for USB)
        tbin     = float(subint['TBIN'])       # s, time per sample
        nchan    = int(subint['NCHAN'])        # channels in file

        half_bw = abs(obsbw) / 2.0
        fmin = obsfreq - half_bw
        fmax = obsfreq + half_bw

    return dict(fmin=fmin, fmax=fmax, tsamp=tbin, nchan_fits=nchan)


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot FRB dynamic spectrum from .npy file')
parser.add_argument('filename', type=str,
                    help='Path to the .npy file')
parser.add_argument('--dm', type=float, required=True,
                    help='Dispersion measure (pc/cm^3)')
parser.add_argument('--fmin', type=float, default=None,
                    help='Minimum frequency in MHz (auto-detected from companion FITS if omitted)')
parser.add_argument('--fmax', type=float, default=None,
                    help='Maximum frequency in MHz (auto-detected from companion FITS if omitted)')
parser.add_argument('--tsamp', type=float, default=None,
                    help='Time sampling in seconds (auto-detected from companion FITS if omitted)')
parser.add_argument('--frb-name', type=str, default=None,
                    help='FRB name for plot title (default: derived from filename)')
parser.add_argument('-f', '--freq-downsample', type=int, default=1,
                    help='Frequency downsampling factor (default: 1)')
parser.add_argument('-t', '--time-downsample', type=int, default=1,
                    help='Time downsampling factor (default: 1)')
parser.add_argument('--cut', type=float, default=None,
                    help='Auto-find pulse and cut +/-N ms around it (e.g., --cut 100 for +/-100ms window)')
parser.add_argument('--flag-freq', type=str, default=None,
                    help='Manually flag frequency bands (e.g., "1540-1560" or "1540-1560,1200-1220")')
parser.add_argument('-s', '--sym', action='store_true', dest='symmetric_plot',
                    help='Create symmetric plot centered on pulse (t=0 at pulse center)')
parser.add_argument('--window-size', type=float, default=10.0,
                    help='Time window size as multiple of pulse width (default: 10, meaning +/-10x pulse width)')
parser.add_argument('--edge-flag', type=float, default=0,
                    help='Width in MHz to flag at band edges (default: 0)')
parser.add_argument('--no-dedisperse', action='store_true',
                    help='Skip dedispersion (data is already dedispersed)')
parser.add_argument('--save-png', action='store_true',
                    help='Save plot as PNG file (default: do not save)')
parser.add_argument('--save-npy', action='store_true',
                    help='Save processed (dedispersed, downsampled, flagged) windowed data as a new .npy file')
parser.add_argument('--output', '-o', type=str, default=None,
                    help='Output plot filename (overrides --save-png default name)')

args = parser.parse_args()
filename = args.filename
FREQ_DOWNSAMPLE = args.freq_downsample
TIME_DOWNSAMPLE = args.time_downsample

if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found!")
    sys.exit(1)

# Auto-detect frequency range and time resolution from companion FITS file
# if not explicitly supplied on the command line.
_need_auto = (args.fmin is None) or (args.fmax is None) or (args.tsamp is None)
if _need_auto:
    if not HAS_ASTROPY:
        print("Warning: astropy not available; cannot auto-detect parameters from FITS.")
        print("         Install astropy or pass --fmin / --fmax / --tsamp explicitly.")
    else:
        _fits_path = find_companion_fits(filename)
        if _fits_path:
            try:
                _meta = read_fits_metadata(_fits_path)
                print(f"\nAuto-detected parameters from: {_fits_path}")
                if args.fmin is None:
                    args.fmin = _meta['fmin']
                    print(f"  fmin  = {args.fmin:.4f} MHz")
                if args.fmax is None:
                    args.fmax = _meta['fmax']
                    print(f"  fmax  = {args.fmax:.4f} MHz")
                if args.tsamp is None:
                    args.tsamp = _meta['tsamp']
                    print(f"  tsamp = {args.tsamp:.8f} s  ({args.tsamp*1e3:.4f} ms)")
                    if _meta['nchan_fits'] is not None:
                        print(f"  (FITS has {_meta['nchan_fits']} channels; .npy has been resampled)")
            except Exception as e:
                print(f"Warning: Could not read FITS metadata from '{_fits_path}': {e}")
        else:
            print(f"\nNo companion FITS file found for '{os.path.basename(filename)}'.")

# Fall back to sensible defaults if still unset
if args.fmin  is None: args.fmin  = 1000.0;  print("  fmin  defaulting to 1000.0 MHz (use --fmin to override)")
if args.fmax  is None: args.fmax  = 1500.0;  print("  fmax  defaulting to 1500.0 MHz (use --fmax to override)")
if args.tsamp is None: args.tsamp = 1e-4;    print("  tsamp defaulting to 1e-4 s / 0.1 ms (use --tsamp to override)")

# Load .npy file
print(f"\n{'='*70}")
print(f"NPY File Information: {filename}")
print(f"{'='*70}\n")

raw_data = np.load(filename)

print(f"Data loaded:")
print(f"  Shape: {raw_data.shape}")
print(f"  Dtype: {raw_data.dtype}")
nan_count = np.sum(np.isnan(raw_data))
if nan_count > 0:
    print(f"  NaN count: {nan_count} ({100*nan_count/raw_data.size:.1f}% of data) - will replace with 0")
    raw_data = np.nan_to_num(raw_data, nan=0.0)
print(f"  Min: {np.min(raw_data):.6e}")
print(f"  Max: {np.max(raw_data):.6e}")
print(f"  Mean: {np.mean(raw_data):.6e}")
print(f"  Std: {np.std(raw_data):.6e}")

# Determine data orientation: assume (time, frequency)
ntime, nchan = raw_data.shape

# Create frequency and time arrays
freqs = np.linspace(args.fmin, args.fmax, nchan)
times = np.arange(ntime) * args.tsamp

print(f"\nAssumed parameters:")
print(f"  Number of time samples: {ntime}")
print(f"  Number of frequency channels: {nchan}")
print(f"  Frequency range: {args.fmin:.1f} - {args.fmax:.1f} MHz")
print(f"  Time resolution: {args.tsamp*1e3:.4f} ms")
print(f"  Total duration: {times[-1]*1000:.3f} ms")

# Dedisperse
if not args.no_dedisperse:
    print(f"\nDedispersing at DM = {args.dm:.2f} pc/cm^3...")
    data_dedisp = dedisperse(raw_data, args.dm, freqs, args.tsamp)
else:
    print(f"\nSkipping dedispersion (--no-dedisperse)")
    data_dedisp = raw_data.copy()

# Transpose to (frequency, time) to match h5 convention
data = data_dedisp.T  # Now shape is (nchan, ntime)

# Flag band edges if requested
if args.edge_flag > 0:
    freq_range = freqs[-1] - freqs[0]
    edge_channels = int((args.edge_flag / freq_range) * len(freqs))
    print(f"\nFlagging {edge_channels} channels ({args.edge_flag} MHz) at each band edge")
    data[:edge_channels, :] = 0
    data[-edge_channels:, :] = 0

# If --cut mode, find pulse and extract window
if args.cut is not None:
    print(f"\nAuto-cut mode: extracting +/-{args.cut:.1f} ms around pulse...")
    time_series_initial = np.mean(data, axis=0)
    pulse_idx_initial = np.argmax(time_series_initial)

    tsamp_s = np.median(np.diff(times))
    window_samples = int((args.cut / 1000.0) / tsamp_s)
    start_idx = max(0, pulse_idx_initial - window_samples)
    end_idx = min(data.shape[1], pulse_idx_initial + window_samples)

    print(f"  Pulse found at sample {pulse_idx_initial} (t={times[pulse_idx_initial]:.6f} s)")
    print(f"  Cutting {start_idx}:{end_idx} ({end_idx-start_idx} samples, {(end_idx-start_idx)*tsamp_s*1000:.1f} ms)")

    data = data[:, start_idx:end_idx]
    times = times[start_idx:end_idx]
    print(f"  Data cut to {data.shape}")

# Build manual flag mask on the original fine frequency grid first,
# then propagate to the downsampled grid so narrow bands (smaller than a
# coarse channel) are still caught after frequency downsampling.
manual_rfi_bands = []
if args.flag_freq is not None:
    for band_str in args.flag_freq.split(','):
        parts = band_str.strip().split('-')
        if len(parts) == 2:
            try:
                freq_low = float(parts[0])
                freq_high = float(parts[1])
                if freq_low > freq_high:
                    freq_low, freq_high = freq_high, freq_low
                manual_rfi_bands.append((freq_low, freq_high))
            except ValueError:
                print(f"  Warning: Could not parse frequency band '{band_str}'")

# Apply flags on the fine grid before downsampling
manual_flag_fine = np.zeros(len(freqs), dtype=bool)
if manual_rfi_bands:
    print(f"\nManual RFI flagging (on original fine grid):")
    for freq_low, freq_high in manual_rfi_bands:
        band_mask = (freqs >= freq_low) & (freqs <= freq_high)
        if np.sum(band_mask) > 0:
            manual_flag_fine |= band_mask
            print(f"  {freq_low}-{freq_high} MHz -> {np.sum(band_mask)} fine channels flagged")
        else:
            print(f"  Warning: No fine channels found in range {freq_low}-{freq_high} MHz")

# Downsample data
if FREQ_DOWNSAMPLE > 1 or TIME_DOWNSAMPLE > 1:
    print(f"\nDownsampling data...")
    print(f"  Frequency downsampling factor: {FREQ_DOWNSAMPLE}")
    print(f"  Time downsampling factor: {TIME_DOWNSAMPLE}")
    print(f"  Original shape: {data.shape}")

    data, nfreq_trim, ntime_trim = downsample_2d(data, FREQ_DOWNSAMPLE, TIME_DOWNSAMPLE)
    freqs, _ = downsample_1d(freqs, FREQ_DOWNSAMPLE)
    times, _ = downsample_1d(times, TIME_DOWNSAMPLE)

    print(f"  Downsampled shape: {data.shape}")
    print(f"  New frequency resolution: {np.median(np.diff(freqs)):.3f} MHz")
    print(f"  New time resolution: {np.median(np.diff(times))*1000:.6f} ms")

    # Propagate fine-grid flag to coarse grid:
    # flag any coarse channel that contains at least one flagged fine channel
    rfi_flag_original = manual_flag_fine[:nfreq_trim].reshape(-1, FREQ_DOWNSAMPLE).any(axis=1)
else:
    rfi_flag_original = manual_flag_fine.copy()

# Zero out the flagged channels on the (possibly downsampled) grid
if np.sum(rfi_flag_original) > 0:
    data[rfi_flag_original, :] = 0
    print(f"\nManual RFI flagging (coarse grid): zeroed {np.sum(rfi_flag_original)} channels")
    for freq_low, freq_high in manual_rfi_bands:
        hits = rfi_flag_original & (freqs >= freq_low - 20) & (freqs <= freq_high + 20)
        if np.sum(hits) > 0:
            print(f"  {freq_low}-{freq_high} MHz -> coarse channels at {freqs[hits]} MHz")

print(f"\nData Statistics (post-processing):")
print("-" * 70)
print(f"  Shape: {data.shape}")
print(f"  Min: {np.nanmin(data):.6e}")
print(f"  Max: {np.nanmax(data):.6e}")
print(f"  Mean: {np.nanmean(data):.6e}")
print(f"  Std: {np.nanstd(data):.6e}")
print(f"  Median: {np.nanmedian(data):.6e}")

# Extract metadata
dm = args.dm
frb_name = args.frb_name if args.frb_name else os.path.splitext(os.path.basename(filename))[0]

print(f"\nFRB: {frb_name}")
print(f"DM: {dm:.4f} pc cm^-3")

# Find pulse and determine its center for time reference
time_series_raw = np.mean(data, axis=0)

# Adaptive smoothing based on time resolution
tsamp_s = np.median(np.diff(times))
tsamp_us = tsamp_s * 1e6

if tsamp_us < 10:
    sigma_samples = max(5, int(100 / tsamp_us))
elif tsamp_us < 50:
    sigma_samples = max(3, int(50 / tsamp_us))
else:
    sigma_samples = 2

time_series_smooth = gaussian_filter1d(time_series_raw, sigma=sigma_samples)

print(f"\nPulse search parameters:")
print(f"  Time resolution: {tsamp_us:.2f} us")
print(f"  Smoothing sigma: {sigma_samples} samples ({sigma_samples * tsamp_us:.1f} us)")

# Find the brightest peak after smoothing
peak_idx = np.argmax(time_series_smooth)

# Robust threshold: use quiet part of time series for noise estimation
baseline_region_size = max(50, len(time_series_smooth)//3)
baseline_initial = np.median(time_series_smooth[:baseline_region_size])
noise_initial = np.std(time_series_smooth[:baseline_region_size])

min_sigma = 5 if tsamp_us < 10 else 3
threshold = baseline_initial + min_sigma * noise_initial

peak_significance = (time_series_smooth[peak_idx] - baseline_initial) / noise_initial if noise_initial > 0 else 0
print(f"  Peak significance: {peak_significance:.1f} sigma")

if peak_significance < min_sigma:
    print(f"  Warning: Peak is only {peak_significance:.1f} sigma, may be RFI. Using simple maximum.")

# Find pulse extent
pulse_mask = time_series_smooth > threshold
if np.sum(pulse_mask) > 0:
    pulse_indices = np.where(pulse_mask)[0]
    pulse_start = max(0, pulse_indices[0] - 5)
    pulse_end = min(len(time_series_smooth), pulse_indices[-1] + 5)

    pulse_region = time_series_smooth[pulse_start:pulse_end]
    pulse_weights = pulse_region - baseline_initial
    pulse_weights = np.maximum(pulse_weights, 0)

    if np.sum(pulse_weights) > 0:
        pulse_center_idx = pulse_start + int(np.sum(np.arange(len(pulse_weights)) * pulse_weights) / np.sum(pulse_weights))
    else:
        pulse_center_idx = peak_idx
else:
    pulse_center_idx = peak_idx

peak_idx = pulse_center_idx

# Convert times to relative time in milliseconds, centered on pulse center
times_rel = (times - times[peak_idx]) * 1000  # ms
peak_time = times_rel[peak_idx]  # Should be ~0

# Clean data: apply RFI flagging
data_clean = data.copy()

# Define off-pulse region
off_pulse_start = 0
off_pulse_end = max(min(20, peak_idx - 50), int(len(times) * 0.2))
if off_pulse_end < 10:
    off_pulse_end = min(int(len(times) * 0.2), len(times) - 1)

print(f"\nRFI Flagging:")
print(f"  Using off-pulse region: {off_pulse_start} - {off_pulse_end} samples")

# Calculate RFI flags from off-pulse statistics
off_pulse_region = data_clean[:, off_pulse_start:off_pulse_end]

channel_median = np.median(off_pulse_region, axis=1)
channel_std = np.std(off_pulse_region, axis=1)
channel_max = np.max(off_pulse_region, axis=1)

valid_channels = (channel_median > 0) & (channel_std > 0)
if np.sum(valid_channels) > 10:
    median_of_medians = np.median(channel_median[valid_channels])
    std_of_medians = np.std(channel_median[valid_channels])
    median_of_stds = np.median(channel_std[valid_channels])
    std_of_stds = np.std(channel_std[valid_channels])
    median_of_max = np.median(channel_max[valid_channels])
    std_of_max = np.std(channel_max[valid_channels])

    rfi_flag = np.zeros(len(freqs), dtype=bool)
    rfi_flag |= (channel_median > median_of_medians + 10 * std_of_medians)
    rfi_flag |= (channel_std > median_of_stds + 10 * std_of_stds)
    rfi_flag |= (channel_max > median_of_max + 10 * std_of_max)
    rfi_flag |= (channel_median < median_of_medians * 0.1)

    # Include manual flags
    rfi_flag |= rfi_flag_original

    if np.sum(rfi_flag) > len(freqs) * 0.5:
        print(f"  Warning: RFI flagging would remove {np.sum(rfi_flag)}/{len(freqs)} channels (>50%)")
        print(f"  Skipping auto RFI flagging to preserve data")
        rfi_flag = rfi_flag_original.copy()
else:
    rfi_flag = rfi_flag_original.copy()

n_rfi = np.sum(rfi_flag)
if n_rfi > 0:
    print(f"  Flagged {n_rfi}/{len(freqs)} channels ({100*n_rfi/len(freqs):.1f}%)")
    rfi_indices = np.where(rfi_flag)[0]

    # Print flagged frequency ranges
    if len(rfi_indices) > 0:
        print(f"  RFI frequencies: ", end="")
        ranges = []
        start = rfi_indices[0]
        for i in range(1, len(rfi_indices)):
            if rfi_indices[i] != rfi_indices[i-1] + 1:
                if start == rfi_indices[i-1]:
                    ranges.append(f"{freqs[start]:.1f}")
                else:
                    ranges.append(f"{freqs[start]:.1f}-{freqs[rfi_indices[i-1]]:.1f}")
                start = rfi_indices[i]
        if start == rfi_indices[-1]:
            ranges.append(f"{freqs[start]:.1f}")
        else:
            ranges.append(f"{freqs[start]:.1f}-{freqs[rfi_indices[-1]]:.1f}")
        print(", ".join(ranges[:5]) + ("..." if len(ranges) > 5 else "") + " MHz")

    data_clean[rfi_flag, :] = 0
else:
    print(f"  No RFI channels flagged")

# Normalize: subtract off-pulse median, divide by noise for SNR units
data_normalized = data_clean.copy()
for i in range(data_clean.shape[0]):
    if not rfi_flag[i]:
        channel_data = data_clean[i, :]
        off_pulse = channel_data[off_pulse_start:off_pulse_end]
        if len(off_pulse) > 5:
            baseline_chan = np.median(off_pulse)
            mad = np.median(np.abs(off_pulse - baseline_chan))
            std_chan = 1.4826 * mad if mad > 0 else np.std(off_pulse)
            if std_chan > 0:
                data_normalized[i, :] = (channel_data - baseline_chan) / std_chan
            else:
                data_normalized[i, :] = channel_data - baseline_chan
        else:
            data_normalized[i, :] = 0

# Recalculate peak position on cleaned and normalized data
good_mask_recenter = ~rfi_flag
data_norm_recenter = data_normalized[good_mask_recenter, :]
n_good_recenter = np.sum(good_mask_recenter)
if n_good_recenter > 0:
    time_series_recenter = np.sum(data_norm_recenter, axis=0) / np.sqrt(n_good_recenter)
else:
    time_series_recenter = np.sum(data_normalized, axis=0) / np.sqrt(data_normalized.shape[0])
    n_good_recenter = data_normalized.shape[0]

peak_idx_final = np.argmax(time_series_recenter)

# Re-center times on the actual peak after all processing
times_rel = (times - times[peak_idx_final]) * 1000
peak_time = times_rel[peak_idx_final]  # Should be exactly 0

print(f"  Re-centered on final peak at time index {peak_idx_final}")

# Recalculate pulse width for zoom window sizing
baseline_rough_final = np.median(time_series_recenter[:max(10, peak_idx_final - 50)])
noise_rough_final = np.std(time_series_recenter[:max(10, peak_idx_final - 50)])
time_series_sub_rough_final = time_series_recenter - baseline_rough_final
peak_value_rough_final = time_series_sub_rough_final[peak_idx_final]

threshold_rough_final = peak_value_rough_final * 0.1

# Find pulse start
pulse_start_rough_final = peak_idx_final
for i in range(peak_idx_final - 1, -1, -1):
    if time_series_sub_rough_final[i] < threshold_rough_final:
        pulse_start_rough_final = i + 1
        break
else:
    pulse_start_rough_final = 0

# Find pulse end
pulse_end_rough_final = peak_idx_final + 1
for i in range(peak_idx_final + 1, len(time_series_sub_rough_final)):
    if time_series_sub_rough_final[i] < threshold_rough_final:
        pulse_end_rough_final = i
        break
else:
    pulse_end_rough_final = len(time_series_sub_rough_final)

width_samples_rough_final = pulse_end_rough_final - pulse_start_rough_final
pulse_width_ms_rough = width_samples_rough_final * tsamp_s * 1000
print(f"  Final pulse width estimate: {pulse_width_ms_rough:.3f} ms ({width_samples_rough_final} samples)")

peak_freq_idx = np.argmax(data_clean[:, peak_idx_final])
print(f"\nPulse identification:")
print(f"  Pulse center time: {peak_time:.3f} ms (index {peak_idx_final})")
print(f"  Peak frequency: {freqs[peak_freq_idx]:.2f} MHz")
print(f"  Peak value: {data_clean[peak_freq_idx, peak_idx_final]:.3f}")

# Define zoom window around the burst
if args.symmetric_plot:
    half_window = args.window_size * pulse_width_ms_rough
    zoom_start = -half_window
    zoom_end = half_window
    print(f"  Symmetric window: +/-{args.window_size:.1f}x pulse width = +/-{half_window:.2f} ms")
else:
    total_time_ms = times_rel[-1] - times_rel[0]
    if total_time_ms < 50:
        window_ms = min(20, total_time_ms * 0.8)
    elif total_time_ms < 100:
        window_ms = 30
    else:
        window_ms = min(150, total_time_ms * 0.3)

    zoom_start = max(times_rel[0], peak_time - window_ms * 0.3)
    zoom_end = min(times_rel[-1], peak_time + window_ms * 0.7)

zoom_indices = (times_rel >= zoom_start) & (times_rel <= zoom_end)

# Create the dynamic spectrum plot with time series and frequency spectrum
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 3],
                       hspace=0.05, wspace=0.05)
ax2 = fig.add_subplot(gs[0, 0])  # Time series (top)
ax1 = fig.add_subplot(gs[1, 0], sharex=ax2)  # Dynamic spectrum (bottom)
ax3 = fig.add_subplot(gs[1, 1], sharey=ax1)  # Frequency spectrum

# Extract zoomed data
data_zoom = data_normalized[:, zoom_indices]
times_zoom = times_rel[zoom_indices]

# Save windowed burst as a new .npy file if requested
if args.save_npy:
    base = os.path.splitext(os.path.basename(filename))[0]
    suffix_parts = [f'DM{args.dm:.4f}']
    if FREQ_DOWNSAMPLE > 1:
        suffix_parts.append(f'f{FREQ_DOWNSAMPLE}')
    if TIME_DOWNSAMPLE > 1:
        suffix_parts.append(f't{TIME_DOWNSAMPLE}')
    suffix_parts.append(f'win{args.window_size:.0f}x' if args.symmetric_plot else 'win')
    suffix = '_' + '_'.join(suffix_parts)
    out_npy = f"{base}{suffix}.npy"
    # Save as (time, freq) to match input convention
    np.save(out_npy, data_clean[:, zoom_indices].T.astype(np.float32))
    print(f"\nSaved processed NPY to: {out_npy}")
    print(f"  Shape: {data_clean[:, zoom_indices].T.shape} (time × freq)")
    print(f"  Freq range: {freqs[0]:.4f} – {freqs[-1]:.4f} MHz")
    print(f"  tsamp: {args.tsamp*1e6:.2f} µs")

# Plot dynamic spectrum (zoomed)
data_zoom_nonzero = data_zoom[data_zoom != 0]
if len(data_zoom_nonzero) > 0:
    vmax = np.percentile(data_zoom_nonzero, 99.5)
    vmin = -2
    vmax = max(vmax, 5)
else:
    vmin, vmax = data_zoom.min(), data_zoom.max()

im = ax1.imshow(data_zoom, aspect='auto', origin='upper',
               extent=[times_zoom[0], times_zoom[-1], freqs[-1], freqs[0]],
               interpolation='nearest', cmap='viridis', vmin=vmin, vmax=vmax)

ax1.set_ylabel('Frequency (MHz)')
ax1.set_xlabel('Time (ms)')
ax2.tick_params(labelbottom=False)


# Plot time series (collapsed in frequency) - zoomed
good_mask = ~rfi_flag
data_norm_for_ts = data_normalized[good_mask, :]
n_good_channels = np.sum(good_mask)
if n_good_channels > 0:
    time_series_full = np.sum(data_norm_for_ts, axis=0) / np.sqrt(n_good_channels)
else:
    n_good_channels = data_normalized.shape[0]
    time_series_full = np.sum(data_normalized, axis=0) / np.sqrt(n_good_channels)

# Calculate baseline and noise from full time series
off_pulse_full = time_series_full[off_pulse_start:off_pulse_end]
baseline_full = np.median(off_pulse_full)
noise_full = np.std(off_pulse_full)

# Extract zoomed time series
time_series_zoom = time_series_full[zoom_indices]

# Boxcar matched filtering for S/N
peak_idx_zoom = np.argmin(np.abs(times_zoom))

baseline_zoom = baseline_full
noise_zoom = noise_full

time_series_sub = time_series_zoom - baseline_zoom

max_width_samples = min(200, len(time_series_sub) // 2)
best_snr = 0
best_width = 1
best_start = peak_idx_zoom

search_start = max(0, peak_idx_zoom - max_width_samples)
search_end = min(len(time_series_sub), peak_idx_zoom + max_width_samples)

print(f"  Boxcar search: widths 1-{max_width_samples}, positions {search_start}-{search_end}")

for width in range(1, max_width_samples + 1):
    for start in range(search_start, min(search_end - width + 1, len(time_series_sub) - width + 1)):
        end = start + width
        pulse_sum = np.sum(time_series_sub[start:end])
        snr = pulse_sum / (noise_zoom * np.sqrt(width))
        if snr > best_snr:
            best_snr = snr
            best_width = width
            best_start = start

pulse_start = best_start
pulse_end = best_start + best_width
width_samples = best_width
peak_snr = best_snr

tsamp_ms = (times_zoom[1] - times_zoom[0])
pulse_width_ms = width_samples * tsamp_ms

print(f"  Pulse width: {width_samples} samples ({pulse_width_ms:.3f} ms)")
print(f"  Time series S/N: {peak_snr:.2f}")

ax2.plot(times_zoom, time_series_zoom, 'k-', linewidth=0.8)
ax2.set_ylabel('Mean Intensity')
ax2.set_xlim(times_zoom[0], times_zoom[-1])
ax2.grid(True, alpha=0.3)
ax2.axvline(times_zoom[peak_idx_zoom], color='r', linestyle='--', alpha=0.5, linewidth=1)

# Mark pulse width on the time series
pulse_start_time = times_zoom[pulse_start] - tsamp_ms
pulse_end_time = times_zoom[pulse_end - 1] + tsamp_ms
pulse_width_ms = pulse_end_time - pulse_start_time

print(f"  Width region: {pulse_start_time:.3f} to {pulse_end_time:.3f} ms (span: {pulse_width_ms:.3f} ms)")

ax2.axvspan(pulse_start_time, pulse_end_time, alpha=0.2, color='orange', label=f'Width ({pulse_width_ms:.2f} ms)')
ax2.legend(loc='upper right', fontsize=12)

# Build title
base_name = os.path.splitext(os.path.basename(filename))[0]

title_line2_parts = []
title_line2_parts.append(f"DM={dm:.2f} pc cm$^{{-3}}$")
title_line2_parts.append(f"S/N~{peak_snr:.1f}")
title_line2_parts.append(f"W~{pulse_width_ms:.2f} ms")

freq_res_mhz = np.abs(freqs[1] - freqs[0])
title_line2_parts.append(f"dt={tsamp_ms:.4f} ms")
title_line2_parts.append(f"df={freq_res_mhz:.3f} MHz")
title_str = frb_name + '\n' + ' - '.join(title_line2_parts)
ax2.set_title(title_str)

# Add more x-axis ticks
ax1.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.tick_params(which='minor', length=3, width=0.5)
ax1.tick_params(which='major', length=6, width=1)

# Plot frequency spectrum (averaged over pulse region)
freq_spectrum = np.mean(data_zoom[:, pulse_start:pulse_end], axis=1)

# Ensure manually flagged frequency ranges show as zero
if args.flag_freq is not None:
    manual_rfi_bands = []
    for band_str in args.flag_freq.split(','):
        parts = band_str.strip().split('-')
        if len(parts) == 2:
            try:
                freq_low = float(parts[0])
                freq_high = float(parts[1])
                if freq_low > freq_high:
                    freq_low, freq_high = freq_high, freq_low
                manual_rfi_bands.append((freq_low, freq_high))
            except ValueError:
                pass

    for freq_low, freq_high in manual_rfi_bands:
        band_mask = (freqs >= freq_low) & (freqs <= freq_high)
        freq_spectrum[band_mask] = 0

ax3.plot(freq_spectrum, freqs, 'k-', linewidth=0.8)
ax3.set_xlabel('Mean Intensity')
ax3.set_ylabel('')
ax3.set_ylim(freqs.min(), freqs.max())
ax3.grid(True, alpha=0.3)
ax3.yaxis.tick_right()
ax3.tick_params(labelleft=False)

plt.tight_layout()

# Save plot if requested
save_filename = None
if args.output:
    save_filename = args.output
elif args.save_png:
    save_filename = f'{base_name}_waterfall.png'

if save_filename:
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {save_filename}')

plt.show()
