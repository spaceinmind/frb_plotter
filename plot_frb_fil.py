#!/usr/bin/env python3
"""
Script to read and plot FRB filterbank data from .fil file

Usage:
    python plot_frb_fil.py <filename.fil> [options]
"""

import sys
import os
import argparse
import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from sigpyproc.readers import FilReader

# Set larger font sizes for better readability
plt.rcParams.update({
    'font.family': 'serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'mathtext.default': 'regular',  # Use regular font for math text
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

def print_fil_info(fil):
    """
    Print information about the filterbank file
    
    Parameters:
    -----------
    fil : FilReader
        Filterbank reader object
    """
    print(f"\n{'='*70}")
    print(f"Filterbank File Information: {fil.header.filename}")
    print(f"{'='*70}\n")
    
    print("Header Information:")
    print("-" * 70)
    print(f"  Source name: {fil.header.source}")
    print(f"  Telescope: {fil.header.telescope}")
    print(f"  Data type: {fil.header.data_type}")
    print(f"  Number of channels: {fil.header.nchans}")
    print(f"  Number of samples: {fil.header.nsamples}")
    print(f"  Number of IFs: {fil.header.nifs}")
    print(f"  Number of bits: {fil.header.nbits}")
    print(f"  Center frequency: {fil.header.fch1:.3f} MHz")
    print(f"  Channel bandwidth: {fil.header.foff:.6f} MHz")
    print(f"  Frequency range: {fil.header.fch1 + (fil.header.nchans-1)*fil.header.foff:.3f} - {fil.header.fch1:.3f} MHz")
    print(f"  Time resolution: {fil.header.tsamp*1000:.6f} ms")
    print(f"  Total duration: {fil.header.tobs:.3f} s ({fil.header.tobs*1000:.3f} ms)")
    print(f"  MJD start: {fil.header.tstart:.10f}")
    
    if hasattr(fil.header, 'dm'):
        print(f"  Reference DM: {fil.header.dm:.4f} pc cm^-3")
    
    if hasattr(fil.header, 'ra') and hasattr(fil.header, 'dec'):
        print(f"  RA: {fil.header.ra}")
        print(f"  Dec: {fil.header.dec}")
    
    # Check for any RFI-related or extended header fields
    rfi_attrs = ['rfi_mask', 'rfi_flags', 'bad_channels', 'zapped_channels', 
                 'channel_mask', 'mask']
    found_rfi = False
    for attr in rfi_attrs:
        if hasattr(fil.header, attr):
            val = getattr(fil.header, attr)
            print(f"  {attr}: {val}")
            found_rfi = True
    
    if not found_rfi:
        print(f"  (No RFI flags found in header)")
    
    print(f"\n{'='*70}\n")

def read_frb_fil(filename, start_time=None, duration=None, dm=None):
    """
    Read FRB filterbank data from .fil file
    
    Parameters:
    -----------
    filename : str
        Path to the filterbank file
    start_time : float, optional
        Start time in seconds from beginning of file
    duration : float, optional
        Duration to read in seconds (None = read all)
    dm : float, optional
        DM for dedispersion (pc cm^-3). If provided, dedisperses while reading
        
    Returns:
    --------
    data : ndarray
        Dynamic spectrum data (frequency x time)
    freqs : ndarray
        Frequency array in MHz
    times : ndarray
        Time array in seconds
    metadata : dict
        Dictionary containing file header information
    """
    fil = FilReader(filename)
    
    # Get frequency array (in MHz)
    freqs = fil.header.fch1 + np.arange(fil.header.nchans) * fil.header.foff
    
    # Read data - sigpyproc returns (nchans, nsamples) which is what we want (frequency, time)
    if start_time is not None:
        start_sample = int(start_time / fil.header.tsamp)
        if duration is not None:
            nsamps = int(duration / fil.header.tsamp)
        else:
            nsamps = fil.header.nsamples - start_sample
    else:
        start_sample = 0
        nsamps = fil.header.nsamples
    
    # Read the data first
    data = fil.read_block(start_sample, nsamps)
    
    # Manual dedispersion if DM is provided (sigpyproc's read_dedisp_block has issues)
    if dm is not None and dm > 0:
        # Calculate dispersion delays relative to highest frequency
        # Lower frequencies arrive later, so they need to be shifted backward (earlier) in time
        freq_ref = freqs.max()  # Reference frequency (highest, channel 0)
        delays_s = 4.148808e3 * dm * (freqs**-2 - freq_ref**-2)  # Delay in seconds
        delays_samples = np.round(delays_s / fil.header.tsamp).astype(int)
        
        # Apply dedispersion by shifting each channel
        # Positive delay means data arrives late, so shift it back (left)
        data_dedisp = np.zeros_like(data)
        for i in range(len(freqs)):
            delay = delays_samples[i]
            if delay > 0:
                # Shift data left by 'delay' samples
                data_dedisp[i, :-delay] = data[i, delay:]
            else:
                data_dedisp[i, :] = data[i, :]
        
        data = data_dedisp
    
    # Get time array (in seconds)
    times = np.arange(data.shape[1]) * fil.header.tsamp
    if start_time is not None:
        times += start_time
    
    # Extract metadata
    metadata = {
        'source_name': fil.header.source,
        'telescope': fil.header.telescope,
        'fch1': fil.header.fch1,
        'foff': fil.header.foff,
        'tsamp': fil.header.tsamp,
        'tstart': fil.header.tstart,
        'nchans': fil.header.nchans,
        'nsamples': fil.header.nsamples,
    }
    
    if hasattr(fil.header, 'dm'):
        metadata['dm'] = fil.header.dm
    
    return data, freqs, times, metadata, fil

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot FRB dynamic spectrum from filterbank file')
parser.add_argument('filename', type=str, 
                    help='Path to the filterbank (.fil) file')
parser.add_argument('-f', '--freq-downsample', type=int, default=1,
                    help='Frequency downsampling factor (default: 1)')
parser.add_argument('-t', '--time-downsample', type=int, default=1,
                    help='Time downsampling factor (default: 1)')
parser.add_argument('--cut', type=float, default=None,
                    help='Auto-find pulse and cut ±N ms around it (e.g., --cut 100 for ±100ms window)')
parser.add_argument('--dm', type=float, default=None,
                    help='DM for dedispersion (pc cm^-3). If not provided, uses header DM or no dedispersion')
parser.add_argument('--no-dedisperse', action='store_true',
                    help='Skip dedispersion even if DM is available')
parser.add_argument('--flag-freq', type=str, default=None,
                    help='Manually flag frequency bands (e.g., "1540-1560" or "1540-1560,1200-1220")')
parser.add_argument('-s', '--sym', action='store_true', dest='symmetric_plot',
                    help='Create symmetric plot centered on pulse (t=0 at pulse center)')
parser.add_argument('--window-size', type=float, default=10.0,
                    help='Time window size as multiple of pulse width (default: 10, meaning ±10× pulse width)')
parser.add_argument('--save-png', action='store_true',
                    help='Save plot as PNG file (default: do not save)')

args = parser.parse_args()
filename = args.filename
FREQ_DOWNSAMPLE = args.freq_downsample
TIME_DOWNSAMPLE = args.time_downsample

if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found!")
    sys.exit(1)

# First, read the file to get metadata (including header DM)
print(f"Reading filterbank file: {filename}")
data, freqs, times, metadata, fil = read_frb_fil(filename, start_time=None, duration=None, dm=None)

# Print file information
print_fil_info(fil)

# Determine DM for dedispersion
dm_value = None
if not args.no_dedisperse:
    if args.dm is not None:
        # User-specified DM takes priority
        dm_value = args.dm
        print(f"Using user-specified DM: {dm_value:.4f} pc cm^-3")
    elif 'dm' in metadata and metadata['dm'] > 0:
        # Use header DM if available and non-zero
        dm_value = metadata['dm']
        print(f"Using DM from file header: {dm_value:.4f} pc cm^-3")
    else:
        print("No DM available - plotting without dedispersion")

# Re-read with dedispersion if DM is available
if dm_value is not None and dm_value > 0:
    print(f"Dedispersing at DM = {dm_value:.4f} pc cm^-3...")
    data, freqs, times, metadata, fil = read_frb_fil(filename, start_time=None, duration=None, dm=dm_value)
    is_dedispersed = True
    print(f"  Dedispersion complete (manual method)")
else:
    is_dedispersed = False

# If --cut mode, find pulse and extract window from already-dedispersed data
if args.cut is not None:
    print(f"\nAuto-cut mode: extracting ±{args.cut:.1f} ms around pulse...")
    # Quick collapse to find pulse in dedispersed data
    time_series_initial = np.mean(data, axis=0)
    pulse_idx_initial = np.argmax(time_series_initial)
    
    # Calculate window in samples
    window_samples = int((args.cut / 1000.0) / metadata['tsamp'])
    start_idx = max(0, pulse_idx_initial - window_samples)
    end_idx = min(len(times), pulse_idx_initial + window_samples)
    
    print(f"  Pulse found at sample {pulse_idx_initial} (t={times[pulse_idx_initial]:.6f} s)")
    print(f"  Cutting {start_idx}:{end_idx} ({end_idx-start_idx} samples, {(end_idx-start_idx)*metadata['tsamp']*1000:.1f} ms)")
    
    # Cut the already-dedispersed data
    data = data[:, start_idx:end_idx]
    times = times[start_idx:end_idx]
    print(f"  Data cut to {data.shape}")

# Downsample data
def downsample_2d(data, freq_factor, time_factor):
    """
    Downsample 2D array by averaging over freq_factor x time_factor blocks
    """
    nfreq, ntime = data.shape
    
    # Trim to be divisible by downsampling factors
    nfreq_trim = (nfreq // freq_factor) * freq_factor
    ntime_trim = (ntime // time_factor) * time_factor
    
    data_trim = data[:nfreq_trim, :ntime_trim]
    
    # Reshape and average
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

# Manual frequency-based RFI flagging BEFORE downsampling to prevent RFI bleeding
rfi_flag_original = np.zeros(len(freqs), dtype=bool)
if args.flag_freq is not None:
    print(f"\nPre-downsampling RFI flagging:")
    manual_rfi_bands = []
    for band_str in args.flag_freq.split(','):
        parts = band_str.strip().split('-')
        if len(parts) == 2:
            try:
                freq_low = float(parts[0])
                freq_high = float(parts[1])
                manual_rfi_bands.append((freq_low, freq_high))
            except ValueError:
                print(f"  Warning: Could not parse frequency band '{band_str}'")
    
    for freq_low, freq_high in manual_rfi_bands:
        # Ensure freq_low < freq_high
        if freq_low > freq_high:
            freq_low, freq_high = freq_high, freq_low
        
        band_mask = (freqs >= freq_low) & (freqs <= freq_high)
        if np.sum(band_mask) > 0:
            rfi_flag_original |= band_mask
            flagged_freqs = freqs[band_mask]
            print(f"  Manual RFI flagging: {freq_low}-{freq_high} MHz → {flagged_freqs[0]:.2f}-{flagged_freqs[-1]:.2f} MHz ({np.sum(band_mask)} channels)")
        else:
            print(f"  Warning: No channels found in range {freq_low}-{freq_high} MHz")
    
    # Zero out flagged channels before downsampling
    if np.sum(rfi_flag_original) > 0:
        print(f"  Zeroing {np.sum(rfi_flag_original)} channels before downsampling")
        data[rfi_flag_original, :] = 0

if FREQ_DOWNSAMPLE > 1 or TIME_DOWNSAMPLE > 1:
    print(f"\nDownsampling data...")
    print(f"  Frequency downsampling factor: {FREQ_DOWNSAMPLE}")
    print(f"  Time downsampling factor: {TIME_DOWNSAMPLE}")
    print(f"  Original shape: {data.shape}")
    
    # Downsample data
    data_down, nfreq_trim, ntime_trim = downsample_2d(data, FREQ_DOWNSAMPLE, TIME_DOWNSAMPLE)
    freqs_down, _ = downsample_1d(freqs, FREQ_DOWNSAMPLE)
    times_down, _ = downsample_1d(times, TIME_DOWNSAMPLE)
    
    print(f"  Downsampled shape: {data_down.shape}")
    print(f"  New frequency resolution: {abs(np.median(np.diff(freqs_down))):.3f} MHz")
    print(f"  New time resolution: {np.median(np.diff(times_down))*1000:.6f} ms")
    
    # Replace original data with downsampled data
    data = data_down
    freqs = freqs_down
    times = times_down

print(f"\nData Statistics:")
print("-" * 70)
print(f"  Shape: {data.shape}")
print(f"  Min: {np.min(data):.6e}")
print(f"  Max: {np.max(data):.6e}")
print(f"  Mean: {np.mean(data):.6e}")
print(f"  Std: {np.std(data):.6e}")
print(f"  Median: {np.median(data):.6e}")

# Find pulse and determine its center for time reference
# First, collapse in frequency to get a time series
time_series_raw = np.mean(data, axis=0)

# Adaptive smoothing based on time resolution and expected pulse width
# For high time resolution data (< 10 us), smooth more to suppress RFI
tsamp_us = metadata['tsamp'] * 1e6  # Convert to microseconds
if tsamp_us < 10:
    # High time resolution: smooth over ~100 us to suppress narrow RFI
    sigma_samples = max(5, int(100 / tsamp_us))
elif tsamp_us < 50:
    # Medium resolution: moderate smoothing
    sigma_samples = max(3, int(50 / tsamp_us))
else:
    # Low resolution: minimal smoothing
    sigma_samples = 2

time_series_smooth = gaussian_filter1d(time_series_raw, sigma=sigma_samples)

print(f"\nPulse search parameters:")
print(f"  Time resolution: {tsamp_us:.2f} µs")
print(f"  Smoothing sigma: {sigma_samples} samples ({sigma_samples * tsamp_us:.1f} µs)")

# Find the brightest peak after smoothing
peak_idx = np.argmax(time_series_smooth)

# Use a more robust threshold: require peak to be significantly above background
# Use the quieter part of the time series (first 30%) for noise estimation
baseline_region_size = max(50, len(time_series_smooth)//3)
baseline_initial = np.median(time_series_smooth[:baseline_region_size])
noise_initial = np.std(time_series_smooth[:baseline_region_size])

# Require at least 5-sigma detection for high time resolution data
min_sigma = 5 if tsamp_us < 10 else 3
threshold = baseline_initial + min_sigma * noise_initial

# Verify the peak is significant
peak_significance = (time_series_smooth[peak_idx] - baseline_initial) / noise_initial
print(f"  Peak significance: {peak_significance:.1f} σ")

if peak_significance < min_sigma:
    print(f"  Warning: Peak is only {peak_significance:.1f}σ, may be RFI. Using simple maximum.")

# Find pulse extent
pulse_mask = time_series_smooth > threshold
if np.sum(pulse_mask) > 0:
    pulse_indices = np.where(pulse_mask)[0]
    pulse_start = max(0, pulse_indices[0] - 5)  # Add small buffer
    pulse_end = min(len(time_series_smooth), pulse_indices[-1] + 5)
    
    # Calculate center of mass of the pulse region
    pulse_region = time_series_smooth[pulse_start:pulse_end]
    pulse_weights = pulse_region - baseline_initial
    pulse_weights = np.maximum(pulse_weights, 0)  # Ensure positive
    
    if np.sum(pulse_weights) > 0:
        pulse_center_idx = pulse_start + int(np.sum(np.arange(len(pulse_weights)) * pulse_weights) / np.sum(pulse_weights))
    else:
        pulse_center_idx = peak_idx
else:
    # Fallback to simple peak
    pulse_center_idx = peak_idx

peak_idx = pulse_center_idx  # Use pulse center as reference
peak_freq_idx = np.argmax(data[:, peak_idx])

# Convert times to relative time in milliseconds, centered on pulse center
times_rel = (times - times[peak_idx]) * 1000  # Convert to ms, centered on pulse
peak_time = times_rel[peak_idx]  # Should be ~0

print(f"\nPulse identification:")
print(f"  Pulse center time: {peak_time:.3f} ms (index {peak_idx})")
print(f"  Peak frequency: {freqs[peak_freq_idx]:.2f} MHz")
print(f"  Peak value: {data[peak_freq_idx, peak_idx]:.3f}")

# Define off-pulse region - use 50 samples before peak or first 20% of data
off_pulse_start = 0
off_pulse_end = max(min(20, peak_idx - 50), int(len(times) * 0.2))
if off_pulse_end < 10:
    off_pulse_end = min(int(len(times) * 0.2), len(times) - 1)
print(f"\nRFI Flagging:")
print(f"  Using off-pulse region: {off_pulse_start} - {off_pulse_end} samples")

off_pulse_region = data[:, off_pulse_start:off_pulse_end]

# Calculate statistics per channel in off-pulse region
channel_median = np.median(off_pulse_region, axis=1)
channel_std = np.std(off_pulse_region, axis=1)
channel_max = np.max(off_pulse_region, axis=1)

# Calculate robust statistics using valid channels
valid_channels = (channel_median > 0) & (channel_std > 0)
if np.sum(valid_channels) > 10:
    median_of_medians = np.median(channel_median[valid_channels])
    std_of_medians = np.std(channel_median[valid_channels])
    median_of_stds = np.median(channel_std[valid_channels])
    std_of_stds = np.std(channel_std[valid_channels])
    median_of_max = np.median(channel_max[valid_channels])
    std_of_max = np.std(channel_max[valid_channels])
    
    # Flag channels that are extreme outliers (>10 sigma for very conservative flagging)
    # Use OR of conditions but require more extreme deviations
    rfi_flag = np.zeros(len(freqs), dtype=bool)
    rfi_flag |= (channel_median > median_of_medians + 10 * std_of_medians)
    rfi_flag |= (channel_std > median_of_stds + 10 * std_of_stds)
    rfi_flag |= (channel_max > median_of_max + 10 * std_of_max)
    
    # Also flag channels with zero or extremely low values (likely bad data)
    rfi_flag |= (channel_median < median_of_medians * 0.1)
    
    # If too many channels flagged (>50%), skip RFI flagging entirely
    if np.sum(rfi_flag) > len(freqs) * 0.5:
        print(f"  Warning: RFI flagging would remove {np.sum(rfi_flag)}/{len(freqs)} channels (>{50}%)")
        print(f"  Skipping RFI flagging to preserve data")
        rfi_flag = np.zeros(len(freqs), dtype=bool)
else:
    # Not enough valid channels, skip RFI flagging
    rfi_flag = np.zeros(len(freqs), dtype=bool)

n_rfi = np.sum(rfi_flag)

# Note: Manual frequency flagging was already applied before downsampling
# No need to re-apply here as it would flag wrong channels after downsampling

n_rfi = np.sum(rfi_flag)
if n_rfi > 0:
    print(f"  Total flagged: {n_rfi}/{len(freqs)} channels ({100*n_rfi/len(freqs):.1f}%)")
    rfi_indices = np.where(rfi_flag)[0]
    
    # Print frequency ranges of flagged channels
    print(f"  RFI frequencies: ", end="")
    ranges = []
    if len(rfi_indices) > 0:
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
else:
    print(f"  No RFI channels flagged")

# Clean data by zeroing RFI channels
data_clean = data.copy()
if n_rfi > 0:
    data_clean[rfi_flag, :] = 0

# Normalize: subtract off-pulse median and use robust noise estimate
data_normalized = data_clean.copy()
for i in range(data_clean.shape[0]):
    if not rfi_flag[i]:  # Only normalize good channels
        channel_data = data_clean[i, :]
        off_pulse = channel_data[off_pulse_start:off_pulse_end]
        if len(off_pulse) > 5:
            baseline_chan = np.median(off_pulse)
            # Use median absolute deviation for robust noise estimate
            mad = np.median(np.abs(off_pulse - baseline_chan))
            # Convert MAD to std (assuming Gaussian)
            std_chan = 1.4826 * mad if mad > 0 else np.std(off_pulse)
            if std_chan > 0:
                # Subtract baseline and divide by noise for SNR units
                data_normalized[i, :] = (channel_data - baseline_chan) / std_chan
            else:
                data_normalized[i, :] = channel_data - baseline_chan
        else:
            data_normalized[i, :] = 0

# Recalculate peak position on the cleaned and normalized data
# This ensures the peak is truly at t=0
if n_rfi > 0:
    good_mask_recenter = ~rfi_flag
    data_norm_recenter = data_normalized[good_mask_recenter, :]
    n_good_recenter = np.sum(good_mask_recenter)
    time_series_recenter = np.sum(data_norm_recenter, axis=0) / np.sqrt(n_good_recenter)
else:
    n_good_recenter = data_normalized.shape[0]
    time_series_recenter = np.sum(data_normalized, axis=0) / np.sqrt(n_good_recenter)

# Find the peak in the cleaned time series
peak_idx_final = np.argmax(time_series_recenter)

# Re-center times on the actual peak after all processing
times_rel = (times - times[peak_idx_final]) * 1000  # Convert to ms, centered on true peak
peak_time = times_rel[peak_idx_final]  # Should be exactly 0

print(f"  Re-centered on final peak at time index {peak_idx_final}")

# Recalculate rough pulse width for zoom window sizing using re-centered data
baseline_rough_final = np.median(time_series_recenter[:max(10, peak_idx_final - 50)])
noise_rough_final = np.std(time_series_recenter[:max(10, peak_idx_final - 50)])
time_series_sub_rough = time_series_recenter - baseline_rough_final
peak_value_rough = time_series_sub_rough[peak_idx_final]

threshold_rough = peak_value_rough * 0.1  # Use 10% for rough width estimate

# Find pulse start
pulse_start_rough = peak_idx_final
for i in range(peak_idx_final - 1, -1, -1):
    if time_series_sub_rough[i] < threshold_rough:
        pulse_start_rough = i + 1
        break
else:
    pulse_start_rough = 0

# Find pulse end
pulse_end_rough = peak_idx_final + 1
for i in range(peak_idx_final + 1, len(time_series_sub_rough)):
    if time_series_sub_rough[i] < threshold_rough:
        pulse_end_rough = i
        break
else:
    pulse_end_rough = len(time_series_sub_rough)

width_samples_rough = pulse_end_rough - pulse_start_rough
pulse_width_ms_rough = width_samples_rough * metadata['tsamp'] * 1000  # Convert to ms
print(f"  Final pulse width estimate: {pulse_width_ms_rough:.3f} ms ({width_samples_rough} samples)")

# Define zoom window around the burst
if args.symmetric_plot:
    # Symmetric plot: user-specified multiple of pulse width centered on pulse
    half_window = args.window_size * pulse_width_ms_rough  # User-defined multiplier on each side
    zoom_start = -half_window
    zoom_end = half_window
    print(f"  Symmetric window: ±{args.window_size:.1f}× pulse width = ±{half_window:.2f} ms")
else:
    # Default: adaptive window showing burst context
    total_time_ms = times_rel[-1] - times_rel[0]
    if total_time_ms < 50:
        window_ms = min(20, total_time_ms * 0.8)  # Very short observation
    elif total_time_ms < 100:
        window_ms = 30  # Short burst like this FRB
    else:
        window_ms = min(100, total_time_ms * 0.3)  # Longer observations
    
    zoom_start = max(times_rel[0], peak_time - window_ms * 0.3)  # Put burst closer to left (30% from edge)
    zoom_end = min(times_rel[-1], peak_time + window_ms * 0.7)  # More space after burst

zoom_indices = (times_rel >= zoom_start) & (times_rel <= zoom_end)

# Create the dynamic spectrum plot with time series and frequency spectrum
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1], 
                       hspace=0.05, wspace=0.05)
ax1 = fig.add_subplot(gs[0, 0])  # Dynamic spectrum
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Time series
ax3 = fig.add_subplot(gs[0, 1], sharey=ax1)  # Frequency spectrum

# Extract zoomed data
data_zoom = data_normalized[:, zoom_indices]
times_zoom = times_rel[zoom_indices]

# Plot dynamic spectrum (zoomed)
# Use percentiles for better scaling, but symmetric around 0 for SNR data
data_zoom_nonzero = data_zoom[data_zoom != 0]
if len(data_zoom_nonzero) > 0:
    # For normalized data, use symmetric scaling around 0
    vmax = np.percentile(data_zoom_nonzero, 99.5)
    # Use -2 to vmax scaling to show structure above noise floor
    vmin = -2
    vmax = max(vmax, 5)  # Ensure we show at least up to 5-sigma
else:
    vmin, vmax = data_zoom.min(), data_zoom.max()

im = ax1.imshow(data_zoom, aspect='auto', origin='upper',
               extent=[times_zoom[0], times_zoom[-1], freqs[-1], freqs[0]],
               interpolation='nearest', cmap='viridis', vmin=vmin, vmax=vmax)

ax1.set_ylabel('Frequency (MHz)')
ax1.tick_params(labelbottom=False)

# Mark RFI channels on the plot
if n_rfi > 0:
    rfi_indices = np.where(rfi_flag)[0]
    # Draw a small patch on the left side to indicate RFI channels
    patch_width = (times_zoom[-1] - times_zoom[0]) * 0.02  # 2% of time range
    for i in range(len(rfi_indices)):
        freq = freqs[rfi_indices[i]]
        ax1.plot([times_zoom[0], times_zoom[0] + patch_width], [freq, freq], 
                color='lightgray', alpha=0.7, linewidth=1.0, linestyle='-')

# Plot time series (collapsed in frequency) - zoomed
# When data is in SNR units per channel, combining N channels increases SNR by sqrt(N)
# So we sum the normalized data and divide by sqrt(number of good channels)
if n_rfi > 0:
    good_mask = ~rfi_flag
    data_norm_for_ts = data_normalized[good_mask, :]
    n_good_channels = np.sum(good_mask)
    # Sum SNR units - this adds in quadrature
    time_series_full = np.sum(data_norm_for_ts, axis=0) / np.sqrt(n_good_channels)
else:
    n_good_channels = data_normalized.shape[0]
    time_series_full = np.sum(data_normalized, axis=0) / np.sqrt(n_good_channels)

# Calculate baseline and noise from the FULL time series using the original off-pulse region
# This ensures consistency regardless of zoom window
off_pulse_full = time_series_full[off_pulse_start:off_pulse_end]
baseline_full = np.median(off_pulse_full)
noise_full = np.std(off_pulse_full)

# Now extract the zoomed time series
time_series_zoom = time_series_full[zoom_indices]

# Calculate SNR from the zoomed time series accounting for pulse width
# Since data is already normalized to SNR units per channel, the time series is roughly in SNR units too
peak_idx_zoom = np.argmax(time_series_zoom)
# Find which part of the original times array this corresponds to
zoom_start_idx = np.argmin(np.abs(times_rel - times_zoom[0]))
peak_idx_full = zoom_start_idx + peak_idx_zoom

# Use baseline and noise calculated from full time series for consistency
baseline_zoom = baseline_full
noise_zoom = noise_full

# Calculate pulse width and width-corrected SNR using boxcar matched filtering
# This matches standard single pulse search algorithms
# Subtract baseline from time series
time_series_sub = time_series_zoom - baseline_zoom

# Test different boxcar widths to find optimal match (similar to single_pulse_search)
# Search from 1 sample up to reasonable maximum
# Allow search up to 1/2 of zoom window or 200 samples, whichever is smaller
max_width_samples = min(200, len(time_series_sub) // 2)
best_snr = 0
best_width = 1
best_start = peak_idx_zoom
best_sum = 0

# Search window: look within ±max_width of the peak
search_start = max(0, peak_idx_zoom - max_width_samples)
search_end = min(len(time_series_sub), peak_idx_zoom + max_width_samples)

print(f"  Boxcar search: widths 1-{max_width_samples}, positions {search_start}-{search_end}")

for width in range(1, max_width_samples + 1):
    # Slide boxcar across the region around the peak
    for start in range(search_start, min(search_end - width + 1, len(time_series_sub) - width + 1)):
        end = start + width
        pulse_region = time_series_sub[start:end]
        pulse_sum = np.sum(pulse_region)
        # S/N for boxcar of this width: sum / (noise * sqrt(width))
        snr = pulse_sum / (noise_zoom * np.sqrt(width))
        if snr > best_snr:
            best_snr = snr
            best_width = width
            best_start = start
            best_sum = pulse_sum

# Use the best-fit boxcar parameters
pulse_start = best_start
pulse_end = best_start + best_width
width_samples = best_width
peak_snr = best_snr

# Calculate pulse width in milliseconds
tsamp_ms = (times_zoom[1] - times_zoom[0])  # Time resolution in ms
pulse_width_ms = width_samples * tsamp_ms

print(f"  Pulse width: {width_samples} samples ({pulse_width_ms:.3f} ms)")
print(f"  Time series S/N: {peak_snr:.2f}")

ax2.plot(times_zoom, time_series_zoom, 'k-', linewidth=0.8)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Mean Intensity')
ax2.set_xlim(times_zoom[0], times_zoom[-1])
ax2.grid(True, alpha=0.3)
ax2.axvline(0, color='r', linestyle='--', alpha=0.5, linewidth=1)  # Mark burst center

# Mark pulse width on the time series
# Use bin edges so the shaded region includes the full start and end bins
pulse_start_time = times_zoom[pulse_start] - tsamp_ms
pulse_end_time = times_zoom[pulse_end - 1] + tsamp_ms
pulse_width_ms = pulse_end_time - pulse_start_time

print(f"  Width region: {pulse_start_time:.3f} to {pulse_end_time:.3f} ms (span: {pulse_width_ms:.3f} ms)")

ax2.axvspan(pulse_start_time, pulse_end_time, alpha=0.2, color='orange', label=f'Width ({pulse_width_ms:.2f} ms)')
ax2.legend(loc='upper right', fontsize=12)

# Build title from metadata
base_name = os.path.splitext(os.path.basename(filename))[0]
telescope = metadata.get('telescope', '')

# Build second line with telescope, DM, S/N, width
title_line2_parts = []
if telescope:
    title_line2_parts.append(f"({telescope})")
if dm_value is not None:
    title_line2_parts.append(f"DM={dm_value:.2f} pc cm$^{{-3}}$")
title_line2_parts.append(f"S/N~{peak_snr:.1f}")
title_line2_parts.append(f"W~{pulse_width_ms:.2f} ms")

# Combine with newline
title_str = base_name + '\n' + ' - '.join(title_line2_parts)
ax1.set_title(title_str)

# Add more x-axis ticks
ax2.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
ax2.tick_params(which='minor', length=3, width=0.5)
ax2.tick_params(which='major', length=6, width=1)

# Plot frequency spectrum (averaged over pulse region)
# Use the pulse region determined earlier
freq_spectrum = np.mean(data_zoom[:, pulse_start:pulse_end], axis=1)

# Ensure manually flagged frequency ranges show as zero
# Re-apply the frequency flagging to the downsampled data for visualization
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
ax3.set_ylim(freqs[0], freqs[-1])
ax3.grid(True, alpha=0.3)
ax3.yaxis.tick_right()
ax3.tick_params(labelleft=False)

plt.tight_layout()

# Save plot if requested
if args.save_png:
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    save_filename = f'{base_filename}_waterfall.png'
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {save_filename}')

plt.show()
