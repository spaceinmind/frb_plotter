#!/usr/bin/env python3
"""
Script to read and plot FRB filterbank data from HDF5 file

Usage:
    python plot_frb_filterbank.py <filename.h5>
"""

import sys
import os
import argparse
import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend for cluster

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.colors import LogNorm

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

def print_h5_info(filename):
    """
    Print information about the HDF5 file structure and contents
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
    """
    print(f"\n{'='*70}")
    print(f"HDF5 File Information: {filename}")
    print(f"{'='*70}\n")
    
    with h5py.File(filename, 'r') as f:
        # Print file structure
        print("File Structure:")
        print("-" * 70)
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}")
                print(f"    Shape: {obj.shape}")
                print(f"    Dtype: {obj.dtype}")
                if obj.size < 10:
                    print(f"    Values: {obj[:]}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        
        f.visititems(print_structure)
        
        # Print attributes
        print(f"\n{'='*70}")
        print("File Attributes:")
        print("-" * 70)
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        # Print detailed data information
        if 'data' in f:
            data = f['data'][:]
            print(f"\n{'='*70}")
            print("Data Statistics:")
            print("-" * 70)
            print(f"  Shape: {data.shape}")
            print(f"  Min: {np.min(data):.6e}")
            print(f"  Max: {np.max(data):.6e}")
            print(f"  Mean: {np.mean(data):.6e}")
            print(f"  Std: {np.std(data):.6e}")
            print(f"  Median: {np.median(data):.6e}")
        
        # Print frequency information
        if 'index_map/freqs' in f:
            freqs = f['index_map/freqs'][:]
            print(f"\n{'='*70}")
            print("Frequency Information:")
            print("-" * 70)
            print(f"  Number of channels: {len(freqs)}")
            print(f"  Frequency range: {freqs.min():.3f} - {freqs.max():.3f} MHz")
            print(f"  Frequency resolution: {np.median(np.diff(freqs)):.3f} MHz")
        
        # Print time information
        if 'index_map/times' in f:
            times = f['index_map/times'][:]
            print(f"\n{'='*70}")
            print("Time Information:")
            print("-" * 70)
            print(f"  Number of time samples: {len(times)}")
            print(f"  Time range: {times.min():.6f} - {times.max():.6f} s")
            print(f"  Time resolution: {np.median(np.diff(times))*1000:.6f} ms")
            print(f"  Total duration: {(times.max() - times.min())*1000:.3f} ms")
    
    print(f"\n{'='*70}\n")

def read_frb_h5(filename):
    """
    Read FRB filterbank data from HDF5 file
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
        
    Returns:
    --------
    data : ndarray
        Dynamic spectrum data (frequency x time)
    freqs : ndarray
        Frequency array in MHz
    times : ndarray
        Time array in seconds
    flag : ndarray
        Flag array
    metadata : dict
        Dictionary containing file attributes
    """
    with h5py.File(filename, 'r') as f:
        # Read the dynamic spectrum data
        data = f['data'][:]
        
        # Read frequency and time axes
        freqs = f['index_map/freqs'][:]
        times = f['index_map/times'][:]
        
        # Read other potentially useful info
        flag = f['flag'][:] if 'flag' in f else None
        good_freq = f['good_freq'][:] if 'good_freq' in f else None
        
        # Read all metadata from attributes
        metadata = {}
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]
        
    return data, freqs, times, flag, good_freq, metadata

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot FRB dynamic spectrum from HDF5 file')
parser.add_argument('filename', type=str,
                    help='Path to the HDF5 file')
parser.add_argument('-f', '--freq-downsample', type=int, default=1,
                    help='Frequency downsampling factor (default: 1)')
parser.add_argument('-t', '--time-downsample', type=int, default=1,
                    help='Time downsampling factor (default: 1)')
parser.add_argument('--cut', type=float, default=None,
                    help='Auto-find pulse and cut ±N ms around it (e.g., --cut 100 for ±100ms window)')
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

# Print HDF5 file information
print_h5_info(filename)

# Read the data
data, freqs, times, flag, good_freq, metadata = read_frb_h5(filename)

# If --cut mode, find pulse and extract window
if args.cut is not None:
    print(f"\nAuto-cut mode: extracting ±{args.cut:.1f} ms around pulse...")
    # Quick collapse to find pulse
    time_series_initial = np.mean(data, axis=0)
    pulse_idx_initial = np.argmax(time_series_initial)
    
    # Calculate window in samples
    tsamp_s = np.median(np.diff(times))
    window_samples = int((args.cut / 1000.0) / tsamp_s)
    start_idx = max(0, pulse_idx_initial - window_samples)
    end_idx = min(data.shape[1], pulse_idx_initial + window_samples)
    
    print(f"  Pulse found at sample {pulse_idx_initial} (t={times[pulse_idx_initial]:.6f} s)")
    print(f"  Cutting {start_idx}:{end_idx} ({end_idx-start_idx} samples, {(end_idx-start_idx)*tsamp_s*1000:.1f} ms)")
    
    # Cut the data
    data = data[:, start_idx:end_idx]
    times = times[start_idx:end_idx]
    if flag is not None:
        flag = flag[:, start_idx:end_idx]
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
            print(f"  {freq_low}-{freq_high} MHz → {np.sum(band_mask)} fine channels flagged")
        else:
            print(f"  Warning: No fine channels found in range {freq_low}-{freq_high} MHz")

if FREQ_DOWNSAMPLE > 1 or TIME_DOWNSAMPLE > 1:
    print(f"\nDownsampling data...")
    print(f"  Frequency downsampling factor: {FREQ_DOWNSAMPLE}")
    print(f"  Time downsampling factor: {TIME_DOWNSAMPLE}")
    print(f"  Original shape: {data.shape}")
    
    # Downsample data
    data_down, nfreq_trim, ntime_trim = downsample_2d(data, FREQ_DOWNSAMPLE, TIME_DOWNSAMPLE)
    freqs_down, _ = downsample_1d(freqs, FREQ_DOWNSAMPLE)
    times_down, _ = downsample_1d(times, TIME_DOWNSAMPLE)
    
    if flag is not None:
        # For flags, use 'any' operation - if any pixel is flagged as bad, the downsampled pixel is bad
        flag_down, _, _ = downsample_2d(flag.astype(float), FREQ_DOWNSAMPLE, TIME_DOWNSAMPLE)
        flag_down = flag_down > 0.5  # More than half of pixels are good
    else:
        flag_down = None
    
    if good_freq is not None:
        # For good_freq, a channel is good only if all original channels were good
        good_freq_trim = good_freq[:nfreq_trim]
        good_freq_down = good_freq_trim.reshape(len(freqs_down), FREQ_DOWNSAMPLE).all(axis=1)
    else:
        good_freq_down = None
    
    print(f"  Downsampled shape: {data_down.shape}")
    print(f"  New frequency resolution: {np.median(np.diff(freqs_down)):.3f} MHz")
    print(f"  New time resolution: {np.median(np.diff(times_down))*1000:.6f} ms")
    
    # Propagate fine-grid flag to coarse grid:
    # flag any coarse channel that contains at least one flagged fine channel
    rfi_flag_original = manual_flag_fine[:nfreq_trim].reshape(-1, FREQ_DOWNSAMPLE).any(axis=1)
    
    # Replace original data with downsampled data
    data = data_down
    freqs = freqs_down
    times = times_down
    flag = flag_down
    good_freq = good_freq_down
else:
    rfi_flag_original = manual_flag_fine.copy()

# Zero out the flagged channels on the (possibly downsampled) grid
if np.sum(rfi_flag_original) > 0:
    data[rfi_flag_original, :] = 0
    print(f"\nManual RFI flagging (coarse grid): zeroed {np.sum(rfi_flag_original)} channels")
    for freq_low, freq_high in manual_rfi_bands:
        hits = rfi_flag_original & (freqs >= freq_low - 20) & (freqs <= freq_high + 20)
        if np.sum(hits) > 0:
            print(f"  {freq_low}-{freq_high} MHz → coarse channels at {freqs[hits]} MHz")

print(f"\nData Statistics:")
print("-" * 70)
print(f"  Shape: {data.shape}")
print(f"  Min: {np.min(data):.6e}")
print(f"  Max: {np.max(data):.6e}")
print(f"  Mean: {np.mean(data):.6e}")
print(f"  Std: {np.std(data):.6e}")
print(f"  Median: {np.median(data):.6e}")

# Extract metadata
dm_incoherent = metadata.get('dm_incoherent', 0.0)
is_dedispersed = metadata.get('is_dedispersed', False)
frb_name = metadata.get('tns_name', metadata.get('repeater_name', 'Unknown FRB'))
stokes = metadata.get('stokes', 'I')
beam_number = metadata.get('beam_number', '')

print(f"\nFRB: {frb_name}")
print(f"Stokes: {stokes}")
if beam_number:
    print(f"Beam: {beam_number}")
print(f"Data already dedispersed: {is_dedispersed}")
print(f"DM from file header: {dm_incoherent:.4f} pc cm^-3\n")

# Find pulse and determine its center for time reference
# First, collapse in frequency to get a time series
time_series_raw = np.mean(data, axis=0)

# Adaptive smoothing based on time resolution and expected pulse width
# For high time resolution data (< 10 us), smooth more to suppress RFI
tsamp_s = np.median(np.diff(times))  # Time resolution in seconds
tsamp_us = tsamp_s * 1e6  # Convert to microseconds
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

# Convert times to relative time in milliseconds, centered on pulse center
times_rel = (times - times[peak_idx]) * 1000  # Convert to ms, centered on pulse
peak_time = times_rel[peak_idx]  # Should be ~0

# Flag==True means good data. Use only good data, set rest to 0 for cleaner display
data_clean = data.copy()
if flag is not None:
    # Keep only flagged (good) data
    data_clean[flag == False] = 0
    n_good = np.sum(flag)
    n_total = flag.size
    print(f"Good data points: {n_good}/{n_total} ({100*n_good/n_total:.1f}%)")

peak_freq_idx = np.argmax(data_clean[:, peak_idx])

print(f"\nPulse identification:")
print(f"  Pulse center time: {peak_time:.3f} ms (index {peak_idx})")
print(f"  Peak frequency: {freqs[peak_freq_idx]:.2f} MHz")
print(f"  Peak value: {data_clean[peak_freq_idx, peak_idx]:.3f}")

# Calculate rough pulse width for zoom window sizing
# Use the smoothed time series for width estimation
baseline_rough = baseline_initial
noise_rough = noise_initial
time_series_sub_rough = time_series_smooth - baseline_rough
peak_value_rough = time_series_sub_rough[peak_idx]

threshold_10pct_rough = peak_value_rough * 0.1
threshold_2sigma_rough = 2 * noise_rough
pulse_threshold_rough = min(threshold_10pct_rough, threshold_2sigma_rough)

# Find pulse start
pulse_start_rough = peak_idx
for i in range(peak_idx - 1, -1, -1):
    if time_series_sub_rough[i] < pulse_threshold_rough:
        pulse_start_rough = i + 1
        break
else:
    pulse_start_rough = 0

# Find pulse end
pulse_end_rough = peak_idx + 1
for i in range(peak_idx + 1, len(time_series_sub_rough)):
    if time_series_sub_rough[i] < pulse_threshold_rough:
        pulse_end_rough = i
        break
else:
    pulse_end_rough = len(time_series_sub_rough)

width_samples_rough = pulse_end_rough - pulse_start_rough
pulse_width_ms_rough = width_samples_rough * tsamp_s * 1000  # Convert to ms
print(f"  Estimated pulse width: {pulse_width_ms_rough:.3f} ms ({width_samples_rough} samples)")

# Define off-pulse region - use 50 samples before peak or first 20% of data
off_pulse_start = 0
off_pulse_end = max(min(20, peak_idx - 50), int(len(times) * 0.2))
if off_pulse_end < 10:
    off_pulse_end = min(int(len(times) * 0.2), len(times) - 1)

print(f"\nRFI Flagging:")
print(f"  Using off-pulse region: {off_pulse_start} - {off_pulse_end} samples")

# Use RFI flags from header (good_freq)
if good_freq is not None:
    rfi_flag = ~good_freq  # Invert: good_freq=True means good, rfi_flag=True means bad
    n_rfi = np.sum(rfi_flag)
    print(f"  From file header: Flagged {n_rfi}/{len(freqs)} channels ({100*n_rfi/len(freqs):.1f}%)")
    
    # Print frequency ranges of flagged channels
    if n_rfi > 0:
        rfi_indices = np.where(rfi_flag)[0]
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
        
        # Zero out RFI channels
        data_clean[rfi_flag, :] = 0
else:
    # Fallback: calculate RFI flags if not in header
    print(f"  No RFI flags in header, calculating...")
    off_pulse_region = data_clean[:, off_pulse_start:off_pulse_end]
    
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
    if n_rfi > 0:
        print(f"  Calculated: Flagged {n_rfi}/{len(freqs)} channels ({100*n_rfi/len(freqs):.1f}%)")
        rfi_indices = np.where(rfi_flag)[0]
        data_clean[rfi_flag, :] = 0
    else:
        print(f"  No RFI channels flagged")

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
# Create time series from normalized data
if good_freq is not None:
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

# Recalculate rough pulse width for zoom window sizing using the re-centered times
# This needs to happen after re-centering to get accurate window bounds
baseline_rough_final = np.median(time_series_recenter[:max(10, peak_idx_final - 50)])
noise_rough_final = np.std(time_series_recenter[:max(10, peak_idx_final - 50)])
time_series_sub_rough_final = time_series_recenter - baseline_rough_final
peak_value_rough_final = time_series_sub_rough_final[peak_idx_final]

threshold_rough_final = peak_value_rough_final * 0.1  # Use 10% for full width

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
pulse_width_ms_rough = width_samples_rough_final * tsamp_s * 1000  # Convert to ms
print(f"  Final pulse width estimate: {pulse_width_ms_rough:.3f} ms ({width_samples_rough_final} samples)")

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
        window_ms = 30  # Short burst
    else:
        window_ms = min(300, total_time_ms * 0.6)  # Longer observations
    
    zoom_start = max(times_rel[0], peak_time - window_ms * 0.2)  # Put burst closer to left (20% from edge)
    zoom_end = min(times_rel[-1], peak_time + window_ms * 0.8)  # More space after burst for scattering tail

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
ax1.set_xlabel('Time (ms)')
ax2.tick_params(labelbottom=False)

# Plot time series (collapsed in frequency) - zoomed
# When data is in SNR units per channel, combining N channels increases SNR by sqrt(N)
# So we sum the normalized data and divide by sqrt(number of good channels)
if good_freq is not None:
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
# The peak should be at t=0 since we already centered times on the pulse
# Find index in zoomed window closest to t=0
peak_idx_zoom = np.argmin(np.abs(times_zoom))

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

# Use the best-fit boxcar parameters for S/N
peak_snr = best_snr

# Calculate width containing 95% of the fluence (W95)
# First we need to define the full pulse region over which to calculate fluence
# We'll use the threshold method to find the broad bounds, then refine for W95

peak_search_idx = np.argmax(time_series_sub)
threshold_edge = 1.0 * noise_zoom # Use 1-sigma for broad bounds
required_consecutive = 3

# Search backwards for broad start
broad_start = 0
consecutive_below = 0
for i in range(peak_search_idx, -1, -1):
    if time_series_sub[i] < threshold_edge and time_series_sub[i] < time_series_sub[peak_search_idx]*0.1:
        consecutive_below += 1
    else:
        consecutive_below = 0
    
    if consecutive_below >= required_consecutive:
        broad_start = i + consecutive_below 
        break

# Search forwards for broad end
broad_end = len(time_series_sub)
consecutive_below = 0
for i in range(peak_search_idx, len(time_series_sub)):
    if time_series_sub[i] < threshold_edge and time_series_sub[i] < time_series_sub[peak_search_idx]*0.1:
        consecutive_below += 1
    else:
        consecutive_below = 0
    
    if consecutive_below >= required_consecutive:
        broad_end = i - consecutive_below + 1
        break

# Extract the pulse profile within broad bounds
pulse_profile = time_series_sub[broad_start:broad_end]
# Ensure all values are positive (baseline already subtracted)
pulse_profile = np.maximum(0, pulse_profile)

if np.sum(pulse_profile) > 0:
    # Calculate cumulative fluence
    cumulative_fluence = np.cumsum(pulse_profile)
    total_fluence = cumulative_fluence[-1]
    
    # Find indices for 2.5% and 97.5% (containing 95%)
    idx_start_95 = np.searchsorted(cumulative_fluence, 0.025 * total_fluence)
    idx_end_95 = np.searchsorted(cumulative_fluence, 0.975 * total_fluence)
    
    # Map back to original indices
    pulse_start = broad_start + idx_start_95
    pulse_end = broad_start + idx_end_95
    
    print(f"  W95 calculation: Broad bounds {broad_start}-{broad_end}, W95 bounds {pulse_start}-{pulse_end}")
else:
    print("  Warning: W95 calculation failed, falling back to boxcar")
    pulse_start = best_start
    pulse_end = best_start + best_width

width_samples = pulse_end - pulse_start

# Calculate pulse width in milliseconds
tsamp_ms = (times_zoom[1] - times_zoom[0])  # Time resolution in ms
pulse_width_ms = width_samples * tsamp_ms

print(f"  Pulse width: {width_samples} samples ({pulse_width_ms:.3f} ms)")
print(f"  Time series S/N: {peak_snr:.2f}")

ax2.plot(times_zoom, time_series_zoom, 'k-', linewidth=0.8)
ax2.set_ylabel('Mean Intensity')
ax2.set_xlim(times_zoom[0], times_zoom[-1])
ax2.grid(True, alpha=0.3)
# Mark burst center at the actual peak location
ax2.axvline(times_zoom[peak_idx_zoom], color='r', linestyle='--', alpha=0.5, linewidth=1)

# Mark pulse width on the time series
# Use bin edges so the shaded region includes the full start and end bins
pulse_start_time = times_zoom[pulse_start] - tsamp_ms
pulse_end_time = times_zoom[pulse_end - 1] + tsamp_ms
pulse_width_ms = pulse_end_time - pulse_start_time

print(f"  Width region: {pulse_start_time:.3f} to {pulse_end_time:.3f} ms (span: {pulse_width_ms:.3f} ms)")

ax2.axvspan(pulse_start_time, pulse_end_time, alpha=0.2, color='blue', label=f'W95 ({pulse_width_ms:.2f} ms)')
ax2.legend(loc='upper right', fontsize=12)

# Build title from metadata
base_name = os.path.splitext(os.path.basename(filename))[0]

# Build second line with telescope info, DM, S/N, width
title_line2_parts = []
if beam_number:
    title_line2_parts.append(f"(Beam {beam_number})")
title_line2_parts.append(f"DM={dm_incoherent:.2f} pc cm$^{{-3}}$")
title_line2_parts.append(f"S/N~{peak_snr:.1f}")
title_line2_parts.append(f"W95~{pulse_width_ms:.2f} ms")

# Combine with newline
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
# Use the pulse region determined earlier
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
if args.save_png:
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    save_filename = f'{base_filename}_waterfall.png'
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {save_filename}')

plt.show()
