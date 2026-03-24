#!/usr/bin/env python3
"""
Script to read and plot FRB data from PSRFITS (.fits) file

Supports standard PSRFITS format as produced by DSPSR, PREPFOLD, SIGPROC, etc.
Expects a SUBINT binary table extension with DATA column of shape
(nsub, npol, nchan, nbin) or (nsub, nchan, nbin).

Usage:
    python plot_frb_fits.py <filename.fits> [options]
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
from astropy.io import fits

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


# ---------------------------------------------------------------------------
# FITS / PSRFITS helpers
# ---------------------------------------------------------------------------

def find_subint_hdu(hdul):
    """Return the SUBINT HDU from an open HDUList, or None."""
    for hdu in hdul:
        if hdu.name in ('SUBINT', 'subint'):
            return hdu
        # Some files store it as an unnamed BinTableHDU after primary
        if isinstance(hdu, fits.BinTableHDU) and 'DATA' in hdu.columns.names:
            return hdu
    return None


def print_fits_info(hdul, filename):
    """Print information about the PSRFITS file."""
    print(f"\n{'='*70}")
    print(f"PSRFITS File Information: {filename}")
    print(f"{'='*70}\n")

    primary = hdul[0].header
    print("Primary Header:")
    print("-" * 70)
    for key in ['TELESCOP', 'OBSERVER', 'SRC_NAME', 'FRONTEND', 'BACKEND',
                'OBS_MODE', 'DATE-OBS', 'CHAN_BW', 'OBSFREQ', 'OBSBW',
                'OBSNCHAN', 'NPOL', 'FD_POLN']:
        val = primary.get(key, '(not set)')
        print(f"  {key:<14}: {val}")

    subint = find_subint_hdu(hdul)
    if subint is not None:
        sh = subint.header
        print("\nSUBINT Header:")
        print("-" * 70)
        for key in ['NAXIS2', 'NCHAN', 'NPOL', 'NBIN', 'TBIN', 'CHAN_BW',
                    'DM', 'RM', 'EPOCHS', 'ZERO_OFF']:
            val = sh.get(key, '(not set)')
            print(f"  {key:<14}: {val}")

    print(f"\n{'='*70}\n")


def read_psrfits(filename, start_time=None, duration=None, dm=None):
    """
    Read dynamic spectrum data from a PSRFITS file.

    Parameters
    ----------
    filename : str
        Path to the .fits file.
    start_time : float, optional
        Start offset in seconds from the beginning.
    duration : float, optional
        Duration to read in seconds (None = all).
    dm : float, optional
        DM for incoherent dedispersion (pc cm^-3).

    Returns
    -------
    data : ndarray, shape (nchan, ntime)
        Dynamic spectrum (frequency × time), Stokes I or total power.
    freqs : ndarray
        Frequency array in MHz (top channel first, matching PSRFITS convention).
    times : ndarray
        Time array in seconds relative to file start.
    metadata : dict
        Key observing parameters.
    hdul : fits.HDUList
        Open HDUList (caller should close if desired).
    """
    hdul = fits.open(filename, memmap=True)

    primary = hdul[0].header
    subint_hdu = find_subint_hdu(hdul)
    if subint_hdu is None:
        hdul.close()
        raise ValueError(
            f"No SUBINT/DATA extension found in '{filename}'. "
            "Is this a PSRFITS dynamic-spectrum file?"
        )

    sh = subint_hdu.header

    # ---- truncation-safe read -------------------------------------------
    # Truncated files raise "buffer is too small" when the whole DATA column
    # is memory-mapped at once.  Read one sub-integration row at a time and
    # stop gracefully on failure.
    nrows_hdr = sh.get('NAXIS2', 0) or len(subint_hdu.data)

    try:
        row0 = np.squeeze(subint_hdu.data['DATA'][0])
    except Exception as e:
        hdul.close()
        raise ValueError(f"Cannot read even the first DATA row: {e}")

    rows = [row0]
    for _ri in range(1, nrows_hdr):
        try:
            rows.append(np.squeeze(subint_hdu.data['DATA'][_ri]))
        except Exception:
            print(f"  WARNING: file truncated after row {_ri}/{nrows_hdr} -- using {_ri} sub-integrations")
            break

    data_col = np.stack(rows, axis=0)   # (nsub_actual, *per-row-shape-after-squeeze)
    raw_shape = data_col.shape
    squeezed_shape = raw_shape
    print(f"  DATA shape after truncation-safe read: {squeezed_shape}  (declared {nrows_hdr} rows)")

    # After squeezing we expect 2-D, 3-D, or 4-D:
    #   2-D: (nsub*nbin, nchan)  or  (nchan, nsub*nbin)  – rare flat dump
    #   3-D: (nsub, nchan, nbin) or (nsub, npol, nbin) – need header to tell
    #   4-D: (nsub, npol, nchan, nbin)
    ndim = data_col.ndim
    nchan_hdr = sh.get('NCHAN', primary.get('OBSNCHAN', 0))
    npol_hdr  = sh.get('NPOL',  primary.get('NPOL', 1))
    nbin_hdr  = sh.get('NBIN',  0)

    if ndim == 4:
        # Standard: (nsub, npol, nchan, nbin)
        nsub, npol, nchan, nbin = data_col.shape
        data_col = data_col.mean(axis=1)              # collapse pol → (nsub, nchan, nbin)

    elif ndim == 3:
        nsub, d1, d2 = data_col.shape
        # Heuristic: match header values where possible
        if nchan_hdr and d1 == nchan_hdr:
            # (nsub, nchan, nbin)
            nchan, nbin = d1, d2
        elif nchan_hdr and d2 == nchan_hdr:
            # (nsub, nbin, nchan) – transpose
            data_col = data_col.transpose(0, 2, 1)
            nchan, nbin = d2, d1
        elif nbin_hdr and d2 == nbin_hdr:
            nchan, nbin = d1, d2
        elif nbin_hdr and d1 == nbin_hdr:
            data_col = data_col.transpose(0, 2, 1)
            nchan, nbin = d2, d1
        else:
            # Fall back: assume larger inner dim is nchan
            if d1 >= d2:
                nchan, nbin = d1, d2
            else:
                data_col = data_col.transpose(0, 2, 1)
                nchan, nbin = d2, d1

    elif ndim == 2:
        # Flat dump – treat entire array as one sub-integration
        d0, d1 = data_col.shape
        if nchan_hdr and d1 == nchan_hdr:
            # (ntime, nchan)
            data_col = data_col[np.newaxis, :, :].transpose(0, 2, 1)   # (1, nchan, ntime)
        elif nchan_hdr and d0 == nchan_hdr:
            # (nchan, ntime)
            data_col = data_col[np.newaxis, :, :]
        else:
            # Guess: assume (nchan, ntime) with larger dim being time
            if d0 <= d1:
                data_col = data_col[np.newaxis, :, :]
            else:
                data_col = data_col.T[np.newaxis, :, :]
        nsub, nchan, nbin = data_col.shape

    else:
        hdul.close()
        raise ValueError(
            f"Cannot interpret DATA shape {raw_shape} (squeezed: {squeezed_shape}). "
            "Please open a GitHub issue or inspect with astropy.io.fits directly."
        )

    # Final shape after all reshaping
    nsub, nchan, nbin = data_col.shape
    print(f"  Interpreted as: nsub={nsub}, nchan={nchan}, nbin={nbin}")

    # ---- time resolution ------------------------------------------------
    # TBIN is the sample period inside a sub-integration bin (seconds)
    # If TBIN is absent, fall back to SUBINT_TBIN or compute from TSUBINT/NBIN
    tsamp = sh.get('TBIN', None)
    if tsamp is None or tsamp <= 0:
        tsubint = sh.get('TSUBINT', None)
        if tsubint and tsubint > 0:
            tsamp = tsubint / nbin
        else:
            # Last resort: try column
            try:
                tsamp = float(subint_hdu.data['TSUBINT'][0]) / nbin
            except Exception:
                tsamp = 1.0 / sh.get('CHAN_BW', 1.0)  # rough fallback

    total_samples = nsub * nbin

    # ---- frequency array ------------------------------------------------
    # Try CHAN_FREQ column first (per-subint freq, take first row)
    if 'CHAN_FREQ' in subint_hdu.columns.names:
        freqs = subint_hdu.data['CHAN_FREQ'][0].astype(float)  # MHz
    else:
        # Reconstruct from primary header
        f_centre = primary.get('OBSFREQ', sh.get('OBSFREQ', 1400.0))
        chan_bw  = primary.get('CHAN_BW', sh.get('CHAN_BW', -1.0))
        if chan_bw == 0:
            chan_bw = -1.0
        # PSRFITS convention: channel 0 is top (highest freq) when foff < 0
        freqs = f_centre + chan_bw * (np.arange(nchan) - nchan / 2.0 + 0.5)

    # ---- flatten sub-integrations into a single time axis ---------------
    # data_col: (nsub, nchan, nbin) → reshape to (nchan, nsub*nbin)
    # We need (nchan, ntime); data layout in PSRFITS is (nsub, nchan, nbin)
    data_2d = data_col.transpose(1, 0, 2).reshape(nchan, total_samples).astype(np.float32)

    # ---- optional scale / offset columns --------------------------------
    for col_name, op in [('DAT_SCL', 'scale'), ('DAT_OFFS', 'offset')]:
        if col_name in subint_hdu.columns.names:
            arr = subint_hdu.data[col_name]       # shape (nsub, nchan) typically
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            # Per-subint, per-channel – apply to each block of nbin samples
            vals_flat = np.repeat(arr.T, nbin, axis=1)  # (nchan, total_samples)
            if op == 'scale':
                data_2d *= vals_flat
            else:
                data_2d += vals_flat

    # ---- sample selection -----------------------------------------------
    if start_time is not None:
        start_sample = int(start_time / tsamp)
        if duration is not None:
            nsamps = int(duration / tsamp)
        else:
            nsamps = total_samples - start_sample
    else:
        start_sample = 0
        nsamps = total_samples

    nsamps = min(nsamps, total_samples - start_sample)
    data = data_2d[:, start_sample:start_sample + nsamps]

    # ---- dedispersion ---------------------------------------------------
    if dm is not None and dm > 0:
        # Reference = highest frequency (arrives first, zero delay).
        # delta_t = K * DM * (f^-2 - f_ref^-2), K=4148.808 MHz^2 pc^-1 cm^3 s^-1
        # All delays >= 0: lower freqs arrive later, so shift their data LEFT.
        freq_ref = freqs.max()
        delays_s = 4.148808e3 * dm * (freqs**-2 - freq_ref**-2)
        delays_samples = np.round(delays_s / tsamp).astype(int)

        print(f"  Dedispersion: freq order={'high->low' if freqs[0]>freqs[-1] else 'low->high'}, "
              f"delay range {delays_samples.min()}..{delays_samples.max()} samples")

        data_dedisp = np.zeros_like(data)
        ntime = data.shape[1]
        for i in range(len(freqs)):
            delay = int(delays_samples[i])
            if delay == 0:
                data_dedisp[i, :] = data[i, :]
            elif delay > 0:
                # channel arrived 'delay' samples late -> shift left
                data_dedisp[i, :ntime - delay] = data[i, delay:]
            else:
                # delay < 0: channel arrived early -> shift right (rare edge case)
                shift = -delay
                data_dedisp[i, shift:] = data[i, :ntime - shift]
        data = data_dedisp

    # ---- time array -----------------------------------------------------
    times = np.arange(data.shape[1]) * tsamp
    if start_time is not None:
        times += start_time

    # ---- metadata -------------------------------------------------------
    metadata = {
        'source_name': primary.get('SRC_NAME', sh.get('SRC_NAME', 'Unknown')),
        'telescope':   primary.get('TELESCOP', 'Unknown'),
        'fch1':        float(freqs[0]),
        'foff':        float(freqs[1] - freqs[0]) if len(freqs) > 1 else -1.0,
        'tsamp':       float(tsamp),
        'tstart':      primary.get('STT_IMJD', 0) + primary.get('STT_SMJD', 0) / 86400.0,
        'nchans':      nchan,
        'nsamples':    total_samples,
    }
    dm_hdr = sh.get('DM', primary.get('DM', None))
    if dm_hdr is not None:
        try:
            metadata['dm'] = float(dm_hdr)
        except (TypeError, ValueError):
            pass

    return data, freqs, times, metadata, hdul


# ---------------------------------------------------------------------------
# Write PSRFITS-like output (simple single-HDU dynamic spectrum)
# ---------------------------------------------------------------------------

def write_psrfits_simple(outname, data_nchan_ntime, freqs, tsamp, tstart_mjd,
                          source_name='Unknown', telescope='Unknown'):
    """
    Write a simplified PSRFITS file: one SUBINT HDU with DATA shape
    (1, nchan, ntime), float32.  Not a full PSRFITS archive but readable
    by most radio astronomy tools.
    """
    nchan, ntime = data_nchan_ntime.shape
    arr = data_nchan_ntime.astype(np.float32)

    # Primary HDU
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['TELESCOP'] = telescope
    primary_hdu.header['SRC_NAME'] = source_name
    primary_hdu.header['OBSFREQ']  = float(np.mean(freqs))
    primary_hdu.header['OBSBW']    = float(freqs[-1] - freqs[0])
    primary_hdu.header['OBSNCHAN'] = nchan
    primary_hdu.header['STT_IMJD'] = int(tstart_mjd)
    primary_hdu.header['STT_SMJD'] = int((tstart_mjd - int(tstart_mjd)) * 86400)

    # SUBINT BinTableHDU
    # DATA column: (1, nchan, ntime) – one sub-integration containing all samples
    data_cube = arr[np.newaxis, :, :]  # (1, nchan, ntime)

    col_data  = fits.Column(name='DATA',     format=f'{nchan*ntime}E',
                             dim=f'({ntime},{nchan})', array=data_cube.reshape(1, -1))
    col_freq  = fits.Column(name='CHAN_FREQ', format=f'{nchan}D',
                             unit='MHz',     array=freqs[np.newaxis, :])
    col_tsubint = fits.Column(name='TSUBINT', format='1D',
                               unit='s',     array=np.array([[ntime * tsamp]]))

    subint_hdu = fits.BinTableHDU.from_columns([col_data, col_freq, col_tsubint])
    subint_hdu.name = 'SUBINT'
    subint_hdu.header['TBIN']   = tsamp
    subint_hdu.header['NBIN']   = ntime
    subint_hdu.header['NCHAN']  = nchan
    subint_hdu.header['NPOL']   = 1
    subint_hdu.header['CHAN_BW'] = float(freqs[1] - freqs[0]) if nchan > 1 else 1.0

    hdul = fits.HDUList([primary_hdu, subint_hdu])
    hdul.writeto(outname, overwrite=True)
    print(f"Saved processed FITS to: {outname}")
    print(f"  Shape: {nchan} chans × {ntime} samples (float32)")
    print(f"  Freq range: {freqs.min():.4f} – {freqs.max():.4f} MHz")
    print(f"  tsamp={tsamp*1e6:.2f} µs")


# ---------------------------------------------------------------------------
# Downsampling helpers
# ---------------------------------------------------------------------------

def downsample_2d(data, freq_factor, time_factor):
    nfreq, ntime = data.shape
    nfreq_trim = (nfreq // freq_factor) * freq_factor
    ntime_trim = (ntime // time_factor) * time_factor
    data_trim = data[:nfreq_trim, :ntime_trim]
    data_down = data_trim.reshape(nfreq_trim // freq_factor, freq_factor,
                                   ntime_trim // time_factor, time_factor)
    return data_down.mean(axis=(1, 3)), nfreq_trim, ntime_trim


def downsample_1d(arr, factor):
    n = len(arr)
    n_trim = (n // factor) * factor
    arr_trim = arr[:n_trim]
    return arr_trim.reshape(n_trim // factor, factor).mean(axis=1), n_trim


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Plot FRB dynamic spectrum from PSRFITS file'
)
parser.add_argument('filename', type=str,
                    help='Path to the PSRFITS (.fits) file')
parser.add_argument('-f', '--freq-downsample', type=int, default=1,
                    help='Frequency downsampling factor (default: 1)')
parser.add_argument('-t', '--time-downsample', type=int, default=1,
                    help='Time downsampling factor (default: 1)')
parser.add_argument('--cut', type=float, default=None,
                    help='Auto-find pulse and cut ±N ms around it')
parser.add_argument('--dm', type=float, default=None,
                    help='DM for dedispersion (pc cm^-3)')
parser.add_argument('--no-dedisperse', action='store_true',
                    help='Skip dedispersion even if DM is available')
parser.add_argument('--flag-freq', type=str, default=None,
                    help='Manually flag frequency bands e.g. "1540-1560,1200-1220"')
parser.add_argument('-s', '--sym', action='store_true', dest='symmetric_plot',
                    help='Symmetric plot centred on pulse (t=0 at pulse centre)')
parser.add_argument('--window-size', type=float, default=10.0,
                    help='Time window as multiple of pulse width (default: 10)')
parser.add_argument('--save-png', action='store_true',
                    help='Save plot as PNG file')
parser.add_argument('--save-fits', action='store_true',
                    help='Save processed data as a new PSRFITS file')

args = parser.parse_args()
filename = args.filename
FREQ_DOWNSAMPLE = args.freq_downsample
TIME_DOWNSAMPLE = args.time_downsample

if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found!")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Initial read (no dedispersion yet – just to get header DM)
# ---------------------------------------------------------------------------
print(f"Reading PSRFITS file: {filename}")
try:
    data, freqs, times, metadata, hdul = read_psrfits(filename)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

with fits.open(filename) as hdul_info:
    print_fits_info(hdul_info, filename)

# ---------------------------------------------------------------------------
# Determine DM
# ---------------------------------------------------------------------------
dm_value = None
if not args.no_dedisperse:
    if args.dm is not None:
        dm_value = args.dm
        print(f"Using user-specified DM: {dm_value:.4f} pc cm^-3")
    elif 'dm' in metadata and metadata['dm'] > 0:
        dm_value = metadata['dm']
        print(f"Using DM from file header: {dm_value:.4f} pc cm^-3")
    else:
        print("No DM available – plotting without dedispersion")

# Re-read with dedispersion applied during load
if dm_value is not None and dm_value > 0:
    print(f"Dedispersing at DM = {dm_value:.4f} pc cm^-3...")
    data, freqs, times, metadata, hdul = read_psrfits(filename, dm=dm_value)
    is_dedispersed = True
    print("  Dedispersion complete")
else:
    is_dedispersed = False

# ---------------------------------------------------------------------------
# Auto-cut around pulse
# ---------------------------------------------------------------------------
if args.cut is not None:
    print(f"\nAuto-cut mode: extracting ±{args.cut:.1f} ms around pulse...")
    time_series_initial = np.mean(data, axis=0)
    pulse_idx_initial = np.argmax(time_series_initial)
    window_samples = int((args.cut / 1000.0) / metadata['tsamp'])
    start_idx = max(0, pulse_idx_initial - window_samples)
    end_idx   = min(len(times), pulse_idx_initial + window_samples)
    print(f"  Pulse found at sample {pulse_idx_initial} (t={times[pulse_idx_initial]:.6f} s)")
    print(f"  Cutting {start_idx}:{end_idx} ({end_idx-start_idx} samples, "
          f"{(end_idx-start_idx)*metadata['tsamp']*1000:.1f} ms)")
    data  = data[:, start_idx:end_idx]
    times = times[start_idx:end_idx]
    print(f"  Data cut to {data.shape}")

# ---------------------------------------------------------------------------
# Manual RFI flagging on fine grid
# ---------------------------------------------------------------------------
manual_rfi_bands = []
if args.flag_freq is not None:
    for band_str in args.flag_freq.split(','):
        parts = band_str.strip().split('-')
        if len(parts) == 2:
            try:
                fl, fh = float(parts[0]), float(parts[1])
                if fl > fh:
                    fl, fh = fh, fl
                manual_rfi_bands.append((fl, fh))
            except ValueError:
                print(f"  Warning: Could not parse frequency band '{band_str}'")

manual_flag_fine = np.zeros(len(freqs), dtype=bool)
if manual_rfi_bands:
    print(f"\nManual RFI flagging (fine grid):")
    for fl, fh in manual_rfi_bands:
        mask = (freqs >= fl) & (freqs <= fh)
        if mask.any():
            manual_flag_fine |= mask
            print(f"  {fl}-{fh} MHz → {mask.sum()} fine channels flagged")
        else:
            print(f"  Warning: no channels in {fl}-{fh} MHz")

# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------
if FREQ_DOWNSAMPLE > 1 or TIME_DOWNSAMPLE > 1:
    print(f"\nDownsampling: {FREQ_DOWNSAMPLE}× freq, {TIME_DOWNSAMPLE}× time")
    print(f"  Original shape: {data.shape}")
    data_down, nfreq_trim, ntime_trim = downsample_2d(data, FREQ_DOWNSAMPLE, TIME_DOWNSAMPLE)
    freqs_down, _ = downsample_1d(freqs, FREQ_DOWNSAMPLE)
    times_down, _ = downsample_1d(times, TIME_DOWNSAMPLE)
    print(f"  Downsampled shape: {data_down.shape}")
    nfreq_trim_flag = (len(manual_flag_fine) // FREQ_DOWNSAMPLE) * FREQ_DOWNSAMPLE
    rfi_flag_original = manual_flag_fine[:nfreq_trim_flag].reshape(-1, FREQ_DOWNSAMPLE).any(axis=1)
    data, freqs, times = data_down, freqs_down, times_down
else:
    rfi_flag_original = manual_flag_fine.copy()

# Zero flagged channels
if rfi_flag_original.any():
    data[rfi_flag_original, :] = 0
    print(f"\nManual RFI: zeroed {rfi_flag_original.sum()} channels on coarse grid")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
print(f"\nData Statistics:")
print("-" * 70)
print(f"  Shape: {data.shape}")
print(f"  Min: {np.min(data):.6e}  Max: {np.max(data):.6e}")
print(f"  Mean: {np.mean(data):.6e}  Std: {np.std(data):.6e}")
print(f"  Median: {np.median(data):.6e}")

# ---------------------------------------------------------------------------
# Pulse finding
# ---------------------------------------------------------------------------
time_series_raw = np.mean(data, axis=0)
tsamp_us = metadata['tsamp'] * 1e6
if tsamp_us < 10:
    sigma_samples = max(5, int(100 / tsamp_us))
elif tsamp_us < 50:
    sigma_samples = max(3, int(50 / tsamp_us))
else:
    sigma_samples = 2

time_series_smooth = gaussian_filter1d(time_series_raw, sigma=sigma_samples)

print(f"\nPulse search:")
print(f"  Time resolution: {tsamp_us:.2f} µs, smoothing σ={sigma_samples} samples")

peak_idx = np.argmax(time_series_smooth)
baseline_region_size = max(50, len(time_series_smooth) // 3)
baseline_initial = np.median(time_series_smooth[:baseline_region_size])
noise_initial    = np.std(time_series_smooth[:baseline_region_size])
min_sigma = 5 if tsamp_us < 10 else 3
threshold = baseline_initial + min_sigma * noise_initial
peak_significance = (time_series_smooth[peak_idx] - baseline_initial) / noise_initial
print(f"  Peak significance: {peak_significance:.1f} σ")

pulse_mask = time_series_smooth > threshold
if pulse_mask.any():
    pulse_indices = np.where(pulse_mask)[0]
    pulse_start_s = max(0, pulse_indices[0] - 5)
    pulse_end_s   = min(len(time_series_smooth), pulse_indices[-1] + 5)
    pulse_region  = time_series_smooth[pulse_start_s:pulse_end_s]
    pulse_weights = np.maximum(pulse_region - baseline_initial, 0)
    if pulse_weights.sum() > 0:
        pulse_center_idx = pulse_start_s + int(
            np.sum(np.arange(len(pulse_weights)) * pulse_weights) / pulse_weights.sum()
        )
    else:
        pulse_center_idx = peak_idx
else:
    pulse_center_idx = peak_idx

peak_idx = pulse_center_idx
peak_freq_idx = np.argmax(data[:, peak_idx])
times_rel = (times - times[peak_idx]) * 1000  # ms, centred on pulse
print(f"  Pulse centre index: {peak_idx}  freq: {freqs[peak_freq_idx]:.2f} MHz")

# Off-pulse region
off_pulse_start = 0
off_pulse_end = max(min(20, peak_idx - 50), int(len(times) * 0.2))
if off_pulse_end < 10:
    off_pulse_end = min(int(len(times) * 0.2), len(times) - 1)

off_pulse_region = data[:, off_pulse_start:off_pulse_end]
channel_median = np.median(off_pulse_region, axis=1)
channel_std    = np.std(off_pulse_region, axis=1)
channel_max    = np.max(off_pulse_region, axis=1)

valid_channels = (channel_median > 0) & (channel_std > 0) & (~rfi_flag_original)
if valid_channels.sum() > 10:
    m_med = np.median(channel_median[valid_channels])
    s_med = np.std(channel_median[valid_channels])
    m_std = np.median(channel_std[valid_channels])
    s_std = np.std(channel_std[valid_channels])
    m_max = np.median(channel_max[valid_channels])
    s_max = np.std(channel_max[valid_channels])
    rfi_flag = rfi_flag_original.copy()
    rfi_flag |= channel_median > m_med + 10 * s_med
    rfi_flag |= channel_std    > m_std + 10 * s_std
    rfi_flag |= channel_max    > m_max + 10 * s_max
    n_auto = (rfi_flag & ~rfi_flag_original).sum()
    if n_auto > len(freqs) * 0.5:
        print(f"  Auto-RFI would flag {n_auto} extra channels (>50%) – skipping")
        rfi_flag = rfi_flag_original.copy()
else:
    rfi_flag = rfi_flag_original.copy()

n_rfi = rfi_flag.sum()
print(f"\nRFI Flagging: {n_rfi}/{len(freqs)} channels flagged ({100*n_rfi/len(freqs):.1f}%)")

data_clean = data.copy()
if n_rfi > 0:
    data_clean[rfi_flag, :] = 0

# Per-channel normalisation (baseline subtract + MAD normalise)
data_normalized = data_clean.copy()
for i in range(data_clean.shape[0]):
    if not rfi_flag[i]:
        ch = data_clean[i, :]
        off = ch[off_pulse_start:off_pulse_end]
        if len(off) > 5:
            bl  = np.median(off)
            mad = np.median(np.abs(off - bl))
            std = 1.4826 * mad if mad > 0 else np.std(off)
            data_normalized[i, :] = (ch - bl) / std if std > 0 else ch - bl
        else:
            data_normalized[i, :] = 0

# Re-centre on clean peak
good_mask_rc = ~rfi_flag
n_good_rc    = good_mask_rc.sum()
if n_good_rc > 0:
    ts_rc = np.sum(data_normalized[good_mask_rc, :], axis=0) / np.sqrt(n_good_rc)
else:
    ts_rc = np.sum(data_normalized, axis=0) / np.sqrt(data_normalized.shape[0])

peak_idx_final = np.argmax(ts_rc)
times_rel = (times - times[peak_idx_final]) * 1000
print(f"  Re-centred on index {peak_idx_final}")

# Rough pulse width
bl_rough   = np.median(ts_rc[:max(10, peak_idx_final - 50)])
ts_sub_r   = ts_rc - bl_rough
pk_val_r   = ts_sub_r[peak_idx_final]
thr_rough  = pk_val_r * 0.1

ps_rough = peak_idx_final
for i in range(peak_idx_final - 1, -1, -1):
    if ts_sub_r[i] < thr_rough:
        ps_rough = i + 1
        break

pe_rough = peak_idx_final + 1
for i in range(peak_idx_final + 1, len(ts_sub_r)):
    if ts_sub_r[i] < thr_rough:
        pe_rough = i
        break

pulse_width_ms_rough = (pe_rough - ps_rough) * metadata['tsamp'] * 1000

# ---------------------------------------------------------------------------
# Zoom window
# ---------------------------------------------------------------------------
if args.symmetric_plot:
    half_window = args.window_size * pulse_width_ms_rough
    zoom_start, zoom_end = -half_window, half_window
else:
    total_time_ms = times_rel[-1] - times_rel[0]
    if total_time_ms < 50:
        window_ms = min(20, total_time_ms * 0.8)
    elif total_time_ms < 100:
        window_ms = 30
    else:
        window_ms = min(100, total_time_ms * 0.3)
    zoom_start = max(times_rel[0],  -window_ms * 0.3)
    zoom_end   = min(times_rel[-1],  window_ms * 0.7)

zoom_indices = (times_rel >= zoom_start) & (times_rel <= zoom_end)
data_zoom  = data_normalized[:, zoom_indices]
times_zoom = times_rel[zoom_indices]

# ---------------------------------------------------------------------------
# Optionally save processed FITS
# ---------------------------------------------------------------------------
if args.save_fits:
    zoom_sample_indices = np.where(zoom_indices)[0]
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = []
    if dm_value:           parts.append(f'DM{dm_value:.4f}')
    if FREQ_DOWNSAMPLE > 1: parts.append(f'f{FREQ_DOWNSAMPLE}')
    if TIME_DOWNSAMPLE > 1: parts.append(f't{TIME_DOWNSAMPLE}')
    parts.append('win')
    out_fits = f"{base}_{'_'.join(parts)}.fits"
    new_tsamp = metadata['tsamp'] * TIME_DOWNSAMPLE
    tstart_window = metadata['tstart'] + times[zoom_sample_indices[0]] / 86400.0
    write_psrfits_simple(
        out_fits,
        data_clean[:, zoom_indices],
        freqs,
        new_tsamp,
        tstart_window,
        source_name=metadata.get('source_name', 'Unknown'),
        telescope=metadata.get('telescope', 'Unknown'),
    )

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9))
gs  = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 3],
                        hspace=0.05, wspace=0.05)
ax2 = fig.add_subplot(gs[0, 0])               # time series
ax1 = fig.add_subplot(gs[1, 0], sharex=ax2)  # dynamic spectrum
ax3 = fig.add_subplot(gs[1, 1], sharey=ax1)  # frequency spectrum

# Dynamic spectrum
data_zoom_nz = data_zoom[data_zoom != 0]
if len(data_zoom_nz) > 0:
    vmax = np.percentile(data_zoom_nz, 99.5)
    vmin, vmax = -2, max(vmax, 5)
else:
    vmin, vmax = data_zoom.min(), data_zoom.max()

im = ax1.imshow(data_zoom, aspect='auto', origin='upper',
                extent=[times_zoom[0], times_zoom[-1], freqs[-1], freqs[0]],
                interpolation='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
ax1.set_ylabel('Frequency (MHz)')
ax1.set_xlabel('Time (ms)')
ax2.tick_params(labelbottom=False)

# Time series
good_mask = ~rfi_flag
n_good_channels = good_mask.sum()
if n_good_channels > 0:
    time_series_full = np.sum(data_normalized[good_mask, :], axis=0) / np.sqrt(n_good_channels)
else:
    time_series_full = np.sum(data_normalized, axis=0) / np.sqrt(data_normalized.shape[0])

off_pulse_full = time_series_full[off_pulse_start:off_pulse_end]
baseline_full  = np.median(off_pulse_full)
noise_full     = np.std(off_pulse_full)

time_series_zoom = time_series_full[zoom_indices]
tsamp_ms         = float(times_zoom[1] - times_zoom[0])

# Boxcar matched filter
time_series_sub = time_series_zoom - baseline_full
peak_idx_zoom   = np.argmax(time_series_sub)
max_width       = min(200, len(time_series_sub) // 2)
search_start    = max(0, peak_idx_zoom - max_width)
search_end      = min(len(time_series_sub), peak_idx_zoom + max_width)

best_snr, best_width, best_start = 0, 1, peak_idx_zoom
for width in range(1, max_width + 1):
    for start in range(search_start, min(search_end - width + 1,
                                         len(time_series_sub) - width + 1)):
        snr = np.sum(time_series_sub[start:start+width]) / (noise_full * np.sqrt(width))
        if snr > best_snr:
            best_snr, best_width, best_start = snr, width, start

pulse_start = best_start
pulse_end   = best_start + best_width
pulse_start_time = times_zoom[pulse_start] - tsamp_ms
pulse_end_time   = times_zoom[pulse_end - 1] + tsamp_ms
pulse_width_ms   = pulse_end_time - pulse_start_time
peak_snr         = best_snr

ax2.plot(times_zoom, time_series_zoom, 'k-', linewidth=0.8)
ax2.set_ylabel('Mean Intensity')
ax2.set_xlim(times_zoom[0], times_zoom[-1])
ax2.grid(True, alpha=0.3)
ax2.axvline(0, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvspan(pulse_start_time, pulse_end_time, alpha=0.2, color='orange',
            label=f'Width ({pulse_width_ms:.2f} ms)')
ax2.legend(loc='upper right', fontsize=12)

# Title
base_name  = os.path.splitext(os.path.basename(filename))[0]
telescope  = metadata.get('telescope', '')
freq_res   = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 0
title_l1   = base_name + (f" ({telescope})" if telescope else "")
title_parts = []
if dm_value is not None:
    title_parts.append(f"DM={dm_value:.2f} pc cm$^{{-3}}$")
title_parts += [f"S/N~{peak_snr:.1f}", f"W~{pulse_width_ms:.2f} ms",
                f"dt={tsamp_ms:.4f} ms", f"df={freq_res:.3f} MHz"]
ax2.set_title(title_l1 + '\n' + ' - '.join(title_parts))

ax1.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.tick_params(which='minor', length=3, width=0.5)
ax1.tick_params(which='major', length=6, width=1)

# Frequency spectrum
freq_spectrum = np.mean(data_zoom[:, pulse_start:pulse_end], axis=1)
for fl, fh in manual_rfi_bands:
    freq_spectrum[(freqs >= fl) & (freqs <= fh)] = 0

ax3.plot(freq_spectrum, freqs, 'k-', linewidth=0.8)
ax3.set_xlabel('Mean Intensity')
ax3.set_ylim(freqs.min(), freqs.max())
ax3.grid(True, alpha=0.3)
ax3.yaxis.tick_right()
ax3.tick_params(labelleft=False)

plt.tight_layout()

if args.save_png:
    save_fn = f"{os.path.splitext(os.path.basename(filename))[0]}_waterfall.png"
    plt.savefig(save_fn, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {save_fn}')

plt.show()