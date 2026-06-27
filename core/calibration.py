"""
calibration.py
Calibration routines for Dicke-switched HI observations.

The Dicke switch alternates between the sky (antenna) and a 50-ohm
reference load.  Dividing antenna power by reference power removes
common-mode gain variations and the receiver bandpass shape, leaving
only the sky signal.

Key operations
--------------
- Compute calibrated ratio spectra  (P_ant / P_ref)
- Estimate system temperature (Tsys) from reference power
- Remove the bandpass (polynomial baseline subtraction)
- Flag and mask RFI channels
- Stack (average) multiple calibrated spectra

Typical usage
-------------
    from core.reader import load_files
    from core.calibration import calibrate, stack_spectra, flag_rfi

    obs_files = load_files(paths)
    cal = calibrate(obs_files[0])          # shape (n_pairs, 256)
    flagged = flag_rfi(cal)                # NaN in bad channels
    stacked = stack_spectra(flagged)       # shape (256,)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import medfilt


# ---------------------------------------------------------------------------
# Per-file calibration
# ---------------------------------------------------------------------------

def calibrate(obs_file, subtract_baseline: bool = False,
              poly_order: int = 5) -> np.ndarray:
    """
    Compute calibrated spectra for all pairs in an ObsFile.

    Steps:
    1. P_ant / P_ref  — removes gain + bandpass shape
    2. Optional polynomial baseline subtraction to remove residual
       bandpass curvature, leaving only spectral features

    Parameters
    ----------
    obs_file         : ObsFile from reader.load_file()
    subtract_baseline: if True, fit and subtract a polynomial baseline
    poly_order       : order of the polynomial fit

    Returns
    -------
    np.ndarray of shape (n_pairs, freq_bins) — calibrated ratios
    """
    stack = obs_file.calibrated_stack()   # shape (n_pairs, freq_bins)

    if subtract_baseline:
        stack = _subtract_baseline(stack, poly_order=poly_order)

    return stack


def _subtract_baseline(spectra: np.ndarray, poly_order: int = 5,
                        edge_frac: float = 0.15) -> np.ndarray:
    """
    Fit and subtract a polynomial baseline from each spectrum.

    Uses the edge channels (where HI signal is absent) to anchor the fit,
    avoiding the HI emission region in the centre of the band.

    Parameters
    ----------
    spectra   : shape (n_spectra, n_channels)
    poly_order: polynomial degree
    edge_frac : fraction of channels at each edge used for fitting

    Returns
    -------
    Baseline-subtracted spectra, same shape as input.
    """
    n_spec, n_chan = spectra.shape
    n_edge = max(4, int(n_chan * edge_frac))
    x = np.arange(n_chan, dtype=float)

    # Channels used for baseline fit: edges only
    fit_mask = np.zeros(n_chan, dtype=bool)
    fit_mask[:n_edge] = True
    fit_mask[-n_edge:] = True

    result = np.empty_like(spectra)
    for i in range(n_spec):
        y = spectra[i]
        valid = fit_mask & np.isfinite(y)
        if valid.sum() < poly_order + 1:
            result[i] = y
            continue
        coeffs = np.polyfit(x[valid], y[valid], poly_order)
        baseline = np.polyval(coeffs, x)
        result[i] = y - baseline

    return result


# ---------------------------------------------------------------------------
# RFI flagging
# ---------------------------------------------------------------------------

def flag_rfi(spectra: np.ndarray,
             sigma_threshold: float = 4.0,
             kernel_size: int = 11) -> np.ndarray:
    """
    Flag RFI-contaminated channels by replacing them with NaN.

    Method: compare each channel value to a median-filtered version of
    the spectrum.  Channels that deviate by more than sigma_threshold *
    MAD (median absolute deviation) are flagged.

    Parameters
    ----------
    spectra         : shape (n_spectra, n_channels)
    sigma_threshold : flag channels > this many sigma above median
    kernel_size     : median filter window size (must be odd)

    Returns
    -------
    Spectra with flagged channels set to NaN.
    """
    result = spectra.copy().astype(float)

    for i in range(result.shape[0]):
        y = result[i]
        if not np.any(np.isfinite(y)):
            continue

        # Median-filtered baseline as RFI-free reference
        y_med = medfilt(np.where(np.isfinite(y), y, np.nanmedian(y)),
                        kernel_size=kernel_size)
        residual = y - y_med

        # MAD-based threshold
        mad = np.nanmedian(np.abs(residual - np.nanmedian(residual)))
        sigma = 1.4826 * mad   # converts MAD to equivalent Gaussian sigma

        if sigma > 0:
            bad = np.abs(residual) > sigma_threshold * sigma
            result[i, bad] = np.nan

    return result


def flag_persistent_rfi(spectra: np.ndarray,
                         threshold_frac: float = 0.3) -> np.ndarray:
    """
    Flag channels that are bad in more than threshold_frac of spectra.

    Persistent RFI (like from WiFi or LTE) shows up in the same channels
    every integration.  This flags those channels across all spectra.

    Parameters
    ----------
    spectra        : shape (n_spectra, n_channels) — may contain NaNs
    threshold_frac : flag channel if bad in > this fraction of spectra

    Returns
    -------
    Spectra with persistently bad channels set to NaN.
    """
    result = spectra.copy()
    nan_frac = np.mean(~np.isfinite(spectra), axis=0)
    persistent = nan_frac > threshold_frac
    result[:, persistent] = np.nan
    return result


# ---------------------------------------------------------------------------
# Stacking (averaging)
# ---------------------------------------------------------------------------

def stack_spectra(spectra: np.ndarray,
                  weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Average multiple calibrated spectra into a single spectrum.

    Uses nanmean so NaN (flagged) channels don't contaminate the average.
    Optionally applies per-spectrum weights (e.g. inverse variance).

    Parameters
    ----------
    spectra : shape (n_spectra, n_channels)
    weights : shape (n_spectra,) optional per-spectrum weights

    Returns
    -------
    np.ndarray of shape (n_channels,) — averaged spectrum
    """
    if weights is None:
        return np.nanmean(spectra, axis=0)

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    weighted = spectra * weights[:, np.newaxis]
    # nansum / sum of weights for channels with valid data
    valid_weights = np.where(np.isfinite(spectra),
                             weights[:, np.newaxis], 0.0)
    w_sum = valid_weights.sum(axis=0)
    result = np.nansum(weighted, axis=0)
    result = np.where(w_sum > 0, result / w_sum, np.nan)
    return result


def stack_obs_files(obs_files: list,
                    subtract_baseline: bool = True,
                    flag: bool = True,
                    sigma_threshold: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrate, flag, and stack all pairs from a list of ObsFile objects.

    Intended for combining multiple nights at the same elevation to
    improve SNR by sqrt(N_nights).

    Parameters
    ----------
    obs_files        : list of ObsFile (should all be same elevation)
    subtract_baseline: apply polynomial baseline subtraction
    flag             : apply RFI flagging
    sigma_threshold  : RFI flagging threshold in sigma

    Returns
    -------
    (stacked_spectrum, freq_axis_mhz)
    stacked_spectrum : shape (n_channels,)
    freq_axis_mhz   : shape (n_channels,)
    """
    all_spectra = []

    for obs in obs_files:
        cal = calibrate(obs, subtract_baseline=subtract_baseline)
        if flag:
            cal = flag_rfi(cal, sigma_threshold=sigma_threshold)
            cal = flag_persistent_rfi(cal)
        all_spectra.append(cal)

    # Combine all spectra from all files
    combined = np.vstack(all_spectra)
    stacked  = stack_spectra(combined)

    # Use frequency axis from first file
    freq_axis = obs_files[0].header.freq_axis_mhz

    return stacked, freq_axis


# ---------------------------------------------------------------------------
# Tsys estimation
# ---------------------------------------------------------------------------

def estimate_tsys(obs_file,
                  t_load_k: float = 290.0) -> np.ndarray:
    """
    Estimate system temperature from reference (50-ohm load) samples.

    Using the Y-factor method:
        T_sys = T_load / (P_ant/P_ref - 1)

    This is a rough estimate — a proper Tsys measurement requires
    observations of a source of known temperature (hot/cold load or
    a calibrator source).

    Parameters
    ----------
    obs_file  : ObsFile
    t_load_k  : physical temperature of the 50-ohm load (K)
                Assumes room temperature ≈ 290K

    Returns
    -------
    np.ndarray of shape (n_pairs,) — Tsys estimate per integration
    """
    tsys_estimates = []
    for pair in obs_file.pairs:
        ratio = pair.calibrated          # P_ant / P_ref per channel
        # Y-factor: Y = P_ant/P_ref = (T_sky + T_sys) / (T_load + T_sys)
        # Solving for T_sys at each channel, then take median
        y = np.nanmedian(ratio)
        if y > 1.0:
            tsys = t_load_k / (y - 1.0)
        else:
            tsys = np.nan
        tsys_estimates.append(tsys)

    return np.array(tsys_estimates)


def rms_noise(spectrum: np.ndarray) -> float:
    """
    Estimate the RMS noise in a spectrum from the edge channels
    (assumed to contain only noise, no HI signal).

    Parameters
    ----------
    spectrum : shape (n_channels,)

    Returns
    -------
    float — RMS noise level
    """
    n = len(spectrum)
    n_edge = max(4, n // 10)
    edge = np.concatenate([spectrum[:n_edge], spectrum[-n_edge:]])
    valid = edge[np.isfinite(edge)]
    if len(valid) < 4:
        return np.nan
    return float(np.std(valid))


# ---------------------------------------------------------------------------
# Multi-night sidereal stacking
# ---------------------------------------------------------------------------

def sidereal_stack(obs_files: list,
                   ra_bin_deg: float = 0.5,
                   flag: bool = True,
                   sigma_threshold: float = 4.0) -> tuple:
    """
    Stack multiple observations by aligning on RA (sidereal time).

    For drift-scan observations at the same elevation, the beam sweeps
    the same strip of sky at the same sidereal time each night. By
    binning spectra into RA slots and averaging across nights, we build
    up SNR exactly as in astrophotography stacking — noise reduces by
    sqrt(N_nights) while the sky signal accumulates coherently.

    Parameters
    ----------
    obs_files        : list of ObsFile (should all be same elevation)
    ra_bin_deg       : width of each RA bin in degrees
                       (~0.5° ≈ 2 minutes of RA, matches ~20s integration
                       at the Earth's 15°/hour rotation rate)
    flag             : apply RFI flagging before stacking
    sigma_threshold  : RFI flagging sigma threshold

    Returns
    -------
    (ra_centres, stacked_spectra, counts, freq_axis)

    ra_centres      : np.ndarray shape (n_bins,) — RA of each bin (degrees)
    stacked_spectra : np.ndarray shape (n_bins, n_channels) — mean spectrum
    counts          : np.ndarray shape (n_bins,) — observations per bin
    freq_axis       : np.ndarray shape (n_channels,) — frequency axis (MHz)
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.spectrum import assign_radec

    # Collect all calibrated SpectrumRecords from all files
    all_records = []
    for obs in obs_files:
        cal = calibrate(obs, subtract_baseline=False)
        if flag:
            cal = flag_rfi(cal, sigma_threshold=sigma_threshold)
            cal = flag_persistent_rfi(cal)
        records = assign_radec(obs)
        for i, rec in enumerate(records):
            if i < len(cal):
                rec.spectrum = cal[i]
        all_records.extend(records)

    if not all_records:
        raise ValueError("No records found in obs_files")

    freq_axis = obs_files[0].header.freq_axis_mhz
    n_chan = len(freq_axis)

    # Build RA bins covering 0–360°
    ra_edges   = np.arange(0, 360 + ra_bin_deg, ra_bin_deg)
    ra_centres = (ra_edges[:-1] + ra_edges[1:]) / 2.0
    n_bins     = len(ra_centres)

    accum  = np.zeros((n_bins, n_chan), dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for rec in all_records:
        i_bin = int(rec.ra_deg / ra_bin_deg)
        if 0 <= i_bin < n_bins:
            valid = np.isfinite(rec.spectrum)
            accum[i_bin, valid]  += rec.spectrum[valid]
            counts[i_bin]        += 1

    # Average — NaN where no data
    with np.errstate(invalid="ignore", divide="ignore"):
        stacked = np.where(counts[:, np.newaxis] > 0,
                           accum / counts[:, np.newaxis],
                           np.nan)

    return ra_centres, stacked, counts, freq_axis