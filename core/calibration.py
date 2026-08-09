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

def bandpass_correct(spectra: np.ndarray,
                     edge_frac: float = 0.15) -> np.ndarray:
    """
    Remove residual bandpass slope by dividing each spectrum by a
    normalized median bandpass template built from the edge channels.

    This is the radio equivalent of flat-fielding in optical imaging.
    The edge channels contain no HI signal, so their shape reflects
    only the receiver bandpass. We fit a low-order polynomial to the
    median spectrum using only those edge channels, then divide every
    spectrum by that smooth template.

    Parameters
    ----------
    spectra   : shape (n_spectra, n_channels) — calibrated sky/ref ratios
    edge_frac : fraction of channels at each edge used to anchor the fit

    Returns
    -------
    Corrected spectra, same shape as input.
    """
    n_spec, n_chan = spectra.shape
    n_edge = max(4, int(n_chan * edge_frac))
    x = np.arange(n_chan, dtype=float)

    # Median spectrum across all integrations — robust against RFI
    median_spec = np.nanmedian(spectra, axis=0)

    # Fit a polynomial using only edge channels (no HI signal there)
    fit_mask = np.zeros(n_chan, dtype=bool)
    fit_mask[:n_edge] = True
    fit_mask[-n_edge:] = True
    valid = fit_mask & np.isfinite(median_spec)

    if valid.sum() < 4:
        return spectra  # not enough points, return unchanged

    coeffs = np.polyfit(x[valid], median_spec[valid], 3)
    bandpass = np.polyval(coeffs, x)

    # Normalize so we divide by shape, not level
    bandpass_norm = bandpass / np.mean(bandpass)

    # Guard against near-zero values
    bandpass_norm = np.where(np.abs(bandpass_norm) < 0.01, 1.0, bandpass_norm)

    return spectra / bandpass_norm[np.newaxis, :]


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
        cal = bandpass_correct(cal)          # <-- add this line
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
                  t_load_k: float = 290.0,
                  t_sky_k: float = 15.0) -> np.ndarray:
    """
    Estimate system temperature from reference (50-ohm load) samples.

    Using the Y-factor method, with Y defined the way this code actually
    computes it (Y = P_ant/P_ref):
        Y = (T_sky + T_sys) / (T_load + T_sys)

    Solving for T_sys:
        T_sys = (Y*T_load - T_sky) / (1 - Y)

    NOTE: an earlier version of this function solved the algebra for the
    opposite convention (Y = P_ref/P_ant, i.e. assuming the reference is
    colder than the antenna). For this setup the sky is colder than the
    290K reference load, so the actual ratio Y = P_ant/P_ref is always
    < 1 -- the old formula's "if y > 1.0" guard silently discarded every
    single estimate as a result. This is a rough estimate; a proper Tsys
    measurement still requires observations of a source of known
    temperature (hot/cold load or a calibrator source).

    Parameters
    ----------
    obs_file  : ObsFile
    t_load_k  : physical temperature of the 50-ohm load (K)
                Assumes room temperature ~= 290K
    t_sky_k   : assumed sky temperature (K), away from strong sources.
                ~10-20K is typical for 21cm continuum + CMB away from
                the galactic plane; adjust if you have a better estimate
                for your specific pointing.

    Returns
    -------
    np.ndarray of shape (n_pairs,) — Tsys estimate per integration
    """
    tsys_estimates = []
    for pair in obs_file.pairs:
        ratio = pair.calibrated          # P_ant / P_ref per channel
        y = np.nanmedian(ratio)
        if 0 < y < 1.0:
            tsys = (y * t_load_k - t_sky_k) / (1.0 - y)
        else:
            tsys = np.nan
        tsys_estimates.append(tsys)

    return np.array(tsys_estimates)


# ---------------------------------------------------------------------------
# Continuum (total power) light curves
# ---------------------------------------------------------------------------

def continuum_lightcurve(obs_file, edge_frac: float = 0.1,
                          exclude_hi_line: bool = True,
                          hi_line_frac: float = 0.34,
                          flag: bool = True,
                          sigma_threshold: float = 4.0) -> np.ndarray:
    """
    Build a total-power (broadband continuum) light curve from a single
    drift-scan observation, for detecting bright compact sources as they
    transit the beam -- Cas A, Cyg A, Tau A, the Sun, etc.

    This is deliberately different from the HI spectral-line pipeline:
    it does NOT apply per-row DC-offset removal or bandpass shape
    correction. For HI work the DC level is a nuisance to remove so the
    tiny (~1-2%) line signal is visible; for continuum work the DC level
    of the calibrated ratio (P_ant/P_ref) IS the signal -- it rises when
    a strong source enters the beam and falls when it leaves. Removing
    it would remove exactly what you're trying to measure.

    IMPORTANT: the galactic HI line itself varies strongly with RA/Dec
    (that's the whole HI sky map), and its excess power is typically
    LARGER than a compact continuum source's signal. If it's left in
    the average, it swamps and can easily be mistaken for a genuine
    continuum detection. By default the middle hi_line_frac of the band
    (where the HI line lives, same region excluded from the bandpass
    fit elsewhere in this codebase) is excluded, leaving only genuinely
    line-free channels for the continuum estimate.

    Channel-level RFI flagging is still safe to apply here: a genuine
    continuum source raises ALL channels together (it's broadband), so
    it doesn't look like an outlier relative to a smooth per-row
    baseline the way narrowband RFI does, and won't get flagged.

    Parameters
    ----------
    obs_file        : ObsFile from reader.load_file()
    edge_frac       : fraction of channels at each band edge to exclude
                       from the average (edge roll-off is instrumental,
                       not sky signal)
    exclude_hi_line : also exclude the middle hi_line_frac of the band,
                       where real HI emission lives. Strongly recommended
                       -- only disable if you specifically want the
                       combined HI+continuum power for some reason.
    hi_line_frac    : fraction of the band (centered on the middle)
                       excluded as the HI line region
    flag            : apply channel-level RFI flagging before averaging
    sigma_threshold : RFI flagging threshold in sigma

    Returns
    -------
    np.ndarray of shape (n_pairs,) -- broadband-averaged ratio per
    integration, in the same units as the calibrated ratio (P_ant/P_ref,
    dimensionless). NaN for any integration with no valid channels.
    """
    cal = calibrate(obs_file, subtract_baseline=False)
    if flag:
        cal = flag_rfi(cal, sigma_threshold=sigma_threshold)
        cal = flag_persistent_rfi(cal)

    n_chan = cal.shape[1]
    n_edge = max(1, int(n_chan * edge_frac))
    band_mask = np.ones(n_chan, dtype=bool)
    band_mask[:n_edge] = False
    band_mask[-n_edge:] = False

    if exclude_hi_line:
        half_width = int(n_chan * hi_line_frac / 2)
        mid = n_chan // 2
        band_mask[mid - half_width: mid + half_width] = False

    with np.errstate(invalid="ignore"):
        return np.nanmean(cal[:, band_mask], axis=1)


def flux_calibrate(values: np.ndarray, baseline: float,
                    tsys_k: float, aperture_m2: float,
                    t_load_k: float = 290.0) -> np.ndarray:
    """
    Convert a continuum light curve (raw P_ant/P_ref ratio) into an
    implied flux density in Jy, relative to a given off-source baseline.

    Derivation: with Y = P_ant/P_ref = (T_sky+T_sys)/(T_load+T_sys),
    a source contributing extra antenna temperature dT above the
    baseline shows up as dY = dT / (T_load+T_sys). A point source of
    flux density S contributes dT = S*A_eff/(2k). Combining:

        S = (Y - baseline) * (T_load + T_sys) * 2k / A_eff

    This assumes the source is unresolved (smaller than the beam) and
    that aperture_m2 is the EFFECTIVE aperture (physical area x
    aperture efficiency), not the physical dish area.

    Parameters
    ----------
    values      : light curve, shape (n,), raw ratio units
    baseline    : off-source reference ratio (e.g. from the quiet part
                  of the same scan)
    tsys_k      : system temperature (K)
    aperture_m2 : effective aperture (m^2)
    t_load_k    : reference load temperature (K)

    Returns
    -------
    np.ndarray of shape (n,) -- implied flux density in Jy
    """
    k = 1.380649e-23  # Boltzmann constant, J/K
    dT = (values - baseline) * (t_load_k + tsys_k)
    S_W = dT * 2.0 * k / aperture_m2       # W / m^2 / Hz
    return S_W / 1e-26                      # Jy


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