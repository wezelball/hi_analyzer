"""
continuum.py
Total-power (continuum) drift-scan light curve display, for detecting
bright compact radio sources (Cas A, Cyg A, Tau A, the Sun, etc.) as
they transit a fixed Az/El beam.

Typical usage
-------------
    from core.reader import load_file
    from core.calibration import continuum_lightcurve, estimate_tsys, flux_calibrate
    from core.spectrum import assign_radec
    from ui.continuum import ContinuumLightCurve

    obs = load_file(path)
    values = continuum_lightcurve(obs)
    records = assign_radec(obs)
    ra = [r.ra_deg for r in records]
    times = [r.timestamp for r in records]

    lc = ContinuumLightCurve(times, ra, values, source_name="Cas A")
    lc.show()
"""

from __future__ import annotations

from typing import List, Optional, Sequence
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Style (matches skymap.py)
# ---------------------------------------------------------------------------

DARK_BG    = "#0D1117"
TEXT_COLOR = "#E2E8F0"
GRID_COLOR = "#2D3748"
LINE_COLOR = "#F0A030"
PEAK_COLOR = "#00BFFF"


class ContinuumLightCurve:
    """
    Total-power light curve for a single drift-scan pointing.

    Parameters
    ----------
    times        : sequence of UTC datetimes, one per integration
    ra_deg       : sequence of RA (degrees), one per integration
    values       : sequence of broadband-averaged ratio values (from
                   calibration.continuum_lightcurve()), one per integration
    source_name  : optional label, e.g. "Cas A" -- used in the title
    az_hpbw_deg  : antenna azimuth HPBW (degrees), used to auto-size the
                   despeckling smoothing kernel to the real beam width,
                   the same approach used in skymap.py
    smooth_kernel: override the smoothing kernel size (in samples).
                   If None, computed from az_hpbw_deg and the actual
                   sample cadence.
    tsys_k       : system temperature (K), for flux calibration
    aperture_m2  : effective aperture (m^2), for flux calibration
    t_load_k     : reference load temperature (K)
    """

    def __init__(
        self,
        times: Sequence[datetime],
        ra_deg: Sequence[float],
        values: Sequence[float],
        source_name: Optional[str] = None,
        az_hpbw_deg: float = 15.0,
        smooth_kernel: Optional[int] = None,
        tsys_k: float = 400.0,
        aperture_m2: float = 0.283,
        t_load_k: float = 290.0,
        expected_source_ra: Optional[float] = None,
        exclusion_halfwidth_deg: Optional[float] = None,
    ):
        self.times   = list(times)
        self.ra_deg  = np.asarray(ra_deg, dtype=float)
        self.values  = np.asarray(values, dtype=float)
        self.source_name = source_name
        self.tsys_k     = tsys_k
        self.aperture_m2 = aperture_m2
        self.t_load_k    = t_load_k
        # If you know where the source SHOULD be (e.g. from its catalog
        # RA), pass it here. This switches to trend-aware detection：fit
        # a smooth baseline from data OUTSIDE this region, then test
        # whether the region around expected_source_ra is significantly
        # above that trend -- rather than just reporting wherever the
        # highest point happens to be, which is easily fooled by slow
        # gain drift over the course of a scan (a real risk for total
        # power measurements, since -- unlike the HI pipeline -- the DC
        # level can't be removed without removing the signal itself).
        self.expected_source_ra = expected_source_ra
        self.exclusion_halfwidth_deg = exclusion_halfwidth_deg or (az_hpbw_deg / 2.0)

        if smooth_kernel is None:
            # Estimate sample cadence from the timestamps to convert the
            # real beam width (degrees) into a kernel size (samples),
            # same physical reasoning as skymap.py's RA smoothing.
            if len(self.times) > 1:
                dt_s = (self.times[-1] - self.times[0]).total_seconds() / max(1, len(self.times) - 1)
                deg_per_sample = dt_s * 15.0 / 3600.0  # sidereal ~15 deg/hr
                smooth_kernel = max(1, int(round(az_hpbw_deg / max(deg_per_sample, 1e-6))))
                if smooth_kernel % 2 == 0:
                    smooth_kernel += 1
            else:
                smooth_kernel = 1
        self.smooth_kernel = smooth_kernel

        self._smoothed: Optional[np.ndarray] = None
        self._baseline: Optional[float] = None
        self._peak_idx: Optional[int] = None
        self._trend: Optional[np.ndarray] = None
        self._snr: Optional[float] = None
        self._detection_value: Optional[float] = None

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self) -> np.ndarray:
        """
        Despeckle the raw light curve (median filter, matched to the
        real beam width) and estimate an off-source baseline.

        If expected_source_ra was given, ALSO fits a smooth trend from
        data outside the expected source region and tests whether that
        region is significantly above the trend (self.snr). This is the
        honest way to check for a specific known source -- it doesn't
        get fooled by slow drift over the scan, and it tells you when a
        result is NOT statistically significant rather than just
        reporting the highest point as if it were a detection.

        Returns
        -------
        Smoothed light curve, same shape as self.values.
        """
        values = self.values.copy()
        valid = np.isfinite(values)

        if self.smooth_kernel > 1 and valid.sum() >= self.smooth_kernel:
            from scipy.ndimage import median_filter
            filled = np.where(valid, values, np.nanmedian(values[valid]))
            smoothed = median_filter(filled, size=self.smooth_kernel, mode="nearest")
            smoothed = np.where(valid, smoothed, np.nan)
        else:
            smoothed = values
        self._smoothed = smoothed

        if self.expected_source_ra is not None:
            self._process_trend_aware(values, valid)
        else:
            # Simple mode (no known target): bottom 20th percentile as
            # baseline. Fine for exploratory scans; for a specific known
            # source, pass expected_source_ra instead.
            finite = smoothed[np.isfinite(smoothed)]
            self._baseline = float(np.percentile(finite, 20)) if finite.size else np.nan
            self._peak_idx = int(np.nanargmax(smoothed)) if finite.size else None

        return smoothed

    def _process_trend_aware(self, values: np.ndarray, valid: np.ndarray) -> None:
        near_source = np.abs(self.ra_deg - self.expected_source_ra) < self.exclusion_halfwidth_deg
        fit_mask = valid & ~near_source

        if fit_mask.sum() < 5:
            # Not enough off-source data to fit a trend; fall back to a
            # flat baseline (median of everything outside the exclusion
            # zone, or everything if that's also too small)
            ref = valid & ~near_source if (valid & ~near_source).sum() >= 3 else valid
            trend_const = float(np.median(values[ref])) if ref.sum() else np.nan
            self._trend = np.full_like(values, trend_const)
        else:
            coeffs = np.polyfit(self.ra_deg[fit_mask], values[fit_mask], 2)
            self._trend = np.polyval(coeffs, self.ra_deg)

        residual = values - self._trend
        noise_std = float(np.nanstd(residual[fit_mask])) if fit_mask.sum() >= 5 else np.nan

        in_zone = valid & near_source
        n_in_zone = int(in_zone.sum())
        mean_residual = float(np.nanmean(residual[in_zone])) if n_in_zone else np.nan

        # SNR of the AVERAGE residual within the expected source region,
        # not of any single sample -- averaging n_in_zone independent-ish
        # samples reduces the noise by roughly sqrt(n_in_zone), same
        # logic as everywhere else in this codebase.
        if np.isfinite(noise_std) and noise_std > 0 and n_in_zone > 0:
            self._snr = mean_residual / (noise_std / np.sqrt(n_in_zone))
        else:
            self._snr = np.nan

        self._detection_value = mean_residual
        # baseline for flux_calibrate purposes: local trend value AT the
        # expected source RA (not a flat percentile, since we've already
        # detrended)
        self._baseline = float(np.polyval(np.polyfit(self.ra_deg[fit_mask], values[fit_mask], 2),
                                           self.expected_source_ra)) if fit_mask.sum() >= 5 else np.nan
        self._peak_idx = int(np.argmin(np.abs(self.ra_deg - self.expected_source_ra))) if n_in_zone else None

    @property
    def snr(self) -> float:
        """Detection significance (only meaningful if expected_source_ra was given)."""
        if self._smoothed is None:
            self.process()
        return self._snr if self._snr is not None else np.nan

    @property
    def detection_value(self) -> float:
        """Mean residual (raw ratio units) within the expected source region."""
        if self._smoothed is None:
            self.process()
        return self._detection_value if self._detection_value is not None else np.nan

    @property
    def baseline(self) -> float:
        if self._baseline is None:
            self.process()
        return self._baseline

    @property
    def peak_value(self) -> float:
        if self._smoothed is None:
            self.process()
        return float(self._smoothed[self._peak_idx]) if self._peak_idx is not None else np.nan

    @property
    def peak_ra(self) -> float:
        if self._peak_idx is None:
            self.process()
        return float(self.ra_deg[self._peak_idx]) if self._peak_idx is not None else np.nan

    @property
    def peak_time(self):
        if self._peak_idx is None:
            self.process()
        return self.times[self._peak_idx] if self._peak_idx is not None else None

    @property
    def peak_flux_jy(self) -> float:
        """
        Implied flux density (Jy), above baseline.

        In trend-aware mode (expected_source_ra given), this uses the
        mean residual across the whole expected source region rather
        than a single peak sample -- much less sensitive to one noisy
        point, and it's what self.snr's significance actually refers to.
        Check self.snr before trusting this as a real detection rather
        than noise.
        """
        from core.calibration import flux_calibrate
        if self.expected_source_ra is not None:
            value = self.detection_value + self.baseline
        else:
            value = self.peak_value
        s = flux_calibrate(np.array([value]), self.baseline,
                            self.tsys_k, self.aperture_m2, self.t_load_k)
        return float(s[0])

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def render(self, figsize=(12, 5), title: Optional[str] = None) -> Figure:
        if self._smoothed is None:
            self.process()

        fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)

        # Sort by RA for a clean left-to-right transit plot (raw file
        # order is already chronological/RA-monotonic for a drift scan,
        # but sorting is cheap insurance against wraparound at RA=0/360)
        order = np.argsort(self.ra_deg)
        ra_sorted = self.ra_deg[order]
        raw_sorted = self.values[order]
        smooth_sorted = self._smoothed[order]

        ax.plot(ra_sorted, raw_sorted, color=LINE_COLOR, alpha=0.25,
                linewidth=0.6, label="raw")
        ax.plot(ra_sorted, smooth_sorted, color=LINE_COLOR, linewidth=1.6,
                label=f"smoothed (kernel={self.smooth_kernel})")

        if self.expected_source_ra is not None:
            trend_sorted = self._trend[order]
            ax.plot(ra_sorted, trend_sorted, color=GRID_COLOR, linewidth=1.0,
                    linestyle="--", label="fitted trend (off-source)")
            lo = self.expected_source_ra - self.exclusion_halfwidth_deg
            hi = self.expected_source_ra + self.exclusion_halfwidth_deg
            ax.axvspan(lo, hi, color=PEAK_COLOR, alpha=0.08)
            ax.axvline(x=self.expected_source_ra, color=PEAK_COLOR,
                       linewidth=0.8, linestyle=":", alpha=0.8)

            snr = self.snr
            sig_word = "SIGNIFICANT" if np.isfinite(snr) and abs(snr) >= 3 else "not significant"
            label = (f"{self.source_name or 'source'} region: "
                     f"SNR={snr:.1f} ({sig_word})\n"
                     f"implied ΔS ≈ {self.peak_flux_jy:.0f} Jy")
            ax.text(0.02, 0.95, label, transform=ax.transAxes,
                    color=PEAK_COLOR, fontsize=10, va="top")
        else:
            if self.baseline is not None and np.isfinite(self.baseline):
                ax.axhline(y=self.baseline, color=GRID_COLOR, linewidth=0.8,
                           linestyle="--", label="baseline (20th pct)")
            if self._peak_idx is not None:
                ax.axvline(x=self.peak_ra, color=PEAK_COLOR, linewidth=0.8,
                           linestyle=":", alpha=0.8)
                label = (f"peak: RA={self.peak_ra:.1f}°  "
                         f"S≈{self.peak_flux_jy:.0f} Jy\n"
                         f"(no expected_source_ra given -- this is just "
                         f"the highest point, not a tested detection)")
                ax.text(0.02, 0.95, label, transform=ax.transAxes,
                        color=PEAK_COLOR, fontsize=9, va="top")

        ax.set_xlabel("Right Ascension (°)", color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("P_ant / P_ref (broadband)", color=TEXT_COLOR, fontsize=10)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.4, alpha=0.5)

        legend = ax.legend(loc="upper right", fontsize=8, facecolor=DARK_BG,
                           edgecolor=GRID_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        if title is None:
            name = self.source_name or "Continuum"
            title = f"{name} Total-Power Drift Scan  |  {len(self.values)} integrations"
        fig.suptitle(title, color=TEXT_COLOR, fontsize=11, fontweight="bold")

        plt.tight_layout()
        return fig

    def show(self, **kwargs) -> None:
        self.render(**kwargs)
        plt.show()

    def save(self, path: str, dpi: int = 150, **kwargs) -> None:
        fig = self.render(**kwargs)
        fig.savefig(path, dpi=dpi, facecolor=DARK_BG, bbox_inches="tight")
        plt.close(fig)
        print(f"Continuum light curve saved to: {path}")