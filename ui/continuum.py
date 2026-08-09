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
    ):
        self.times   = list(times)
        self.ra_deg  = np.asarray(ra_deg, dtype=float)
        self.values  = np.asarray(values, dtype=float)
        self.source_name = source_name
        self.tsys_k     = tsys_k
        self.aperture_m2 = aperture_m2
        self.t_load_k    = t_load_k

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

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self) -> np.ndarray:
        """
        Despeckle the raw light curve (median filter, matched to the
        real beam width) and estimate an off-source baseline.

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

        # Baseline: bottom 20th percentile, i.e. assume most of the scan
        # is off-source and only a modest fraction (near the transit) is
        # elevated. Reasonable for a single ~1-hour-wide transit inside
        # a ~24-hour scan.
        finite = smoothed[np.isfinite(smoothed)]
        self._baseline = float(np.percentile(finite, 20)) if finite.size else np.nan
        self._peak_idx = int(np.nanargmax(smoothed)) if finite.size else None

        self._smoothed = smoothed
        return smoothed

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
        """Implied flux density (Jy) of the peak, above baseline."""
        from core.calibration import flux_calibrate
        s = flux_calibrate(np.array([self.peak_value]), self.baseline,
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

        if self.baseline is not None and np.isfinite(self.baseline):
            ax.axhline(y=self.baseline, color=GRID_COLOR, linewidth=0.8,
                       linestyle="--", label="baseline (20th pct)")

        if self._peak_idx is not None:
            ax.axvline(x=self.peak_ra, color=PEAK_COLOR, linewidth=0.8,
                       linestyle=":", alpha=0.8)
            label = (f"peak: RA={self.peak_ra:.1f}°  "
                     f"S≈{self.peak_flux_jy:.0f} Jy")
            ax.text(0.02, 0.95, label, transform=ax.transAxes,
                    color=PEAK_COLOR, fontsize=10, va="top")

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
