"""
skymap.py
Build and display a 2D radio sky brightness map from drift scan data.

Multiple drift scans at different elevations produce horizontal strips
on the sky at different declinations.  This module grids those strips
onto a regular RA/Dec map and displays the result.

The brightness value at each map pixel is the integrated HI emission
over a selected velocity range, or the peak brightness temperature
in that range.

Typical usage
-------------
    from core.reader import load_files
    from core.spectrum import assign_radec
    from ui.skymap import SkyMap

    obs_files = load_files(paths)
    records = []
    for obs in obs_files:
        records.extend(assign_radec(obs))

    smap = SkyMap(records)
    smap.show()
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from core.spectrum import SpectrumRecord, freq_to_velocity, HI_REST_FREQ_MHZ


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

DARK_BG    = "#0D1117"
TEXT_COLOR = "#E2E8F0"
GRID_COLOR = "#2D3748"


# ---------------------------------------------------------------------------
# SkyMap class
# ---------------------------------------------------------------------------

class SkyMap:
    """
    2D radio sky map built from drift scan SpectrumRecord objects.

    The map shows integrated HI brightness over a velocity range,
    gridded onto a regular RA/Dec pixel grid.

    Parameters
    ----------
    records       : list of SpectrumRecord from spectrum.assign_radec()
    vel_min_kms   : minimum velocity for integration window (km/s)
    vel_max_kms   : maximum velocity for integration window (km/s)
    ra_bins       : number of RA pixels
    dec_bins      : number of Dec pixels
    ra_range      : (min, max) RA range in degrees. Auto if None.
    dec_range     : (min, max) Dec range in degrees. Auto if None.
    """

    def __init__(
        self,
        records: List[SpectrumRecord],
        vel_min_kms: float = -150.0,
        vel_max_kms: float =  150.0,
        ra_bins:  int = 360,
        dec_bins: int = 180,
        ra_range:  Optional[Tuple[float, float]] = None,
        dec_range: Optional[Tuple[float, float]] = None,
        beam_halfwidth_deg: float = 1.5,
    ):
        self.records     = records
        self.vel_min     = vel_min_kms
        self.vel_max     = vel_max_kms
        self.ra_bins     = ra_bins
        self.dec_bins    = dec_bins
        # Each drift-scan record is really a beam-width-wide strip of sky,
        # not an infinitesimal point in Dec. Without this, every strip
        # collapses onto a single pixel row and is invisible on any map
        # that spans more than a degree or two of Dec. This is a display
        # approximation, not a measured beam pattern.
        self.beam_halfwidth_deg = beam_halfwidth_deg

        # Determine map extent from data
        all_ra  = np.array([r.ra_deg  for r in records])
        all_dec = np.array([r.dec_deg for r in records])

        if ra_range is None:
            pad = 5.0
            self.ra_range = (max(0, all_ra.min() - pad),
                             min(360, all_ra.max() + pad))
        else:
            self.ra_range = ra_range

        if dec_range is None:
            pad = 5.0
            self.dec_range = (max(-90, all_dec.min() - pad),
                              min(90,  all_dec.max() + pad))
        else:
            self.dec_range = dec_range

        self._image: Optional[np.ndarray] = None
        self._counts: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Gridding
    # ------------------------------------------------------------------

    def build(self) -> np.ndarray:
        """
        Grid all SpectrumRecord objects onto the RA/Dec pixel map.

        Each record contributes its integrated power over the velocity
        window to the nearest pixel. Multiple records in the same pixel
        are averaged.

        Returns
        -------
        2D np.ndarray of shape (dec_bins, ra_bins) — mean brightness.
        NaN where no data exists.
        """
        ra_edges  = np.linspace(*self.ra_range,  self.ra_bins  + 1)
        dec_edges = np.linspace(*self.dec_range, self.dec_bins + 1)
        dec_centers = (dec_edges[:-1] + dec_edges[1:]) / 2.0

        image  = np.zeros((self.dec_bins, self.ra_bins), dtype=float)
        counts = np.zeros((self.dec_bins, self.ra_bins), dtype=int)

        for rec in self.records:
            # Velocity window
            vel = freq_to_velocity(rec.freq_mhz)
            in_window = (vel >= self.vel_min) & (vel <= self.vel_max)
            if not in_window.any():
                continue

            # Integrated power over velocity window
            power_in_window = rec.spectrum[in_window]
            if not np.any(np.isfinite(power_in_window)):
                continue
            value = float(np.nanmean(power_in_window))

            # RA: nearest pixel (RA coverage is continuous, one bin is fine)
            i_ra  = np.searchsorted(ra_edges,  rec.ra_deg)  - 1
            if not (0 <= i_ra < self.ra_bins):
                continue

            # Dec: spread this integration across every pixel row within
            # beam_halfwidth_deg, since a single drift-scan pointing covers
            # a beam-width-wide strip of sky, not one infinitesimal pixel.
            dec_rows = np.where(np.abs(dec_centers - rec.dec_deg)
                                 <= self.beam_halfwidth_deg)[0]
            if dec_rows.size == 0:
                # Fallback: nearest single row (e.g. beam_halfwidth_deg
                # smaller than one pixel)
                i_dec = np.searchsorted(dec_edges, rec.dec_deg) - 1
                if 0 <= i_dec < self.dec_bins:
                    dec_rows = np.array([i_dec])
                else:
                    continue

            image[dec_rows, i_ra]  += value
            counts[dec_rows, i_ra] += 1

        # Average pixels with multiple observations
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(counts > 0, image / counts, np.nan)

        self._image  = result
        self._counts = counts
        return result

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def render(self, figsize=(14, 6),
               cmap: str = "inferno",
               title: Optional[str] = None) -> Figure:
        """
        Render the sky map as a rectangular RA/Dec plot.

        Parameters
        ----------
        figsize : figure dimensions in inches
        cmap    : matplotlib colormap
        title   : plot title (auto-generated if None)
        """
        if self._image is None:
            self.build()

        image = self._image

        fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)

        # Percentile clip for display
        vmin = float(np.nanpercentile(image, 2))
        vmax = float(np.nanpercentile(image, 98))

        im = ax.imshow(
            image,
            origin="lower",
            extent=[*self.ra_range, *self.dec_range],
            aspect="auto",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation="bilinear",
        )

        # Grid lines
        for ra in range(0, 361, 30):
            if self.ra_range[0] <= ra <= self.ra_range[1]:
                ax.axvline(x=ra, color=GRID_COLOR, linewidth=0.5,
                           linestyle="--", alpha=0.6)
        for dec in range(-90, 91, 15):
            if self.dec_range[0] <= dec <= self.dec_range[1]:
                ax.axhline(y=dec, color=GRID_COLOR, linewidth=0.5,
                           linestyle="--", alpha=0.6)

        # Celestial equator
        ax.axhline(y=0, color="#4A6A8A", linewidth=0.8,
                   linestyle="-", alpha=0.5)

        # Axes labels
        ax.set_xlabel("Right Ascension (°)", color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("Declination (°)",     color=TEXT_COLOR, fontsize=10)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label(
            f"Mean ΔP/P_ref, DC subtracted  "
            f"(v = {self.vel_min:.0f} to {self.vel_max:.0f} km/s)",
            color=TEXT_COLOR, fontsize=8
        )
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR, fontsize=7)

        # Title
        if title is None:
            n_el = len(set(r.el_deg for r in self.records))
            n_rec = len(self.records)
            title = (f"HI Sky Map  |  {n_rec} integrations  |  "
                     f"{n_el} elevation strip(s)  |  "
                     f"v = [{self.vel_min:.0f}, {self.vel_max:.0f}] km/s")
        fig.suptitle(title, color=TEXT_COLOR, fontsize=10,
                     fontweight="bold", y=0.99)

        # Coverage map inset (pixel count)
        self._add_coverage_text(ax)

        plt.tight_layout()
        return fig

    def _add_coverage_text(self, ax: Axes) -> None:
        """Annotate each Dec strip with its coverage statistics."""
        el_groups: dict = {}
        for rec in self.records:
            el = rec.el_deg
            if el not in el_groups:
                el_groups[el] = []
            el_groups[el].append(rec.dec_deg)

        for el, decs in el_groups.items():
            dec_mean = np.mean(decs)
            n = len(decs)
            if self.dec_range[0] <= dec_mean <= self.dec_range[1]:
                ax.text(self.ra_range[0] + 1, dec_mean + 0.5,
                        f"El={el:.0f}°  Dec={dec_mean:.1f}°  n={n}",
                        color="#00BFFF", fontsize=7, alpha=0.8)

    def show(self, **kwargs) -> None:
        """Render and display the sky map interactively."""
        self.render(**kwargs)
        plt.show()

    def save(self, path: str, dpi: int = 150, **kwargs) -> None:
        """Render and save the sky map to a file."""
        fig = self.render(**kwargs)
        fig.savefig(path, dpi=dpi, facecolor=DARK_BG, bbox_inches="tight")
        plt.close(fig)
        print(f"Sky map saved to: {path}")