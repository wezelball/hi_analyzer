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
        beam_halfwidth_deg: float = 12.5,
        az_hpbw_deg: float = 15.0,
        smooth_ra_kernel: Optional[int] = None,
    ):
        self.records     = records
        self.vel_min     = vel_min_kms
        self.vel_max     = vel_max_kms
        self.ra_bins     = ra_bins
        self.dec_bins    = dec_bins
        # Each drift-scan record is really a beam-width-wide strip of sky,
        # not an infinitesimal point in Dec. Default is half this dish's
        # measured elevation HPBW (~25 deg, 100x60cm Nooelec wire-grid
        # parabolic). Update if you change dishes.
        self.beam_halfwidth_deg = beam_halfwidth_deg
        # Median-filter kernel size (RA direction) applied to the final
        # gridded image, same technique already proven in
        # cmd_stack_waterfall: removes single-pixel noise spikes while
        # preserving real structure. A single integration was never
        # really a 1-degree-wide sample -- the antenna's own azimuth
        # beam (~15 deg HPBW per its spec sheet) already smears it over
        # roughly this much sky, so the kernel is computed from that real
        # beam width and the actual RA bin size, rather than picked
        # arbitrarily. Pass smooth_ra_kernel explicitly to override.
        if smooth_ra_kernel is None:
            ra_bin_width_deg = (ra_range[1] - ra_range[0]) / ra_bins if ra_range \
                else 360.0 / ra_bins
            smooth_ra_kernel = max(1, int(round(az_hpbw_deg / ra_bin_width_deg)))
            if smooth_ra_kernel % 2 == 0:
                smooth_ra_kernel += 1  # median_filter wants an odd kernel
        self.az_hpbw_deg      = az_hpbw_deg
        self.smooth_ra_kernel = smooth_ra_kernel

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

        image   = np.zeros((self.dec_bins, self.ra_bins), dtype=float)
        weights = np.zeros((self.dec_bins, self.ra_bins), dtype=float)
        counts  = np.zeros((self.dec_bins, self.ra_bins), dtype=int)

        # beam_halfwidth_deg is treated as the beam's half-width-at-half-
        # max (HWHM). Converting to a Gaussian sigma gives a smooth taper
        # that matches a real antenna beam much better than a uniform box
        # of equal weight out to a hard edge -- the previous box weighting
        # is why two overlapping strips could show up with identical
        # values in their overlap zone (edge-of-beam and center-of-beam
        # samples were being weighted the same).
        dec_sigma = self.beam_halfwidth_deg / 1.1774
        search_radius = 2.5 * dec_sigma  # generous cutoff, negligible tail beyond this

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

            # RA: nearest pixel (RA coverage is continuous, one bin is fine;
            # beam-width smoothing in RA is applied afterward as a proper
            # filter, see smooth_ra_kernel below)
            i_ra  = np.searchsorted(ra_edges,  rec.ra_deg)  - 1
            if not (0 <= i_ra < self.ra_bins):
                continue

            # Dec: Gaussian-weighted contribution to every pixel row within
            # search_radius, since a single drift-scan pointing covers a
            # beam-width-wide strip of sky, not one infinitesimal pixel.
            dec_rows = np.where(np.abs(dec_centers - rec.dec_deg)
                                 <= search_radius)[0]
            if dec_rows.size == 0:
                i_dec = np.searchsorted(dec_edges, rec.dec_deg) - 1
                if 0 <= i_dec < self.dec_bins:
                    dec_rows = np.array([i_dec])
                    row_weights = np.array([1.0])
                else:
                    continue
            else:
                delta = dec_centers[dec_rows] - rec.dec_deg
                row_weights = np.exp(-0.5 * (delta / dec_sigma) ** 2)

            image[dec_rows, i_ra]   += value * row_weights
            weights[dec_rows, i_ra] += row_weights
            counts[dec_rows, i_ra]  += 1

        # Weighted average of pixels with multiple contributing observations
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(counts > 0, image / weights, np.nan)

        # Despeckle + resolution-match in RA: median filter with a kernel
        # width matched to the dish's actual azimuth HPBW (~15 deg per its
        # spec sheet), converted to pixels from the real RA bin width so
        # it stays correct even if ra_bins changes. A single integration
        # was never really a 1-degree-wide sample -- the antenna's own
        # beam already smears it over roughly this much sky, so this is
        # matching the map's resolution to the real instrument rather
        # than picking an arbitrary smoothing amount.
        if self.smooth_ra_kernel and self.smooth_ra_kernel > 1:
            from scipy.ndimage import median_filter
            k = self.smooth_ra_kernel
            smoothed = result.copy()
            for i_dec in range(self.dec_bins):
                row = result[i_dec]
                valid = np.isfinite(row)
                if valid.sum() < k:
                    continue
                # Median-filter only the finite run(s); leave true gaps as NaN
                filled = np.where(valid, row, np.nanmedian(row[valid]))
                filtered = median_filter(filled, size=k, mode="nearest")
                smoothed[i_dec] = np.where(valid, filtered, np.nan)
            result = smoothed

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

        # Color scale from the noise floor, not a blind percentile clip --
        # same approach already proven in cmd_stack_waterfall. Using the
        # interquartile range as a robust noise estimate keeps a handful
        # of very bright or very dark pixels from washing out the whole
        # scale the way a plain 2nd/98th percentile can.
        finite = image[np.isfinite(image)]
        p25  = float(np.nanpercentile(finite, 25))
        p75  = float(np.nanpercentile(finite, 75))
        p99  = float(np.nanpercentile(finite, 99))
        noise_sigma = (p75 - p25) / 1.35
        vmin = -2.0 * noise_sigma
        vmax = p99

        # NaN (true no-data) gets its own distinct color so it's never
        # confused with real-but-faint data at the dark end of the
        # colormap (inferno's darkest values are nearly black, same as
        # the plot background).
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#324057")  # muted slate blue-grey

        im = ax.imshow(
            image,
            origin="lower",
            extent=[*self.ra_range, *self.dec_range],
            aspect="auto",
            cmap=cmap_obj,
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
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