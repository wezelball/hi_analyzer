"""
plots.py
Visualization routines for HI spectra, waterfalls, and stacking results.

Plots produced
--------------
- Single calibrated spectrum (power vs velocity)
- Waterfall plot (time vs velocity, color = power)
- Stacked spectrum with noise estimate
- Tsys vs time

Typical usage
-------------
    from core.reader import load_file
    from core.calibration import calibrate, flag_rfi, stack_spectra
    from core.spectrum import freq_to_velocity
    from ui.plots import plot_spectrum, plot_waterfall

    obs = load_file("data/WGA_260624_04.txt")
    cal = calibrate(obs)
    vel = freq_to_velocity(obs.header.freq_axis_mhz)

    plot_spectrum(vel, stack_spectra(cal), title="Stacked HI spectrum")
    plot_waterfall(vel, cal, obs.timestamps())
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from core.spectrum import HI_REST_FREQ_MHZ, freq_to_velocity


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

DARK_BG    = "#0D1117"
TEXT_COLOR = "#E2E8F0"
GRID_COLOR = "#2D3748"
HI_COLOR   = "#00BFFF"
WARN_COLOR = "#FFE66D"


def _dark_ax(ax, fig=None):
    """Apply dark theme to axes."""
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.4, alpha=0.6)
    if fig:
        fig.patch.set_facecolor(DARK_BG)


# ---------------------------------------------------------------------------
# Single spectrum plot
# ---------------------------------------------------------------------------

def plot_spectrum(
    vel_kms: np.ndarray,
    spectrum: np.ndarray,
    freq_mhz: Optional[np.ndarray] = None,
    title: str = "HI Spectrum",
    xlabel: str = "LSR Velocity (km/s)",
    ylabel: str = "Calibrated Power (P_ant / P_ref)",
    show_hi_line: bool = True,
    show_noise: bool = True,
    figsize=(12, 5),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot a single calibrated HI spectrum.

    Parameters
    ----------
    vel_kms      : velocity axis (km/s)
    spectrum     : calibrated power values
    freq_mhz     : optional frequency axis for secondary x-axis
    title        : plot title
    show_hi_line : draw a vertical line at v=0 (HI rest frequency)
    show_noise   : annotate RMS noise level
    figsize      : figure size
    save_path    : if given, save to this path instead of displaying
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
    _dark_ax(ax, fig)

    # Main spectrum
    ax.plot(vel_kms, spectrum, color=HI_COLOR, linewidth=1.2, alpha=0.9,
            label="Calibrated spectrum")

    # HI rest velocity line
    if show_hi_line:
        ax.axvline(x=0, color=WARN_COLOR, linewidth=1.0,
                   linestyle="--", alpha=0.7, label="HI rest (v=0)")

    # Noise estimate from edge channels
    if show_noise:
        n_edge = max(4, len(spectrum) // 10)
        edge = np.concatenate([spectrum[:n_edge], spectrum[-n_edge:]])
        rms = float(np.nanstd(edge[np.isfinite(edge)]))
        if np.isfinite(rms):
            ax.axhspan(
                np.nanmean(spectrum) - rms,
                np.nanmean(spectrum) + rms,
                color="#444444", alpha=0.25, label=f"±1σ noise ({rms:.4f})"
            )

    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR,
              facecolor=DARK_BG, edgecolor=GRID_COLOR)

    # Optional secondary frequency axis
    if freq_mhz is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        # Map velocity ticks back to frequency for secondary axis
        vel_ticks = ax.get_xticks()
        freq_ticks = HI_REST_FREQ_MHZ * (1 - vel_ticks / 299792.458)
        ax2.set_xticks(vel_ticks)
        ax2.set_xticklabels([f"{f:.2f}" for f in freq_ticks],
                            fontsize=7, color=TEXT_COLOR)
        ax2.set_xlabel("Frequency (MHz)", color=TEXT_COLOR, fontsize=9)
        ax2.tick_params(colors=TEXT_COLOR)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Waterfall plot (time vs velocity)
# ---------------------------------------------------------------------------

def plot_waterfall(
    vel_kms: np.ndarray,
    spectra: np.ndarray,
    timestamps: List[datetime],
    title: str = "HI Waterfall",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "inferno",
    figsize=(12, 7),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot a waterfall (time vs velocity, color = calibrated power).

    Each row is one integration. Time increases downward.
    HI emission appears as a bright vertical streak at the emission velocity.

    Parameters
    ----------
    vel_kms    : velocity axis (km/s), shape (n_channels,)
    spectra    : calibrated spectra, shape (n_integrations, n_channels)
    timestamps : list of datetime for each integration
    vmin/vmax  : color scale limits (auto if None)
    cmap       : matplotlib colormap
    figsize    : figure size
    save_path  : save to path instead of displaying
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
    _dark_ax(ax, fig)

    if vmin is None:
        vmin = float(np.nanpercentile(spectra, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(spectra, 98))

    # Build time extent for imshow
    t0 = matplotlib.dates.date2num(timestamps[0])
    t1 = matplotlib.dates.date2num(timestamps[-1])
    v0, v1 = vel_kms[0], vel_kms[-1]

    im = ax.imshow(
        spectra,
        origin="upper",
        aspect="auto",
        extent=[v0, v1, t1, t0],
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )

    # HI rest line
    ax.axvline(x=0, color=WARN_COLOR, linewidth=1.0,
               linestyle="--", alpha=0.7, label="v=0 (HI rest)")

    # Time axis formatting
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("LSR Velocity (km/s)", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("UTC Time", color=TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label("P_ant / P_ref", color=TEXT_COLOR, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR, fontsize=7)

    ax.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR,
              facecolor=DARK_BG, edgecolor=GRID_COLOR)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Stacked spectrum comparison
# ---------------------------------------------------------------------------

def plot_stacked_comparison(
    vel_kms: np.ndarray,
    spectra_dict: dict,
    title: str = "Stacked Spectra Comparison",
    figsize=(12, 5),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Overlay multiple stacked spectra for comparison.

    Parameters
    ----------
    vel_kms      : velocity axis (km/s)
    spectra_dict : dict mapping label -> spectrum array
    """
    colors = [HI_COLOR, "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF"]
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
    _dark_ax(ax, fig)

    for (label, spec), color in zip(spectra_dict.items(), colors):
        ax.plot(vel_kms, spec, color=color, linewidth=1.3,
                alpha=0.85, label=label)

    ax.axvline(x=0, color=WARN_COLOR, linewidth=1.0, linestyle="--",
               alpha=0.7, label="HI rest (v=0)")
    ax.set_xlabel("LSR Velocity (km/s)", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Calibrated Power (P_ant / P_ref)", color=TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR,
              facecolor=DARK_BG, edgecolor=GRID_COLOR)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Tsys vs time
# ---------------------------------------------------------------------------

def plot_tsys(
    timestamps: List[datetime],
    tsys: np.ndarray,
    title: str = "System Temperature vs Time",
    figsize=(12, 4),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot estimated Tsys over the course of an observation.

    A stable Tsys indicates a stable receiver and no large RFI events.
    Spikes indicate RFI or hardware issues.
    A slow drift indicates thermal changes in the LNA or feed.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
    _dark_ax(ax, fig)

    valid = np.isfinite(tsys)
    ax.plot(timestamps, np.where(valid, tsys, np.nan),
            color=HI_COLOR, linewidth=1.0, alpha=0.8)

    if valid.any():
        median_tsys = float(np.nanmedian(tsys))
        ax.axhline(y=median_tsys, color=WARN_COLOR, linewidth=1.0,
                   linestyle="--", alpha=0.7,
                   label=f"Median Tsys = {median_tsys:.0f} K")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, color=TEXT_COLOR)
    ax.set_xlabel("UTC Time", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Tsys proxy (K)", color=TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT_COLOR,
              facecolor=DARK_BG, edgecolor=GRID_COLOR)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
        plt.close(fig)
    return fig