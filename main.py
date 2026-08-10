"""
main.py
HI Analyzer — command-line entry point.

Commands
--------
    inspect   — show file header and statistics, no plots
    spectrum  — calibrated spectrum plot for one or more files
    waterfall — waterfall (time vs velocity) plot
    stack     — stack multiple files at same elevation, show result
    skymap    — build 2D sky map from all input files

Usage examples
--------------
    # Inspect a file
    python main.py inspect data/WGA_260624_04.txt

    # Plot a single calibrated spectrum
    python main.py spectrum data/WGA_260624_04.txt

    # Stack two nights at the same elevation
    python main.py stack data/WGA_260624_04.txt data/WGA_260625_04.txt

    # Build a sky map from all files
    python main.py skymap data/*.txt

    # Save outputs instead of displaying
    python main.py spectrum data/WGA_260624_04.txt --save output/spectrum.png

    # Adjust velocity window
    python main.py spectrum data/WGA_260624_04.txt --vel-min -200 --vel-max 200
"""

from __future__ import annotations

import argparse
import glob
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.reader import load_file, load_files, group_by_elevation
from core.calibration import (calibrate, flag_rfi, flag_persistent_rfi,
                               stack_spectra, stack_obs_files,
                               sidereal_stack, estimate_tsys,
                               continuum_lightcurve, flux_calibrate)
from core.spectrum import (freq_to_velocity, assign_radec,
                           records_to_arrays, HI_REST_FREQ_MHZ)
from ui.plots import (plot_spectrum, plot_waterfall,
                      plot_stacked_comparison, plot_tsys)
from ui.skymap import SkyMap
from ui.continuum import ContinuumLightCurve


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HI Analyzer — drift scan radio astronomy pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("command",
                   choices=["inspect", "spectrum", "waterfall", "stack", "stack-waterfall", "skymap", "continuum"],
                   help="Analysis command to run")

    p.add_argument("files", nargs="+",
                   help="Input ezRA observation file(s). Glob patterns accepted.")

    # Velocity window
    p.add_argument("--vel-min", type=float, default=-200.0,
                   help="Minimum LSR velocity for display/integration (km/s)")
    p.add_argument("--vel-max", type=float, default=200.0,
                   help="Maximum LSR velocity for display/integration (km/s)")

    # Calibration
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip polynomial baseline subtraction (default: off)")
    p.add_argument("--subtract-baseline", action="store_true",
                   help="Enable polynomial baseline subtraction (experimental)")
    p.add_argument("--no-flag", action="store_true",
                   help="Skip RFI flagging")
    p.add_argument("--sigma", type=float, default=4.0,
                   help="RFI flagging threshold (sigma)")
    p.add_argument("--poly-order", type=int, default=5,
                   help="Polynomial order for baseline subtraction")
    p.add_argument("--bandpass-correct", action="store_true",
               help="Apply bandpass correction to remove receiver gain slope "
                    "(radio equivalent of flat-fielding)")

    # Output
    p.add_argument("--save", type=str, default=None,
                   help="Save plot to this file instead of displaying")
    p.add_argument("--report", type=str, default=None,
                   help="Save text report to this file")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output image DPI")

    # Stacking options
    p.add_argument("--ra-bin-deg", type=float, default=0.5,
                   help="RA bin width for sidereal stacking (degrees). "
                        "~0.5° = ~2 min of RA, matches a 20s integration")

    # Sky map options
    p.add_argument("--ra-bins", type=int, default=360,
                   help="RA pixels in sky map")
    p.add_argument("--dec-bins", type=int, default=90,
                   help="Dec pixels in sky map")
    p.add_argument("--beam-halfwidth-deg", type=float, default=12.5,
                   help="Sky map: dec smearing half-width per integration "
                        "(deg). Defaults to half this dish's elevation "
                        "HPBW (~25 deg). Widen/narrow if you change dishes.")

    # Continuum (total power) options
    p.add_argument("--source-name", type=str, default=None,
                   help="Label for the continuum plot title, e.g. 'Cas A'")
    p.add_argument("--edge-frac", type=float, default=0.1,
                   help="Continuum: fraction of band edges excluded from "
                        "the broadband average (edge roll-off is "
                        "instrumental, not sky signal)")
    p.add_argument("--tsys-k", type=float, default=None,
                   help="Continuum: system temperature (K) for flux "
                        "calibration. If not given, estimated automatically "
                        "from this file's own reference-load levels.")
    p.add_argument("--aperture-m2", type=float, default=0.283,
                   help="Continuum: effective aperture (m^2) for flux "
                        "calibration. Default matches the 100x60cm "
                        "Nooelec wire-grid dish at ~60%% efficiency.")
    p.add_argument("--source-ra", type=float, default=None,
                   help="Continuum: expected RA (deg) of a known target "
                        "source (e.g. Cyg A = 299.87). If given, switches "
                        "to trend-aware detection with a real significance "
                        "test (SNR), instead of just reporting the highest "
                        "point in the scan -- strongly recommended when "
                        "you know what you're looking for.")

    return p


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_inspect(files, args) -> None:
    """Print file headers and statistics, no plots."""
    import numpy as np

    print(f"\n{'='*70}")
    print(f"  HI Analyzer — File Inspection")
    print(f"{'='*70}\n")

    for obs in files:
        print(obs)
        print(f"\n  Sample pairs   : {obs.n_pairs}")
        print(f"  Duration       : {obs.duration_hours:.2f} hours")
        if obs.pairs:
            # Quick Tsys estimate
            tsys = estimate_tsys(obs)
            valid = tsys[np.isfinite(tsys)]
            if len(valid) > 0:
                print(f"  Tsys proxy     : {np.median(valid):.1f} K  "
                      f"(median, range {valid.min():.0f}–{valid.max():.0f} K)")

            # HI channel check
            from core.spectrum import hi_channel_index
            hi_ch = hi_channel_index(obs.header.freq_axis_mhz)
            if hi_ch is not None:
                print(f"  HI rest freq   : channel {hi_ch} of {obs.header.freq_bins}  "
                      f"({obs.header.freq_axis_mhz[hi_ch]:.4f} MHz)")
            else:
                print(f"  HI rest freq   : NOT in band "
                      f"({obs.header.freq_min_mhz:.3f}–"
                      f"{obs.header.freq_max_mhz:.3f} MHz)")

        print()


def cmd_spectrum(obs_files, args) -> None:
    """Calibrate and plot a spectrum for each file (or stacked)."""
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np

    for obs in obs_files:
        freq  = obs.header.freq_axis_mhz
        vel   = freq_to_velocity(freq)

        cal = calibrate(obs,
                        subtract_baseline=args.subtract_baseline,
                        poly_order=args.poly_order)
        if not args.no_flag:
            cal = flag_rfi(cal, sigma_threshold=args.sigma)
            cal = flag_persistent_rfi(cal)

        stacked = stack_spectra(cal)

        # Velocity window
        in_win = (vel >= args.vel_min) & (vel <= args.vel_max)

        title = (f"{obs.path.name}  |  "
                 f"El={obs.header.el_deg:.0f}°  Dec={obs.header.dec_deg:+.1f}°  |  "
                 f"{obs.n_pairs} integrations  |  "
                 f"{obs.duration_hours:.1f}h")

        save = args.save or None
        if args.save and len(obs_files) > 1:
            stem = Path(obs.path).stem
            save = str(Path(args.save).with_stem(stem))

        plot_spectrum(vel[in_win], stacked[in_win],
                      freq_mhz=freq[in_win],
                      title=title,
                      save_path=save)

        if not args.save:
            plt.show()
            plt.close("all")


def cmd_waterfall(obs_files, args) -> None:
    """Plot waterfall for each file."""
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for obs in obs_files:
        freq  = obs.header.freq_axis_mhz
        vel   = freq_to_velocity(freq)
        in_win = (vel >= args.vel_min) & (vel <= args.vel_max)

        cal = calibrate(obs,
                        subtract_baseline=args.subtract_baseline,
                        poly_order=args.poly_order)
        if not args.no_flag:
            cal = flag_rfi(cal, sigma_threshold=args.sigma)
            cal = flag_persistent_rfi(cal)

        timestamps = obs.timestamps()
        title = (f"Waterfall: {obs.path.name}  |  "
                 f"El={obs.header.el_deg:.0f}°  "
                 f"Dec={obs.header.dec_deg:+.1f}°")

        save = args.save or None
        if args.save and len(obs_files) > 1:
            stem = Path(obs.path).stem
            save = str(Path(args.save).with_stem(f"{stem}_waterfall"))

        plot_waterfall(vel[in_win], cal[:, in_win], timestamps,
                       title=title, save_path=save)

        if not args.save:
            plt.show()
            plt.close("all")


def cmd_stack(obs_files, args) -> None:
    """Stack multiple files and compare by elevation group."""
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = group_by_elevation(obs_files)

    print(f"\nElevation groups found: {len(groups)}")
    for el, files in sorted(groups.items()):
        names = [f.path.name for f in files]
        print(f"  El={el:.0f}°  ({len(files)} file(s)): {', '.join(names)}")

    spectra_dict = {}
    freq_axis = None

    for el, files in sorted(groups.items()):
        stacked, freq = stack_obs_files(
            files,
            subtract_baseline=args.subtract_baseline,
            flag=not args.no_flag,
            sigma_threshold=args.sigma,
        )
        n_total = sum(f.n_pairs for f in files)
        dec = files[0].header.dec_deg
        label = f"El={el:.0f}°  Dec={dec:+.1f}°  (n={n_total})"
        spectra_dict[label] = stacked
        if freq_axis is None:
            freq_axis = freq

    vel = freq_to_velocity(freq_axis)
    in_win = (vel >= args.vel_min) & (vel <= args.vel_max)

    spectra_windowed = {k: v[in_win] for k, v in spectra_dict.items()}

    title = (f"Stacked HI Spectra by Elevation  |  "
             f"{len(groups)} strip(s)  |  "
             f"v = [{args.vel_min:.0f}, {args.vel_max:.0f}] km/s")

    plot_stacked_comparison(vel[in_win], spectra_windowed,
                            title=title, save_path=args.save)

    if not args.save:
        plt.show()


def cmd_stack_waterfall(obs_files, args) -> None:
    """
    Stack multiple observations by RA alignment, display as waterfall.

    Aligns spectra from all input files by RA (sidereal time) and averages
    them, then displays the result as a waterfall plot with RA on the Y-axis
    instead of time.  Multiple nights at the same elevation stack coherently,
    improving SNR by sqrt(N_nights) just like astrophotography stacking.
    """
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from core.calibration import sidereal_stack

    # Group by elevation
    groups = group_by_elevation(obs_files)
    print(f"\nElevation groups: {len(groups)}")
    for el, files in sorted(groups.items()):
        n_nights = len(files)
        n_pairs  = sum(f.n_pairs for f in files)
        dec      = files[0].header.dec_deg
        print(f"  El={el:.0f}°  Dec={dec:+.1f}°  "
              f"{n_nights} night(s)  {n_pairs} total integrations")

    for el, files in sorted(groups.items()):
        dec = files[0].header.dec_deg
        freq_axis = files[0].header.freq_axis_mhz

        print(f"\nStacking El={el:.0f}° ({len(files)} file(s)) by RA...")
        ra_centres, stacked, counts, freq = sidereal_stack(
            files,
            ra_bin_deg=args.ra_bin_deg,
            flag=not args.no_flag,
            sigma_threshold=args.sigma,
        )

        # Only keep RA bins that have data
        has_data = counts > 0
        ra_data      = ra_centres[has_data]
        stacked_data = stacked[has_data]

        print(f"  RA coverage: {ra_data.min():.1f}° – {ra_data.max():.1f}°  "
              f"({has_data.sum()} bins,  max stack depth={counts.max()})")

        # Velocity window
        from core.spectrum import freq_to_velocity
        vel = freq_to_velocity(freq)
        in_win = (vel >= args.vel_min) & (vel <= args.vel_max)

        title = (
            f"Sidereal-Stacked Waterfall  |  El={el:.0f}°  Dec={dec:+.1f}°  |  "
            f"{len(files)} night(s)  |  RA bin={args.ra_bin_deg:.1f}°  |  "
            f"v=[{args.vel_min:.0f},{args.vel_max:.0f}] km/s"
        )

        # --- Prepare display array ---
        display = stacked_data[:, in_win].copy()

        # Step 1: Bandpass correction + DC offset removal.
        if args.bandpass_correct:
            n_chan = display.shape[1]
            n_edge = max(4, int(n_chan * 0.15))
            x = np.arange(n_chan, dtype=float)

            median_spec = np.nanmedian(display, axis=0)

            fit_mask = np.zeros(n_chan, dtype=bool)
            fit_mask[:] = True
            fit_mask[:n_edge] = False
            fit_mask[-n_edge:] = False
            hi_start = n_chan // 3
            hi_end   = 2 * n_chan // 3
            fit_mask[hi_start:hi_end] = False

            valid = fit_mask & np.isfinite(median_spec) & (median_spec > 0.1)

            if valid.sum() >= 2:
                coeffs = np.polyfit(x[valid], median_spec[valid], 2)
                bandpass = np.polyval(coeffs, x)
                # Remove only the shape, not the DC level
                bandpass_ac = bandpass - np.mean(bandpass)
                display = display - bandpass_ac[np.newaxis, :]

        # Always subtract per-row median to remove DC offset
        row_medians = np.nanmedian(display, axis=1, keepdims=True)
        display -= row_medians
 
        # Step 2: Flag DC-offset RFI columns FIRST (before variance flagging
        # removes channels and causes nanmean to warn on empty slices)
        col_mean    = np.nanmean(display, axis=0)
        cm_med      = np.nanmedian(col_mean)
        cm_mad      = np.nanmedian(np.abs(col_mean - cm_med)) * 1.4826
        if cm_mad > 0:
            dc_rfi_cols = np.abs(col_mean - cm_med) > args.sigma * cm_mad
        else:
            dc_rfi_cols = np.zeros(display.shape[1], dtype=bool)
        n_dc = dc_rfi_cols.sum()
        if n_dc > 0:
            print(f"  Flagging {n_dc} DC-offset RFI channel(s)")
            display[:, dc_rfi_cols] = np.nan

        # Step 3: Flag channels where column variance is anomalously high.
        # Real sky signal varies smoothly with RA; RFI is erratic.
        # ddof=0 avoids RuntimeWarning when a column has only 1 valid value
        col_std     = np.nanstd(display, axis=0, ddof=0)
        col_med_std = np.nanmedian(col_std)
        col_mad_std = np.nanmedian(np.abs(col_std - col_med_std))
        col_sigma   = 1.4826 * col_mad_std
        if col_sigma > 0:
            rfi_cols = col_std > (col_med_std + args.sigma * col_sigma)
        else:
            rfi_cols = np.zeros(display.shape[1], dtype=bool)
        n_flagged = rfi_cols.sum()
        if n_flagged > 0:
            print(f"  Flagging {n_flagged} persistent RFI channel(s) "
                  f"({n_flagged/display.shape[1]*100:.1f}% of band)")
            display[:, rfi_cols] = np.nan

        # Step 4: Interpolate flagged columns from their neighbours
        for j in range(display.shape[1]):
            col = display[:, j]
            nans = ~np.isfinite(col)
            if nans.any() and (~nans).sum() > 2:
                x = np.arange(len(col))
                col[nans] = np.interp(x[nans], x[~nans], col[~nans])
                display[:, j] = col

        # Step 5: Two-pass interpolation for isolated bad pixels —
        # first row-wise, then column-wise to catch any remaining NaNs
        for i in range(display.shape[0]):
            row = display[i]
            nans = ~np.isfinite(row)
            if nans.any() and (~nans).sum() > 2:
                x = np.arange(len(row))
                row[nans] = np.interp(x[nans], x[~nans], row[~nans])
                display[i] = row
        for j in range(display.shape[1]):
            col = display[:, j]
            nans = ~np.isfinite(col)
            if nans.any() and (~nans).sum() > 2:
                x = np.arange(len(col))
                col[nans] = np.interp(x[nans], x[~nans], col[~nans])
                display[:, j] = col
        # Replace any remaining NaN with the overall median
        overall_med = float(np.nanmedian(display))
        display = np.where(np.isfinite(display), display, overall_med)

        # Step 5b: Median filter to eliminate isolated outlier pixels.
        # A 3x3 median filter replaces each pixel with the median of its
        # 8 surrounding neighbours, removing single-pixel spikes while
        # preserving the broader HI signal structure.
        from scipy.ndimage import median_filter
        display = median_filter(display, size=3)

        # Step 6: Set color scale from the noise floor only.
        # Estimate noise as the MAD of the central 50% of values —
        # this ignores both the deep negative outliers (RFI artifacts)
        # and the bright HI signal peaks.
        
        # Exclude edge channels from colorscale calculation —
        # they may have uncorrected rolloff artifacts
        n_edge_disp = max(4, int(display.shape[1] * 0.15))
        interior = display[:, n_edge_disp:-n_edge_disp]

        p25  = float(np.nanpercentile(interior, 25))
        p75  = float(np.nanpercentile(interior, 75))
        p99  = float(np.nanpercentile(interior, 99))
        noise_sigma = (p75 - p25) / 1.35
        # vmin: just below the noise floor (show a little negative room)
        # vmax: at p99 so the brightest HI signal is fully visible
        vmin = -2.0 * noise_sigma
        vmax = p99

        print(f"  Display shape: {display.shape}")
        print(f"  Display min: {np.nanmin(display):.5f}  max: {np.nanmax(display):.5f}")
        print(f"  Interior min: {np.nanmin(interior):.5f}  max: {np.nanmax(interior):.5f}")
        print(f"  p25: {p25:.5f}  p75: {p75:.5f}  p99: {p99:.5f}")
        print(f"  noise_sigma: {noise_sigma:.5f}")
        print(f"  vmin: {vmin:.5f}  vmax: {vmax:.5f}")

        # Hard clip at these limits so outliers don't affect rendering
        display = np.clip(display, vmin, vmax)

        n_zeros = np.sum(display == 0.0)
        n_below = np.sum(display < vmin)
        print(f"  Noise sigma: {noise_sigma:.5f}  "
              f"Color scale: {vmin:.5f} – {vmax:.5f}")
        print(f"  Zero pixels: {n_zeros}  "
              f"Below vmin (clipped): {n_below}")

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")
        display = np.fliplr(display)
        im = ax.imshow(
            display,
            origin="upper",
            aspect="auto",
            extent=[vel[in_win][-1], vel[in_win][0],
                    ra_data[-1], ra_data[0]],
            vmin=vmin, vmax=vmax,
            cmap="inferno",
            interpolation="nearest",
        )

        # Mark flagged frequency channels with subtle vertical lines
        # so the viewer knows data was removed rather than absent
        flagged_mask = rfi_cols | dc_rfi_cols
        flagged_vel  = vel[in_win][flagged_mask]
        for fv in flagged_vel:
            ax.axvline(x=fv, color="#444466", linewidth=0.5,
                       linestyle="-", alpha=0.5)

        # HI rest velocity line
        ax.axvline(x=0, color="#00BFFF", linewidth=1.2,
                   linestyle="--", alpha=0.8, label="v=0 (HI rest)")

        ax.set_xlabel("LSR Velocity (km/s)", color="#E2E8F0", fontsize=10)
        ax.set_ylabel("Right Ascension (°)", color="#E2E8F0", fontsize=10)
        ax.tick_params(colors="#E2E8F0")
        for spine in ax.spines.values():
            spine.set_color("#2D3748")

        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label("ΔP / P_ref  (median subtracted)", color="#E2E8F0", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#E2E8F0")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#E2E8F0", fontsize=7)

        ax.legend(fontsize=8, framealpha=0.3, labelcolor="#E2E8F0",
                  facecolor="#0D1117", edgecolor="#2D3748")
        fig.suptitle(title, color="#E2E8F0", fontsize=10,
                     fontweight="bold", y=0.99)
        plt.tight_layout()

        save = args.save
        if args.save and len(groups) > 1:
            from pathlib import Path
            save = str(Path(args.save).with_stem(
                f"{Path(args.save).stem}_el{el:.0f}"))

        if save:
            fig.savefig(save, dpi=args.dpi, facecolor="#0D1117",
                        bbox_inches="tight")
            print(f"  Saved to: {save}")
            plt.close(fig)
        else:
            plt.show()


def cmd_skymap(obs_files, args) -> None:
    """Build and display a 2D sky map from all files."""
    import matplotlib
    if args.save:
        matplotlib.use("Agg")

    import numpy as np

    print(f"\nAssigning RA/Dec coordinates to {sum(o.n_pairs for o in obs_files)} integrations...")
    all_records = []
    for obs in obs_files:
        # Calibrate first
        cal = calibrate(obs, subtract_baseline=args.subtract_baseline,
                        poly_order=args.poly_order)
        if not args.no_flag:
            cal = flag_rfi(cal, sigma_threshold=args.sigma)
            cal = flag_persistent_rfi(cal)

        # --- Bandpass correction + DC-offset removal ---
        # Same approach as cmd_stack_waterfall. The raw calibrated ratio
        # sits at a DC level of ~0.595 with the HI signal only ~0.017
        # above it, so without this step the sky map is dominated by
        # DC level / bandpass shape rather than actual HI structure.
        n_chan = cal.shape[1]

        if args.bandpass_correct:
            n_edge = max(4, int(n_chan * 0.15))
            x = np.arange(n_chan, dtype=float)
            median_spec = np.nanmedian(cal, axis=0)

            fit_mask = np.ones(n_chan, dtype=bool)
            fit_mask[:n_edge] = False
            fit_mask[-n_edge:] = False
            hi_start = n_chan // 3
            hi_end   = 2 * n_chan // 3
            fit_mask[hi_start:hi_end] = False

            valid = fit_mask & np.isfinite(median_spec) & (median_spec > 0.1)
            if valid.sum() >= 2:
                coeffs = np.polyfit(x[valid], median_spec[valid], 2)
                bandpass = np.polyval(coeffs, x)
                # Remove only the shape, not the DC level
                bandpass_ac = bandpass - np.mean(bandpass)
                cal = cal - bandpass_ac[np.newaxis, :]

        # Always subtract per-row median to remove the DC offset
        row_medians = np.nanmedian(cal, axis=1, keepdims=True)
        cal = cal - row_medians

        print(f"  {obs.path.name}: bandpass_correct={args.bandpass_correct}  "
              f"row-median DC removed  "
              f"(cal range after correction: "
              f"{np.nanmin(cal):.5f} to {np.nanmax(cal):.5f})")

        # Replace pair spectra with calibrated versions
        records = assign_radec(obs)
        for i, rec in enumerate(records):
            rec.spectrum = cal[i] if i < len(cal) else rec.spectrum
        all_records.extend(records)

    print(f"  Total records: {len(all_records)}")
    print(f"  RA range: {min(r.ra_deg for r in all_records):.1f}° – "
          f"{max(r.ra_deg for r in all_records):.1f}°")
    print(f"  Dec strips: {sorted(set(round(r.dec_deg, 1) for r in all_records))}")

    smap = SkyMap(
        all_records,
        vel_min_kms=args.vel_min,
        vel_max_kms=args.vel_max,
        ra_bins=args.ra_bins,
        dec_bins=args.dec_bins,
        beam_halfwidth_deg=args.beam_halfwidth_deg,
    )

    print("Building sky map grid...")
    smap.build()

    if args.save:
        smap.save(args.save, dpi=args.dpi)
    else:
        smap.show()


def cmd_continuum(obs_files, args) -> None:
    """
    Build a total-power (continuum) light curve to detect bright compact
    sources -- Cas A, Cyg A, Tau A, the Sun, etc. -- as they transit the
    beam. Unlike the HI commands, this does NOT remove the DC level of
    the calibrated ratio: the DC level rising as a source transits IS
    the signal.

    Accepts multiple files (e.g. several nights at the same pointing) and
    combines them into a single light curve, the same stacking approach
    already used for the sky map -- averaging down noise by roughly
    sqrt(N_nights).
    """
    import numpy as np

    all_times, all_ra, all_values = [], [], []
    all_tsys = []

    for obs in obs_files:
        print(f"\n{obs.path.name}: {obs.header}")

        values = continuum_lightcurve(obs, edge_frac=args.edge_frac,
                                       flag=not args.no_flag,
                                       sigma_threshold=args.sigma)
        records = assign_radec(obs)

        if args.tsys_k is None:
            tsys_arr = estimate_tsys(obs)
            valid = tsys_arr[np.isfinite(tsys_arr)]
            if valid.size:
                all_tsys.extend(valid.tolist())
                print(f"  This file's median Tsys: {np.median(valid):.1f} K "
                      f"({valid.size} valid integrations)")

        for rec, v in zip(records, values):
            all_times.append(rec.timestamp)
            all_ra.append(rec.ra_deg)
            all_values.append(v)

    if args.tsys_k is not None:
        tsys_k = args.tsys_k
        print(f"\nUsing supplied Tsys: {tsys_k:.1f} K")
    elif all_tsys:
        tsys_k = float(np.median(all_tsys))
        print(f"\nCombined Tsys across all {len(obs_files)} file(s): "
              f"{tsys_k:.1f} K (median of {len(all_tsys)} integrations)")
    else:
        tsys_k = 400.0
        print(f"\nNo valid Tsys estimates -- falling back to default {tsys_k:.1f} K")

    lc = ContinuumLightCurve(
        all_times, all_ra, all_values,
        source_name=args.source_name,
        tsys_k=tsys_k,
        aperture_m2=args.aperture_m2,
        expected_source_ra=args.source_ra,
    )
    lc.process()

    print(f"\n{'='*60}")
    print(f"Combined: {len(obs_files)} file(s), {len(all_values)} total integrations")
    if args.source_ra is not None:
        sig = "SIGNIFICANT (>=3 sigma)" if np.isfinite(lc.snr) and abs(lc.snr) >= 3 else "not significant"
        print(f"Target: {args.source_name or 'source'} at RA={args.source_ra:.2f} deg")
        print(f"SNR: {lc.snr:.2f}  ({sig})")
        print(f"Implied flux: {lc.peak_flux_jy:.0f} Jy")
        if np.isfinite(lc.snr) and abs(lc.snr) < 3:
            print("Not a confirmed detection with this data -- consider "
                  "adding more nights to improve SNR (stack more files "
                  "the same way this command already combines them).")
    else:
        print(f"Baseline (20th pct): {lc.baseline:.5f}")
        print(f"Peak: RA={lc.peak_ra:.1f}°  value={lc.peak_value:.5f}  "
              f"implied flux ≈ {lc.peak_flux_jy:.0f} Jy")
        print("(No --source-ra given -- this is just the highest point, "
              "not a statistically tested detection. Pass --source-ra "
              "for a real significance test.)")
    print(f"{'='*60}")

    if args.save:
        lc.save(args.save, dpi=args.dpi)
    else:
        lc.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    import numpy as np   # needed by command functions

    parser = build_parser()
    args = parser.parse_args(argv)

    # Expand glob patterns in file list
    expanded = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if matches:
            expanded.extend(matches)
        elif Path(pattern).exists():
            expanded.append(pattern)
        else:
            print(f"WARNING: no files matched: {pattern}")

    if not expanded:
        print("ERROR: no input files found.")
        sys.exit(1)

    print(f"\nLoading {len(expanded)} file(s)...")
    obs_files = load_files(expanded)

    if not obs_files:
        print("ERROR: no files loaded successfully.")
        sys.exit(1)

    # Dispatch command
    cmd = args.command
    if cmd == "inspect":
        import numpy as np
        cmd_inspect(obs_files, args)
    elif cmd == "spectrum":
        cmd_spectrum(obs_files, args)
    elif cmd == "waterfall":
        cmd_waterfall(obs_files, args)
    elif cmd == "stack":
        cmd_stack(obs_files, args)
    elif cmd == "stack-waterfall":
        cmd_stack_waterfall(obs_files, args)
    elif cmd == "skymap":
        cmd_skymap(obs_files, args)
    elif cmd == "continuum":
        cmd_continuum(obs_files, args)


if __name__ == "__main__":
    main()