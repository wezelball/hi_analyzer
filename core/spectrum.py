"""
spectrum.py
HI spectral analysis: velocity axis, coordinate assignment, and
per-spectrum metadata for building sky maps.

The 21-cm HI line rest frequency is 1420.405751 MHz.
Doppler velocity relative to the Local Standard of Rest (LSR) is:
    v_LSR = c * (f_HI - f_obs) / f_HI   [km/s]

Positive velocity = receding gas (redshift).
Negative velocity = approaching gas (blueshift).

Typical usage
-------------
    from core.spectrum import freq_to_velocity, SpectrumRecord, assign_radec

    vel = freq_to_velocity(freq_axis_mhz)
    records = assign_radec(obs_file)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

HI_REST_FREQ_MHZ = 1420.405751   # MHz — rest frequency of 21-cm HI line
C_KMS = 299792.458                # speed of light km/s


# ---------------------------------------------------------------------------
# Frequency / velocity conversions
# ---------------------------------------------------------------------------

def freq_to_velocity(freq_mhz: np.ndarray,
                     rest_freq_mhz: float = HI_REST_FREQ_MHZ) -> np.ndarray:
    """
    Convert frequency array to LSR Doppler velocity (km/s).

    v = c * (f_rest - f_obs) / f_rest

    Positive v = gas moving away (redshift, higher freq bins → more negative v).
    The standard radio convention: v increases toward lower frequency.

    Parameters
    ----------
    freq_mhz      : frequency axis in MHz
    rest_freq_mhz : rest frequency (default: HI 1420.405751 MHz)

    Returns
    -------
    np.ndarray of velocities in km/s
    """
    return C_KMS * (rest_freq_mhz - freq_mhz) / rest_freq_mhz


def velocity_to_freq(vel_kms: float,
                     rest_freq_mhz: float = HI_REST_FREQ_MHZ) -> float:
    """Convert a Doppler velocity (km/s) to observed frequency (MHz)."""
    return rest_freq_mhz * (1.0 - vel_kms / C_KMS)


def hi_channel_index(freq_axis_mhz: np.ndarray) -> Optional[int]:
    """
    Return the channel index closest to the HI rest frequency.

    Parameters
    ----------
    freq_axis_mhz : frequency axis array

    Returns
    -------
    Integer index, or None if HI rest frequency is outside the band.
    """
    if (HI_REST_FREQ_MHZ < freq_axis_mhz.min() or
            HI_REST_FREQ_MHZ > freq_axis_mhz.max()):
        return None
    return int(np.argmin(np.abs(freq_axis_mhz - HI_REST_FREQ_MHZ)))


# ---------------------------------------------------------------------------
# Coordinate assignment
# ---------------------------------------------------------------------------

def altaz_to_radec(az_deg: float, el_deg: float,
                   lat_deg: float, lon_deg: float, elev_m: float,
                   timestamp: datetime) -> SkyCoord:
    """
    Convert antenna Az/El pointing to RA/Dec at a given UTC time.

    Parameters
    ----------
    az_deg    : azimuth (degrees, 0=N, 90=E)
    el_deg    : elevation (degrees)
    lat_deg   : observer latitude (degrees N)
    lon_deg   : observer longitude (degrees E)
    elev_m    : observer elevation (metres)
    timestamp : UTC datetime

    Returns
    -------
    astropy SkyCoord in ICRS (J2000 RA/Dec)
    """
    location = EarthLocation(lat=lat_deg * u.deg,
                             lon=lon_deg * u.deg,
                             height=elev_m * u.m)
    t = Time(timestamp)
    frame = AltAz(obstime=t, location=location)
    altaz = SkyCoord(az=az_deg * u.deg, alt=el_deg * u.deg, frame=frame)
    return altaz.icrs


def lsr_correction_kms(ra_deg: float, dec_deg: float,
                        timestamp: datetime) -> float:
    """
    Estimate the LSR velocity correction for a given sky direction and time.

    The Earth's orbital velocity has a component along any line of sight
    that shifts the observed HI frequency. A full LSR correction requires
    the Sun's motion relative to the LSR (~20 km/s toward apex at
    RA=18h, Dec=+30°) plus the Earth's orbital velocity (~30 km/s).

    This returns an approximate heliocentric correction only.
    For a full LSR correction, use astropy's radial_velocity_correction.

    Parameters
    ----------
    ra_deg    : right ascension (degrees)
    dec_deg   : declination (degrees)
    timestamp : UTC observation time

    Returns
    -------
    float — approximate velocity correction in km/s
            Add this to observed velocity to get LSR velocity.
    """
    from astropy.coordinates import get_body_barycentric_posvel, ICRS
    from astropy.time import Time

    t = Time(timestamp)
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')

    try:
        # Earth's barycentric velocity in km/s
        _, earth_vel = get_body_barycentric_posvel('earth', t)
        # Project onto line of sight
        target_cart = target.cartesian.xyz.value
        earth_vel_kms = earth_vel.xyz.to(u.km / u.s).value
        correction = float(np.dot(earth_vel_kms, target_cart))
        return correction
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# SpectrumRecord — a single calibrated spectrum with full metadata
# ---------------------------------------------------------------------------

@dataclass
class SpectrumRecord:
    """
    A fully annotated calibrated spectrum with sky coordinates.

    This is the fundamental unit for building sky maps — each drift
    scan integration produces one SpectrumRecord.
    """
    timestamp:    datetime        # UTC mid-point of the integration
    ra_deg:       float           # RA of beam centre (degrees, J2000)
    dec_deg:      float           # Dec of beam centre (degrees, J2000)
    az_deg:       float           # antenna azimuth (degrees)
    el_deg:       float           # antenna elevation (degrees)
    spectrum:     np.ndarray      # calibrated power, shape (n_channels,)
    freq_mhz:     np.ndarray      # frequency axis, shape (n_channels,)
    source_file:  str = ""        # originating filename

    @property
    def velocity_kms(self) -> np.ndarray:
        """Doppler velocity axis in km/s (LSR, approximate)."""
        return freq_to_velocity(self.freq_mhz)

    @property
    def peak_velocity_kms(self) -> float:
        """Velocity of the peak spectral channel (km/s)."""
        idx = np.nanargmax(self.spectrum)
        return float(self.velocity_kms[idx])

    @property
    def peak_power(self) -> float:
        """Peak calibrated power value."""
        return float(np.nanmax(self.spectrum))

    @property
    def coord(self) -> SkyCoord:
        """Beam pointing as an astropy SkyCoord."""
        return SkyCoord(ra=self.ra_deg * u.deg,
                        dec=self.dec_deg * u.deg, frame='icrs')


# ---------------------------------------------------------------------------
# Assign RA/Dec to all pairs in an ObsFile
# ---------------------------------------------------------------------------

def assign_radec(obs_file,
                 apply_lsr: bool = False) -> List[SpectrumRecord]:
    """
    Convert all SamplePairs in an ObsFile into SpectrumRecord objects
    with proper RA/Dec coordinates assigned from the Az/El pointing
    and the timestamp of each integration.

    Parameters
    ----------
    obs_file   : ObsFile from reader.load_file()
    apply_lsr  : if True, apply approximate LSR velocity correction

    Returns
    -------
    List of SpectrumRecord, one per SamplePair.
    """
    hdr    = obs_file.header
    freq   = hdr.freq_axis_mhz
    records = []

    for pair in obs_file.pairs:
        ts = pair.timestamp

        # Convert Az/El → RA/Dec at this timestamp
        coord = altaz_to_radec(
            az_deg=hdr.az_deg,
            el_deg=hdr.el_deg,
            lat_deg=hdr.latitude,
            lon_deg=hdr.longitude,
            elev_m=hdr.elevation_m,
            timestamp=ts,
        )

        spectrum = pair.calibrated

        rec = SpectrumRecord(
            timestamp=ts,
            ra_deg=float(coord.ra.deg),
            dec_deg=float(coord.dec.deg),
            az_deg=hdr.az_deg,
            el_deg=hdr.el_deg,
            spectrum=spectrum,
            freq_mhz=freq,
            source_file=obs_file.path.name,
        )
        records.append(rec)

    return records


def records_to_arrays(records: List[SpectrumRecord]):
    """
    Convert a list of SpectrumRecord to numpy arrays for plotting/analysis.

    Returns
    -------
    ra    : shape (n,)  — RA in degrees
    dec   : shape (n,)  — Dec in degrees
    times : shape (n,)  — timestamps as datetime objects
    spectra : shape (n, n_channels) — calibrated spectra
    freq  : shape (n_channels,) — frequency axis MHz
    """
    ra      = np.array([r.ra_deg   for r in records])
    dec     = np.array([r.dec_deg  for r in records])
    times   = [r.timestamp for r in records]
    spectra = np.vstack([r.spectrum for r in records])
    freq    = records[0].freq_mhz
    return ra, dec, times, spectra, freq