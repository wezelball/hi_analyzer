"""
reader.py
Parse ezRA-format observation files into structured data.

Each file contains:
  - A 7-line header with site, frequency, and antenna parameters
  - Data rows: timestamp followed by 256 RMS power values per frequency bin
  - Rows ending with 'R' are reference (50-ohm load) samples
  - Rows without 'R' are antenna (sky) samples
  - Reference and antenna samples alternate strictly, ~20s apart

Typical usage
-------------
    from core.reader import ObsFile, load_files

    obs = ObsFile("data/WGA_260624_04.txt")
    print(obs)

    # Load and combine multiple files
    dataset = load_files(["data/WGA_260624_04.txt", "data/WGA_260625_04.txt"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ObsHeader:
    """Parsed header from an ezRA observation file."""
    source_script: str
    latitude:  float          # degrees N
    longitude: float          # degrees E
    elevation_m: float        # metres above sea level
    site_name: str
    freq_min_mhz: float
    freq_max_mhz: float
    freq_bins: int
    az_deg: float             # antenna azimuth (degrees, 0=N)
    el_deg: float             # antenna elevation (degrees)
    gain_db: float            # receiver gain (dB)

    @property
    def freq_axis_mhz(self) -> np.ndarray:
        """Frequency axis for the 256 bins in MHz."""
        return np.linspace(self.freq_min_mhz, self.freq_max_mhz, self.freq_bins)

    @property
    def freq_resolution_khz(self) -> float:
        """Frequency resolution per bin in kHz."""
        return (self.freq_max_mhz - self.freq_min_mhz) / (self.freq_bins - 1) * 1e3

    @property
    def dec_deg(self) -> float:
        """
        Declination of the beam pointing (degrees).
        For a south-facing dish (Az=180°): Dec = Lat - (90 - El).
        At El=90° (zenith): Dec = Lat.
        """
        if self.el_deg >= 90.0:
            return self.latitude
        return self.latitude - (90.0 - self.el_deg)

    def __str__(self) -> str:
        return (
            f"Site: {self.site_name}  "
            f"Lat={self.latitude:+.2f}°  Lon={self.longitude:+.2f}°  "
            f"El={self.elevation_m:.0f}m\n"
            f"Freq: {self.freq_min_mhz:.3f}–{self.freq_max_mhz:.3f} MHz  "
            f"({self.freq_bins} bins, {self.freq_resolution_khz:.1f} kHz/bin)\n"
            f"Antenna: Az={self.az_deg:.1f}°  El={self.el_deg:.1f}°  "
            f"→ Dec={self.dec_deg:+.2f}°  Gain={self.gain_db:.1f} dB"
        )


@dataclass
class Sample:
    """A single antenna or reference sample."""
    timestamp: datetime       # UTC
    power: np.ndarray         # RMS power, shape (freq_bins,)
    is_reference: bool        # True = 50-ohm load; False = sky


@dataclass
class SamplePair:
    """
    A matched antenna/reference pair.
    The calibrated spectrum is antenna.power / reference.power.
    """
    antenna:   Sample
    reference: Sample

    @property
    def timestamp(self) -> datetime:
        """Mid-point timestamp between antenna and reference samples."""
        ant_ts = self.antenna.timestamp.timestamp()
        ref_ts = self.reference.timestamp.timestamp()
        mid    = (ant_ts + ref_ts) / 2.0
        return datetime.fromtimestamp(mid, tz=timezone.utc)

    @property
    def calibrated(self) -> np.ndarray:
        """
        Calibrated spectrum: P_antenna / P_reference.
        Removes gain variations and bandpass shape common to both samples.
        Values > 1 indicate sky signal above the noise floor.
        """
        ref = self.reference.power
        # Guard against zero reference values
        ref = np.where(ref > 0, ref, np.nan)
        return self.antenna.power / ref

    @property
    def tsys_proxy(self) -> float:
        """
        Approximate system temperature proxy from the reference power.
        A 50-ohm load at room temperature (290K) gives T_load ≈ 290K,
        so mean(P_ref) is proportional to T_sys + T_load.
        """
        return float(np.nanmean(self.reference.power))


@dataclass
class ObsFile:
    """
    A single parsed ezRA observation file.

    Attributes
    ----------
    path     : source file path
    header   : parsed ObsHeader
    pairs    : list of matched SamplePair objects
    """
    path:   Path
    header: ObsHeader
    pairs:  List[SamplePair] = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def duration_hours(self) -> float:
        if self.n_pairs < 2:
            return 0.0
        t0 = self.pairs[0].timestamp.timestamp()
        t1 = self.pairs[-1].timestamp.timestamp()
        return (t1 - t0) / 3600.0

    @property
    def start_time(self) -> Optional[datetime]:
        return self.pairs[0].timestamp if self.pairs else None

    @property
    def end_time(self) -> Optional[datetime]:
        return self.pairs[-1].timestamp if self.pairs else None

    def calibrated_stack(self) -> np.ndarray:
        """
        Return all calibrated spectra stacked as a 2D array.
        Shape: (n_pairs, freq_bins)
        """
        return np.vstack([p.calibrated for p in self.pairs])

    def timestamps(self) -> List[datetime]:
        return [p.timestamp for p in self.pairs]

    def __str__(self) -> str:
        return (
            f"ObsFile: {self.path.name}\n"
            f"  {self.header}\n"
            f"  Pairs: {self.n_pairs}  "
            f"Duration: {self.duration_hours:.2f}h  "
            f"Start: {self.start_time.isoformat()[:19] if self.start_time else 'N/A'}"
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_header(lines: List[str]) -> ObsHeader:
    """Parse the 7-line ezRA file header."""
    # Line 0: 'from ezCol250313a.py'
    source = lines[0].strip()

    # Line 1: 'lat 37.8 long -77.9 amsl 116.0 name TwoMice_RA'
    m = re.match(
        r'lat\s+([\d.\-]+)\s+long\s+([\d.\-]+)\s+amsl\s+([\d.\-]+)\s+name\s+(\S+)',
        lines[1]
    )
    if not m:
        raise ValueError(f"Cannot parse site line: {lines[1]!r}")
    lat, lon, amsl, name = float(m[1]), float(m[2]), float(m[3]), m[4]

    # Line 2: 'freqMin 1419.2 freqMax 1421.61 freqBinQty 256'
    m = re.match(
        r'freqMin\s+([\d.]+)\s+freqMax\s+([\d.]+)\s+freqBinQty\s+(\d+)',
        lines[2]
    )
    if not m:
        raise ValueError(f"Cannot parse frequency line: {lines[2]!r}")
    freq_min, freq_max, freq_bins = float(m[1]), float(m[2]), int(m[3])

    # Line 3: 'azDeg 180 elDeg 60'
    m = re.match(r'azDeg\s+([\d.]+)\s+elDeg\s+([\d.]+)', lines[3])
    if not m:
        raise ValueError(f"Cannot parse antenna line: {lines[3]!r}")
    az, el = float(m[1]), float(m[2])

    # Lines 4-5: comments — extract gain if present
    gain = 0.0
    for line in lines[4:7]:
        m = re.search(r'gain\s+([\d.]+)', line)
        if m:
            gain = float(m[1])
            break

    return ObsHeader(
        source_script=source,
        latitude=lat,
        longitude=lon,
        elevation_m=amsl,
        site_name=name,
        freq_min_mhz=freq_min,
        freq_max_mhz=freq_max,
        freq_bins=freq_bins,
        az_deg=az,
        el_deg=el,
        gain_db=gain,
    )


def _parse_sample(line: str) -> Tuple[datetime, np.ndarray, bool]:
    """
    Parse a single data row.
    Returns (timestamp, power_array, is_reference).
    """
    is_ref = line.endswith(' R') or line.endswith('\tR')
    if is_ref:
        line = line[:-2]

    parts = line.split()
    ts = datetime.fromisoformat(parts[0]).replace(tzinfo=timezone.utc)
    power = np.array([float(v) for v in parts[1:]], dtype=np.float64)
    return ts, power, is_ref


def load_file(path: str | Path) -> ObsFile:
    """
    Parse a single ezRA observation file.

    Parameters
    ----------
    path : path to the .txt file

    Returns
    -------
    ObsFile with header and all sample pairs populated.

    Raises
    ------
    ValueError if the file format is unrecognised or pairs cannot be matched.
    """
    path = Path(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Split header (first 7 lines) from data
    header_lines = [l.rstrip() for l in lines[:7]]
    header = _parse_header(header_lines)

    # Parse all data rows
    samples: List[Sample] = []
    data_re = re.compile(r'^\d{4}-\d{2}-\d{2}T')

    for line in lines[7:]:
        line = line.strip()
        if not line or not data_re.match(line):
            continue
        try:
            ts, power, is_ref = _parse_sample(line)
            if len(power) == header.freq_bins:
                samples.append(Sample(ts, power, is_ref))
        except (ValueError, IndexError):
            continue   # skip malformed rows

    # Match antenna/reference pairs
    # Pattern is strictly alternating: R, ant, R, ant, ...
    # The reference precedes its paired antenna sample
    pairs: List[SamplePair] = []
    i = 0
    while i < len(samples) - 1:
        s0 = samples[i]
        s1 = samples[i + 1]
        if s0.is_reference and not s1.is_reference:
            pairs.append(SamplePair(antenna=s1, reference=s0))
            i += 2
        elif not s0.is_reference and s1.is_reference:
            # antenna comes before reference — pair them anyway
            pairs.append(SamplePair(antenna=s0, reference=s1))
            i += 2
        else:
            # Consecutive same type — skip one and resync
            i += 1

    return ObsFile(path=path, header=header, pairs=pairs)


def load_files(paths: List[str | Path]) -> List[ObsFile]:
    """
    Load and return a list of ObsFile objects from multiple paths.
    Files are sorted by start time.

    Parameters
    ----------
    paths : list of file paths

    Returns
    -------
    List of ObsFile, sorted chronologically.
    """
    obs_files = []
    for p in paths:
        try:
            obs = load_file(p)
            obs_files.append(obs)
            print(f"  Loaded: {obs}")
        except Exception as e:
            print(f"  WARNING: could not load {p}: {e}")

    # Sort by start time
    obs_files.sort(key=lambda o: o.start_time or datetime.min.replace(tzinfo=timezone.utc))
    return obs_files


def group_by_elevation(obs_files: List[ObsFile],
                        tolerance_deg: float = 1.0) -> dict:
    """
    Group ObsFile objects by antenna elevation within a tolerance.

    Parameters
    ----------
    obs_files     : list of loaded ObsFile objects
    tolerance_deg : files within this many degrees are considered the same elevation

    Returns
    -------
    dict mapping representative elevation (float) → list of ObsFile
    """
    groups: dict = {}
    for obs in obs_files:
        el = obs.header.el_deg
        matched = None
        for key in groups:
            if abs(key - el) <= tolerance_deg:
                matched = key
                break
        if matched is not None:
            groups[matched].append(obs)
        else:
            groups[el] = [obs]
    return groups