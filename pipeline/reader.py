"""
pipeline/reader.py — Auto-discover and load SoLEXS / HEL1OS Level-1 FITS files.

Two real-world naming conventions are handled, because they're genuinely different:

  SoLEXS  — filenames ARE self-describing:
            AL1_SOLEXS_YYYYMMDD_SDD1_L1.lc[.gz]
            AL1_SOLEXS_YYYYMMDD_SDD1_L1.gti[.gz]
            Date and detector are parsed straight from the filename.

  HEL1OS — filenames carry NO date or detector info at all (e.g. just
            "lightcurve_czt1.fits"). Every HEL1OS file must be OPENED to
            find out what it is:
              - observation date  -> ISOSTART / MJDSTART in the primary header
              - detector name     -> embedded in the table extension names,
                                     e.g. "CZT1_LC_BAND_20.00KEV_TO_40.00KEV" -> CZT1
            HEL1OS also ships in 12-hour chunks (one file per half-day), so
            multiple files can legitimately belong to the same observation
            date + detector and need to be concatenated, not overwritten.
"""

import gzip
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

# SoLEXS filenames are self-describing — fast path, no need to open the file
_SOLEXS_FNAME_RE = re.compile(
    r"AL1_SOLEXS_(\d{8})_([\w]+)_L1[._](lc|gti)",
    re.IGNORECASE,
)

# HEL1OS extension names look like: CZT1_LC_BAND_20.00KEV_TO_40.00KEV
_BAND_RE = re.compile(r"^([A-Z]+\d*)_LC_BAND_([\d.]+)KEV_TO_([\d.]+)KEV", re.IGNORECASE)

# MJD 40587 = 1970-01-01T00:00:00 UTC (the Unix epoch) — used to convert MJD -> Unix seconds
_MJD_UNIX_EPOCH = 40587.0


# ── Low-level FITS helpers ────────────────────────────────────────────────────

def _open_fits(path: Path):
    """Open a FITS file, transparently decompressing .gz if needed."""
    p = str(path)
    if p.endswith(".gz"):
        with gzip.open(path, "rb") as gz:
            return fits.open(io.BytesIO(gz.read()))
    return fits.open(path)


def _to_native(arr: np.ndarray) -> np.ndarray:
    """
    Convert a FITS-derived array (typically big-endian) to native byte order.
    Needed because pandas/cython operations (resample().max(), groupby, etc.)
    raise 'Big-endian buffer not supported on little-endian compiler' otherwise.
    """
    if arr.dtype.byteorder not in ("=", "|"):
        return arr.byteswap().view(arr.dtype.newbyteorder("="))
    return arr


def _peek_header(path: Path) -> "fits.Header | None":
    """Open just enough of a FITS file to read its primary header."""
    try:
        with _open_fits(path) as hdul:
            return hdul[0].header.copy()
    except Exception:
        return None


def _header_obs_date(header) -> str | None:
    """
    Extract an observation date (YYYYMMDD) from whatever date-ish keyword
    is present. SoLEXS uses OBS_DATE directly; HEL1OS uses ISOSTART/MJDSTART.
    """
    if "OBS_DATE" in header:
        val = str(header["OBS_DATE"])
        return val[:8] if len(val) >= 8 else None

    if "ISOSTART" in header:
        try:
            return pd.Timestamp(header["ISOSTART"]).strftime("%Y%m%d")
        except Exception:
            pass

    if "MJDSTART" in header:
        try:
            unix_s = (float(header["MJDSTART"]) - _MJD_UNIX_EPOCH) * 86400.0
            return pd.Timestamp(unix_s, unit="s", tz="UTC").strftime("%Y%m%d")
        except Exception:
            pass

    return None


def _detector_from_extensions(path: Path) -> str | None:
    """Open a HEL1OS file and read the detector name straight out of an extension name."""
    try:
        with _open_fits(path) as hdul:
            for hdu in hdul[1:]:
                m = _BAND_RE.match(hdu.name or "")
                if m:
                    return m.group(1).upper()
    except Exception:
        pass
    return None


# ── Table extraction (handles both SoLEXS and HEL1OS column schemes) ─────────

def _extract_table(hdu) -> "pd.DataFrame | None":
    """
    Read one FITS table extension into a DataFrame with at least
    TIME_UTC (UTC-aware) and RATE (counts/s) columns.

    Handles two real-world column schemes:
      SoLEXS  : TIME (Unix seconds), COUNTS
      HEL1OS  : ISOT (ISO-8601 string) or MJD, CTR (counts/s)
    """
    if hdu.data is None or len(hdu.data) == 0:
        return None

    cols = [c.name for c in hdu.columns]

    rate_col = next((c for c in ("COUNTS", "RATE", "COUNT_RATE", "CTR") if c in cols), None)
    if rate_col is None:
        return None
    rate = _to_native(np.array(hdu.data[rate_col])).flatten().astype(float)

    if "TIME" in cols:
        t = _to_native(np.array(hdu.data["TIME"])).flatten().astype(float)
        time_utc = pd.to_datetime(t, unit="s", utc=True)
    elif "ISOT" in cols:
        raw = np.array(hdu.data["ISOT"]).astype(str)
        time_utc = pd.to_datetime(raw, utc=True, format="ISO8601", errors="coerce")
    elif "MJD" in cols:
        mjd = _to_native(np.array(hdu.data["MJD"])).flatten().astype(float)
        unix_s = (mjd - _MJD_UNIX_EPOCH) * 86400.0
        time_utc = pd.to_datetime(unix_s, unit="s", utc=True)
    else:
        return None

    df = pd.DataFrame({"TIME_UTC": time_utc, "RATE": rate}).dropna(subset=["TIME_UTC"])
    # NOTE: do NOT use .astype('int64') here — pandas may store TIME_UTC at
    # second or nanosecond resolution depending on construction path, so a
    # fixed '/1e9' silently produces garbage for one of the two cases.
    # Timedelta division is resolution-agnostic and always yields seconds.
    df["TIME"] = (df["TIME_UTC"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    return df.sort_values("TIME").reset_index(drop=True)


# ── Public loaders ────────────────────────────────────────────────────────────

def load_lightcurve(path: Path) -> pd.DataFrame:
    """
    Load a SoLEXS or HEL1OS light-curve FITS file (single file, single
    detector — half-day HEL1OS chunks are concatenated by load_all(), not here).

    Returns a DataFrame with:
      TIME      — Unix seconds (float64)
      TIME_UTC  — UTC-aware pandas Timestamp
      COUNTS    — broadband count rate (counts/s) — the widest energy band
                  if multiple bands are present, or the only series if not
      BAND_<lo>_<hi>KEV — per-energy-band columns (HEL1OS multi-band files only)
    """
    path = Path(path)
    bands: dict[str, tuple[pd.DataFrame, float]] = {}  # label -> (df, energy_span)

    with _open_fits(path) as hdul:
        for hdu in hdul[1:]:
            if not hasattr(hdu, "columns"):
                continue
            df = _extract_table(hdu)
            if df is None:
                continue

            m = _BAND_RE.match(hdu.name or "")
            if m:
                lo, hi = float(m.group(2)), float(m.group(3))
                label = f"BAND_{lo:g}_{hi:g}KEV"
                span = hi - lo
            else:
                # SoLEXS-style single-extension file (e.g. "RATE") -> treat as broadband
                label = "COUNTS"
                span = float("inf")

            bands[label] = (df, span)

    if not bands:
        raise ValueError(f"No usable light-curve table found in {path.name}")

    # The widest energy span is the broadband channel -> becomes COUNTS
    broadband_label = max(bands, key=lambda k: bands[k][1])
    merged = bands[broadband_label][0][["TIME", "TIME_UTC", "RATE"]].rename(
        columns={"RATE": "COUNTS"}
    )

    for label, (df, _) in bands.items():
        if label == broadband_label:
            continue
        band_df = df[["TIME", "RATE"]].rename(columns={"RATE": label})
        merged = pd.merge_asof(
            merged.sort_values("TIME"), band_df.sort_values("TIME"),
            on="TIME", direction="nearest", tolerance=2.0,
        )

    return merged.sort_values("TIME").reset_index(drop=True)


def load_gti(path: Path) -> list:
    """
    Load GTI file. Returns list of (start_unix, stop_unix) tuples.
    Empty list = no GTI extension found, or no valid data recorded.
    """
    path = Path(path)
    with _open_fits(path) as hdul:
        try:
            gti = hdul["GTI"]
        except KeyError:
            return []
        if gti.data is None or len(gti.data) == 0:
            return []
        starts = _to_native(np.array(gti.data["START"])).flatten()
        stops  = _to_native(np.array(gti.data["STOP"])).flatten()
        return list(zip(starts.tolist(), stops.tolist()))


def apply_gti(df: pd.DataFrame, intervals: list, time_col: str = "TIME") -> pd.DataFrame:
    """Keep only rows whose TIME falls inside at least one good-time interval."""
    if not intervals:
        return df
    mask = pd.Series(False, index=df.index)
    t = df[time_col]
    for start, stop in intervals:
        mask |= (t >= start) & (t <= stop)
    return df[mask].reset_index(drop=True)


# ── File discovery ────────────────────────────────────────────────────────────

def discover_files(data_dir: Path) -> dict:
    """
    Scan data_dir for SoLEXS and HEL1OS files, regardless of which of the
    two very different naming conventions each instrument uses.

    Returns:
      { (instrument, date_str, detector): {"lc": [paths], "gti": [paths]} }

    Values are LISTS of paths because HEL1OS ships in 12-hour chunks —
    two files can legitimately share the same (instrument, date, detector) key.
    """
    data_dir = Path(data_dir)
    found: dict = {}

    def _add(key, ftype, fpath):
        found.setdefault(key, {}).setdefault(ftype, []).append(fpath)

    for fpath in sorted(data_dir.iterdir()):
        if not fpath.is_file() or fpath.name.startswith("."):
            continue

        stem = fpath.name[:-3] if fpath.name.endswith(".gz") else fpath.name

        # ── Fast path: SoLEXS self-describing filename ──
        m = _SOLEXS_FNAME_RE.search(stem)
        if m:
            date_str, detector, ftype = m.group(1), m.group(2).upper(), m.group(3).lower()
            _add(("SOLEXS", date_str, detector), ftype, fpath)
            continue

        # ── Slow path: open the file and read its header / extensions ──
        header = _peek_header(fpath)
        if header is None:
            print(f"  [reader] Skipping unreadable file: {fpath.name}")
            continue

        instrument = str(header.get("INSTRUME", "")).upper()
        if not instrument:
            print(f"  [reader] Skipping file with no INSTRUME keyword: {fpath.name}")
            continue

        date_str = _header_obs_date(header)
        if date_str is None:
            print(f"  [reader] Skipping {fpath.name} — could not determine observation date")
            continue

        content = str(header.get("CONTENT", "")).upper()
        if "GOOD TIME" in content or "GTI" in content:
            ftype = "gti"
            detector = str(header.get("DETECTOR", "UNKNOWN")).upper()
        else:
            ftype = "lc"
            detector = _detector_from_extensions(fpath) or "UNKNOWN"

        _add((instrument, date_str, detector), ftype, fpath)

    return found


def load_all(data_dir: Path, verbose: bool = True) -> dict:
    """
    Load all SoLEXS / HEL1OS files found in data_dir (with GTI filtering,
    and HEL1OS half-day chunks concatenated automatically).

    Returns nested dict:
      {
        date_str: {
          "SOLEXS": {"SDD1": df, "SDD2": df},
          "HEL1OS": {"CZT1": df, "CZT2": df, "CDTE1": df, ...},
        },
        ...
      }
    """
    file_map = discover_files(data_dir)
    if not file_map:
        print(f"[reader] No SoLEXS/HEL1OS files found in {data_dir}/")
        return {}

    result: dict = {}

    for (instrument, date_str, detector), files in sorted(file_map.items()):
        lc_paths = files.get("lc", [])
        if not lc_paths:
            if verbose:
                print(f"  [reader] Skipping {instrument}/{detector} {date_str} — no light curve file")
            continue

        chunks = []
        for p in lc_paths:
            try:
                chunks.append(load_lightcurve(p))
            except Exception as exc:
                print(f"  [reader] ERROR loading {p.name}: {exc}")

        if not chunks:
            continue

        df = pd.concat(chunks, ignore_index=True).sort_values("TIME").drop_duplicates("TIME")
        n_chunks_msg = f" ({len(chunks)} chunks merged)" if len(chunks) > 1 else ""

        # GTI filtering (concatenate intervals across any GTI files found for this key)
        gti_paths = files.get("gti", [])
        intervals = []
        for p in gti_paths:
            intervals.extend(load_gti(p))

        if gti_paths:
            if intervals:
                before = len(df)
                df = apply_gti(df, intervals)
                if verbose:
                    print(f"  [reader] {instrument}/{detector} {date_str}{n_chunks_msg}: "
                          f"GTI filtered {before:,} → {len(df):,} rows")
            else:
                if verbose:
                    print(f"  [reader] {instrument}/{detector} {date_str}{n_chunks_msg}: "
                          f"GTI empty — using all {len(df):,} rows")
        else:
            if verbose:
                print(f"  [reader] {instrument}/{detector} {date_str}{n_chunks_msg}: "
                      f"{len(df):,} rows (no GTI file)")

        result.setdefault(date_str, {}).setdefault(instrument, {})[detector] = df

    return result
