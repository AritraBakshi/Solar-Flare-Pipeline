"""
pipeline/preprocessor.py — Calibration, resampling, and multi-detector fusion.

Steps applied to raw DataFrames from reader.py:
  1. Resample both SoLEXS detectors to a common 1-second grid, take max across SDD1/SDD2
  2. Resample HEL1OS detectors similarly
  3. Convert SoLEXS broadband counts/s → GOES XRS-B equivalent W/m²
     (either using the nominal config factor or an auto-fitted factor from GOES CSV)
  4. Merge SXR and HXR series into one DataFrame for the forecaster
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

import config


# ── Resampling ────────────────────────────────────────────────────────────────

def resample_to_grid(df: pd.DataFrame,
                     freq: str = "1s",
                     agg: str = "max") -> pd.Series:
    """
    Resample a light-curve DataFrame to a regular time grid.

    Parameters
    ----------
    df   : DataFrame with TIME_UTC (UTC-aware) and COUNTS columns
    freq : pandas offset string for the target grid (default "1s")
    agg  : aggregation method — "max" preserves peak flux within each bin

    Returns
    -------
    pd.Series indexed by UTC DatetimeIndex, values = resampled counts/s
    """
    s = df.set_index("TIME_UTC")["COUNTS"].dropna()
    return s.resample(freq).agg(agg).interpolate(method="time", limit=5)


def merge_detectors(detector_dict: dict, freq: str = "1s") -> pd.Series:
    """
    Merge multiple detector DataFrames (e.g. SDD1 + SDD2) into one series.
    Strategy: resample each, then take the maximum across detectors at each timestep.
    This keeps the brightest signal when both detectors observe the same flare.
    """
    series = []
    for det_name, df in detector_dict.items():
        if df is not None and len(df) > 0:
            s = resample_to_grid(df, freq=freq)
            s.name = det_name
            series.append(s)

    if not series:
        return pd.Series(dtype=float)

    combined = pd.concat(series, axis=1)
    return combined.max(axis=1)


# ── GOES cross-calibration ────────────────────────────────────────────────────

def _goes_calib_from_csv(goes_csv: Path) -> float | None:
    """
    Auto-fit the SoLEXS-to-GOES calibration factor from a GOES XRS event list CSV.

    Expects a CSV with columns: peak_time, xrs_b_flux (W/m²)
    If the CSV is not available, returns None and the nominal config factor is used.
    """
    try:
        goes = pd.read_csv(goes_csv, parse_dates=["peak_time"])
        goes = goes.dropna(subset=["xrs_b_flux"])
        return goes  # caller uses this to do the matching
    except Exception:
        return None


def calibrate_to_flux(counts: pd.Series,
                      goes_csv: Path | None = None,
                      nominal_factor: float = config.SOLEXS_CALIB_FACTOR) -> pd.Series:
    """
    Convert SoLEXS count rate (counts/s) to GOES XRS-B equivalent flux (W/m²).

    If goes_csv is provided and contains peak flare data that overlaps the
    observation period, the calibration factor is fitted automatically.
    Otherwise the nominal config factor is used.

    Returns pd.Series of flux in W/m².
    """
    factor = nominal_factor

    if goes_csv is not None and Path(goes_csv).exists():
        goes_data = _goes_calib_from_csv(goes_csv)
        if goes_data is not None:
            # Find overlapping GOES events in the observation window
            obs_start = counts.index.min()
            obs_end   = counts.index.max()
            mask = (
                (goes_data["peak_time"] >= obs_start) &
                (goes_data["peak_time"] <= obs_end)
            )
            overlap = goes_data[mask]

            if len(overlap) >= 2:
                # For each GOES event, find the SoLEXS peak within ±5 minutes
                solexs_peaks, goes_peaks = [], []
                for _, row in overlap.iterrows():
                    t0 = row["peak_time"]
                    window = counts[
                        (counts.index >= t0 - pd.Timedelta("5min")) &
                        (counts.index <= t0 + pd.Timedelta("5min"))
                    ]
                    if len(window) > 0:
                        solexs_peaks.append(window.max())
                        goes_peaks.append(row["xrs_b_flux"])

                if len(solexs_peaks) >= 2:
                    slope, _, r, _, _ = linregress(solexs_peaks, goes_peaks)
                    if r**2 > 0.8:  # good correlation → use fitted factor
                        factor = slope
                        print(f"  [calibration] Auto-fitted factor: {factor:.3e} W/m² per cts/s "
                              f"(R²={r**2:.3f}, N={len(solexs_peaks)} events)")
                    else:
                        print(f"  [calibration] Poor correlation (R²={r**2:.3f}), "
                              f"using nominal factor {factor:.3e}")

    flux = counts * factor
    flux.name = "FLUX_WM2"
    return flux


def assign_goes_class(flux_wm2: float) -> str:
    """Return the GOES flare class (A/B/C/M/X) for a given peak flux in W/m²."""
    for cls, (lo, hi) in config.GOES_CLASS_THRESHOLDS.items():
        if lo <= flux_wm2 < hi:
            return cls
    return "A"  # below A threshold → treat as A-class micro-event


# ── Combined daily DataFrame ──────────────────────────────────────────────────

def build_combined_series(date_data: dict,
                          freq: str = "1s",
                          goes_csv: Path | None = config.GOES_CSV_PATH) -> pd.DataFrame:
    """
    Build a single merged DataFrame for one observation day.

    Parameters
    ----------
    date_data : dict from reader.load_all(), e.g.
                {"SOLEXS": {"SDD1": df, "SDD2": df}, "HEL1OS": {"CZT": df, "CDTE": df}}

    Returns
    -------
    DataFrame with columns:
      SXR_COUNTS   — merged SoLEXS count rate (counts/s)
      SXR_FLUX     — GOES XRS-B equivalent flux (W/m²)
      HXR_COUNTS   — merged HEL1OS broadband count rate
      HXR_HARDNESS — HEL1OS high-band / low-band ratio (if available)
    Index: UTC DatetimeIndex at 1-second cadence
    """
    result = pd.DataFrame()

    # SoLEXS
    if "SOLEXS" in date_data and date_data["SOLEXS"]:
        sxr_counts = merge_detectors(date_data["SOLEXS"], freq=freq)
        result["SXR_COUNTS"] = sxr_counts
        result["SXR_FLUX"]   = calibrate_to_flux(
            sxr_counts, goes_csv=goes_csv
        )

    # HEL1OS
    if "HEL1OS" in date_data and date_data["HEL1OS"]:
        hxr_counts = merge_detectors(date_data["HEL1OS"], freq=freq)
        result["HXR_COUNTS"] = hxr_counts

        # Compute hardness ratio if per-band columns exist
        all_bands = {}
        for det, df in date_data["HEL1OS"].items():
            band_cols = [c for c in df.columns if c.startswith("BAND_")]
            if band_cols and len(band_cols) >= 2:
                all_bands[det] = df

        if all_bands:
            # Use first detector with multi-band data
            det, df_b = next(iter(all_bands.items()))
            band_cols = sorted([c for c in df_b.columns if c.startswith("BAND_")])
            mid = len(band_cols) // 2
            lo_bands = band_cols[:mid]
            hi_bands = band_cols[mid:]

            lo_rate = df_b.set_index("TIME_UTC")[lo_bands].sum(axis=1).resample(freq).max()
            hi_rate = df_b.set_index("TIME_UTC")[hi_bands].sum(axis=1).resample(freq).max()
            hardness = hi_rate / (lo_rate + 1e-6)   # avoid zero division
            result["HXR_HARDNESS"] = hardness

    if result.empty:
        return result

    result = result.sort_index()

    # Forward-fill short gaps (instrument data gaps up to 5 seconds)
    result = result.ffill(limit=5)

    return result
