"""
pipeline/nowcaster.py — Solar flare nowcasting (real-time detection).

Two independent detectors:
  detect_sxr()  — Background subtraction + 5σ threshold + duration filter (SoLEXS)
  detect_hxr()  — Derivative spike detector (HEL1OS)

Then:
  merge_catalogues()  — Match SXR and HXR events via Neupert-effect time window
  classify_flare()    — Assign GOES A/B/C/M/X class from peak flux in W/m²
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

import config
from pipeline.preprocessor import assign_goes_class


# ── Background estimation ─────────────────────────────────────────────────────

def _bridge_short_gaps(above: np.ndarray, max_gap_samples: int) -> np.ndarray:
    """
    Fill in short False-runs that are sandwiched between True regions, so a
    burst with several sub-pulses (quasi-periodic pulsations) doesn't get
    fragmented into many separate catalogue entries just because flux dipped
    below threshold for a few seconds between pulses. Gaps at the very start
    or end of the array (not sandwiched) are left alone.
    """
    above = above.copy()
    n = len(above)
    i = 0
    while i < n:
        if above[i]:
            i += 1
            continue
        j = i
        while j < n and not above[j]:
            j += 1
        gap_len = j - i
        if i > 0 and j < n and above[i - 1] and above[j] and gap_len <= max_gap_samples:
            above[i:j] = True
        i = j
    return above


def _rolling_background(
    series: pd.Series, window_s: int, quantile: float = 0.5
) -> tuple[pd.Series, pd.Series]:
    """
    Estimate background and its uncertainty using a rolling quantile and MAD.

    quantile=0.5 (median) is robust to short, sharp contamination — good for
    SXR's gradual events. For HXR, a lower quantile (e.g. 0.2) stays anchored
    to the true quiet floor even when a longer-duration burst occupies a
    large fraction of the window, which the median would not survive.

    sigma is estimated as 1.4826 × MAD (consistent with Gaussian sigma),
    floored at the Poisson level (sqrt(background)).
    """
    min_periods = max(60, window_s // 10)

    if quantile == 0.5:
        bg = series.rolling(window=window_s, center=True, min_periods=min_periods).median()
    else:
        bg = series.rolling(window=window_s, center=True, min_periods=min_periods).quantile(quantile)

    residuals = (series - bg).abs()
    mad = residuals.rolling(window=window_s, center=True, min_periods=min_periods).median()
    sigma = 1.4826 * mad
    sigma = sigma.clip(lower=np.sqrt(bg.clip(lower=1)))  # Poisson floor

    return bg, sigma


# ── SXR detector (SoLEXS — gradual events) ───────────────────────────────────

def detect_sxr(
    series: pd.Series,
    bg_window_s:    int   = config.SXR_BG_WINDOW_S,
    threshold_sigma: float = config.SXR_THRESHOLD_SIGMA,
    min_duration_s:  int   = config.SXR_MIN_DURATION_S,
) -> pd.DataFrame:
    """
    Detect solar flares in a SoLEXS (soft X-ray) light curve.

    Algorithm:
      1. Estimate rolling background + sigma
      2. Flag samples > background + threshold_sigma × sigma
      3. Group consecutive flagged samples into candidate events
      4. Keep events lasting ≥ min_duration_s seconds

    Parameters
    ----------
    series : pd.Series with UTC DatetimeIndex, values = counts/s or flux

    Returns
    -------
    DataFrame with columns:
      start, peak, end  — UTC timestamps
      peak_counts       — peak value at the event maximum
      background        — estimated background at peak time
      snr               — peak / background ratio
      duration_s        — event duration in seconds
      instrument        — "SoLEXS"
    """
    series = series.dropna().sort_index()
    if len(series) < 120:
        return pd.DataFrame()

    bg, sigma     = _rolling_background(series, bg_window_s)
    threshold     = bg + threshold_sigma * sigma
    above         = (series > threshold).fillna(False)

    events = []
    in_event = False
    start_idx = 0

    idx_arr = np.where(above.values)[0]

    # Group consecutive True positions into run segments
    above_arr = above.values
    times     = series.index

    i = 0
    n = len(above_arr)
    while i < n:
        if above_arr[i]:
            start_pos = i
            while i < n and above_arr[i]:
                i += 1
            end_pos = i  # exclusive

            duration = (times[end_pos - 1] - times[start_pos]).total_seconds()
            if duration >= min_duration_s:
                seg       = series.iloc[start_pos:end_pos]
                peak_pos  = start_pos + int(np.argmax(seg.values))
                peak_time = times[peak_pos]
                peak_val  = series.iloc[peak_pos]
                bg_val    = bg.iloc[peak_pos] if peak_pos < len(bg) else np.nan

                events.append({
                    "start":       times[start_pos],
                    "peak":        peak_time,
                    "end":         times[end_pos - 1],
                    "peak_counts": peak_val,
                    "background":  bg_val,
                    "snr":         peak_val / max(bg_val, 1),
                    "duration_s":  duration,
                    "instrument":  "SoLEXS",
                })
        else:
            i += 1

    return pd.DataFrame(events)


# ── HXR detector (HEL1OS — impulsive events) ─────────────────────────────────

def detect_hxr(
    series: pd.Series,
    bg_window_s:    int   = config.HXR_BG_WINDOW_S,
    bg_quantile:    float = config.HXR_BG_QUANTILE,
    threshold_sigma: float = config.HXR_THRESHOLD_SIGMA,
    min_duration_s:  int   = config.HXR_MIN_DURATION_S,
    deriv_smooth_s:  int   = config.HXR_DERIV_SMOOTH_S,
    merge_gap_s:     int   = config.HXR_MERGE_GAP_S,
) -> pd.DataFrame:
    """
    Detect impulsive hard X-ray bursts in a HEL1OS light curve.

    Algorithm:
      1. Smooth the series slightly to reduce single-sample noise spikes
      2. Compute the time derivative (rate of change)
      3. Apply background-subtraction threshold on the absolute count rate
         AND require positive derivative (rising flux) to flag onset.
         Background uses a low percentile (not median) over a long window —
         see _rolling_background's docstring for why the median fails here.
      4. Bridge short below-threshold gaps (quasi-periodic sub-pulses) so one
         burst envelope isn't fragmented into many catalogue entries
      5. Group and duration-filter as in detect_sxr

    Returns DataFrame with same columns as detect_sxr.
    """
    series = series.dropna().sort_index()
    if len(series) < 30:
        return pd.DataFrame()

    # Smooth to reduce noise before differentiation
    smoothed     = series.rolling(window=deriv_smooth_s, center=True, min_periods=1).mean()
    deriv        = smoothed.diff().fillna(0)

    bg, sigma    = _rolling_background(series, bg_window_s, quantile=bg_quantile)
    threshold    = bg + threshold_sigma * sigma

    # Flag: above threshold AND rising (positive derivative), with short
    # below-threshold dips between sub-pulses bridged so one burst envelope
    # isn't fragmented into many catalogue entries.
    above = ((series > threshold) & (deriv > 0)).fillna(False).values
    above = _bridge_short_gaps(above, max_gap_samples=merge_gap_s)

    events = []
    above_arr = above
    times     = series.index
    n         = len(above_arr)
    i         = 0

    while i < n:
        if above_arr[i]:
            start_pos = i
            while i < n and above_arr[i]:
                i += 1
            end_pos = i  # exclusive

            duration = (times[end_pos - 1] - times[start_pos]).total_seconds()
            if duration >= min_duration_s:
                seg       = series.iloc[start_pos:end_pos]
                peak_pos  = start_pos + int(np.argmax(seg.values))
                peak_time = times[peak_pos]
                peak_val  = series.iloc[peak_pos]
                bg_val    = bg.iloc[peak_pos] if peak_pos < len(bg) else np.nan

                events.append({
                    "start":       times[start_pos],
                    "peak":        peak_time,
                    "end":         times[end_pos - 1],
                    "peak_counts": peak_val,
                    "background":  bg_val,
                    "snr":         peak_val / max(bg_val, 1),
                    "duration_s":  duration,
                    "instrument":  "HEL1OS",
                })
        else:
            i += 1

    return pd.DataFrame(events)


# ── Catalogue merge ───────────────────────────────────────────────────────────

def merge_catalogues(
    sxr_cat: pd.DataFrame,
    hxr_cat: pd.DataFrame,
    sxr_flux: pd.Series | None = None,
    neupert_window_s: int = config.NEUPERT_WINDOW_S,
) -> pd.DataFrame:
    """
    Merge SXR and HXR event catalogues into a unified master catalogue.

    Matching rule (Neupert effect):
      HXR burst peak typically precedes or coincides with SXR peak.
      Match an HXR event to an SXR event if their time windows overlap
      within ± neupert_window_s.

    GOES classification is applied using the peak SXR flux in W/m²
    (requires sxr_flux series in W/m²; if absent, raw counts are used
    and class is labelled "unknown").

    Returns DataFrame with columns:
      start_utc, peak_utc, end_utc, duration_s
      sxr_peak_counts, sxr_peak_flux_wm2, goes_class, goes_class_num
      hxr_peak_counts, hxr_detected
      snr_sxr, snr_hxr
      source   — "SXR_only" | "HXR_only" | "SXR+HXR"
    """
    rows = []

    matched_hxr = set()

    # ── Process SXR events ──
    for _, sxr in (sxr_cat.iterrows() if len(sxr_cat) > 0 else []):
        row = {
            "start_utc":        sxr["start"],
            "peak_utc":         sxr["peak"],
            "end_utc":          sxr["end"],
            "duration_s":       sxr["duration_s"],
            "sxr_peak_counts":  sxr["peak_counts"],
            "snr_sxr":          sxr["snr"],
            "hxr_detected":     False,
            "hxr_peak_counts":  np.nan,
            "snr_hxr":          np.nan,
            "source":           "SXR_only",
        }

        # Get W/m² flux at peak
        if sxr_flux is not None:
            peak_window = sxr_flux[
                (sxr_flux.index >= sxr["start"]) &
                (sxr_flux.index <= sxr["end"])
            ]
            peak_flux   = peak_window.max() if len(peak_window) > 0 else np.nan
        else:
            peak_flux   = np.nan

        row["sxr_peak_flux_wm2"] = peak_flux
        row["goes_class"]         = assign_goes_class(peak_flux) if not np.isnan(peak_flux) else "?"
        row["goes_class_num"]     = _class_to_num(row["goes_class"])

        # Look for a matching HXR event — among ALL candidates whose window
        # overlaps within the Neupert tolerance, pick the one with the
        # highest peak_counts (the dominant burst), not just the first one
        # encountered in time order. A weak early sub-pulse can otherwise
        # "steal" the match from the real, much stronger counterpart.
        if len(hxr_cat) > 0:
            sxr_start_ts = sxr["start"].timestamp()
            sxr_end_ts   = sxr["end"].timestamp()
            best_idx, best_peak = None, -1

            for hxr_idx, hxr in hxr_cat.iterrows():
                if hxr_idx in matched_hxr:
                    continue
                hxr_start_ts = hxr["start"].timestamp()
                hxr_end_ts   = hxr["end"].timestamp()
                overlap = (
                    hxr_start_ts <= sxr_end_ts + neupert_window_s and
                    hxr_end_ts   >= sxr_start_ts - neupert_window_s
                )
                if overlap and hxr["peak_counts"] > best_peak:
                    best_idx, best_peak = hxr_idx, hxr["peak_counts"]

            if best_idx is not None:
                best_hxr = hxr_cat.loc[best_idx]
                row["hxr_detected"]    = True
                row["hxr_peak_counts"] = best_hxr["peak_counts"]
                row["snr_hxr"]         = best_hxr["snr"]
                row["source"]          = "SXR+HXR"
                matched_hxr.add(best_idx)

        rows.append(row)

    # ── HXR-only events (no SXR counterpart) ──
    if len(hxr_cat) > 0:
        for hxr_idx, hxr in hxr_cat.iterrows():
            if hxr_idx in matched_hxr:
                continue
            rows.append({
                "start_utc":           hxr["start"],
                "peak_utc":            hxr["peak"],
                "end_utc":             hxr["end"],
                "duration_s":          hxr["duration_s"],
                "sxr_peak_counts":     np.nan,
                "sxr_peak_flux_wm2":   np.nan,
                "goes_class":          "?",
                "goes_class_num":      0,
                "hxr_detected":        True,
                "hxr_peak_counts":     hxr["peak_counts"],
                "snr_sxr":             np.nan,
                "snr_hxr":             hxr["snr"],
                "source":              "HXR_only",
            })

    master = pd.DataFrame(rows)
    if len(master) > 0:
        master = master.sort_values("peak_utc").reset_index(drop=True)

    return master


def _class_to_num(cls: str) -> int:
    return {"A": 1, "B": 1, "C": 2, "M": 3, "X": 4}.get(cls, 0)


# ── Convenience: run full nowcasting pipeline for one day ─────────────────────

def nowcast_day(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full nowcasting pipeline on a combined day DataFrame
    (output of preprocessor.build_combined_series).

    Returns (sxr_catalogue, hxr_catalogue, master_catalogue)
    """
    sxr_cat = pd.DataFrame()
    hxr_cat = pd.DataFrame()

    if "SXR_COUNTS" in combined.columns:
        print("  [nowcaster] Running SXR detector ...")
        sxr_cat = detect_sxr(combined["SXR_COUNTS"])
        print(f"    → {len(sxr_cat)} SXR events detected")

    if "HXR_COUNTS" in combined.columns:
        print("  [nowcaster] Running HXR detector ...")
        hxr_cat = detect_hxr(combined["HXR_COUNTS"])
        print(f"    → {len(hxr_cat)} HXR events detected")

    sxr_flux = combined["SXR_FLUX"] if "SXR_FLUX" in combined.columns else None
    master   = merge_catalogues(sxr_cat, hxr_cat, sxr_flux=sxr_flux)
    
    # Check if master has records to prevent KeyError
    confirmed_count = (master['source'] == 'SXR+HXR').sum() if not master.empty else 0
    
    print(f"  [nowcaster] Master catalogue: {len(master)} events "
          f"({confirmed_count} confirmed SXR+HXR)")

    return sxr_cat, hxr_cat, master
