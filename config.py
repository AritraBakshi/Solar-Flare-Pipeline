"""
config.py — All tunable parameters for the Aditya-L1 solar flare pipeline.
Adjust thresholds and model settings here without touching pipeline code.
"""

from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")      # drop AL1_* FITS files here
OUTPUT_DIR  = Path("outputs")   # catalogues, model, plots saved here

# ── SoLEXS (SXR) nowcasting detector ─────────────────────────────────────────
SXR_BG_WINDOW_S    = 1800   # rolling background window (s) — 30 minutes
SXR_THRESHOLD_SIGMA = 5.0   # flag when flux > background + N×sigma
SXR_MIN_DURATION_S  = 60    # minimum event duration to survive noise filter

# ── HEL1OS (HXR) nowcasting detector ─────────────────────────────────────────
# HXR events are impulsive but the overall envelope of a major flare can still
# span 20-30 minutes (multiple particle injections, extended decay). A 5-minute
# median-based background gets contaminated by the event itself in that case —
# the background silently rises with the signal and nothing ever looks
# anomalous. Use a longer window AND a low percentile (not the median) so the
# estimate stays anchored to the true quiet floor even when a large fraction
# of the window is event-contaminated.
HXR_BG_WINDOW_S     = 1800   # 30-minute rolling background (same scale as SXR)
HXR_BG_QUANTILE     = 0.20   # 20th percentile, not median — robust to long bursts
HXR_THRESHOLD_SIGMA = 4.0
HXR_MIN_DURATION_S  = 10     # the FLAGGED excursion can still be short
HXR_DERIV_SMOOTH_S  = 5      # smoothing window for rate-of-change calculation
HXR_MERGE_GAP_S      = 90    # bridge gaps shorter than this between flagged
                              # excursions — HXR bursts are genuinely "spiky"
                              # (quasi-periodic pulsations) and a few seconds
                              # below threshold between sub-pulses shouldn't
                              # fragment one burst into many catalogue entries

# ── Catalogue merging (Neupert effect) ───────────────────────────────────────
# HXR peak typically precedes or coincides with SXR peak (Neupert effect).
# Match SXR and HXR events that overlap within this window.
NEUPERT_WINDOW_S = 600      # 10-minute matching window

# ── GOES calibration (SoLEXS counts/s → W/m²) ───────────────────────────────
# Empirical factor derived from the 3 known flares on June 3 2026 (SDD2):
#   M9.3 → 6580 cts/s,  M7.7 → 5343 cts/s,  X1.0 → 7190 cts/s
#   ratios: 1.413e-8, 1.441e-8, 1.391e-8  →  averaged ≈ 1.415e-8
# A single-point fit (using only the X1.0 pair) put X1.0 right at the M/X boundary
# and misclassified it as M-class — averaging across all 3 known events fixes this.
# Set GOES_CSV_PATH to refine this automatically from a real GOES CSV download.
SOLEXS_CALIB_FACTOR = 1.415e-8   # W/m² per count/s (nominal; refine per detector)
GOES_CSV_PATH       = None       # path to GOES XRS CSV for auto cross-calibration

# GOES class thresholds (GOES XRS-B 1–8 Å peak flux, W/m²)
GOES_CLASS_THRESHOLDS = {
    "A": (1e-8, 1e-7),
    "B": (1e-7, 1e-6),
    "C": (1e-6, 1e-5),
    "M": (1e-5, 1e-4),
    "X": (1e-4, float("inf")),
}

# ── Forecasting ───────────────────────────────────────────────────────────────
FORECAST_HORIZON_MIN  = 30      # predict flare occurrence in next N minutes
FEATURE_WINDOWS_MIN   = [1, 5, 15, 60]  # rolling window sizes for features (minutes)
PRECURSOR_WINDOW_MIN  = 30      # look-back window to identify pre-flare precursor rises

# Label encoding for multi-class model
# 0 = quiet  1 = A/B class  2 = C class  3 = M class  4 = X class
LABEL_CLASS_MAP = {"quiet": 0, "A": 1, "B": 1, "C": 2, "M": 3, "X": 4}
NUM_CLASSES     = 5
CLASS_NAMES     = ["Quiet", "A/B", "C", "M", "X"]

# ── XGBoost model ────────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "objective":     "multi:softprob",
    "num_class":     NUM_CLASSES,
    "n_estimators":  300,
    "max_depth":     6,
    "learning_rate": 0.05,
    "subsample":     0.8,
    "colsample_bytree": 0.8,
    "eval_metric":   "mlogloss",
    "random_state":  42,
    "n_jobs":        -1,
}

# Use chronological 80/20 split if no explicit split date given
TRAIN_TEST_SPLIT_RATIO = 0.80   # ignored if TRAIN_TEST_SPLIT_DATE is set
TRAIN_TEST_SPLIT_DATE  = None   # e.g. "2026-06-10" to split before/after this date

# ── Dashboard replay ─────────────────────────────────────────────────────────
REPLAY_STEP_S     = 60      # seconds per replay tick
DISPLAY_WINDOW_MIN = 120    # rolling window shown in live chart (minutes)
