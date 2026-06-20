"""
pipeline/forecaster.py — Solar flare forecasting model.

Pipeline:
  1. build_features()  — rolling statistical features from SXR + HXR time series
  2. build_labels()    — multi-class labels (0=quiet, 1=A/B, 2=C, 3=M, 4=X)
                         for a forecast horizon of N minutes
  3. train()           — XGBoost multi-class model with chronological train/test split
  4. predict()         — real-time probability vector for the 5 classes
  5. save() / load()   — persist trained model to disk
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import linregress
from xgboost import XGBClassifier

import config


# ── Feature engineering ───────────────────────────────────────────────────────

def _slope(x: np.ndarray) -> float:
    """Linear trend slope over a 1D array (handles NaN gracefully)."""
    valid = ~np.isnan(x)
    if valid.sum() < 3:
        return 0.0
    t = np.arange(len(x))[valid].astype(float)
    slope, *_ = linregress(t, x[valid])
    return slope


def build_features(combined: pd.DataFrame,
                   windows_min: list[int] = config.FEATURE_WINDOWS_MIN,
                   resample_freq: str = "1min") -> pd.DataFrame:
    """
    Compute rolling statistical features for the forecasting model.

    Input:  combined DataFrame from preprocessor (SXR_FLUX, HXR_COUNTS, …)
    Output: feature DataFrame at 1-minute cadence

    Features per channel × window:
      _mean     — rolling mean flux
      _std      — rolling standard deviation
      _slope    — linear trend slope (rising vs. decaying)
      _max      — rolling maximum
      _deriv    — mean rate of change (finite difference)

    Cross-channel features:
      hardness_ratio — HXR_COUNTS / SXR_COUNTS (non-thermal vs. thermal)
      sxr_excess     — SXR_FLUX / rolling_background (relative enhancement)
    """
    # Resample to 1-minute to reduce noise before feature extraction
    df = combined.resample(resample_freq).mean()

    feats = pd.DataFrame(index=df.index)

    channels = {}
    if "SXR_FLUX" in df.columns:
        channels["sxr"] = df["SXR_FLUX"].clip(lower=0)
    if "SXR_COUNTS" in df.columns and "sxr" not in channels:
        channels["sxr"] = df["SXR_COUNTS"].clip(lower=0)
    if "HXR_COUNTS" in df.columns:
        channels["hxr"] = df["HXR_COUNTS"].clip(lower=0)

    for ch_name, series in channels.items():
        for w_min in windows_min:
            prefix = f"{ch_name}_w{w_min:02d}"
            roll   = series.rolling(window=w_min, min_periods=max(1, w_min // 2))

            feats[f"{prefix}_mean"]  = roll.mean()
            feats[f"{prefix}_std"]   = roll.std().fillna(0)
            feats[f"{prefix}_max"]   = roll.max()

            # Slope: rolling apply with linregress
            feats[f"{prefix}_slope"] = roll.apply(
                lambda x: _slope(np.array(x)), raw=True
            )

            # Rate of change: difference between current and previous window
            feats[f"{prefix}_deriv"] = series.diff(periods=w_min) / w_min

        # Background-relative enhancement (precursor signal)
        bg = series.rolling(window=60, min_periods=30, center=True).median()
        feats[f"{ch_name}_excess"] = series / (bg + 1e-20)

    # Cross-channel features
    if "sxr" in channels and "hxr" in channels:
        feats["hardness_ratio"] = (
            channels["hxr"] / (channels["sxr"].clip(lower=1e-20))
        )
        feats["hardness_slope"] = feats.get(
            "hxr_w05_slope", pd.Series(0, index=df.index)
        ) / (feats.get("sxr_w05_slope", pd.Series(1e-20, index=df.index)).abs() + 1e-20)

    # Time features (solar flares cluster by active region visibility)
    feats["hour_of_day"]  = df.index.hour + df.index.minute / 60.0
    feats["cos_hour"]     = np.cos(2 * np.pi * feats["hour_of_day"] / 24)
    feats["sin_hour"]     = np.sin(2 * np.pi * feats["hour_of_day"] / 24)
    feats.drop(columns=["hour_of_day"], inplace=True)

    return feats.dropna(how="all")


# ── Labeling ──────────────────────────────────────────────────────────────────

def build_labels(
    feats: pd.DataFrame,
    master_catalogue: pd.DataFrame,
    horizon_min: int = config.FORECAST_HORIZON_MIN,
) -> pd.Series:
    """
    Create multi-class labels for each feature timestamp.

    Label = highest GOES class of any flare whose START falls within
            the next `horizon_min` minutes after the timestamp.

    Class encoding:
      0 = quiet (no flare in horizon)
      1 = A or B class
      2 = C class
      3 = M class
      4 = X class

    Uses chronological lookahead (no future leakage into features).
    """
    labels = pd.Series(0, index=feats.index, dtype=int, name="label")

    if master_catalogue is None or len(master_catalogue) == 0:
        return labels

    horizon = pd.Timedelta(minutes=horizon_min)
    cat = master_catalogue[master_catalogue["goes_class"].isin(list("ABCMX"))].copy()

    for ts in feats.index:
        window_end = ts + horizon
        # Flares starting within the forecast horizon
        upcoming = cat[
            (cat["start_utc"] > ts) &
            (cat["start_utc"] <= window_end)
        ]
        if len(upcoming) == 0:
            continue
        max_class_num = upcoming["goes_class_num"].max()
        labels.at[ts] = int(max_class_num)

    return labels


# ── Train / evaluate ──────────────────────────────────────────────────────────

class FlareForecastModel:
    """
    Thin wrapper around an XGBClassifier that handles datasets where not all
    5 flare classes are present (e.g. a single observation day with no C-class
    events). Internally trains on whatever classes ARE present using a compact
    label encoding, then pads predict_proba() back out to the full 5-class
    space [Quiet, A/B, C, M, X] so downstream code never has to know the
    difference.
    """

    def __init__(self, xgb_model, present_classes: np.ndarray):
        self.xgb_model = xgb_model
        self.present_classes = np.asarray(present_classes)

    def predict_proba(self, X) -> np.ndarray:
        raw = self.xgb_model.predict_proba(X)              # shape (n, k)
        full = np.zeros((raw.shape[0], config.NUM_CLASSES))  # shape (n, 5)
        for i, cls in enumerate(self.present_classes):
            full[:, int(cls)] = raw[:, i]
        return full

    def predict(self, X) -> np.ndarray:
        """Return the predicted class (0-4) for each row, decoded to the full label space."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def train(
    feats: pd.DataFrame,
    labels: pd.Series,
    split_ratio: float = config.TRAIN_TEST_SPLIT_RATIO,
    split_date: str | None = config.TRAIN_TEST_SPLIT_DATE,
    model_params: dict = config.XGBOOST_PARAMS,
) -> tuple[FlareForecastModel, dict]:
    """
    Train a multi-class XGBoost model with a chronological train/test split.

    Handles the case where not every GOES class is present in the data
    (common with limited observation days) by encoding only the classes
    that actually occur, then padding predictions back to the full
    5-class space via FlareForecastModel.

    Returns (model, metrics_dict)
    """
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder

    # Align features and labels on index
    common = feats.index.intersection(labels.index)
    X = feats.loc[common].fillna(0)
    y_raw = labels.loc[common]

    # Compact-encode whatever classes are actually present in this dataset
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
    n_present = len(le.classes_)
    if n_present < 2:
        raise ValueError(
            f"Only {n_present} class present in the data — need at least 2 "
            f"(e.g. Quiet + one flare class) to train. Add more observation days."
        )

    # Chronological split (never random — would leak future into past)
    if split_date is not None:
        split_ts = pd.Timestamp(split_date, tz="UTC")
        train_mask = X.index < split_ts
    else:
        n_train = int(len(X) * split_ratio)
        train_mask = pd.Series(False, index=X.index)
        train_mask.iloc[:n_train] = True

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y_enc[train_mask], y_enc[~train_mask]

    print(f"  [forecaster] Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    print(f"  [forecaster] Classes present: "
          f"{[config.CLASS_NAMES[c] for c in le.classes_]} "
          f"({n_present}/{config.NUM_CLASSES} of the full class space)")

    params = dict(model_params)
    params["num_class"] = n_present
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    model = FlareForecastModel(xgb_model, le.classes_)

    # Decode predictions back to the original 0-4 label space for reporting
    y_pred_enc = xgb_model.predict(X_test)
    y_pred  = le.inverse_transform(y_pred_enc)
    y_test_orig = le.inverse_transform(y_test)

    report = classification_report(
        y_test_orig, y_pred,
        labels=list(range(config.NUM_CLASSES)),
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test_orig, y_pred, labels=list(range(config.NUM_CLASSES)))

    # Per-class TPR and FPR
    tpr, fpr = {}, {}
    for i, cls in enumerate(config.CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fp - fn
        tpr[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr[cls] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm,
        "tpr_per_class": tpr,
        "fpr_per_class": fpr,
        "overall_accuracy": report["accuracy"],
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "classes_present": [config.CLASS_NAMES[c] for c in le.classes_],
    }

    print(f"  [forecaster] Overall accuracy: {report['accuracy']:.3f}")
    for cls in config.CLASS_NAMES[1:]:  # skip Quiet
        print(f"    {cls}: TPR={tpr[cls]:.3f}  FPR={fpr[cls]:.3f}")

    return model, metrics


def compute_lead_times(
    model: "FlareForecastModel",
    feats: pd.DataFrame,
    master_catalogue: pd.DataFrame,
    threshold: float = 0.3,
    horizon_min: int = config.FORECAST_HORIZON_MIN,
) -> pd.DataFrame:
    """
    Compute forecast lead times for each flare in master_catalogue.

    Lead time = (earliest alert timestamp) − (flare start timestamp)
    An alert is triggered when the model's flare-class probability exceeds `threshold`.

    Returns DataFrame with columns:
      peak_utc, goes_class, lead_time_min, alerted
    """
    if len(master_catalogue) == 0 or len(feats) == 0:
        return pd.DataFrame()

    X = feats.fillna(0)
    proba = model.predict_proba(X)   # shape (n_samples, 5)
    # Sum of non-quiet class probabilities
    flare_proba = pd.Series(
        1 - proba[:, 0], index=feats.index, name="flare_prob"
    )

    records = []
    for _, flare in master_catalogue.iterrows():
        if flare.get("goes_class", "?") not in list("ABCMX"):
            continue
        flare_start = flare["start_utc"]
        look_back   = flare_start - pd.Timedelta(hours=2)

        # Alerts in the 2-hour pre-flare window
        pre_flare_proba = flare_proba[
            (flare_proba.index >= look_back) &
            (flare_proba.index <  flare_start)
        ]
        alerts = pre_flare_proba[pre_flare_proba >= threshold]

        if len(alerts) > 0:
            first_alert  = alerts.index[0]
            lead_time_s  = (flare_start - first_alert).total_seconds()
            lead_time_min = lead_time_s / 60
        else:
            first_alert  = None
            lead_time_min = np.nan

        records.append({
            "peak_utc":       flare["peak_utc"],
            "goes_class":     flare["goes_class"],
            "lead_time_min":  lead_time_min,
            "alerted":        first_alert is not None,
        })

    return pd.DataFrame(records)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model: "FlareForecastModel", path: Path) -> None:
    joblib.dump(model, path)
    print(f"  [forecaster] Model saved → {path}")


def load_model(path: Path) -> "FlareForecastModel":
    return joblib.load(path)


def predict_proba(model: "FlareForecastModel", feats: pd.DataFrame) -> np.ndarray:
    """
    Return class probability array for the most recent feature row.
    Shape: (5,) corresponding to [Quiet, A/B, C, M, X].
    """
    X = feats.iloc[[-1]].fillna(0)
    return model.predict_proba(X)[0]
