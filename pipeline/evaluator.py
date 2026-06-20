"""
pipeline/evaluator.py — Evaluation metrics for the hackathon submission.

Produces:
  • Per-class TPR / FPR table
  • Lead time distribution summary
  • Detection accuracy per flare class
  • Console + CSV summary report
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

import config


def detection_accuracy(master_catalogue: pd.DataFrame,
                       goes_reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute detection accuracy per GOES class.

    If a GOES reference catalogue is provided (DataFrame with peak_utc and goes_class),
    we count True Positives and False Negatives against ground truth.
    Otherwise we report the class distribution of the detected events.

    Returns DataFrame with columns: class, n_detected, [n_reference, TPR, FN] if ref given
    """
    if master_catalogue is None or len(master_catalogue) == 0:
        return pd.DataFrame()

    counts = (
        master_catalogue[master_catalogue["goes_class"].isin(list("ABCMX"))]
        .groupby("goes_class")
        .size()
        .reindex(list("ABCMX"), fill_value=0)
        .reset_index()
    )
    counts.columns = ["goes_class", "n_detected"]

    if goes_reference is not None and len(goes_reference) > 0:
        ref_counts = (
            goes_reference[goes_reference["goes_class"].isin(list("ABCMX"))]
            .groupby("goes_class")
            .size()
            .reindex(list("ABCMX"), fill_value=0)
            .reset_index()
        )
        ref_counts.columns = ["goes_class", "n_reference"]
        counts = counts.merge(ref_counts, on="goes_class", how="left")
        counts["TPR"] = (counts["n_detected"] / counts["n_reference"].clip(lower=1)).clip(upper=1.0)
        counts["FN"]  = (counts["n_reference"] - counts["n_detected"]).clip(lower=0)

    return counts


def forecast_metrics(lead_times_df: pd.DataFrame) -> dict:
    """
    Summarise lead time statistics across all forecasted flares.

    Returns dict with:
      alerted_rate     — fraction of flares for which an alert was issued
      lead_time_mean   — mean lead time in minutes (alerted flares only)
      lead_time_median
      lead_time_min
      lead_time_max
      per_class        — per GOES class breakdown
    """
    if lead_times_df is None or len(lead_times_df) == 0:
        return {}

    alerted = lead_times_df[lead_times_df["alerted"]]

    summary = {
        "total_flares":    len(lead_times_df),
        "alerted":         len(alerted),
        "alerted_rate":    len(alerted) / max(len(lead_times_df), 1),
        "lead_time_mean":  alerted["lead_time_min"].mean()   if len(alerted) else np.nan,
        "lead_time_median":alerted["lead_time_min"].median() if len(alerted) else np.nan,
        "lead_time_min":   alerted["lead_time_min"].min()    if len(alerted) else np.nan,
        "lead_time_max":   alerted["lead_time_min"].max()    if len(alerted) else np.nan,
        "per_class": {},
    }

    for cls in list("ABCMX"):
        cls_rows = lead_times_df[lead_times_df["goes_class"] == cls]
        cls_alert = cls_rows[cls_rows["alerted"]]
        summary["per_class"][cls] = {
            "total":    len(cls_rows),
            "alerted":  len(cls_alert),
            "tpr":      len(cls_alert) / max(len(cls_rows), 1),
            "lead_med": cls_alert["lead_time_min"].median() if len(cls_alert) else np.nan,
        }

    return summary


def false_alarm_rate(model_alerts: pd.Series,
                     master_catalogue: pd.DataFrame,
                     window_min: int = 30) -> float:
    """
    Compute false alarm rate: fraction of model alerts that are NOT followed
    by a real flare within `window_min` minutes.

    Parameters
    ----------
    model_alerts    : boolean pd.Series (True = alert fired at that timestamp)
    master_catalogue: unified flare catalogue with start_utc column
    window_min      : lookahead window for matching alerts to events
    """
    if model_alerts.sum() == 0:
        return 0.0

    alert_times = model_alerts[model_alerts].index
    flare_starts = (
        pd.to_datetime(master_catalogue["start_utc"], utc=True).tolist()
        if len(master_catalogue) > 0 else []
    )

    false_alarms = 0
    window = pd.Timedelta(minutes=window_min)

    for at in alert_times:
        # Is there a flare starting within the next window_min minutes?
        followed_by_flare = any(
            at <= fs <= at + window
            for fs in flare_starts
        )
        if not followed_by_flare:
            false_alarms += 1

    return false_alarms / len(alert_times)


def print_report(detection_acc: pd.DataFrame,
                 forecast_summary: dict,
                 far: float | None = None) -> None:
    """Print a formatted evaluation report to the console."""
    print("\n" + "=" * 60)
    print("  ADITYA-L1 SOLAR FLARE PIPELINE — EVALUATION REPORT")
    print("=" * 60)

    print("\n── Nowcasting Detection Accuracy ──")
    print(detection_acc.to_string(index=False))

    if forecast_summary:
        print("\n── Forecasting Lead Time ──")
        print(f"  Flares with alert   : {forecast_summary['alerted']} / {forecast_summary['total_flares']} "
              f"({forecast_summary['alerted_rate']*100:.1f}%)")
        print(f"  Mean lead time      : {forecast_summary['lead_time_mean']:.1f} min")
        print(f"  Median lead time    : {forecast_summary['lead_time_median']:.1f} min")
        print(f"  Range               : {forecast_summary['lead_time_min']:.1f} – "
              f"{forecast_summary['lead_time_max']:.1f} min")
        print("\n  Per-class breakdown:")
        for cls, stats in forecast_summary.get("per_class", {}).items():
            if stats["total"] > 0:
                print(f"    {cls}: TPR={stats['tpr']:.2f}  "
                      f"lead_time_median={stats['lead_med']:.1f} min  "
                      f"(N={stats['total']})")

    if far is not None:
        print(f"\n── False Alarm Rate: {far*100:.1f}% ──")

    print("=" * 60 + "\n")


def save_report(detection_acc: pd.DataFrame,
                forecast_summary: dict,
                lead_times_df: pd.DataFrame,
                output_dir: Path) -> None:
    """Save CSV outputs for the hackathon submission."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(detection_acc) > 0:
        detection_acc.to_csv(output_dir / "detection_accuracy.csv", index=False)

    if len(lead_times_df) > 0:
        lead_times_df.to_csv(output_dir / "lead_times.csv", index=False)

    # Flat JSON-friendly summary
    import json
    summary_path = output_dir / "forecast_summary.json"
    with open(summary_path, "w") as f:
        # Convert numpy types for JSON serialisation
        def _jsonify(obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, dict):           return {k: _jsonify(v) for k, v in obj.items()}
            return obj
        json.dump(_jsonify(forecast_summary), f, indent=2)

    print(f"  [evaluator] Reports saved to {output_dir}/")
