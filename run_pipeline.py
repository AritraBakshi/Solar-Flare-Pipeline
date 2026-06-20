"""
run_pipeline.py — Single entry point for the Aditya-L1 solar flare pipeline.

Usage:
    python run_pipeline.py

Steps:
  1. Load all SoLEXS + HEL1OS files from data/
  2. Preprocess and calibrate to GOES W/m²
  3. Run nowcasting detectors → master flare catalogue
  4. Build features and labels → train forecasting model
  5. Evaluate: TPR, FPR, lead time
  6. Save all outputs to outputs/

After running, launch the dashboard with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd

# Make sure local imports work regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

import config
from pipeline.reader       import load_all
from pipeline.preprocessor import build_combined_series
from pipeline.nowcaster    import nowcast_day
from pipeline.forecaster   import (
    build_features, build_labels, train,
    compute_lead_times, save_model,
)
from pipeline.evaluator    import (
    detection_accuracy, forecast_metrics,
    false_alarm_rate, print_report, save_report,
)


def main():
    print("\n" + "=" * 60)
    print("  ADITYA-L1 SOLAR FLARE PIPELINE")
    print("=" * 60)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load all data files ─────────────────────────────────────────────
    print(f"\n[1/5] Loading data from {config.DATA_DIR}/")
    all_data = load_all(config.DATA_DIR)
    if not all_data:
        print("  ERROR: No data files found. "
              "Drop AL1_SOLEXS_* and AL1_HEL1OS_* files into data/ and re-run.")
        sys.exit(1)

    dates = sorted(all_data.keys())
    print(f"  Found {len(dates)} observation day(s): {', '.join(dates)}")

    # ── 2. Preprocess each day ─────────────────────────────────────────────
    print("\n[2/5] Preprocessing and calibrating ...")
    combined_all: dict[str, pd.DataFrame] = {}
    for date in dates:
        print(f"  Processing {date} ...")
        combined = build_combined_series(all_data[date])
        if combined.empty:
            print(f"    WARNING: Empty combined series for {date}, skipping.")
            continue
        combined_all[date] = combined
        combined.to_csv(config.OUTPUT_DIR / f"combined_{date}.csv")
        print(f"    Saved combined_{date}.csv  ({len(combined):,} rows)")

    if not combined_all:
        print("  ERROR: No usable combined data produced. Check file formats.")
        sys.exit(1)

    # ── 3. Nowcasting ──────────────────────────────────────────────────────
    print("\n[3/5] Nowcasting (flare detection) ...")
    all_catalogues = []
    for date, combined in combined_all.items():
        print(f"  Detecting flares in {date} ...")
        sxr_cat, hxr_cat, master = nowcast_day(combined)

        sxr_cat.to_csv(config.OUTPUT_DIR / f"sxr_catalogue_{date}.csv", index=False)
        hxr_cat.to_csv(config.OUTPUT_DIR / f"hxr_catalogue_{date}.csv", index=False)
        master.to_csv(config.OUTPUT_DIR  / f"master_catalogue_{date}.csv", index=False)
        all_catalogues.append(master)

    master_all = pd.concat(all_catalogues, ignore_index=True) if all_catalogues else pd.DataFrame()
    master_all.to_csv(config.OUTPUT_DIR / "master_catalogue_ALL.csv", index=False)
    print(f"\n  Master catalogue: {len(master_all)} total events across all days")
    if len(master_all) > 0:
        print(master_all[["peak_utc", "goes_class", "source", "duration_s"]].to_string(index=False))

    # ── 4. Forecasting ─────────────────────────────────────────────────────
    print("\n[4/5] Building forecasting model ...")

    # Concatenate all days into one feature / label set
    feat_frames, label_frames = [], []
    for date, combined in combined_all.items():
        feats = build_features(combined)
        day_master = master_all[
            master_all["peak_utc"].dt.strftime("%Y%m%d") == date
        ] if len(master_all) > 0 else pd.DataFrame()
        labels = build_labels(feats, day_master)
        feat_frames.append(feats)
        label_frames.append(labels)

    feats_all  = pd.concat(feat_frames).sort_index()
    labels_all = pd.concat(label_frames).sort_index()

    n_flare = (labels_all > 0).sum()
    print(f"  Feature matrix: {feats_all.shape}  |  Flare samples: {n_flare} "
          f"({n_flare/len(labels_all)*100:.1f}%)")

    if n_flare < 5:
        print("  WARNING: Very few flare samples for training. "
              "Collect more observation days for a robust model.")

    model, metrics = train(feats_all, labels_all)
    save_model(model, config.OUTPUT_DIR / "flare_model.pkl")

    # Lead times
    lead_times = compute_lead_times(model, feats_all, master_all)
    lead_times.to_csv(config.OUTPUT_DIR / "lead_times.csv", index=False)

    # ── 5. Evaluation ──────────────────────────────────────────────────────
    print("\n[5/5] Evaluating ...")
    det_acc  = detection_accuracy(master_all)
    fc_summ  = forecast_metrics(lead_times)

    # False alarm rate: build binary alert series from model probabilities
    from pipeline.forecaster import predict_proba
    import numpy as np
    proba_mat = model.predict_proba(feats_all.fillna(0))
    flare_prob = pd.Series(1 - proba_mat[:, 0], index=feats_all.index)
    alert_series = flare_prob >= 0.40     # 40% probability threshold for alert
    far = false_alarm_rate(alert_series, master_all)

    print_report(det_acc, fc_summ, far)
    save_report(det_acc, fc_summ, lead_times, config.OUTPUT_DIR)

    print(f"\nAll outputs saved to {config.OUTPUT_DIR}/")
    print("Launch the dashboard with:  streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
