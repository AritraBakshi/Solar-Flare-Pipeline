# Aditya-L1 Solar Flare Nowcasting & Forecasting Pipeline

Bharatiya Antariksh Hackathon — Problem Statement 15

Combines SoLEXS (soft X-ray) and HEL1OS (hard X-ray) Level-1 data from ISRO's
Aditya-L1 mission to detect solar flares in real time (nowcasting) and predict
them ahead of time (forecasting).

## Quick start

```bash
pip install -r requirements.txt

# 1. Drop your downloaded PRADAN files into data/ — see naming convention below
# 2. Run the full pipeline
python run_pipeline.py

# 3. Launch the live dashboard
streamlit run dashboard/app.py
```

## What goes in `data/`

Download Level-1 products from the PRADAN portal
(https://pradan1.issdc.gov.in/al1) and drop them straight into `data/` —
`.gz` or uncompressed both work. **The two instruments use genuinely
different naming conventions, and the reader handles both:**

| Instrument | File pattern                                  | Content                          |
|------------|------------------------------------------------|-----------------------------------|
| SoLEXS     | `AL1_SOLEXS_YYYYMMDD_SDD1_L1.lc[.gz]`          | Soft X-ray light curve, detector 1 |
| SoLEXS     | `AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc[.gz]`          | Soft X-ray light curve, detector 2 |
| SoLEXS     | `AL1_SOLEXS_YYYYMMDD_SDD1_L1.gti[.gz]`         | Good-time-interval mask           |
| HEL1OS     | *anything* — e.g. `lightcurve_czt1.fits`       | Hard X-ray light curve, any detector |

SoLEXS filenames are self-describing (date + detector right in the name).
**HEL1OS filenames carry no date or detector information at all** — the
reader opens every non-SoLEXS file, reads the observation date from the
`ISOSTART`/`MJDSTART` header keyword, and reads the detector name (CZT1,
CZT2, CDTE1, …) straight out of the FITS extension names
(`CZT1_LC_BAND_20.00KEV_TO_40.00KEV`, etc.). You can name HEL1OS files
whatever you want — just drop them in.

HEL1OS also ships in **12-hour chunks** (one file per half-day) rather than
full days. The reader detects when two files share the same observation
date + detector and concatenates them automatically — you don't need to
merge them yourself.

You don't need every file for every day — the pipeline works with whatever
combination is present (e.g. SoLEXS-only days still produce a partial
catalogue). More observation days = a better forecasting model, since a
single day rarely contains every flare class.

## Project layout

```
config.py              All tunable thresholds and parameters — start here to tune behaviour
run_pipeline.py         Single entry point: load → preprocess → nowcast → forecast → evaluate
pipeline/
  reader.py             FITS file discovery, loading, GTI filtering
  preprocessor.py       Resampling, multi-detector merge, GOES W/m² calibration
  nowcaster.py          SXR detector, HXR detector, catalogue merge, GOES classification
  forecaster.py         Feature engineering, multi-class labeling, XGBoost training
  evaluator.py          TPR/FPR tables, lead time stats, false alarm rate
dashboard/
  app.py                Streamlit dashboard: browse mode + simulated live replay
notebooks/
  Pipeline_Walkthrough.ipynb   Read-only-friendly walkthrough for the team — imports
                                and calls the pipeline package directly, doesn't
                                reimplement any logic. Run `python run_pipeline.py`
                                first, then open this for an annotated, plotted
                                walkthrough of the same run.
data/                   Drop your PRADAN downloads here
outputs/                All generated catalogues, model, and reports land here
```

## How it works

**Nowcasting** — `detect_sxr()` estimates a rolling background (median +
MAD-based sigma) and flags sustained excursions above a 5σ threshold lasting
at least 60 seconds (this duration filter is what separates real flares from
noise blips — see `config.SXR_MIN_DURATION_S`). `detect_hxr()` does the same
but additionally requires a positive derivative, since hard X-ray bursts are
impulsive rather than gradual. `merge_catalogues()` then matches HXR bursts
to SXR events within a ±10-minute window (the Neupert effect — hard X-ray
bursts precede or coincide with the soft X-ray peak, not the other way
round), and classifies each merged SXR event into a GOES A/B/C/M/X class
using the calibration in `config.SOLEXS_CALIB_FACTOR`.

**Forecasting** — `build_features()` computes rolling mean/std/slope/max
across multiple time windows (1, 5, 15, 60 minutes) for both channels, plus
a hardness ratio and a background-relative "excess" feature meant to catch
the slow pre-flare rise the judges' slides call "precursor heating."
`build_labels()` assigns a 5-class label (Quiet / A·B / C / M / X) based on
whether — and how strong — a flare starts within the next
`config.FORECAST_HORIZON_MIN` minutes. `train()` fits an XGBoost multi-class
model with a strictly chronological train/test split (never random — that
would leak future information backward).

**Evaluation** — `compute_lead_times()` finds, for every real flare, the
earliest timestamp at which the model's flare probability crossed the alert
threshold, and reports that as minutes of lead time. `false_alarm_rate()`
checks every alert the model fired and verifies whether a real flare
actually followed within a window — alerts that weren't are false alarms.

## Real findings from testing against actual SoLEXS + HEL1OS data

This pipeline has been run end-to-end against real PRADAN downloads for
June 3, 2026 (the X1.0/M9.3/M7.7 day), which surfaced several things worth
knowing about up front:

- **HXR background estimation needs a long window AND a low percentile, not
  a median.** A major flare's hard X-ray envelope (impulsive burst + decay)
  can span 20-30 minutes — long enough that a short median-based background
  window gets contaminated by the event itself and never flags it as
  anomalous. `_rolling_background()` uses a 30-minute window at the 20th
  percentile for HEL1OS specifically to stay anchored to the true quiet
  floor.
- **HXR bursts are genuinely "spiky" (quasi-periodic pulsations) — bridge
  short gaps before grouping into events**, or one real burst gets
  fragmented into many catalogue rows as flux dips below threshold between
  sub-pulses. See `HXR_MERGE_GAP_S` in `config.py`.
- **When matching SXR and HXR events (Neupert effect), pick the strongest
  candidate inside the matching window, not the first one encountered** — a
  weak early sub-pulse can otherwise "steal" the match from the real,
  much stronger burst that actually corresponds to the flare.
- **A large number of short HXR-only bursts with no SXR counterpart is
  expected, not a bug.** On June 3 the detector found 71 hard X-ray
  excursions total but only 3 confirmed SXR+HXR flares — the other ~68 are
  most likely genuine small non-thermal microbursts below the soft X-ray
  flare threshold, a real and documented solar phenomenon. Worth
  highlighting as a finding (it demonstrates sensitivity to low-class
  events) rather than suppressing.
- **GOES classification needs at least 2-3 known flares to calibrate
  against, not 1** — a single-point calibration put the X1.0 flare right at
  the M/X boundary and misclassified it.
- **FITS data is big-endian; pandas/cython resampling will crash on it**
  without an explicit byte-order conversion.
- **`datetime64` resolution varies (`ns` vs `s`) depending on construction
  path in newer pandas** — converting to Unix seconds via `.astype('int64')`
  silently breaks for one of the two cases. Use Timedelta division instead,
  which is resolution-agnostic.

## Known limitations to be upfront about

- **One observation day is not enough to evaluate properly.** With only one
  day of data, all real flares tend to land in the training split, leaving
  the test split with no positive examples — accuracy will look perfect but
  is not a meaningful estimate. Add more days before trusting the TPR/FPR
  numbers (the `data/` folder structure already supports dropping in
  multiple days — file discovery and merging is automatic).
- **GOES cross-calibration is still approximate.** `SOLEXS_CALIB_FACTOR` in
  `config.py` was derived by averaging across the 3 known flares on one day.
  It's good enough for approximate classification, but cross-calibrating
  against the real GOES XRS CSV (set `config.GOES_CSV_PATH`) will make this
  more rigorous — `preprocessor.calibrate_to_flux()` already supports this,
  it just needs the file.
