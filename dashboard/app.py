"""
dashboard/app.py — Aditya-L1 Solar Flare Nowcasting & Forecasting Dashboard

Run with:
    streamlit run dashboard/app.py

Two modes:
  • Replay mode  — step through a historical day as if it were arriving live,
                    triggering nowcast/forecast alerts exactly as they would fire
  • Browse mode  — inspect the full day's catalogue and light curves at once

Reads directly from outputs/ (produced by run_pipeline.py) — run that first.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from pipeline.forecaster import load_model, predict_proba

st.set_page_config(page_title="Aditya-L1 Flare Pipeline", layout="wide", page_icon="☀️")


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data
def list_available_dates():
    pattern = "combined_*.csv"
    files = sorted(config.OUTPUT_DIR.glob(pattern))
    return [f.stem.replace("combined_", "") for f in files]


@st.cache_data
def load_combined(date: str) -> pd.DataFrame:
    df = pd.read_csv(config.OUTPUT_DIR / f"combined_{date}.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


@st.cache_data
def load_master_catalogue() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "master_catalogue_ALL.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["start_utc", "peak_utc", "end_utc"])
    return df


@st.cache_resource
def load_forecast_model():
    path = config.OUTPUT_DIR / "flare_model.pkl"
    if not path.exists():
        return None
    return load_model(path)


@st.cache_data
def load_features_for(date: str) -> pd.DataFrame | None:
    from pipeline.forecaster import build_features
    combined = load_combined(date)
    return build_features(combined)


# ── Sidebar controls ───────────────────────────────────────────────────────────

st.sidebar.title("☀️ Aditya-L1 Flare Pipeline")
available_dates = list_available_dates()

if not available_dates:
    st.error(
        "No processed data found in outputs/.\n\n"
        "Run `python run_pipeline.py` first to process the files in data/."
    )
    st.stop()

selected_date = st.sidebar.selectbox("Observation date", available_dates, index=0)
mode = st.sidebar.radio("Mode", ["Browse (full day)", "Replay (simulated live)"])

combined = load_combined(selected_date)
master_cat = load_master_catalogue()
day_events = master_cat[
    master_cat["peak_utc"].dt.strftime("%Y%m%d") == selected_date
] if len(master_cat) > 0 else pd.DataFrame()

model = load_forecast_model()
feats = load_features_for(selected_date) if model is not None else None


# ── Helper: build the light curve figure ──────────────────────────────────────

def build_lightcurve_fig(df: pd.DataFrame, events: pd.DataFrame,
                         up_to: pd.Timestamp | None = None) -> go.Figure:
    """Build the dual-panel SXR/HXR light curve figure with event markers."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Soft X-ray (SoLEXS) — GOES-equivalent flux",
                        "Hard X-ray (HEL1OS) — count rate"),
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45],
    )

    plot_df = df if up_to is None else df[df.index <= up_to]

    if "SXR_FLUX" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["SXR_FLUX"],
                      mode="lines", name="SXR flux (W/m²)",
                      line=dict(color="#378ADD", width=1)),
            row=1, col=1,
        )
        fig.update_yaxes(type="log", title_text="W/m²", row=1, col=1)

    if "HXR_COUNTS" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["HXR_COUNTS"],
                      mode="lines", name="HXR counts/s",
                      line=dict(color="#D85A30", width=1)),
            row=2, col=1,
        )
        fig.update_yaxes(type="log", title_text="counts/s", row=2, col=1)

    # Event markers
    for _, ev in events.iterrows():
        if up_to is not None and ev["peak_utc"] > up_to:
            continue
        color = {"A": "#999", "B": "#999", "C": "#EF9F27",
                 "M": "#D85A30", "X": "#A6195E"}.get(ev.get("goes_class", "?"), "#999")
        fig.add_vline(
            x=ev["peak_utc"], line_dash="dot", line_color=color, line_width=1.5,
            row=1, col=1,
        )
        fig.add_annotation(
            x=ev["peak_utc"], y=1.05, yref="y domain",
            text=f"{ev.get('goes_class','?')}", showarrow=False,
            font=dict(size=10, color=color), row=1, col=1,
        )

    fig.update_layout(
        height=520, margin=dict(t=40, b=20, l=60, r=20),
        showlegend=False, hovermode="x unified",
    )
    return fig


# ── Main layout ────────────────────────────────────────────────────────────────

st.title("Aditya-L1 Solar Flare Nowcasting & Forecasting")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Events detected", len(day_events))
col2.metric("Confirmed SXR+HXR", int((day_events["source"] == "SXR+HXR").sum()) if len(day_events) else 0)
col3.metric("Highest class", day_events["goes_class"].max() if len(day_events) else "—")
col4.metric("Forecast model", "Loaded ✓" if model is not None else "Not trained")

st.divider()

if mode == "Browse (full day)":
    st.plotly_chart(build_lightcurve_fig(combined, day_events), width='stretch')

    st.subheader("Flare catalogue")
    if len(day_events) > 0:
        display_cols = ["start_utc", "peak_utc", "end_utc", "goes_class",
                        "sxr_peak_flux_wm2", "hxr_detected", "source", "duration_s"]
        st.dataframe(
            day_events[[c for c in display_cols if c in day_events.columns]],
            width='stretch',
        )
    else:
        st.info("No flares detected for this date.")

else:  # Replay mode
    st.caption(
        "Simulates the pipeline running live: steps through the day second-by-second, "
        "showing nowcast alerts the instant the detector fires and forecast probability "
        "as the model would have seen it in real time."
    )

    run = st.sidebar.button("▶ Start replay", width='stretch')
    speed = st.sidebar.slider("Replay speed (minutes per tick)", 1, 60, 15)

    chart_slot  = st.empty()
    alert_slot  = st.empty()
    gauge_slot  = st.empty()

    if run:
        start_t = combined.index.min()
        end_t   = combined.index.max()
        cursor  = start_t
        step    = pd.Timedelta(minutes=speed)

        fired_alerts = set()

        while cursor <= end_t:
            fig = build_lightcurve_fig(combined, day_events, up_to=cursor)
            chart_slot.plotly_chart(fig, width='stretch', key=f"chart_{cursor}")

            # Nowcast alert: any event whose peak has just occurred
            just_fired = day_events[
                (day_events["peak_utc"] > cursor - step) &
                (day_events["peak_utc"] <= cursor)
            ]
            for _, ev in just_fired.iterrows():
                key = ev["peak_utc"]
                if key not in fired_alerts:
                    fired_alerts.add(key)
                    alert_slot.error(
                        f"🚨 NOWCAST ALERT — {ev.get('goes_class','?')}-class flare peak detected "
                        f"at {ev['peak_utc'].strftime('%H:%M:%S')} UTC"
                    )

            # Forecast probability gauge
            if model is not None and feats is not None:
                past_feats = feats[feats.index <= cursor]
                if len(past_feats) > 0:
                    proba = predict_proba(model, past_feats)
                    flare_p = 1 - proba[0]
                    cols = gauge_slot.columns(5)
                    for i, (name, p) in enumerate(zip(config.CLASS_NAMES, proba)):
                        cols[i].metric(name, f"{p*100:.1f}%")
                    if flare_p >= 0.4:
                        alert_slot.warning(
                            f"⚠️ FORECAST ALERT — {flare_p*100:.0f}% probability of a flare "
                            f"in the next {config.FORECAST_HORIZON_MIN} minutes "
                            f"(as of {cursor.strftime('%H:%M:%S')} UTC)"
                        )

            cursor += step
            time.sleep(0.05)

        st.success("Replay complete.")
    else:
        st.info("Press ▶ Start replay in the sidebar to begin the simulated live run.")

st.divider()
st.caption(
    "Aditya-L1 SoLEXS + HEL1OS combined nowcasting/forecasting pipeline · "
    "Bharatiya Antariksh Hackathon — Problem Statement 15"
)
