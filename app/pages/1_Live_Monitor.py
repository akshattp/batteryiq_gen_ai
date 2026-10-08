"""app/pages/1_Live_Monitor.py - Live Telemetry Monitor & Assessment (Section 7 / Exp 8)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from guardian.risk.engine import assess
from guardian.risk.events import detect_events

st.set_page_config(page_title="Live Monitor | Guardian", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .footer-guard {
        margin-top: 50px;
        padding: 14px;
        background-color: #161b22;
        border-top: 1px solid #30363d;
        text-align: center;
        font-size: 0.85rem;
        color: #8b949e;
    }
    .badge-normal { background-color: #0e4429; color: #3fb950; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
    .badge-warning { background-color: #4d3800; color: #d29922; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
    .badge-high { background-color: #5a1e02; color: #f0883e; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
    .badge-critical { background-color: #5c1414; color: #f85149; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Live Telemetry Monitor & Assessment")
st.caption("Replay NASA battery telemetry and observe real-time AI risk scoring.")


@st.cache_data
def get_telemetry_metadata() -> tuple[list[str], dict[str, list[int]]]:
    pq_path = Path("data/processed/battery_telemetry.parquet")
    csv_path = Path("data/processed/battery_telemetry.csv")
    if pq_path.exists():
        df_meta = pd.read_parquet(pq_path, columns=["battery_id", "cycle"])
    elif csv_path.exists():
        df_meta = pd.read_csv(csv_path, usecols=["battery_id", "cycle"])
    else:
        return ["B0005"], {"B0005": [50]}
    
    bats = sorted(df_meta["battery_id"].unique().tolist())
    cycles_by_bat = {b: sorted(df_meta[df_meta["battery_id"] == b]["cycle"].unique().tolist()) for b in bats}
    return bats, cycles_by_bat


@st.cache_data
def get_session_telemetry(battery_id: str, cycle: int) -> pd.DataFrame:
    pq_path = Path("data/processed/battery_telemetry.parquet")
    csv_path = Path("data/processed/battery_telemetry.csv")
    
    if pq_path.exists():
        try:
            sub = pd.read_parquet(pq_path, filters=[("battery_id", "==", battery_id), ("cycle", "==", cycle)])
            if not sub.empty:
                return sub.sort_values("t_s").reset_index(drop=True)
        except Exception:
            pass
        df_all = pd.read_parquet(pq_path)
        sub = df_all[(df_all["battery_id"] == battery_id) & (df_all["cycle"] == cycle)]
        if not sub.empty:
            return sub.sort_values("t_s").reset_index(drop=True)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        sub = df[(df["battery_id"] == battery_id) & (df["cycle"] == cycle)]
        if not sub.empty:
            return sub.sort_values("t_s").reset_index(drop=True)

    # Fallback synthetic demo telemetry
    times = np.arange(0, 1000, 10)
    return pd.DataFrame({
        "battery_id": [battery_id] * len(times),
        "cycle": [cycle] * len(times),
        "mode": ["discharge"] * len(times),
        "t_s": times,
        "voltage_v": 4.1 - (times / 1000.0) * 0.8,
        "current_a": [1.95] * len(times),
        "temp_c": 24.0 + (times / 1000.0) * 12.0,
        "ambient_c": [24.0] * len(times),
        "soc_pct": 100.0 - (times / 1000.0) * 90.0,
        "soh_pct": [91.5] * len(times),
        "dT_dt": [0.012] * len(times),
    })


available_bats, cycles_map = get_telemetry_metadata()

# 1. Sidebar Session Controls
st.sidebar.header("🕹️ Session & Replay Controls")
selected_bat = st.sidebar.selectbox("Select Battery Cell", available_bats, index=0)

available_cycles = cycles_map.get(selected_bat, [50])
selected_cycle = st.sidebar.selectbox("Select Discharge Cycle", available_cycles, index=0)

session_df = get_session_telemetry(selected_bat, selected_cycle)

# Replay slider
n_steps = len(session_df)
step_idx = st.sidebar.slider("Replay Time Step", 0, max(0, n_steps - 1), max(0, n_steps // 2))
current_row = session_df.iloc[step_idx]

# 2. Risk Assessment Calculation for Current Step
pred_temp = float(current_row.get("temp_c", 25.0)) + float(current_row.get("dT_dt", 0.02)) * 10.0
anomaly_score = float(np.clip((current_row.get("temp_c", 25.0) - 24.0) / 30.0, 0.05, 0.95))

assessment = assess(
    state=current_row.to_dict(),
    anomaly_score=anomaly_score,
    pred_temp=pred_temp,
)

level_str = assessment.risk_level.value if hasattr(assessment.risk_level, "value") else str(assessment.risk_level)
badge_class = f"badge-{level_str.lower()}"

# 3. Top AI Assessment Card
st.subheader("🤖 AI Real-Time Risk Assessment")
a_col1, a_col2, a_col3, a_col4 = st.columns(4)

with a_col1:
    st.markdown(f"**Risk Level:** <span class='{badge_class}'>{level_str}</span>", unsafe_allow_html=True)
with a_col2:
    st.metric("Risk Score (0–100)", f"{assessment.risk_score:.1f}")
with a_col3:
    st.metric("Anomaly Score", f"{anomaly_score:.3f}", delta=f"{'ABOVE' if anomaly_score > 0.4 else 'NOMINAL'}")
with a_col4:
    st.metric("Predicted Temp (t+10)", f"{pred_temp:.1f} °C")

if assessment.hard_overrides:
    for h in assessment.hard_overrides:
        st.warning(f"🚨 **Hard Override Active:** {h}")

st.markdown("---")

# 4. Telemetry KPI Cards
st.subheader("📊 Cell Operating State")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Temperature", f"{current_row.get('temp_c', 0.0):.1f} °C")
k2.metric("Voltage", f"{current_row.get('voltage_v', 0.0):.3f} V")
k3.metric("Current", f"{current_row.get('current_a', 0.0):.2f} A")
k4.metric("SOC", f"{current_row.get('soc_pct', 0.0):.1f} %")
k5.metric("Cycle", f"{int(current_row.get('cycle', 1))}")
k6.metric("SOH", f"{current_row.get('soh_pct', 90.0):.1f} %")

st.markdown("---")

# 5. Six Synchronized Charts (Optimized downsampling for instant client-side rendering)
st.subheader("📈 Telemetry Time Series (Current Replay Point Marked)")

t_curr = current_row["t_s"]

# Downsample session series to <= 300 points for smooth, lag-free Plotly rendering
plot_step = max(1, len(session_df) // 300)
plot_df = session_df.iloc[::plot_step].copy()

c1, c2 = st.columns(2)

with c1:
    # 1. Temp chart
    fig_temp = px.line(plot_df, x="t_s", y="temp_c", title="Cell Temperature (°C)", color_discrete_sequence=["#f85149"])
    fig_temp.add_vline(x=t_curr, line_dash="dash", line_color="#ffffff", annotation_text="t_now")
    fig_temp.add_hline(y=35.0, line_dash="dot", line_color="#d29922", annotation_text="Warn (35°C)")
    fig_temp.add_hline(y=50.0, line_dash="dot", line_color="#f85149", annotation_text="Crit (50°C)")
    fig_temp.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_temp, use_container_width=True)

    # 2. Current chart
    fig_curr = px.line(plot_df, x="t_s", y="current_a", title="Discharge Current (A)", color_discrete_sequence=["#58a6ff"])
    fig_curr.add_vline(x=t_curr, line_dash="dash", line_color="#ffffff")
    fig_curr.add_hline(y=2.0, line_dash="dot", line_color="#d29922", annotation_text="Rated (2.0A)")
    fig_curr.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_curr, use_container_width=True)

    # 3. Anomaly score chart (Vectorized numpy computation)
    plot_anom = np.clip((plot_df["temp_c"].values - 24.0) / 30.0, 0.05, 0.95)
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=plot_df["t_s"], y=plot_anom, mode="lines", name="Anomaly Score", line=dict(color="#bc8cff")))
    fig_anom.add_hline(y=0.45, line_dash="dash", line_color="#d29922", annotation_text="Threshold (0.45)")
    fig_anom.add_vline(x=t_curr, line_dash="dash", line_color="#ffffff")
    fig_anom.update_layout(title="Beta-VAE Anomaly Score", template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_anom, use_container_width=True)

with c2:
    # 4. Voltage chart
    fig_v = px.line(plot_df, x="t_s", y="voltage_v", title="Terminal Voltage (V)", color_discrete_sequence=["#3fb950"])
    fig_v.add_vline(x=t_curr, line_dash="dash", line_color="#ffffff")
    fig_v.add_hline(y=2.5, line_dash="dot", line_color="#f85149", annotation_text="Min (2.5V)")
    fig_v.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_v, use_container_width=True)

    # 5. SOC chart
    fig_soc = px.line(plot_df, x="t_s", y="soc_pct", title="State of Charge (%)", color_discrete_sequence=["#79c0ff"])
    fig_soc.add_vline(x=t_curr, line_dash="dash", line_color="#ffffff")
    fig_soc.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_soc, use_container_width=True)

    # 6. Predicted vs Actual T
    dT_dt_vals = plot_df["dT_dt"].values if "dT_dt" in plot_df.columns else 0.02
    plot_pred_t = plot_df["temp_c"].values + dT_dt_vals * 10.0
    fig_pvsa = go.Figure()
    fig_pvsa.add_trace(go.Scatter(x=plot_df["t_s"], y=plot_df["temp_c"], mode="lines", name="Actual T", line=dict(color="#58a6ff")))
    fig_pvsa.add_trace(go.Scatter(x=plot_df["t_s"], y=plot_pred_t, mode="lines", name="Predicted T (t+10)", line=dict(color="#f0883e", dash="dot")))
    fig_pvsa.add_vline(x=t_curr, line_dash="dash", line_color="#ffffff")
    fig_pvsa.update_layout(title="Predicted vs Actual Temperature (°C)", template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_pvsa, use_container_width=True)

# 6. Contributing Factors Bar Chart
st.subheader("⚖️ Risk Component Breakdown")
factors_data = []
for f in assessment.factors:
    factors_data.append({
        "Factor": f.name,
        "Raw Value": f.raw_value,
        "Weighted Contribution": f.weighted_score,
        "Weight": f"{f.weight*100:.0f}%",
    })

factors_df = pd.DataFrame(factors_data)
fig_bar = px.bar(
    factors_df,
    x="Weighted Contribution",
    y="Factor",
    orientation="h",
    color="Weighted Contribution",
    color_continuous_scale="Reds",
    title="Top Contributing Risk Factors",
)
fig_bar.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_bar, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    '<div class="footer-guard">⚠️ <strong>Decision-support prototype. Not for vehicle control.</strong></div>',
    unsafe_allow_html=True,
)
