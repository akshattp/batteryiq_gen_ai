"""app/pages/4_WhatIf.py - Interactive What-If Simulation & OOD Guard (M14 / Section 7)."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from guardian.risk.engine import assess
from guardian.risk.whatif import simulate_whatif

st.set_page_config(page_title="What-If Simulation | Guardian", page_icon="🎛️", layout="wide")

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
    .badge-normal { background-color: #0e4429; color: #3fb950; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-warning { background-color: #4d3800; color: #d29922; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-high { background-color: #5a1e02; color: #f0883e; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-critical { background-color: #5c1414; color: #f85149; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-ood {
        background-color: #ffebe9;
        border: 2px solid #cf222e;
        color: #cf222e;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎛️ Interactive What-If Simulation")
st.caption("Stress-test battery operating conditions, observe forward risk projection, and verify safety guardrails.")

# Baseline parameters
col_base, col_mod = st.columns([1, 1])

with col_base:
    st.subheader("1. Baseline Operating State")
    base_current = st.number_input("Baseline Current (A)", 0.0, 5.0, 1.8, 0.1)
    base_temp = st.number_input("Baseline Temp (°C)", 15.0, 60.0, 26.5, 0.5)
    base_ambient = st.number_input("Baseline Ambient Temp (°C)", 15.0, 50.0, 24.0, 0.5)
    base_soc = st.number_input("Baseline SOC (%)", 0.0, 100.0, 80.0, 5.0)
    base_voltage = st.number_input("Baseline Voltage (V)", 2.0, 4.5, 3.85, 0.05)

with col_mod:
    st.subheader("2. Modified Simulation Parameters")
    mod_current = st.slider("Simulated Discharge Current (A)", 0.0, 6.0, 2.8, 0.1)
    mod_temp = st.slider("Simulated Initial Temp (°C)", 15.0, 70.0, 36.0, 0.5)
    mod_ambient = st.slider("Simulated Ambient Temp (°C)", 15.0, 60.0, 32.0, 0.5)
    mod_soc = st.slider("Simulated State of Charge (%)", 0.0, 100.0, 70.0, 5.0)

# Run Simulations
base_state = {
    "battery_id": "B0005",
    "current_a": base_current,
    "temp_c": base_temp,
    "ambient_c": base_ambient,
    "soc_pct": base_soc,
    "voltage_v": base_voltage,
    "dT_dt": (base_current / 2.0) * 0.03,
}

mod_params = {
    "current_a": mod_current,
    "temp_c": mod_temp,
    "ambient_c": mod_ambient,
    "soc_pct": mod_soc,
}

base_assessment = assess(
    state=base_state,
    anomaly_score=0.1,
    pred_temp=base_temp + (base_current / 2.0) * 0.3,
)

mod_assessment, is_ood = simulate_whatif(base_state, mod_params)

st.markdown("---")
st.subheader("3. Side-by-Side Comparison: Current vs What-If")

# Out of Distribution Warning Banner
if is_ood:
    st.markdown(
        '<div class="badge-ood">⚠️ Outside training range — extrapolation, low confidence</div>',
        unsafe_allow_html=True,
    )
    st.warning("One or more operating parameters exceed the NASA baseline training envelope (Current: 0-3.0A, Temp: 18-45°C).")

c_res1, c_res2 = st.columns(2)

base_lvl = base_assessment.risk_level.value if hasattr(base_assessment.risk_level, "value") else str(base_assessment.risk_level)
mod_lvl = mod_assessment.risk_level.value if hasattr(mod_assessment.risk_level, "value") else str(mod_assessment.risk_level)

with c_res1:
    st.markdown("#### 🔵 Baseline State Results")
    st.markdown(f"**Risk Level:** <span class='badge-{base_lvl.lower()}'>{base_lvl}</span>", unsafe_allow_html=True)
    st.metric("Risk Score", f"{base_assessment.risk_score:.1f}")
    st.metric("Predicted Temp (t+10)", f"{base_assessment.predicted_temp_c:.1f} °C")

with c_res2:
    st.markdown("#### 🟠 Simulated What-If Results")
    st.markdown(f"**Risk Level:** <span class='badge-{mod_lvl.lower()}'>{mod_lvl}</span>", unsafe_allow_html=True)
    score_delta = mod_assessment.risk_score - base_assessment.risk_score
    temp_delta = mod_assessment.predicted_temp_c - base_assessment.predicted_temp_c
    st.metric("Risk Score", f"{mod_assessment.risk_score:.1f}", delta=f"{score_delta:+.1f}")
    st.metric("Predicted Temp (t+10)", f"{mod_assessment.predicted_temp_c:.1f} °C", delta=f"{temp_delta:+.1f} °C")

if mod_assessment.hard_overrides:
    st.markdown("#### 🚨 Active Hard Safety Overrides:")
    for h in mod_assessment.hard_overrides:
        st.error(f"• {h}")

# Comparison Bar Chart
st.markdown("---")
st.subheader("📊 Comparative Risk Metric Delta")

fig = go.Figure()
fig.add_trace(go.Bar(
    name="Baseline State",
    x=["Temperature (°C)", "Predicted Temp (°C)", "Discharge Current (A)", "Risk Score (0-100)"],
    y=[base_temp, base_assessment.predicted_temp_c, base_current, base_assessment.risk_score],
    marker_color="#58a6ff",
))
fig.add_trace(go.Bar(
    name="What-If Scenario",
    x=["Temperature (°C)", "Predicted Temp (°C)", "Discharge Current (A)", "Risk Score (0-100)"],
    y=[mod_temp, mod_assessment.predicted_temp_c, mod_current, mod_assessment.risk_score],
    marker_color="#f85149" if mod_assessment.risk_score > 55 else "#f0883e",
))
fig.update_layout(barmode="group", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    '<div class="footer-guard">⚠️ <strong>Decision-support prototype. Not for vehicle control.</strong></div>',
    unsafe_allow_html=True,
)
