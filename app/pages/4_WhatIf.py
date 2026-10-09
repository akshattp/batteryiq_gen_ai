"""app/pages/4_WhatIf.py - Interactive What-If Simulation & OOD Guard (M14 / Section 7)."""

from __future__ import annotations

import sys
import importlib
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from guardian.risk.engine import assess
from guardian.risk.whatif import simulate_whatif

if "app.ui_theme" in sys.modules:
    importlib.reload(sys.modules["app.ui_theme"])
else:
    import app.ui_theme
from app.ui_theme import inject_theme, render_header, render_footer, make_gauge_chart

st.set_page_config(page_title="What-If Simulation | Guardian", page_icon="🎛️", layout="wide")

render_header(
    title="Interactive What-If Simulation Lab",
    subtitle="Stress-Test Electrochemical Operating Bounds, Observe Thermal Horizon & Verify Safety Guardrails",
    badge_text="SIMULATION TWIN",
    tag="WORKBENCH 04",
)

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

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
st.header("3. Side-by-Side Comparison: Current vs What-If")

# Out of Distribution Warning Banner
if is_ood:
    st.markdown(
        '<div style="margin-bottom: 12px;"><span class="chip chip-warning">⚠️ OUTSIDE TRAINING RANGE &bull; LOW CONFIDENCE EXTRAPOLATION</span></div>',
        unsafe_allow_html=True,
    )
    st.warning("One or more operating parameters exceed the NASA baseline training envelope (Current: 0-3.0A, Temp: 18-45°C).")

c_res1, c_res2 = st.columns(2)

base_lvl = base_assessment.risk_level.value if hasattr(base_assessment.risk_level, "value") else str(base_assessment.risk_level)
mod_lvl = mod_assessment.risk_level.value if hasattr(mod_assessment.risk_level, "value") else str(mod_assessment.risk_level)

with c_res1:
    st.subheader("🔵 Baseline State Results")
    fig_base_gauge = make_gauge_chart(base_assessment.risk_score, "BASELINE RISK")
    st.plotly_chart(fig_base_gauge, use_container_width=True)
    st.markdown(f"**Severity Level:** <span class='chip chip-info'>{base_lvl}</span>", unsafe_allow_html=True)
    st.metric("Predicted Temp (t+10)", f"{base_assessment.predicted_temp_c:.1f} °C")

with c_res2:
    st.subheader("🟠 Simulated What-If Results")
    fig_mod_gauge = make_gauge_chart(mod_assessment.risk_score, "SIMULATED RISK")
    st.plotly_chart(fig_mod_gauge, use_container_width=True)
    score_delta = mod_assessment.risk_score - base_assessment.risk_score
    temp_delta = mod_assessment.predicted_temp_c - base_assessment.predicted_temp_c
    mod_chip = '<span class="chip chip-danger"><span class="chip-pulse"></span> CRITICAL</span>' if mod_assessment.risk_score >= 75 else ('<span class="chip chip-warning">ELEVATED</span>' if mod_assessment.risk_score >= 50 else '<span class="chip chip-info">NOMINAL</span>')
    st.markdown(f"**Severity Level:** {mod_chip}", unsafe_allow_html=True)
    st.metric("Predicted Temp Delta", f"{mod_assessment.predicted_temp_c:.1f} °C", delta=f"{temp_delta:+.1f} °C")

if mod_assessment.hard_overrides:
    st.markdown("<h4 class='h4-title' style='color: #FF6500;'>🚨 Active Hard Safety Overrides:</h4>", unsafe_allow_html=True)
    for h in mod_assessment.hard_overrides:
        st.error(f"• {h}")

# Comparison Bar Chart
st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
st.header("📊 Comparative Risk Metric Delta")

fig = go.Figure()
fig.add_trace(go.Bar(
    name="Baseline State",
    x=["Temperature (°C)", "Predicted Temp (°C)", "Discharge Current (A)", "Risk Score (0-100)"],
    y=[base_temp, base_assessment.predicted_temp_c, base_current, base_assessment.risk_score],
    marker_color="#0091B9",
))
fig.add_trace(go.Bar(
    name="What-If Scenario",
    x=["Temperature (°C)", "Predicted Temp (°C)", "Discharge Current (A)", "Risk Score (0-100)"],
    y=[mod_temp, mod_assessment.predicted_temp_c, mod_current, mod_assessment.risk_score],
    marker_color="#FF6500" if mod_assessment.risk_score > 55 else "#FFD500",
))
fig.update_layout(barmode="group", template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)

render_footer()
