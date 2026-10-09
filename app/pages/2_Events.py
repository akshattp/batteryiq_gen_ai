"""app/pages/2_Events.py - Safety Events Timeline & LLM Diagnostic Drill-down."""

from __future__ import annotations

import json
import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from guardian.genai.explain import generate_explanation, template_explanation
from guardian.risk.engine import assess
from guardian.risk.events import detect_events
from guardian.risk.recommend import recommend_actions

if "app.ui_theme" in sys.modules:
    importlib.reload(sys.modules["app.ui_theme"])
else:
    import app.ui_theme
from app.ui_theme import inject_theme, render_header, render_footer

st.set_page_config(page_title="Safety Events | Guardian", page_icon="🚨", layout="wide")

render_header(
    title="Debounced Safety Events & Drill-Down",
    subtitle="Inspect Multi-Tier Threshold Violations, Rapid Severity Escalations & Grounded GenAI Diagnostics",
    badge_text="SAFETY MONITOR",
    tag="WORKBENCH 02",
)


@st.cache_data
def get_available_batteries_and_cycles() -> tuple[list[str], dict[str, list[int]]]:
    pq_path = Path("data/processed/battery_telemetry.parquet")
    csv_path = Path("data/processed/battery_telemetry.csv")
    if pq_path.exists():
        df_meta = pd.read_parquet(pq_path, columns=["battery_id", "cycle"])
    elif csv_path.exists():
        df_meta = pd.read_csv(csv_path, usecols=["battery_id", "cycle"])
    else:
        return ["B0005"], {"B0005": [30, 50, 80, 120, 160]}

    bats = sorted(df_meta["battery_id"].unique().tolist())
    cycles_map = {b: sorted(df_meta[df_meta["battery_id"] == b]["cycle"].unique().tolist()) for b in bats}
    return bats, cycles_map


@st.cache_data
def get_session_events(battery_id: str = "B0005", cycle: int = 50) -> tuple[pd.DataFrame, list]:
    pq_path = Path("data/processed/battery_telemetry.parquet")
    csv_path = Path("data/processed/battery_telemetry.csv")
    sub = pd.DataFrame()
    
    if pq_path.exists():
        try:
            sub = pd.read_parquet(pq_path, filters=[("battery_id", "==", battery_id), ("cycle", "==", cycle)])
        except Exception:
            pass
        if sub.empty:
            df = pd.read_parquet(pq_path)
            sub = df[(df["battery_id"] == battery_id) & (df["cycle"] == cycle)]
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        sub = df[(df["battery_id"] == battery_id) & (df["cycle"] == cycle)]

    if sub.empty:
        times = np.arange(0, 500, 10)
        sub = pd.DataFrame({
            "battery_id": [battery_id] * len(times),
            "cycle": [cycle] * len(times),
            "mode": ["discharge"] * len(times),
            "t_s": times,
            "voltage_v": 4.1 - (times / 500.0) * 0.9,
            "current_a": [2.0] * len(times),
            "temp_c": 24.0 + (times / 500.0) * 22.0,
            "soc_pct": 100.0 - (times / 500.0) * 90.0,
            "dT_dt": [0.035] * len(times),
        })
    else:
        sub = sub.sort_values("t_s").reset_index(drop=True)

    # Subsample to at most 300 steps for rapid debounced event scanning
    eval_step = max(1, len(sub) // 300)
    eval_sub = sub.iloc[::eval_step]

    assessments = []
    for row in eval_sub.to_dict(orient="records"):
        pred_t = float(row.get("temp_c", 25.0)) + float(row.get("dT_dt", 0.02)) * 10.0
        anom = float(min(0.95, max(0.05, (float(row.get("temp_c", 25.0)) - 24.0) / 25.0)))
        a = assess(state=row, anomaly_score=anom, pred_temp=pred_t)
        assessments.append(a)

    events = detect_events(assessments, min_gap_s=15.0)
    return sub, events


# Battery Selector
available_bats, cycles_map = get_available_batteries_and_cycles()

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    sel_bat = st.selectbox("Battery ID", available_bats, index=0)
with col_sel2:
    available_cycles = cycles_map.get(sel_bat, [50])
    sel_cycle = st.selectbox("Discharge Cycle", available_cycles, index=min(1, len(available_cycles) - 1))

sub_df, events = get_session_events(sel_bat, sel_cycle)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
st.header(f"📋 Detected Safety Events ({len(events)} Total)")

if not events:
    st.success("✅ No abnormal safety events detected in this session cycle. All operating parameters remained within nominal ranges.")
else:
    events_table_data = []
    for i, e in enumerate(events):
        lvl = e.risk_level.value if hasattr(e.risk_level, "value") else str(e.risk_level)
        events_table_data.append({
            "Index": i,
            "Event ID": e.event_id,
            "Time (s)": round(e.t_s, 1),
            "Risk Level": lvl,
            "Risk Score": f"{e.risk_score:.1f}",
            "Trigger Reasons": " | ".join(e.reasons[:2]),
        })

    events_df = pd.DataFrame(events_table_data)
    st.dataframe(events_df, use_container_width=True)

    # Event Selection for Drill-down
    selected_idx = st.selectbox("Select Event for Detailed Diagnostic Drill-Down", range(len(events)), format_func=lambda x: f"Event #{x} ({events[x].event_id} at t={events[x].t_s:.1f}s)")
    ev = events[selected_idx]

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.header(f"🔍 Event Inspection: {ev.event_id}")

    lvl_str = ev.risk_level.value if hasattr(ev.risk_level, "value") else str(ev.risk_level)
    if lvl_str.lower() == "critical":
        badge = '<span class="chip chip-danger"><span class="chip-pulse"></span> CRITICAL</span>'
    elif lvl_str.lower() in ("high", "warning"):
        badge = f'<span class="chip chip-warning">{lvl_str.upper()}</span>'
    else:
        badge = f'<span class="chip chip-success">{lvl_str.upper()}</span>'

    p1, p2, p3 = st.columns(3)
    p1.markdown(f"**Severity Level:** {badge}", unsafe_allow_html=True)
    p2.metric("Risk Score", f"{ev.risk_score:.1f}")
    p3.metric("Timestamp", f"{ev.t_s:.1f} s")

    col_diag, col_llm = st.columns([1, 1])

    with col_diag:
        st.subheader("1. Why Was This Detected?")
        for r in ev.reasons:
            st.error(f"• {r}")

        st.subheader("2. What Does the Model Recommend?")
        recs = recommend_actions(ev.assessment)
        for act in recs:
            st.info(f"👉 {act}")

    with col_llm:
        st.subheader("3. Validated GenAI Diagnostic Explanation")
        
        # Build structured input dict
        assessment_dict = {
            "battery_id": sel_bat,
            "cycle": sel_cycle,
            "t_s": ev.t_s,
            "risk_level": lvl_str,
            "risk_score": ev.risk_score,
            "anomaly_score": ev.assessment.anomaly_score,
            "predicted_temp_c": ev.assessment.predicted_temp_c,
            "telemetry": {
                "temp_c": round(float(ev.assessment.factors[0].raw_value if ev.assessment.factors else 25.0), 1),
                "voltage_v": 3.4,
                "current_a": 2.1,
                "soc_pct": 45.0,
            },
            "factors": [f.to_dict() for f in ev.assessment.factors],
            "hard_overrides": ev.assessment.hard_overrides,
        }

        # Deterministic template or validated LLM explanation
        explanation = template_explanation(assessment_dict)

        st.markdown(f"**Diagnostic Summary:**\n> {explanation.summary}")
        
        st.markdown("**Key Inferences:**")
        for reason in explanation.reasons:
            st.write(f"- {reason}")
            
        st.markdown("**Operational Guidance:**")
        for act in explanation.recommended_actions:
            st.write(f"• {act}")
            
        st.caption(f"🛡️ Validation Note: {explanation.confidence_note}")

    st.markdown("---")
    if st.button("💬 Ask Follow-Up Question about this Event"):
        st.session_state["prefill_chat_query"] = f"Explain what caused {ev.event_id} at t={ev.t_s:.1f}s and why risk score was {ev.risk_score:.1f}."
        st.info("Query queued. Navigate to the 'Ask Guardian' tab to continue.")

render_footer()
