"""app/pages/7_AutoHeal_Report.py - Auto-Heal Audit Report & Real-time Logs (Section 7)."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Auto-Heal Report | Guardian", page_icon="🩹", layout="wide")

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
    .metric-box {
        background-color: #1a1d24;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩹 Auto-Heal Loop Audit Report")
st.caption("Live monitoring of self-healing mechanisms across development-time tests, LLM runtime retries, and sandboxed code execution.")

# Load reports/autoheal/log.jsonl
log_file = Path("reports/autoheal/log.jsonl")
records = []
if log_file.exists():
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

df_logs = pd.DataFrame(records)

# KPI Metrics
total_events = len(df_logs)
fails = len(df_logs[df_logs["status"] == "FAIL"]) if not df_logs.empty else 0
passes = len(df_logs[df_logs["status"] == "PASS"]) if not df_logs.empty else 0
fixes = len(df_logs[df_logs["status"] == "FIX"]) if not df_logs.empty else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Execution Runs", total_events)
m2.metric("Detected Failures", fails)
m3.metric("Verified Fixes / Passes", passes)
heal_rate = f"{(passes / max(1, passes + fails - passes)) * 100:.1f} %" if passes else "100.0 %"
m4.metric("Heal Success Rate", heal_rate)

st.markdown("---")

col_left, col_right = st.columns([1, 1])

if not df_logs.empty and "source" in df_logs.columns:
    with col_left:
        st.subheader("📊 Breakdown by Source")
        source_counts = df_logs["source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Count"]
        fig_source = px.pie(source_counts, values="Count", names="Source", title="Runs by Source", color_discrete_sequence=px.colors.qualitative.Dark24)
        fig_source.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig_source, use_container_width=True)

    with col_right:
        st.subheader("⚠️ Error Taxonomy Distribution")
        err_df = df_logs[df_logs["status"] == "FAIL"]["error_class"].value_counts().reset_index()
        err_df.columns = ["Error Class", "Count"]
        if not err_df.empty:
            fig_err = px.bar(err_df, x="Error Class", y="Count", color="Count", title="Detected Error Classes", color_continuous_scale="Reds")
            fig_err.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig_err, use_container_width=True)
        else:
            st.info("No active errors recorded.")

st.markdown("---")
st.subheader("📋 Auto-Heal Execution Log Table")

if not df_logs.empty:
    display_cols = [c for c in ["ts", "source", "milestone", "attempt", "status", "error_class", "fix_summary", "duration_s"] if c in df_logs.columns]
    st.dataframe(df_logs[display_cols].sort_values("ts", ascending=False), use_container_width=True)
else:
    st.info("No records present in reports/autoheal/log.jsonl yet.")

# Render AUTOHEAL_REPORT.md if present
report_md = Path("reports/AUTOHEAL_REPORT.md")
if report_md.exists():
    st.markdown("---")
    st.subheader("📄 Formal Auto-Heal Report")
    with open(report_md, "r", encoding="utf-8") as f:
        st.markdown(f.read())

st.markdown("---")
st.markdown(
    '<div class="footer-guard">⚠️ <strong>Decision-support prototype. Not for vehicle control.</strong></div>',
    unsafe_allow_html=True,
)
