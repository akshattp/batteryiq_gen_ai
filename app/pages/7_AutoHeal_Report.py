"""app/pages/7_AutoHeal_Report.py - Auto-Heal Audit Report & Real-time Logs (Section 7)."""

from __future__ import annotations

import json
import sys
import importlib
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

if "app.ui_theme" in sys.modules:
    importlib.reload(sys.modules["app.ui_theme"])
else:
    import app.ui_theme
from app.ui_theme import inject_theme, render_header, render_footer

st.set_page_config(page_title="Auto-Heal Report | Guardian", page_icon="🩹", layout="wide")

render_header(
    title="Auto-Heal Audit Report & Diagnostic Traces",
    subtitle="Monitoring Automated Self-Correction Loops across Unit Tests, Runtime LLM Retries & Sandboxed Code Execution",
    badge_text="AUTO-HEAL ONLINE",
    tag="AUDIT 07",
)

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
with m1:
    st.markdown(
        f"""
        <div class="stat-tile" style="border-left: 4px solid #0091B9;">
            <div class="stat-tile-label" style="color: #BAE4F0;">🚀 Total Test Runs</div>
            <div class="stat-tile-val">{total_events}</div>
            <div class="stat-tile-sub">CI/CD & Runtime Invocations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""
        <div class="stat-tile" style="border-left: 4px solid #FF6500;">
            <div class="stat-tile-label" style="color: #FF6500;">⚠️ Detected Failures</div>
            <div class="stat-tile-val" style="color: {'#FF6500' if fails else '#0091B9'};">{fails}</div>
            <div class="stat-tile-sub">Captured Syntax & Logic Errors</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""
        <div class="stat-tile" style="border-left: 4px solid #BAE4F0;">
            <div class="stat-tile-label" style="color: #BAE4F0;">✅ Verified Fixes / Passes</div>
            <div class="stat-tile-val" style="color: #BAE4F0;">{passes}</div>
            <div class="stat-tile-sub">Clean Re-test Passes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m4:
    heal_rate_val = f"{(passes / max(1, passes + fails - passes)) * 100:.1f}%" if passes else "100.0%"
    st.markdown(
        f"""
        <div class="stat-tile" style="border-left: 4px solid #FFD500;">
            <div class="stat-tile-label" style="color: #FFD500;">🛡️ Heal Success Rate</div>
            <div class="stat-tile-val" style="color: #FFD500;">{heal_rate_val}</div>
            <div class="stat-tile-sub">Autonomous Healing Ratio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1])

if not df_logs.empty and "source" in df_logs.columns:
    with col_left:
        st.subheader("📊 Breakdown by Source")
        source_counts = df_logs["source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Count"]
        fig_source = px.pie(
            source_counts,
            values="Count",
            names="Source",
            title="Runs by Source",
            color_discrete_sequence=["#0091B9", "#BAE4F0", "#FFD500", "#FF6500", "#004E9B"],
        )
        fig_source.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280)
        st.plotly_chart(fig_source, use_container_width=True)

    with col_right:
        st.subheader("⚠️ Error Taxonomy Distribution")
        err_df = df_logs[df_logs["status"] == "FAIL"]["error_class"].value_counts().reset_index()
        err_df.columns = ["Error Class", "Count"]
        if not err_df.empty:
            fig_err = px.bar(
                err_df,
                x="Error Class",
                y="Count",
                color="Count",
                title="Detected Error Classes",
                color_continuous_scale=[[0.0, "#FFD500"], [1.0, "#FF6500"]],
            )
            fig_err.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280)
            st.plotly_chart(fig_err, use_container_width=True)
        else:
            st.info("No active errors recorded.")

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
st.header("📋 Auto-Heal Execution Log Table")

if not df_logs.empty:
    display_cols = [c for c in ["ts", "source", "milestone", "attempt", "status", "error_class", "fix_summary", "duration_s"] if c in df_logs.columns]
    st.dataframe(df_logs[display_cols].sort_values("ts", ascending=False), use_container_width=True)
else:
    st.info("No records present in reports/autoheal/log.jsonl yet.")

# Render AUTOHEAL_REPORT.md if present
report_md = Path("reports/AUTOHEAL_REPORT.md")
if report_md.exists():
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.header("📄 Formal Auto-Heal Report")
    with open(report_md, "r", encoding="utf-8") as f:
        st.markdown(
            f"""
            <div class="angular-card" style="padding: 24px 32px;">
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f.read())
        st.markdown("</div>", unsafe_allow_html=True)

render_footer()
