"""app/pages/6_Model_Card.py - Model Card Viewer (Section 7)."""

from __future__ import annotations

import sys
import importlib
from pathlib import Path
import streamlit as st

if "app.ui_theme" in sys.modules:
    importlib.reload(sys.modules["app.ui_theme"])
else:
    import app.ui_theme
from app.ui_theme import inject_theme, render_header, render_footer

st.set_page_config(page_title="Model Card | Guardian", page_icon="📑", layout="wide")

render_header(
    title="AI Model Card & System Governance",
    subtitle="Standardized Architecture Specifications, Evaluation Benchmarks, Training Protocols & Safety Boundaries",
    badge_text="GOVERNANCE AUDIT",
    tag="REPORT 06",
)

card_path = Path("reports/MODEL_CARD.md")
if card_path.exists():
    with open(card_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(
        f"""
        <div class="angular-card" style="padding: 24px 32px;">
        """,
        unsafe_allow_html=True,
    )
    st.markdown(content)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Generating Model Card from current model artifacts...")
    st.markdown(
        """
        # AI Battery Safety Guardian — Model Card
        
        ### Model Details
        - **Organization:** AI Battery Safety Guardian Project
        - **Model Types:** $\\beta$-VAE Anomaly Detector, LightGBM Temperature Predictor, Conditional GAN, DistilBERT Incident Classifier, Qwen2.5-0.5B LoRA Explainer.
        - **Target Domain:** Lithium-ion Battery Energy Storage & Electric Vehicles.
        
        ### Intended Use
        - **Primary Use:** Human-in-the-loop diagnostic decision support and early thermal hazard warning.
        - **Out of Scope:** Direct autonomous vehicle or BMS hardware disconnection.
        """
    )

render_footer()
