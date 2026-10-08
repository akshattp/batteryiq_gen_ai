"""app/pages/6_Model_Card.py - Model Card Viewer (Section 7)."""

from __future__ import annotations

from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Model Card | Guardian", page_icon="📑", layout="wide")

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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📑 AI Model Card & Architecture Specifications")
st.caption("Standardized model documentation covering model architectures, training protocols, evaluation metrics, and safety boundaries.")

card_path = Path("reports/MODEL_CARD.md")
if card_path.exists():
    with open(card_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
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

st.markdown("---")
st.markdown(
    '<div class="footer-guard">⚠️ <strong>Decision-support prototype. Not for vehicle control.</strong></div>',
    unsafe_allow_html=True,
)
