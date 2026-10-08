"""app/Home.py - AI Battery Safety Guardian main portal and mission control."""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="AI Battery Safety Guardian",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Premium Aesthetics
st.markdown(
    """
    <style>
    .kpi-card {
        background-color: #1a1d24;
        border: 1px solid #2d3340;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-normal { background-color: #0e4429; color: #3fb950; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-warning { background-color: #4d3800; color: #d29922; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-high { background-color: #5a1e02; color: #f0883e; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-critical { background-color: #5c1414; color: #f85149; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
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

st.title("⚡ AI Battery Safety Guardian")
st.markdown(
    "### Intelligent Multi-Tier Monitoring, Anomaly Detection & Explainable Decision Support"
)

col_lead, col_status = st.columns([2, 1])

with col_lead:
    st.markdown(
        """
        The **AI Battery Safety Guardian** is an enterprise-grade digital twin and decision-support system
        built to prevent lithium-ion battery failures and thermal runaway propagation in high-voltage energy storage.
        
        #### Core Capabilities:
        - 📈 **Real-Time Telemetry & VAE Reconstruction:** 12 interaction features + $\\beta$-VAE latent manifold anomaly scoring.
        - 🔥 **Forecasted Thermal Modeling:** LightGBM forward temperature horizon ($T_{t+10}$) predictor.
        - 🛡️ **Deterministic Risk Engine:** Transparent, mathematically monotonic Python scoring (0–100) with non-negotiable hard overrides.
        - 🧠 **Validated LLM Explainer & Agent:** Self-correcting explanations, strict numerical grounding, and sandboxed code execution.
        - 📚 **Grounded RAG Assistant:** Section-indexed FAISS retrieval with verifiable engineering citations (`[KB-xx-Cx]`).
        """
    )

with col_status:
    st.subheader("System Health & Artifacts")
    
    # Model Artifact Status Checks
    artifacts = {
        "Beta-VAE Anomaly Detector": Path("artifacts/models/vae_model.pt").exists(),
        "Temperature Predictor": Path("artifacts/models/temp_model.joblib").exists(),
        "Knowledge Base Vector Index": Path("artifacts/index/kb.faiss").exists(),
        "LoRA Explainer Adapter": Path("artifacts/models/lora_explainer/adapter_config.json").exists(),
        "BERT Incident Classifier": Path("artifacts/models/incident_classifier/model.pt").exists(),
        "Fault Window GAN": Path("artifacts/models/gan_generator.pt").exists(),
    }
    
    for name, ready in artifacts.items():
        if ready:
            st.markdown(f"✅ **{name}**: `ONLINE`")
        else:
            st.markdown(f"⚠️ **{name}**: `OFFLINE / PENDING`")

st.markdown("---")
st.subheader("Application Navigation")

n1, n2, n3, n4 = st.columns(4)
with n1:
    st.markdown("#### 1. Live Monitor")
    st.write("Replay battery discharge cycles, inspect 6 synchronized charts, and observe risk escalation.")
with n2:
    st.markdown("#### 2. Safety Events")
    st.write("Debounced event timeline, factor breakdown, recommended actions, and validated LLM diagnostics.")
with n3:
    st.markdown("#### 3. Ask Guardian")
    st.write("Conversational assistant with tool calling, RAG knowledge retrieval, and sandboxed pandas code exec.")
with n4:
    st.markdown("#### 4. What-If Simulation")
    st.write("Simulate elevated currents and ambient heating with out-of-distribution guards.")

n5, n6, n7, _ = st.columns(4)
with n5:
    st.markdown("#### 5. Incident Reports")
    st.write("Triage technician notes with DistilBERT and synthesize fault scenarios with the conditional GAN.")
with n6:
    st.markdown("#### 6. Model Card")
    st.write("Inspect comprehensive model performance, architecture specifications, and safety guidelines.")
with n7:
    st.markdown("#### 7. Auto-Heal Report")
    st.write("Review self-healing audit metrics, runtime repair traces, and test execution history.")

st.markdown("---")
st.markdown(
    '<div class="footer-guard">⚠️ <strong>Decision-support prototype. Not for vehicle control.</strong></div>',
    unsafe_allow_html=True,
)
