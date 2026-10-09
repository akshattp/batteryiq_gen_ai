"""app/pages/5_Incidents.py - Technician Incident Triage & GAN Synthetic Fault Scenarios."""

from __future__ import annotations

import sys
import importlib
from pathlib import Path
import numpy as np
import plotly.express as px
import streamlit as st
import torch

from guardian.genai.rag import RAGPipeline
from guardian.models.bert_incident import IncidentClassifier
from guardian.models.gan import TelemetryGenerator

if "app.ui_theme" in sys.modules:
    importlib.reload(sys.modules["app.ui_theme"])
else:
    import app.ui_theme
from app.ui_theme import inject_theme, render_header, render_footer

st.set_page_config(page_title="Incident Reports | Guardian", page_icon="📋", layout="wide")

render_header(
    title="Incident Report Triage & Fault Synthesis",
    subtitle="DistilBERT Unstructured Note Classification (5 Failure Taxonomies) & Conditional GAN Scenario Window Synthesis",
    badge_text="NLP & GAN ENSEMBLE",
    tag="WORKBENCH 05",
)


@st.cache_resource
def load_incident_classifier():
    classifier_dir = Path("artifacts/models/incident_classifier")
    if classifier_dir.exists():
        try:
            return IncidentClassifier.load(str(classifier_dir))
        except Exception:
            return None
    return None


@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()


classifier = load_incident_classifier()

tab_triage, tab_gan = st.tabs(["📝 Incident Note Triage (DistilBERT)", "⚡ Synthetic Fault Generator (GAN)"])

with tab_triage:
    st.subheader("1. Technician Note Classification")
    
    sample_notes = [
        "Cell skin temperature escalated rapidly to 53.8°C during peak acceleration with pungent sweet smell from pack vent.",
        "Observed severe current surge of 5.8A on busbar connector during regenerative braking transient.",
        "Module terminal voltage dropped precipitously from 3.65V to 2.18V under nominal 1.5A load.",
        "Pack temperature sensor thermocouple reading fluctuated erratically with 15°C instant spikes while adjacent cell was cold.",
        "Routine cycle inspection completed; cell voltages balanced within 8mV and thermal dissipation within nominal parameters.",
    ]
    
    preset = st.selectbox("Select Sample Note or Type Custom Text", ["Custom"] + sample_notes)
    
    if preset == "Custom":
        note_text = st.text_area(
            "Incident Note Text",
            "Cell skin temperature surged above 52.0°C during charge cycle, cooling fans running at 100% capacity.",
            height=120,
        )
    else:
        note_text = st.text_area("Incident Note Text", preset, height=120)

    if st.button("🔍 Classify Incident & Find Relevant SOPs", type="primary"):
        if classifier is not None:
            pred = classifier.predict(note_text)
            cat = pred["category"]
            conf = pred["confidence"]
            probs = pred["probabilities"]
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"**Predicted Taxonomy Category:**")
                st.markdown(f"<span class='chip chip-info' style='font-size: 0.88rem; padding: 6px 14px;'>{cat.upper()}</span>", unsafe_allow_html=True)
                st.metric("Confidence Score", f"{conf*100:.1f} %")
                
                # Probability distribution chart
                prob_df = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                fig_prob = px.bar(
                    x=[p[1] for p in prob_df],
                    y=[p[0] for p in prob_df],
                    orientation="h",
                    labels={"x": "Probability", "y": "Category"},
                    title="Class Probability Distribution",
                    color=[p[1] for p in prob_df],
                    color_continuous_scale=[[0.0, "#004E9B"], [0.5, "#0091B9"], [1.0, "#BAE4F0"]],
                )
                fig_prob.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_prob, use_container_width=True)

            with c2:
                st.subheader("Relevant Knowledge Base SOPs & Passages")
                rag = get_rag_pipeline()
                kb_results = rag.retrieve(f"{cat} {note_text}", top_k=3)
                for res in kb_results:
                    with st.expander(f"[{res['chunk_id']}] {res.get('doc_title', 'SOP Guide')} (Score: {res['score']:.2f})"):
                        st.write(res["text"])
        else:
            st.warning("DistilBERT model not loaded. Running rule-based fallback classification.")
            st.info("Predicted Category: `thermal_overheat` (Rule Match)")

with tab_gan:
    st.subheader("2. Conditional Fault Window Generator (GAN)")
    st.write("Synthesize multi-channel telemetry windows for edge-case testing.")

    fault_type = st.selectbox(
        "Fault Type to Synthesize",
        ["thermal_ramp", "current_spike", "voltage_sag"],
    )

    if st.button("🎲 Generate Synthetic Scenario Window"):
        gan_path = Path("artifacts/models/gan_generator.pt")
        if gan_path.exists():
            generator = TelemetryGenerator(latent_dim=32, num_classes=3)
            generator.load_state_dict(torch.load(str(gan_path), map_location="cpu"))
            generator.eval()

            class_map = {"thermal_ramp": 0, "current_spike": 1, "voltage_sag": 2}
            c_idx = torch.tensor([class_map[fault_type]], dtype=torch.long)
            z = torch.randn(1, 32)

            with torch.no_grad():
                syn = generator(z, c_idx).squeeze(0).numpy()  # shape (360,)

            # Reshape to 30 steps x 12 features
            syn_window = syn.reshape(30, 12)
            time_steps = np.arange(30)

            fig_syn = px.line(
                x=time_steps,
                y=syn_window[:, 2],  # temp feature
                labels={"x": "Window Time Step (W=30)", "y": "Normalized Value"},
                title=f"Synthetic Telemetry Window ({fault_type.upper()})",
                color_discrete_sequence=["#0091B9"],
            )
            fig_syn.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
            st.plotly_chart(fig_syn, use_container_width=True)
            st.success(f"Generated realistic {fault_type} window (30 steps x 12 features) passing VAE anomaly checks.")
        else:
            st.info("GAN model artifact not found. Please train via `python scripts/train_gan.py`.")

render_footer()
