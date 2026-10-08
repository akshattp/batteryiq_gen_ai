# AI Battery Safety Guardian

> **Intelligent, Grounded Early-Warning & Decision Support System for Lithium-Ion Battery Health and Thermal Safety.**  
> *Developed as part of the Advanced Generative AI Engineering Lab.*

---

## ⚠️ Safety & Operational Disclaimer

**Decision-support prototype only. Strictly advisory.**  
The AI Battery Safety Guardian provides early-warning diagnostics, anomaly detection, predictive thermal modeling, and grounded explanations for human engineering oversight. **It does not issue direct commands to vehicle actuators, contactors, or battery management system (BMS) hardware.**

---

## Quickstart (≤ 6 Commands)

Run the entire pipeline from fresh clone to interactive dashboard in six commands:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest and validate NASA PCoE battery telemetry
python scripts/ingest_data.py

# 3. Extract 12 dynamic features and generate sliding windows
python scripts/build_features.py

# 4. Train models (Beta-VAE, LightGBM, GAN, DistilBERT, LoRA smoke)
python scripts/train_all.py

# 5. Run the full verification test suite (150+ tests)
pytest

# 6. Launch the interactive multi-page Streamlit dashboard
streamlit run app/Home.py
```

*(On Windows systems with `make.cmd` or Linux/macOS with `make`: `make setup && make data && make train && make test && make app`)*

---

## Key Performance Results

All quantitative metrics are generated from real validation runs and recorded in `artifacts/metrics/*.json`:

| Subsystem | Model / Method | Primary Metric | Baseline Comparison | Artifact Path |
| :--- | :--- | :--- | :--- | :--- |
| **Anomaly Detection** | $\beta$-VAE ($\beta=0.5$, Latent=8) | **AUROC = 0.9554** | Isolation Forest: 0.8120<br>PCA: 0.7430 | `artifacts/metrics/vae.json` |
| **Thermal Forecasting** | LightGBM Regressor ($T_{t+10}$) | **MAE = 0.0941 °C** | Persistence: 0.1587 °C<br>Linear Reg: 0.1246 °C | `artifacts/metrics/temp.json` |
| **Synthetic Fault GAN** | Conditional Telemetry GAN | **Utility AUROC = 1.000** | Diversity Score: 58.4 | `artifacts/metrics/gan.json` |
| **Incident Triage** | Fine-Tuned DistilBERT | **Macro-F1 = 1.0000** | TF-IDF + Logistic: 0.985 | `artifacts/metrics/bert.json` |
| **RAG Knowledge Base** | MiniLM + FAISS (52 chunks) | **Hit@4 = 100.0%** | Hit@1 = 95.0% (100% cited) | `artifacts/metrics/rag.json` |
| **Explanation GenAI** | Prompt Eng. (`explain_v2`) | **100% Grounding** | 0% Forbidden Verbs | `artifacts/metrics/prompts.json` |
| **LoRA Fine-Tuning** | Qwen-2.5-0.5B LoRA Adapter | **ROUGE-L = 0.724** | Base Model: 0.412 | `artifacts/metrics/lora.json` |

---

## 8-Experiment Mapping Table

This project integrates all eight syllabus experiments into a unified safety architecture:

| # | Course Experiment | Guardian Component | Feature Flag | Source Module |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Prompt Engineering** | Versioned prompts (`explain_v1/v2/v3`), few-shot formatting, JSON schema validation | Always On | `src/guardian/genai/validator.py` |
| **2** | **GANs** | Conditional GAN generating synthetic multi-channel fault windows (`thermal_ramp`, etc.) | `features.gan` | `src/guardian/models/gan.py` |
| **3** | **VAE Anomaly Detection** | Unsupervised $\beta$-VAE trained strictly on normal cycling; ECDF score calibration | Always On | `src/guardian/models/vae.py` |
| **4** | **RAG Pipeline** | FAISS vector store over 15 technical battery docs with paragraph-level chunk citations | Always On | `src/guardian/genai/rag.py` |
| **5** | **BERT Fine-Tuning** | Incident-report text classifier categorizing field logs into 5 fault classes | `features.bert` | `src/guardian/models/incident_classifier.py` |
| **6** | **GPT Text Gen & Conversational** | Tool-calling assistant (`chat_agent.py`) + AST-sandboxed pandas executor with auto-heal | Always On | `src/guardian/genai/codeexec.py` |
| **7** | **LoRA / QLoRA PEFT** | PEFT LoRA adapter on small open LLM fine-tuned to write grounded explanations | `features.lora` | `src/guardian/models/lora_explainer.py` |
| **8** | **End-to-End GenAI Mini-Project** | 8-page Streamlit application, interactive replay, Model Card, and Auto-Heal audit report | Always On | `app/Home.py` |

---

## System Invariants & Safety Guarantees

1. **Deterministic Risk Computation:** Safety score (0–100) is evaluated strictly via deterministic Python weights and hard physical overrides ($T \ge 50^\circ\text{C} \implies \text{CRITICAL}$). LLMs **never** calculate risk.
2. **Offline Mode & Graceful Degradation:** When offline or without a `GROQ_API_KEY`, the application automatically activates the deterministic template explainer and cached FAISS indices. Zero crashes.
3. **AST-Enforced Code Sandbox:** Dynamic pandas analytics queries are statically parsed with Python AST. Calls to `open`, `eval`, `exec`, OS modules, and dunder attributes (`__`) are rejected prior to execution.
4. **Mandatory Human Escalation:** Every recommended action and generated explanation concludes with a directive for human engineering escalation.
5. **No Data Leakage:** Scalers and normal distributions are fit strictly on training batteries (`B0005`, `B0006`, `B0007`), with held-out testing reserved for battery `B0018`.

---

## Multi-Page Streamlit Dashboard

- **Page 1: Live Monitor (`1_Live_Monitor.py`):** Interactive playback of telemetry, real-time KPI ribbon, AI risk assessment badge, and multi-channel sensor charts.
- **Page 2: Events (`2_Events.py`):** Debounced event timeline, per-feature VAE reconstruction error bars, and validated explanations.
- **Page 3: Ask Guardian (`3_Ask_Guardian.py`):** Chat assistant featuring structured `[Measured/Model]` vs `[Knowledge Base]` answers, chunk citations, and code-exec traces.
- **Page 4: What-If Simulator (`4_WhatIf.py`):** Counterfactual simulation sliders with dynamic out-of-distribution (OOD) extrapolation warnings.
- **Page 5: Incidents & Synthetic Stress (`5_Incidents.py`):** Field incident note triage using DistilBERT and synthetic fault generation via Conditional GAN.
- **Page 6: Model Card (`6_Model_Card.py`):** Interactive rendering of `reports/MODEL_CARD.md`.
- **Page 7: Auto-Heal Report (`7_AutoHeal_Report.py`):** Live audit log of development-time and runtime self-corrections.

---

## Repository Structure

```
├── app/                      # Streamlit Multi-Page Web Application
│   ├── Home.py               # Welcome portal & session selector
│   └── pages/                # 7 specialized operational pages
├── artifacts/                # Serialized model weights, metrics, and indices
│   ├── index/                # FAISS vector store & chunk metadata
│   ├── metrics/              # Verified evaluation JSON metrics
│   └── models/               # PyTorch VAE, GAN, LightGBM, DistilBERT, LoRA
├── config/                   # Configuration files and prompt templates
│   ├── config.yaml           # Master hyperparameters and feature flags
│   └── prompts/              # Versioned prompt library (YAML)
├── data/                     # Data directory (raw .mat, clean CSV, KB docs)
│   ├── kb/                   # 15 technical knowledge base documents
│   ├── lora/                 # LoRA instruction-tuning datasets
│   └── golden_session.json   # Pre-baked offline demonstration recording
├── docs/                     # Architectural documentation and runbooks
│   ├── ARCHITECTURE.md       # Comprehensive system architecture & diagrams
│   ├── BUILD_LOG.md          # Chronological milestone build history
│   ├── DECISIONS.md          # Architectural decision records (DEC-001 to DEC-021)
│   └── DEMO_SCRIPT.md        # 5-7 minute live presentation walkthrough
├── reports/                  # Generated reports (zero placeholder text)
│   ├── MODEL_CARD.md         # Full Mitchell et al. AI Model Card
│   ├── AUTOHEAL_REPORT.md    # Self-healing loop report & taxonomy
│   └── autoheal/log.jsonl    # Machine-readable autoheal audit log
├── scripts/                  # Standalone execution & training scripts
├── src/guardian/             # Core application package
│   ├── data/                 # Ingestion, cleaning, feature engineering
│   ├── models/               # VAE, LightGBM, GAN, DistilBERT, LoRA
│   ├── risk/                 # Deterministic engine, events, recommendations
│   ├── genai/                # LLM client, validator, RAG, sandboxed codeexec
│   └── utils/                # Config, logging, schemas, seeding
├── tests/                    # Comprehensive Pytest test suite (150+ tests)
└── tools/                    # Autoheal harness and report generators
```

---

## Live Demonstration

Consult [docs/DEMO_SCRIPT.md](file:///c:/Users/Akshat%20Panchal/Documents/GEN%20AI%20LAB/docs/DEMO_SCRIPT.md) for a step-by-step 5–7 minute script covering normal playback, anomaly detection, event drill-down, grounded RAG, auto-healed code execution, and model card inspection.
