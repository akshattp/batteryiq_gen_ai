# AI Battery Safety Guardian — System Model Card

## 1. Executive Overview & Model Details
- **System Name:** AI Battery Safety Guardian Decision-Support Platform
- **Version:** 1.0.0-rc
- **Model Architectures:**
  1. **Anomaly Detector:** Fully connected $\beta$-Variational Autoencoder ($\beta$-VAE, latent dimension $Z=8$, input dimension $D=360$).
  2. **Thermal Horizon Predictor:** Gradient Boosted Decision Trees (LightGBM, 100 estimators, max depth 6) for forward temperature forecasting ($T_{t+10}$).
  3. **Deterministic Safety Engine:** Pure Python monotonic scoring ($S \in [0, 100]$) with non-negotiable hard overrides.
  4. **Field Incident Classifier:** Fine-tuned Transformer (`distilbert-base-uncased`, 5-class failure taxonomy).
  5. **Knowledge Assistant (RAG):** Dense bi-encoder (`all-MiniLM-L6-v2`) + FAISS Inner Product vector index (52 section-chunked technical passages).
  6. **Diagnostic Explainer:** Quantized Low-Rank Adaptation (LoRA, $r=16, \alpha=32$) on `Qwen/Qwen2.5-0.5B-Instruct` + few-shot prompt engine (`explain_v2`).
  7. **Fault Window Synthesizer:** Class-conditional Telemetry GAN ($Z=32 \rightarrow 360$, 3 fault modes).

---

## 2. Intended Domain & Safety Envelope
- **Primary Use Case:** Continuous telemetry monitoring, early thermal anomaly detection, and explainable decision-support advisory for lithium-ion battery packs in electric mobility and energy storage systems (BESS).
- **Intended Users:** Test bench engineers, battery service technicians, and fleet safety managers.
- **Out-of-Scope & Forbidden Applications:** Direct, autonomous physical actuation (e.g. commanding breaker disconnect, disabling BMS hardware protections, or shutting down vehicle propulsion). The system operates strictly as an advisory copilot. Every advisory mandate concludes with a human-in-the-loop escalation instruction.

---

## 3. Data Ingestion, Featurization & Splitting
- **Primary Dataset:** NASA Prognostics Center of Excellence (PCoE) Battery Aging Dataset (B0005, B0006, B0007, B0018).
- **Ingestion Preprocessing:** Recursive directory scanning, MAT-file extraction, and continuous Coulomb counting ($SOC(t) = SOC_0 - \frac{1}{Q_n} \int I dt$).
- **Interaction Feature Engineering (12 Features):**
  1. `temp_c`: Raw thermocouple temperature
  2. `dT_dt`: First-difference rate of temperature change (°C/min)
  3. `voltage_v`: Cell terminal voltage
  4. `current_a`: Charge/discharge current
  5. `soc_pct`: State of Charge (%)
  6. `power_w`: Instantaneous power ($P = V \times I$)
  7. `apparent_resistance_ohm`: $R = V / (I + \epsilon)$
  8. `joule_heating_proxy`: $I^2 \times R$ resistive dissipation proxy
  9. `v_drop_from_nominal`: Deviation from nominal OCV ($3.8\text{V} - V$)
  10. `thermal_current_ratio`: Decoupling indicator $\frac{dT/dt}{|I| + \epsilon}$
  11. `rolling_temp_mean_5`: 5-step rolling mean temperature
  12. `soh_pct`: State of Health capacity fade percentage
- **Leakage-Safe Partitioning:**
  - **Train Batteries:** B0005, B0006 (19,758 sliding windows, $W=30$, stride=10).
  - **Validation Battery:** B0007 (5,214 windows).
  - **Held-Out Test Battery:** B0018 (5,510 windows). Zero temporal or cell overlap between splits.

---

## 4. Benchmark Evaluation & Empirical Metrics

### 4.1 Anomaly Detection ($\beta$-VAE vs Baselines on Held-Out B0018)
The anomaly detector reconstructs $W=30$ normalized windows ($D=360$). Reconstruction error is normalized via validation median and IQR into an anomaly score $\in [0, 1]$.
- **Beta-VAE Overall AUROC:** 0.9554 (Target $\ge 0.85$: **PASS**)
- **Beta-VAE Overall PR-AUC:** 0.9188
- **Recall at 5% False Positive Rate (FPR):** 0.7988
- **Baseline Comparison:**
  - Isolation Forest AUROC: 0.8410
  - PCA Reconstruction (8 components) AUROC: 0.8872
- **Breakdown Across 5 Injected Fault Modes:**
  - Thermal Runaway Ramp: AUROC 0.9982
  - Internal Short Circuit: AUROC 0.9715
  - Current Surge Spike: AUROC 0.9420
  - Voltage Sag Collapse: AUROC 0.9580
  - Sensor Drift / Failure: AUROC 0.9240

### 4.2 Forecasted Thermal Modeling ($T_{t+10}$ LightGBM)
Predicts cell temperature 10 steps into the future using lag interaction features and heating proxies.
- **Model Test MAE:** 0.0941 °C
- **Persistence Baseline ($T_{t+10} = T_t$) MAE:** 0.1127 °C
- **Improvement over Persistence:** 16.5% error reduction.
- **Monotonicity Compliance:** 100% verified across current demand increments.

### 4.3 Deterministic Risk Engine
Combines multi-factor signals with static weights: Temperature (0.30), Temp Trend (0.20), Current (0.15), Voltage (0.10), Anomaly Score (0.25).
- **Hard Overrides:**
  - $T_{eff} \ge 50.0^\circ\text{C} \rightarrow$ Mandatory `CRITICAL` (Score $\ge 75.0$).
  - $T_{eff} \ge 42.0^\circ\text{C} \rightarrow$ Mandatory `HIGH` (Score $\ge 55.0$).
  - $V < 2.5\text{V}$ or $V > 4.2\text{V} \rightarrow$ Mandatory $\ge$ `HIGH`.
- **Debounce Window:** 15.0 seconds minimum gap for identical event types; immediate bypass on severity escalation.

### 4.4 RAG Knowledge Retrieval & Grounding
FAISS L2-normalized Inner Product index over 52 section-level passages across 15 technical documents.
- **Hit@1 Rate:** 95.0%
- **Hit@4 Rate:** 100.0% (Target $\ge 80.0\%$: **PASS**)
- **Citation Presence & Validity Rate:** 100.0% (all generated claims cited with `[KB-xx-Cx]`).
- **Telemetry-Knowledge Separation:** Strict Markdown separation between `[Measured/Model]` telemetry and `[Knowledge Base]` facts.

### 4.5 Technician Incident Text Classification (DistilBERT)
Evaluated on stratified held-out test split (113 notes) across 5 failure categories:
- **DistilBERT Macro-F1:** 1.0000
- **TF-IDF + Logistic Regression Macro-F1:** 1.0000
- **Caveat:** High F1 reflects synthetic linguistic consistency across domain taxonomy.

### 4.6 LoRA Explainer Adapter (Exp 7)
Fine-tuned on 425 synthetic instruction pairs on `Qwen/Qwen2.5-0.5B-Instruct` with rank $r=16, \alpha=32$.
- **ROUGE-L F1 against Reference:** 0.6350
- **Numeric Grounding Fidelity Rate:** 100.0%
- **Forbidden Directive Violation Rate:** 0.0%

### 4.7 Synthetic Fault Scenario GAN (Exp 2)
Conditional generator ($Z=32$) trained on 19,758 windows.
- **Beta-VAE Detection Utility Rate:** 100.0% (100% of synthetic fault windows detected as anomalies).
- **Diversity Distance Metric:** 57.2 (no mode collapse).

---

## 5. Ethical Considerations & Operational Guardrails
1. **Human-in-the-Loop Requirement:** All high-risk alerts require human engineer sign-off.
2. **Deterministic Risk Invariance:** LLMs are strictly excluded from numerical risk calculation.
3. **AST Sandbox Isolation:** Sandboxed code execution rejects imports, OS calls, and private attributes prior to subprocess execution.
4. **Out-of-Distribution Guard:** Explicit visual badge shown when parameters exceed training bounds ($I > 3.0\text{A}, T > 45.0^\circ\text{C}$).
