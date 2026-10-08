# AI Battery Safety Guardian — Final Live Demo Script (5–7 Minutes)

> **Audience:** Technical evaluators, battery systems engineers, and AI safety auditors.  
> **Mandatory Core Principle:** *Decision support prototype only. Not for autonomous vehicle or BMS control.*  
> **Offline Fallback Ready:** If no internet or API key is available, the system functions 100% deterministically using cached embeddings, pre-baked sessions, and template explanations.

---

## 0. Demo Overview & Timing Breakdown

| Step | Topic | Dashboard Page | Est. Time | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1** | Baseline Nominal Operation | `1_Live_Monitor` | 0:45 | VAE reconstructs normal cycling; risk score remains low (<30). |
| **Step 2** | Anomaly Detection & Risk Escalation | `1_Live_Monitor` | 1:00 | VAE detects abnormal pattern early; risk engine transitions to WARNING/HIGH. |
| **Step 3** | Event Inspection & Guarded Explanation | `2_Events` | 1:00 | Debounced events, per-feature error breakdown, validated LLM explanation. |
| **Step 4** | Grounded RAG & Auto-Healed CodeExec | `3_Ask_Guardian` | 1:15 | Citations in `[Knowledge Base]` vs `[Measured/Model]`; pandas auto-repair in action. |
| **Step 5** | What-If Physics Simulation & OOD Guard | `4_WhatIf` | 0:45 | Thermal prediction with counterfactual current/ambient; OOD badge triggers. |
| **Step 6** | Incident Triage (BERT) & Synthetic Stress (GAN) | `5_Incidents` | 0:45 | Fine-tuned NLP incident classifier & conditional telemetry generation. |
| **Step 7** | Model Card & Auto-Heal Resilience Audit | `6_Model_Card` & `7_AutoHeal` | 0:30 | Transparent benchmarks, zero placeholders, audited self-healing records. |
| **Total** | | | **~6:00** | End-to-end resilient AI engineering adhering to strict safety guardrails. |

---

## 1. Step-by-Step Presentation Walkthrough

### Step 1: Nominal Battery Cycling (Page: Live Monitor)
- **Presenter Action:**
  1. Open browser to `http://localhost:8501`.
  2. In the sidebar, select **Battery**: `B0018` (held-out test battery), **Cycle**: `10` (nominal early discharge cycle).
  3. Ensure **Fault Injection** is set to `None`.
  4. Press the **Play / Step** slider to advance telemetry to $t = 300\text{ s}$.
- **Presenter Dialogue:**
  > *"Welcome to the AI Battery Safety Guardian demonstration. Here we are monitoring held-out battery B0018 during normal discharge. Notice the KPI ribbon: temperature is ~24°C, voltage is steady, and current is at nominal 2A.*  
  > *Our unsupervised $\beta$-VAE anomaly detector computes reconstruction loss against learned healthy cell dynamics. Because the telemetry closely matches the healthy manifold, the Anomaly Score is below 0.15, and our deterministic Risk Score stays in the green NORMAL zone (score 12/100). Crucially, the risk score is computed by pure deterministic Python rules—never an LLM."*
- **Visual Indicators:**
  - KPI Badges: Risk Level: `NORMAL` (Green), Anomaly Score: `< 0.20`.
  - Telemetry charts: Smooth curves, no event markers.

---

### Step 2: Anomaly Detection & Thermal Runaway Early Warning (Page: Live Monitor)
- **Presenter Action:**
  1. Under the session controls, toggle **Fault Injection** to `thermal_ramp` (or select Cycle 50 with simulated thermal stress).
  2. Advance the playback slider past $t = 650\text{ s}$ to $t = 880\text{ s}$.
- **Presenter Dialogue:**
  > *"Now let's observe what happens when subtle internal overheating begins. Look closely at the Anomaly Score chart: before raw cell temperature breaches regulatory thresholds, the VAE anomaly score spikes to 0.78.*  
  > *Simultaneously, our supervised LightGBM thermal forecaster predicts that in 10 steps ($t+10$), temperature will breach 47°C.*  
  > *The deterministic risk engine evaluates temperature, trend ($dT/dt$), current, voltage, and anomaly score, transitioning the safety state from WARNING to HIGH (Risk Score 68.2). Hard safety override rules guarantee that any temperature $\ge 42^\circ\text{C}$ immediately escalates to at least HIGH."*
- **Visual Indicators:**
  - Risk Level Badge: Shifts dynamically to `HIGH` (Orange) or `CRITICAL` (Red).
  - Anomaly Score chart: Purple trace crosses threshold lines ($p95$ and $p99.5$).
  - Temperature chart: Red forecast line leads the actual curve upwards.

---

### Step 3: Event Inspection & Guarded LLM Explanation (Page: Events)
- **Presenter Action:**
  1. Click **Events** in the sidebar navigation.
  2. Point out the debounced timeline table showing detected safety events.
  3. Click on the most recent `HIGH` or `CRITICAL` event row to open the inspection drawer.
- **Presenter Dialogue:**
  > *"The Event Engine debounces rapid fluctuations with a mandatory 15-second gap to prevent alert fatigue. When we inspect this event, we see the complete telemetry snapshot, the top contributing factors, and a per-feature VAE reconstruction error bar chart showing that temperature and current deviated most.*  
  > *Below it is our GenAI explanation. Every explanation passes through our strict AST and regex validator: all reported numbers must match inputs within 1%, risk levels must match identically, and zero autonomous control commands like 'shut down vehicle' are ever permitted. Notice our rule-based recommendations: 'Reduce charge/discharge demand', 'Inspect cooling system', and mandatory escalation to human engineering oversight."*
- **Visual Indicators:**
  - Event Detail Drawer: Displays factors, VAE bar chart, and validated explanation card.
  - Persistent disclaimer banner: *"⚠️ Decision-support prototype. Not for vehicle control."*

---

### Step 4: Ask the Guardian — RAG & Auto-Healed Code Execution (Page: Ask Guardian)
- **Presenter Action:**
  1. Navigate to **Ask Guardian**.
  2. **Query 1 (Knowledge Assistant / RAG):** Type or select:  
     `"Why can high current discharge accelerate thermal degradation?"`
  3. Click **Submit**. Observe the structured response separating `[Measured/Model]` vs `[Knowledge Base]` with exact citations like `[KB-02-C1]`.
  4. **Query 2 (Code Execution with Auto-Heal):** Type:  
     `"What was the max temperature observed in cycle 50 and at what timestamp?"`
  5. Expand the **Tools Used & Code Execution Trace** expander.
- **Presenter Dialogue:**
  > *"Our conversational assistant provides dual capabilities: grounded RAG and sandboxed code execution.*  
  > *First, when asked domain safety questions, the system retrieves relevant chunks from our 15 peer-reviewed battery technical documents via FAISS. It strictly separates measured telemetry facts from retrieved knowledge base literature, citing source IDs directly.*  
  > *Second, for analytical questions, the LLM generates sandboxed pandas code executed in an isolated subprocess. If the initial code attempts forbidden operations like `import os` or fails with a KeyError, our Auto-Heal loop captures the traceback, feeds it to our repair prompt, and recovers transparently. The complete trace and attempt history are visible right here in the UI."*
- **Visual Indicators:**
  - Expandable source citation drawers displaying verified excerpts.
  - Trace box showing executed pandas snippet and auto-heal attempt status (`PASS`).

---

### Step 5: What-If Physics Simulation & OOD Guard (Page: What-If)
- **Presenter Action:**
  1. Navigate to **What-If**.
  2. Observe baseline sliders: Current = 2.0 A, Ambient = 24.0 °C, SOC = 60%.
  3. Drag the **Current** slider up to `15.0 A` and **Ambient Temperature** up to `40.0 °C`.
  4. Observe the side-by-side comparison table and the prominent **OOD Badge**.
- **Presenter Dialogue:**
  > *"Safety engineers often need to simulate counterfactual scenarios. What if ambient temperature rises by 15°C and the driver requests high current?*  
  > *Here, our LightGBM thermal model recomputes predicted temperature, and the risk engine recalculates the risk level.*  
  > *Crucially, we enforce physics monotonicity and out-of-distribution guards: because 15A exceeds the NASA cycling envelope (max ~4.5A), a high-visibility amber warning appears: '⚠️ Outside Training Range — Extrapolation / Low Confidence'. We never mislead engineers with ungrounded extrapolations."*
- **Visual Indicators:**
  - Side-by-side metric diff: Current ($T_{t+10} = 28^\circ\text{C}$, Risk = 22) vs What-If ($T_{t+10} = 53^\circ\text{C}$, Risk = 85).
  - Out-of-Distribution Alert Banner.

---

### Step 6: Incident Triage (BERT) & Synthetic Faults (GAN) (Page: Incidents)
- **Presenter Action:**
  1. Navigate to **Incidents**.
  2. Click the sample button: **"Technician Note: Pack exterior hot to touch after fast charging, smell of burning insulation."**
  3. Click **Classify Incident**.
  4. Scroll down to the **Synthetic Fault GAN** section and click **Generate Scenario: thermal_ramp**.
- **Presenter Dialogue:**
  > *"In Page 5, we support fleet maintenance operations. Field technicians can paste raw maintenance logs. Our fine-tuned DistilBERT classifier triages the text into 5 fault categories—here categorizing it as `thermal_overheat` with 98% confidence and linking directly to standard inspection checklists.*  
  > *Below it, our Conditional GAN generates synthetic multi-channel abnormal windows. These synthetic windows are used to stress-test risk thresholds without endangering physical battery hardware."*
- **Visual Indicators:**
  - BERT Classification: Predicted category, confidence bar chart, linked SOP documents.
  - GAN Plot: Multi-variate generated telemetry overlay ($V, I, T$).

---

### Step 7: Model Card & Auto-Heal Resilience Audit (Pages: Model Card & Auto-Heal)
- **Presenter Action:**
  1. Briefly navigate to **Model Card** (`6_Model_Card.py`) and scroll through the system and per-model benchmark tables.
  2. Navigate to **Auto-Heal Report** (`7_AutoHeal_Report.py`).
- **Presenter Dialogue:**
  > *"Finally, we uphold the highest standards of Responsible AI and transparency.  
  > Page 6 renders our comprehensive Model Card following the Mitchell et al. standard. Every single metric—from VAE AUROC of 0.955 to LightGBM MAE of 0.094°C—is loaded directly from verified JSON artifacts, with zero placeholder text.*  
  > *Page 7 displays our live Auto-Heal Audit Report, documenting all development-time and runtime self-corrections, failure taxonomy distributions, and mean-time-to-repair metrics.  
  > The AI Battery Safety Guardian demonstrates how combining deterministic physics, ML anomaly detection, and guardrailed GenAI creates a safe, reliable, and auditable decision-support system."*
- **Visual Indicators:**
  - Quantitative benchmark comparison tables.
  - Auto-Heal failure taxonomy breakdown and audit table.

---

## 2. Emergency Presenter Troubleshooting & Offline Modes

| Potential Issue | Immediate Action / Fallback |
| :--- | :--- |
| **No Internet / Missing GROQ API Key** | App automatically switches to `LLM_PROVIDER=template` (deterministic template explainer + cached FAISS index). Zero downtime. |
| **Streamlit Cache Delay** | Click the hamburger menu $\rightarrow$ **Clear cache** or use the pre-baked **Golden Session** toggle on the home page. |
| **GPU / CUDA Not Detected** | Normal and expected: all models (VAE, LightGBM, GAN, DistilBERT, LoRA) are CPU-optimized for instant inference on standard laptops. |
