# Architectural Decisions & Log

All design choices, trade-offs, and parameter derivation records are maintained here.

---

## Decision Log

### DEC-001: Repository Scaffolding and Directory Layout (Milestone M0)
- **Date**: 2026-10-08
- **Decision**: Established exact repository structure outlined in Section 4. Created dual execution shims (`Makefile` and `make.cmd`) to ensure cross-platform compatibility across Windows PowerShell and Unix environments.
- **Rationale**: Strict compliance with Section 4 binding spec.

### DEC-002: Raw Dataset Sourcing and File Discovery
- **Date**: 2026-10-08
- **Decision**: Mounted `5. Battery Data Set` directly into `data/raw/` via directory junction. Recursive search pattern `data/raw/**/*.mat` configured in `src/guardian/data/ingest.py` to seamlessly detect `B0005.mat`, `B0006.mat`, `B0007.mat`, and `B0018.mat`.
- **Rationale**: Files are pre-downloaded on the host; bypasses redundant web downloads and handles nested archive extraction folders natively.

### DEC-003: LLM Provider Configuration & Groq Model Selection
- **Date**: 2026-10-08
- **Decision**: Primary LLM provider configured to `groq`. Models documented:
  - Primary default: `llama-3.3-70b-versatile`
  - High-speed / low-latency fallback: `llama-3.1-8b-instant`
  - Supported alternatives in ecosystem: `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`
  - Offline fallback: `template` provider (deterministic Python text builder) activates automatically if `GROQ_API_KEY` is absent or network requests fail.
- **Rationale**: Adheres to Section 3 and Section 8.1; guarantees zero test or app failures when offline.

### DEC-004: Execution Environment and Compute Constraints
- **Date**: 2026-10-08
- **Decision**: Strict CPU execution mode on host workstation. PyTorch installed for CPU operations. GPU-heavy training tasks (LoRA, DistilBERT full-scale) reserved for Colab handoff or executed locally only under `--smoke` flag.
- **Rationale**: User hardware is CPU-only laptop; prevents out-of-memory or freezing conditions.

### DEC-005: Auto-Heal Harness Logging & Self-Test Verification
- **Date**: 2026-10-08
- **Decision**: Implemented `tools/autoheal.py` with structured JSONL records in `reports/autoheal/log.jsonl`. Features automated error classification, `--record-fix` diagnosis linking, and `--selftest` harness verification.
- **Rationale**: Mandated by Section 10 of implementation plan.

### DEC-006: Data Pipeline Normalization & Console Encoding
- **Date**: 2026-10-08
- **Decision**: Coulomb counting was implemented using cumulative rectangular/trapezoidal integration of signed current normalized by each cycle's empirical discharge capacity (clipped to [0, 100]). Out-of-bounds rows (39 rows total) were persisted to `data/interim/rejected_rows.csv` rather than silently dropped. Upgraded `tools/autoheal.py` stdout handling to support utf-8 with error replacement to prevent Windows `cp1252` encoding exceptions on degree Celsius symbols.
- **Rationale**: Conforms to Section 5.2 and 5.3 specifications; ensures deterministic reproducible data ingestion.

### DEC-007: Feature Windows, Normal-Only Scaler, and Fault Suite
- **Date**: 2026-10-08
- **Decision**: Sliding windows of $W=30$ steps (stride 5) are extracted strictly within individual battery-cycle segments to prevent temporal discontinuities. `StandardScaler` is fitted exclusively on normal training split rows (`label == 'normal'`), ensuring zero leakage of validation, test, or synthetic fault statistics into the scaling parameters. Five synthetic fault functions were implemented with explicit physical directionality (`thermal_ramp`, `current_spike`, `voltage_sag`, `sensor_stuck`, `temp_current_decoupling`) marked with `label = 'injected_fault'` and excluded from normal model training.
- **Rationale**: Section 5.4, 5.5, and 5.6 requirements; guarantees mathematically clean training boundaries and realistic evaluation benchmarks.

### DEC-008: Beta-VAE Architecture, Calibration, and Anomaly Thresholding
- **Date**: 2026-10-09
- **Decision**: Implemented MLP Beta-VAE ($[360 \rightarrow 128 \rightarrow 64 \rightarrow 8]$) with $\beta=0.5$ trained exclusively on normal cycling telemetry windows. Calibration uses empirical CDF mapping of validation reconstruction errors to bound anomaly scores to $[0.0, 1.0]$. Thresholds established at $p_{95} = 1.5470$ (Warning) and $p_{99.5} = 8.2344$ (Critical). Evaluation on 42,817 held-out test windows demonstrated AUROC = 0.9554 (exceeding the 0.85 plan threshold), with full baseline comparison against Isolation Forest and PCA.
- **Rationale**: Adheres to Section 6 Milestone 3; delivers explainable, feature-deconstructable anomaly detection without ever training on synthetic fault modes.

### DEC-009: Temperature Predictor Architecture and Horizon
- **Date**: 2026-10-09
- **Decision**: Implemented LightGBM multi-step ahead temperature regressor with forecast horizon $H=10$ steps ahead ($T_{t+10}$) augmented with temporal lag features ($t-1, t-5, t-10$). On 309,306 held-out test steps from battery `B0018`, LightGBM achieved MAE = 0.0941 °C and $R^2 = 0.9968$, beating the persistence baseline (0.1127 °C) by 16.50%. Subgroup metrics recorded across charge (MAE = 0.0778 °C), discharge (MAE = 0.2338 °C), and ambient conditions.
- **Rationale**: Fulfills Section 6 Milestone 4 criteria; enables forward-looking risk anticipation in the deterministic risk engine.

### DEC-010: Deterministic Risk Engine & Hard Override Priority
- **Date**: 2026-10-09
- **Decision**: Risk scoring implemented as a pure Python function (`assess()`) strictly isolating LLMs from safety score calculation. Weights assigned: Temperature (0.30), Temp Trend (0.20), Current (0.15), Voltage (0.10), Anomaly (0.25). Three non-negotiable hard overrides enforce safety: $T_{eff} \ge 50^\circ\text{C} \rightarrow \text{CRITICAL}$, $T_{eff} \ge 42^\circ\text{C} \rightarrow \ge \text{HIGH}$, and $V \notin [2.5, 4.2]\text{V} \rightarrow \ge \text{HIGH}$. Mathematical monotonicity proven across temperature, current, heating rate, and anomaly axes.
- **Rationale**: Conforms to Rule 6 & 7 and Section 6 Milestone 5; guarantees explainable, transparent, and non-hallucinatory risk assessment.

### DEC-011: Debounce Window, Non-Autonomous Advisories & Human Escalation
- **Date**: 2026-10-09
- **Decision**: Event detection debouncing enforced at 15-second minimum gap with immediate severity escalation bypass. Advisory catalog strictly eliminates all autonomous control verbs (`shut down`, `disconnect`, `disable bms`, `emergency stop`), mandating that all recommendations conclude with a human-in-the-loop escalation line.
- **Rationale**: Adheres to Operating Rule 6 and Milestone M6 specifications.

### DEC-012: Groq Model Selection, Prompt Benchmark Study, and Self-Correcting Explanation Engine
- **Date**: 2026-10-09
- **Decision**: Querying `client.models.list()` identified `qwen/qwen3.8-27b` as active and fully functional on Groq cloud with fast JSON-mode completion. Caching keyed on SHA256 was implemented in `artifacts/llm_cache/` alongside exponential backoff on HTTP 429. In a formal benchmark study across 15 diverse assessments, `explain_v2` (few-shot) won over `explain_v1` and `explain_v3` with 100% first-pass validity, 100% numerical fidelity, and the lowest latency (11.78s). Runtime validator and self-correction loop retry up to 2 times on validation failures before falling back cleanly to deterministic `template_explanation()`.
- **Rationale**: Fulfills Section 6 Milestone 10 (Exp 1 + 6) and Operating Rules 6, 7, and 8.

### DEC-013: Conditional GAN Architecture and Synthetic Fault Utility Benchmark
- **Date**: 2026-10-09
- **Decision**: Implemented `TelemetryGenerator` and `TelemetryDiscriminator` conditioned on 3 fault categories (`thermal_ramp`, `current_spike`, `voltage_sag`) using 16-dim class embeddings concatenated with latent noise ($Z=32$) and input telemetry ($W \times F = 360$). Trained on 19,758 windows derived exclusively from training batteries. Utility evaluation through the pretrained Beta-VAE anomaly detector verified a 100% detection rate on synthetic abnormal windows without mode collapse (mean diversity distance 56.4 for thermal ramps, 59.8 for current spikes). Visual overlays persisted to `artifacts/figures/gan_synthetic_overlay.png`.
- **Rationale**: Fulfills Section 6 Milestone 7 (Exp 2); provides synthetic scenario stress-testing for the dashboard and threshold verification.

### DEC-014: Incident Reports Taxonomy, Generation, and DistilBERT Fine-Tuning
- **Date**: 2026-10-09
- **Decision**: Generated 750 balanced synthetic technician incident notes across 5 taxonomy categories (`thermal_overheat`, `overcurrent`, `overvoltage_or_undervoltage`, `sensor_fault`, `normal_observation`) with realistic telemetry references and marked `is_synthetic=True`. Trained TF-IDF + LogisticRegression baseline (100% test accuracy / 1.0 macro-F1) and fine-tuned `distilbert-base-uncased` with PyTorch (1.0 macro-F1 on test split). Caveat recorded in `artifacts/metrics/bert.json` noting high F1 reflects synthetic linguistic consistency.
- **Rationale**: Fulfills Section 6 Milestone 8 (Exp 5); integrates unstructured technician text triage into the Guardian platform.

### DEC-015: RAG Vector Knowledge Base, Dense FAISS Indexing, and Citation Grounding
- **Date**: 2026-10-09
- **Decision**: Assembled 15 curated technical battery safety documents in `data/kb/` spanning thermal runaway, Joule heating, safe operating envelopes, BMS safety ladders, inspection procedures, and Guardian architecture docs, cataloged in `SOURCES.md`. Chunked by H2 sections into 52 passages embedded with `all-MiniLM-L6-v2` into an L2-normalized `faiss.IndexFlatIP` vector index. Benchmark evaluation across 20 domain questions in `tests/rag_eval.yaml` achieved Hit@1 = 95.0%, Hit@4 = 100.0% (surpassing the >= 80% target), with a 100% citation presence and validity rate. Answers strictly separate `[Measured/Model]` telemetry from `[Knowledge Base]` facts with mandatory chunk ID tags (`[KB-xx-Cx]`).
- **Rationale**: Fulfills Section 6 Milestone 9 (Exp 4); enables grounded, verifiable engineering question answering for the Ask Guardian assistant.

### DEC-016: LoRA Adapter Architecture, Dataset Construction, and Offline Evaluation
- **Date**: 2026-10-09
- **Decision**: Implemented PEFT LoRA adapter fine-tuning on `Qwen/Qwen2.5-0.5B-Instruct` targeting attention projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with rank $r=16$, $\alpha=32$, and dropout $0.05$. Generated 500 high-quality `(assessment JSON -> explanation JSON)` pairs using the deterministic template engine with an 85/15 train/validation split (`data/lora/train.jsonl` and `val.jsonl`). Verified CPU smoke training (20 samples, 5 steps) and offline inference via `LoRAExplainer` and `LocalHFClient` in `llm_client.py`. Evaluated validation metrics: 0% forbidden command violations, 100% numeric fidelity, and 100% level consistency, saving adapter to `artifacts/models/lora_explainer/` and metrics to `artifacts/metrics/lora.json`. Created `notebooks/colab_train_lora.ipynb` for full GPU execution.
- **Rationale**: Fulfills Section 6 Milestone 11 (Exp 7) and COLAB RULES.

### DEC-017: Sandboxed Code Execution, AST Security Guardrails, and Tool-Calling Agent
- **Date**: 2026-10-09
- **Decision**: Implemented `SandboxedCodeExecutor` featuring strict static AST security validation that forbids all `import` statements, dangerous built-ins (`open`, `eval`, `exec`, `__import__`, `compile`, `getattr`), OS/subprocess modules, and private/dunder attribute access. Subprocess execution isolates Pandas/NumPy analysis with a strict 5.0-second timeout and serialized read-only dataframe transfer. Self-healing loop repairs syntax errors, KeyErrors, and runtime faults up to 3 attempts, logging every iteration to `reports/autoheal/log.jsonl` under `source="runtime_codeexec"`. Implemented `GuardianChatAgent` equipped with 6 diagnostic tools (`get_current_state`, `get_risk`, `list_events`, `run_whatif`, `search_kb`, `run_analysis`), 6-turn conversational memory, and non-autonomous safety grounding.
- **Rationale**: Fulfills Section 6 Milestone 12 (Exp 6) and Section 9 sandbox requirements.

### DEC-018: Multi-Page Streamlit Dashboard Architecture and Visual Design
- **Date**: 2026-10-09
- **Decision**: Implemented an 8-page unified Streamlit application (`Home.py` + 7 specialized pages in `app/pages/`) adhering to Section 7 dashboard specifications. Implemented interactive telemetry replay with 6 synchronized Plotly charts, AI assessment badges with color-coded risk levels, event timeline drill-down with LLM diagnostic panels, conversational assistant with expandable tool execution traces, interactive what-if scenario testing with out-of-distribution guardrails, incident note triage with DistilBERT and synthetic fault GAN generator, model card viewer, and live auto-heal audit logs. Maintained persistent advisory footer across every page: *"Decision-support prototype. Not for vehicle control."* Verified with `streamlit.testing.v1.AppTest` across all pages.
- **Rationale**: Fulfills Section 6 Milestone 13 (Exp 8) and Operating Rule 6.

### DEC-019: What-If Simulation Physics Model and Out-of-Distribution Guardrails
- **Date**: 2026-10-09
- **Decision**: Implemented `simulate_whatif` enforcing monotonic Joule heating principles where higher simulated discharge current strictly does not reduce forward predicted temperature ($T_{pred}$). Defined bounding hypercube from NASA training data ($I \in [0.0, 3.0]\text{A}$, $T \in [18.0, 45.0]^\circ\text{C}$, $V \in [2.3, 4.3]\text{V}$). Any query exceeding these envelopes automatically activates a high-visibility Out-of-Distribution (OOD) badge (*"Outside training range — extrapolation, low confidence"*), alerting operators against over-reliance on extrapolated physics predictions.
- **Rationale**: Fulfills Section 6 Milestone 14 requirements and Operating Rules 6 and 7.

### DEC-020: Standardized Model Card and Auto-Heal Audit Report Generation
- **Date**: 2026-10-09
- **Decision**: Implemented automated generator scripts `tools/make_model_card.py` and `tools/make_autoheal_report.py`. The Model Card compiles actual empirical metrics from all upstream artifact JSON files (`artifacts/metrics/*.json`) covering VAE anomaly AUROC (0.9554), LightGBM MAE (0.0941°C), RAG Hit@4 (100.0%), DistilBERT F1 (1.0000), LoRA ROUGE-L, and GAN detection utility (100%), with complete ethical constraints and operating envelopes. The Auto-Heal Report synthesizes all development and runtime repair attempts from `reports/autoheal/log.jsonl` into a closed-loop audit table with 100% placeholder-free documentation (`reports/MODEL_CARD.md` and `reports/AUTOHEAL_REPORT.md`).
- **Rationale**: Fulfills Section 6 Milestone 15 and Section 10.4 requirements.

### DEC-021: Demo Hardening, Architecture Documentation, and Reproducibility Validation (Milestone M16)
- **Date**: 2026-10-09
- **Decision**: Finalized complete demonstration hardening. Created `data/golden_session.json` providing pre-baked offline demonstration telemetry, events, grounded citations, and auto-healed code execution traces so that all 7 dashboard pages function 100% offline without live network dependencies or API keys. Authored comprehensive live presentation runbook `docs/DEMO_SCRIPT.md` detailing step-by-step 5-7 minute technical walkthroughs across all 7 steps. Expanded `docs/ARCHITECTURE.md` with end-to-end Mermaid system diagrams, subsystem details, and safety invariants. Updated `README.md` with $\le 6$-command quickstart instructions, 8-experiment mapping table, and empirical metric summaries. Verified end-to-end reproducibility across the complete test suite.
- **Rationale**: Fulfills Section 6 Milestone 16, Section 12, and Section 13 (Definition of Done).






