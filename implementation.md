# AI Battery Safety Guardian — Implementation Plan (Agent Spec v1)

> **Audience:** an autonomous coding agent (Antigravity or similar). Follow this document literally.
> **Where it is ambiguous, choose the simplest option, record the choice in `docs/DECISIONS.md`, and continue.**

---

## 0. Operating rules for the agent (non-negotiable)

1. **Build milestone by milestone** (Section 6). Do not start milestone *N+1* until milestone *N*'s acceptance command passes.
2. **Every acceptance command must be run through the auto-heal harness** (`python tools/autoheal.py --milestone M3 -- <command>`). Never run it bare. This produces the evidence for the Auto-Heal Report (Section 10).
3. **No fabricated results.** Every number in docs, model card or dashboard must come from `artifacts/metrics/*.json` produced by real runs. If a run fails, report it; never hard-code metrics.
4. **Seeds:** all randomness uses `SEED=42` from `config/config.yaml`.
5. **Config over constants:** thresholds, paths, hyperparameters live in `config/config.yaml`. No magic numbers in code.
6. **Decision support only.** The system never issues control commands ("shut down", "disconnect"). It recommends inspection/monitoring/load-reduction for a human engineer.
7. **LLMs never compute the safety score.** The risk score is deterministic Python. LLMs only explain and retrieve.
8. **Core must work offline.** Every LLM/GenAI feature has a deterministic fallback so the demo never dies (Section 8.4).
9. **Optional modules (GAN, BERT, LoRA) sit behind feature flags** in config (`features.gan`, `features.bert`, `features.lora`). If they fail, the core app must still run.
10. Type hints, docstrings, and a pytest test for every public function in `src/`.
11. Keep a running `docs/BUILD_LOG.md` (milestone, date, what was done, what failed).
12. Max **5 heal attempts** per failure. After that, write `docs/BLOCKED.md` with the error, what was tried, and move to a fallback (Section 10.3).

---

## 1. Product definition

**Name:** AI Battery Safety Guardian
**One-liner:** Given Li-ion battery telemetry, *monitor → detect abnormal behaviour → predict thermal behaviour → score risk → explain → recommend action*, with a grounded knowledge assistant and what-if analysis.

**Explicit non-goals**
- No real-time vehicle control or BMS firmware.
- No claim that the VAE "predicts thermal runaway" (it detects deviation from learned normal behaviour).
- No CFD / full digital twin. What-if = re-running the ML temperature model.
- No claim of production readiness. Public dataset, research prototype.

---

## 2. Experiment → component mapping (all 8 used)

| # | Course experiment | Component in the project | Priority | Feature flag |
|---|---|---|---|---|
| 1 | Prompt engineering | Versioned prompt library: explanation, recommendation, RAG-answer, code-gen prompts; JSON-schema outputs, few-shot, guardrails; prompt A/B evaluation | **P0** | always on |
| 2 | GANs | Small conditional GAN (or WGAN-GP) generating synthetic *abnormal* telemetry windows to (a) augment/stress-test risk thresholds, (b) evaluate detectors. Compared against rule-based fault injection | P1 | `features.gan` |
| 3 | VAE anomaly detection | Core anomaly detector trained on normal windows only; reconstruction error → anomaly score | **P0** | always on |
| 4 | RAG pipeline | "Ask the Guardian" over a battery-safety knowledge base with citations | **P0** | always on |
| 5 | BERT fine-tuning | Incident-report classifier (text → fault category) used for triage and as an extra signal on the Incident page | P1 | `features.bert` |
| 6 | GPT text gen & conversational AI | Explanation generator + chat assistant with tool calling (telemetry lookups, what-if, code-exec analytics) | **P0** | always on |
| 7 | LoRA/QLoRA PEFT | Fine-tune a small open model (≈0.5–1.5B) to write the safety explanation in a fixed style; compare with prompted base model | P1 | `features.lora` |
| 8 | Mini project (E2E GenAI app) | The whole Streamlit app + Model Card + Auto-Heal report | **P0** | — |

---

## 3. Tech stack (pinned choices)

- Python 3.11, `uv` or `venv`, `requirements.txt` with pinned versions.
- Data/ML: pandas, numpy, scipy, scikit-learn, **LightGBM**, **PyTorch** (VAE, GAN), joblib.
- GenAI: `transformers`, `datasets`, `peft`, `bitsandbytes` (only on GPU), `sentence-transformers` (`all-MiniLM-L6-v2`), **FAISS-cpu** (vector store; no external DB), `pypdf` / plain markdown loaders.
- LLM access: a single `LLMClient` interface (Section 8.1) with providers `groq | gemini | openai | anthropic | local_hf | template`. **Default provider: `groq`** (OpenAI-compatible API at `https://api.groq.com/openai/v1`, use the `groq` or `openai` Python SDK with that base URL). Env vars: `LLM_PROVIDER=groq`, `GROQ_API_KEY`, `GROQ_MODEL` (default `llama-3.3-70b-versatile`; cheap/fast alternative `llama-3.1-8b-instant`; **verify current model names on Groq's model list at build time and record in DECISIONS.md**). Use JSON mode / tool calling where supported. Groq free tier is rate-limited: implement exponential backoff on HTTP 429, a small on-disk response cache keyed by hash of (prompt, model, temperature) in `artifacts/llm_cache/`, and never loop LLM calls inside Streamlit reruns. If the key is missing or calls fail, fall back to `template` automatically.
- UI: **Streamlit** + Plotly. (No Angular/FastAPI for the prototype.)
- Testing: pytest, pytest-cov. Lint: ruff.
- Heavy training (BERT, LoRA, GAN): scripts must run on **CPU for smoke test** (tiny subset, 1 epoch) and on a **GPU (Colab T4) for full run**. Provide `notebooks/colab_train_lora.ipynb` that clones the repo and runs the same script.

---

## 4. Repository structure (create exactly this)

```
battery-guardian/
├── README.md
├── requirements.txt
├── Makefile                      # make setup | data | train | test | app | report
├── .env.example
├── config/
│   ├── config.yaml               # all thresholds / hyperparameters / flags
│   └── prompts/                  # versioned prompt templates (*.yaml)
├── data/
│   ├── raw/                      # downloaded dataset (gitignored)
│   ├── interim/
│   ├── processed/                # battery_telemetry.csv, features.parquet
│   └── kb/                       # RAG knowledge-base docs (*.md, *.pdf)
│   └── incidents/                # incident_reports.csv (BERT)
├── src/guardian/
│   ├── data/        ingest.py  validate.py  clean.py  features.py  faults.py
│   ├── models/      vae.py  temp_predictor.py  gan.py  bert_incident.py  lora_explainer.py
│   ├── risk/        engine.py  events.py  recommend.py  whatif.py
│   ├── genai/       llm_client.py  prompts.py  explain.py  rag.py  validator.py
│   │                chat_agent.py  codeexec.py
│   └── utils/       config.py  logging.py  seed.py  schemas.py
├── app/
│   ├── Home.py
│   └── pages/ 1_Live_Monitor.py 2_Events.py 3_Ask_Guardian.py 4_WhatIf.py
│              5_Incidents.py 6_Model_Card.py 7_AutoHeal_Report.py
├── tools/
│   ├── autoheal.py               # heal harness (Section 10)
│   ├── make_model_card.py
│   └── make_autoheal_report.py
├── scripts/
│   ├── download_data.py  build_dataset.py
│   ├── train_vae.py  train_temp.py  train_gan.py  train_bert.py  train_lora.py
│   ├── build_kb_index.py  eval_all.py  generate_incidents.py
├── tests/
├── artifacts/
│   ├── models/   metrics/   figures/   index/
├── reports/
│   ├── autoheal/ log.jsonl
│   ├── MODEL_CARD.md
│   └── AUTOHEAL_REPORT.md
└── docs/ DECISIONS.md BUILD_LOG.md BLOCKED.md ARCHITECTURE.md
```

---

## 5. Data specification

### 5.1 Dataset
**Primary: NASA PCoE Li-ion Battery Aging dataset** (batteries B0005, B0006, B0007, B0018 minimum; extend to more if available). It provides per-cycle charge/discharge/impedance records with measured voltage, current, temperature, time and capacity.
- `scripts/download_data.py` tries to download; if the network or source is unavailable it prints exact manual instructions and stops with a clear message (the user places files in `data/raw/`).
- **Fallback dataset** (only if primary is unusable, log in DECISIONS.md): any public Li-ion dataset with V/I/T time series (e.g., CALCE or Oxford). Do not silently switch.

### 5.2 Unified schema — `data/processed/battery_telemetry.csv`
| column | type | note |
|---|---|---|
| battery_id | str | e.g. B0005 |
| cycle | int | cycle index |
| mode | {charge, discharge} | from source record type |
| t_s | float | seconds since start of the cycle |
| timestamp | datetime | synthetic, monotonic: cycle start + t_s (document this) |
| voltage_v | float | measured |
| current_a | float | signed: discharge negative, charge positive (document convention) |
| temp_c | float | measured |
| ambient_c | float | from dataset metadata, else config default |
| soc_pct | float | **derived by coulomb counting** from current, normalised per cycle using that cycle's capacity; clipped to [0,100] |
| capacity_ah | float | per-cycle, if available |
| soh_pct | float | capacity / initial capacity × 100 |
| label | {normal, injected_fault} | `normal` for original data |
| fault_type | str/null | only for injected |

### 5.3 Cleaning rules (`clean.py`)
- Drop duplicate timestamps within a (battery, cycle, mode).
- Resample to uniform step if irregular (config `data.resample_s`, default native step).
- Physical range checks from config: voltage 2.0–4.5 V, temp −20–80 °C, |current| ≤ 50 A; out-of-range rows flagged to `data/interim/rejected_rows.csv`, not silently dropped.
- Missing: forward-fill ≤ 3 steps, else drop segment; log counts.
- Output a **data quality report** `artifacts/metrics/data_quality.json` (rows in/out, per-rule rejects).

### 5.4 Features (`features.py`)
Per time step: `temp_c, dT_dt (°C/min, smoothed), voltage_v, dV_dt, current_a, abs_current_a, soc_pct, cycle, is_charge, temp_minus_ambient, temp_x_abs_current (interaction), soh_pct`.
Windows: sliding windows of `W=30` steps (stride 5) → flattened or 1D tensor for the VAE. Scaling: `StandardScaler` fit **on the training-normal split only**, saved to `artifacts/models/scaler.joblib`.

### 5.5 Splits (leakage-safe)
- Split **by battery/cycle**, never random rows: train = early/mid cycles of batteries A,B,C; val = other cycles; test = a **held-out battery**. Record in `artifacts/metrics/split.json`.
- VAE trains on **normal only**.

### 5.6 Fault injection (`faults.py`) — gives labelled anomalies for evaluation
Implement functions that take a normal window/session and inject, with severity parameter:
1. `thermal_ramp` (temp rises abnormally fast)
2. `current_spike` (short high-current burst)
3. `voltage_sag` (voltage drops abnormally under load)
4. `sensor_stuck` (temp flat-lines)
5. `temp_current_decoupling` (temp rises while current is ~0)
Injected data is marked `label=injected_fault`, and **never used to train the VAE**. Be explicit in docs that this is synthetic evaluation, not real fault data.

---

## 6. Milestones (each with deliverables + acceptance command)

Acceptance commands are run **via autoheal**. Tests live in `tests/`.

### M0 — Scaffold
Create repo structure, `config.yaml`, logging, seed util, `Makefile`, empty modules with docstrings, `tools/autoheal.py` (full implementation, Section 10).
**Accept:** `make test` runs (0 failing); `python tools/autoheal.py --selftest` demonstrates a deliberate failure being logged then healed.

### M1 — Data pipeline
`download_data.py → ingest → validate → clean → battery_telemetry.csv`, with quality report.
**Accept:** `pytest tests/test_data.py` — schema columns exact, no NaN in required columns, ranges valid, ≥ 3 batteries, SOC in [0,100].

### M2 — Features, windows, splits, fault injection
**Accept:** `pytest tests/test_features.py tests/test_faults.py` — window shapes correct, scaler fit on train only (test asserts no val/test stats leak), each fault type changes the signal in the expected direction.

### M3 — VAE anomaly detector (Exp 3)
- Architecture: MLP VAE, input = flattened window (W×F), hidden `[128, 64]`, latent `8`, β-VAE with `beta=0.5` (config), ELBO = MSE reconstruction + β·KL; Adam lr 1e-3, early stopping on val loss.
- **Anomaly score** = mean squared reconstruction error per window → mapped to [0,1] via empirical CDF of **normal validation** errors (`score = ecdf(err)`), saved with the model.
- **Thresholds (from normal-validation percentiles, config):** `< p95 → NORMAL`, `p95–p99.5 → WARNING`, `≥ p99.5 → CRITICAL` (named *Low/Medium/High* in the UI).
- Per-feature reconstruction error retained for explanations ("which parameters contributed").
- Baselines to compare: Isolation Forest, and PCA reconstruction error.
**Accept:** `python scripts/train_vae.py && python scripts/eval_all.py --only vae` writes `artifacts/metrics/vae.json` containing AUROC, PR-AUC, recall@5%FPR on test battery with injected faults, per fault type, vs baselines. **Target (not a gate for faking): AUROC ≥ 0.85 overall.** If missed, report honestly and tune within 3 attempts.

### M4 — Temperature predictor (supervised)
- Target: `temp_c` at `t + H` (H = 10 steps default, config), LightGBM regressor; features = Section 5.4 at t (+ lagged values 1,5,10).
- Baselines: persistence (`T(t+H)=T(t)`), linear regression.
- Metrics: MAE, RMSE, R² on held-out battery; also error by mode (charge/discharge) and by ambient.
- Save feature importance and a predicted-vs-actual figure.
**Accept:** `artifacts/metrics/temp.json` exists; LightGBM MAE **must beat persistence** or the report states it did not.

### M5 — Risk engine (deterministic, explainable)
`risk/engine.py`: pure function `assess(state, anomaly, pred_temp, history) -> RiskAssessment`.

**Components (each 0–1), then weighted:**
| component | formula (thresholds in config) | weight |
|---|---|---|
| Temperature | `clip((max(T, T_pred) − T_warn)/(T_crit − T_warn), 0, 1)` | 0.30 |
| Temp trend | `clip(dT_dt / dTdt_crit, 0, 1)` | 0.20 |
| Current | `clip(abs(I)/I_rated, 0, 1.5)/1.5` | 0.15 |
| Voltage | 1 if outside [V_min, V_max], else distance-to-limit scaled | 0.10 |
| Anomaly | VAE anomaly score | 0.25 |

`RiskScore = 100 × Σ weight·component` (**higher = riskier**; the UI says "Risk Score", not "Safety Score", to avoid ambiguity).
**Levels:** `<30 NORMAL`, `30–55 WARNING`, `55–75 HIGH`, `≥75 CRITICAL`.
**Hard overrides:** `T ≥ T_crit` → CRITICAL; `T ≥ T_high` or `V` outside limits → at least HIGH.
Defaults for the NASA ambient range go in config (e.g. `T_warn=35, T_high=42, T_crit=50` °C; `dTdt_crit=3 °C/min`) — **derive final numbers from the normal-data distribution (e.g., T_warn≈p99 of normal) and document in DECISIONS.md.**
Output also lists **top contributing factors** (components sorted by weighted contribution) as structured data.
**Accept:** `pytest tests/test_risk.py` with ≥ 15 table-driven cases incl. every override, boundary values, monotonicity (raising temperature never lowers risk).

### M6 — Event detection & recommendation engine
- `events.py`: scan a session's assessments, emit events on **level transitions**, anomaly crossing, trend breaches, decoupling; each event has `{ts, type, severity, contributing_factors, values}`; debounce (config `events.min_gap_s`).
- `recommend.py`: **rule-based** recommendation catalog mapping `(level, factors)` → ordered actions (e.g., HIGH+temp trend → "Reduce load", "Inspect thermal management", "Continue monitoring", "Escalate to engineer if temperature keeps rising"). Always ends with a human-in-the-loop line.
**Accept:** `pytest tests/test_events.py tests/test_recommend.py`; a replayed injected-fault session must produce ≥ 1 event, and recommendations never contain forbidden control verbs (test greps).

### M7 — GAN (Exp 2) *(flag: `features.gan`)*
- Conditional GAN or WGAN-GP over **windows** (small MLP/1D-conv) conditioned on fault type (`thermal_ramp`, `current_spike`, `voltage_sag`).
- Train on windows produced by `faults.py` from **train batteries only**.
- Evaluate: (a) visual overlay of real-injected vs GAN samples, (b) simple fidelity metrics (per-feature mean/std diff, a classifier two-sample test AUC), (c) **utility test:** run VAE + risk engine on GAN-generated abnormal windows and report detection rate; (d) mode-collapse check (diversity score).
- Use: threshold sanity-checking and a "synthetic scenario" button in the dashboard.
**Accept:** `artifacts/metrics/gan.json` + figure; document honestly that GAN data is synthetic.

### M8 — BERT incident classifier (Exp 5) *(flag: `features.bert`)*
- `generate_incidents.py` creates **≈ 600–900 synthetic technician/driver incident notes** (templates + LLM paraphrase if available, else template combinatorics with noise) labelled: `thermal_overheat, overcurrent, overvoltage_or_undervoltage, sensor_fault, normal_observation`. Mark dataset as **synthetic** in model card.
- Fine-tune `distilbert-base-uncased` (or `bert-base-uncased` on GPU) with HF `Trainer`; stratified 70/15/15 split; report accuracy, macro-F1, confusion matrix; compare vs TF-IDF + logistic regression baseline.
- App use: paste an incident note → predicted category + confidence → links to relevant KB passages and current telemetry window.
**Accept:** `artifacts/metrics/bert.json`; macro-F1 reported against baseline. (High scores on synthetic text are expected; state this caveat.)

### M9 — RAG knowledge assistant (Exp 4)
- **KB (`data/kb/`)**: 15–30 short markdown/PDF docs the agent assembles from reputable open sources (paraphrase, no copy-paste of copyrighted text; record source URLs + licence in `data/kb/SOURCES.md`): Li-ion thermal runaway basics, BMS functions, why high current heats cells (I²R), temperature/voltage operating windows, charging safety, SOC/SOH concepts, inspection checklist, **plus the project's own docs** (what each risk component means, what the anomaly score means).
- Pipeline: chunk (300–500 tokens, 50 overlap) → embed (MiniLM) → FAISS → top-k=4 → rerank optional → prompt → LLM answer **with chunk-id citations**.
- Answer format distinguishes **[Measured/Model]** facts (from telemetry, passed in structured context) vs **[Knowledge Base]** facts (cited chunks). If retrieval score < threshold → answer "not found in knowledge base".
- Eval set: `tests/rag_eval.yaml` with ≥ 20 questions and expected source doc; metrics: hit@k, citation validity rate, answer-faithfulness spot check (validator, Section 8.3).
**Accept:** `artifacts/metrics/rag.json` (hit@4 ≥ 0.8 target), no answer without citations (test).

### M10 — LLM layer, prompt engineering, self-correcting explanations (Exp 1 + 6)
Implement Section 8 fully: `LLMClient`, prompt library, explanation generator, validator with regenerate loop, deterministic fallback.
- **Prompt engineering study:** ≥ 3 prompt variants for the explanation task (zero-shot, few-shot, structured-JSON+rules). Evaluate on 30 stored assessments with the validator metrics (numeric fidelity, level consistency, forbidden-content rate, length). Save to `artifacts/metrics/prompts.json` and pick the winner by data.
**Accept:** `pytest tests/test_validator.py tests/test_explain_fallback.py`; with `LLM_PROVIDER=template` the full pipeline returns valid output.

### M11 — LoRA/QLoRA fine-tune (Exp 7) *(flag: `features.lora`)*
- Base: a small open instruct model (≈0.5–1.5B params, e.g. Qwen2.5-0.5B/1.5B-Instruct or TinyLlama), chosen for the available GPU.
- Training data: 400–800 `(structured assessment JSON → explanation)` pairs. Targets are produced by the **deterministic template + validated LLM outputs**, never unvalidated text. Hold out 15%.
- QLoRA (4-bit) if GPU supports it, else LoRA fp16/bf16. `r=16, alpha=32, dropout=0.05`, target attention projections.
- Evaluate vs base model (same prompt) and vs the API LLM on the held-out set using the validator metrics + ROUGE-L against reference + a small manual rubric (clarity, correctness; 20 samples).
- Save adapter to `artifacts/models/lora_explainer/`; `llm_client` provider `local_hf` loads it.
**Accept:** `artifacts/metrics/lora.json`. CPU smoke run (`--smoke`: 20 samples, 5 steps) must pass in tests; full run is a Colab step.

### M12 — Tool-calling chat agent + Code Execution with auto-heal (Exp 6)
Chat assistant (`chat_agent.py`) with tools:
- `get_current_state(session, t)`, `get_risk(session, t)`, `list_events(session)`, `run_whatif(...)`, `search_kb(query)`, and
- **`run_analysis(question)` → LLM writes pandas code → sandboxed execution → auto-heal on error** (Section 9).
**Accept:** `pytest tests/test_codeexec.py` including: a seeded buggy generated snippet (KeyError / wrong column) gets repaired within ≤ 3 attempts; a malicious snippet (`import os`, `open(...)`, `__import__`) is **rejected before execution**.

### M13 — Streamlit dashboard
Pages (Section 7). Replay mode (play/pause/speed slider) over a selected battery/cycle; each tick runs preprocessing → VAE → temp model → risk engine → events.
**Accept:** `streamlit run app/Home.py` starts; `pytest tests/test_app_smoke.py` (uses `streamlit.testing.v1.AppTest`) loads every page with no exception.

### M14 — What-if simulation
Sliders: current, SOC, ambient temp (and mode). Re-run temp predictor + risk engine; show side-by-side *Current vs What-if*.
**Guard:** if inputs fall outside the training range (min/max per feature saved at training), show a visible **"Outside training range — extrapolation, low confidence"** badge.
**Accept:** `pytest tests/test_whatif.py` — monotonic sanity (higher current at same state does not reduce predicted temp beyond tolerance), OOD badge triggers.

### M15 — Evaluation, Model Card, reports
`scripts/eval_all.py` regenerates all metrics; `tools/make_model_card.py` and `tools/make_autoheal_report.py` build the two reports from artifacts + logs.
**Accept:** both reports generated with **zero placeholder text** (`grep -n "TODO\|TBD\|XX" reports/*.md` returns nothing).

### M16 — Demo hardening
`README.md` (setup in ≤ 6 commands), `docs/ARCHITECTURE.md` with diagram, demo script (Section 12), pre-baked "golden session" so the demo works without live API.
**Accept:** fresh clone → `make setup && make data && make train && make app` works (document runtime).

---

## 7. Dashboard specification

| Page | Content |
|---|---|
| **Home / Live Monitor** | Session selector (battery, cycle, mode), replay controls. KPI cards: Temp, Voltage, Current, SOC, Cycle, SOH. AI Assessment card: **Risk Level badge, Risk Score, Anomaly Score, Predicted Temp (t+H)**. Six charts (T, I, V, SOC, anomaly score with threshold lines, predicted vs actual T) with event markers. Top-contributing-factors bar. |
| **Events** | Timeline table. Click an event → panel: *Why detected?* (factors + per-feature VAE error bars), *What does the model recommend?*, *LLM explanation* (validated), *Ask follow-up* button. |
| **Ask the Guardian** | Chat (Exp 6) + RAG (Exp 4). Shows answer, **[Measured/Model] vs [Knowledge Base]** sections, citations with expandable source chunks, and a "tools used" trace (incl. generated code + heal attempts when `run_analysis` was used). |
| **What-If** | Sliders, side-by-side comparison, OOD badge. |
| **Incidents** | Paste incident note → BERT category + confidence + similar KB passages; GAN "generate synthetic fault scenario" button (flags). |
| **Model Card** | Renders `reports/MODEL_CARD.md`. |
| **Auto-Heal Report** | Renders `reports/AUTOHEAL_REPORT.md` + table from `log.jsonl`. |

Persistent footer on every page: *"Decision-support prototype. Not for vehicle control."*

---

## 8. GenAI layer specification

### 8.1 `LLMClient`
```python
class LLMClient(Protocol):
    def generate(self, messages: list[dict], *, json_schema: dict | None = None,
                 temperature: float = 0.2, max_tokens: int = 500) -> str: ...
```
Providers: `groq` (default), `gemini`, `openai`, `anthropic`, `local_hf` (optionally with LoRA adapter), `template` (deterministic, no network). Retries with exponential backoff (max 3) on transient errors; on final failure raise `LLMUnavailable` → caller uses fallback.

### 8.2 Prompt library (`config/prompts/*.yaml`, versioned)
Files: `explain_v1/v2/v3.yaml`, `rag_answer_v1.yaml`, `codegen_v1.yaml`, `heal_codegen_v1.yaml`, `incident_assist_v1.yaml`. Each has `id, version, system, user_template, few_shot, output_schema, changelog`. Rules embedded in the system prompt:
- Use **only** numbers from the provided JSON.
- State the risk level exactly as given.
- Do not recommend autonomous control; recommend inspection/monitoring/escalation.
- Output strict JSON: `{"summary": str, "reasons": [str], "recommended_actions": [str], "confidence_note": str}`.

### 8.3 Validator & self-correction loop for explanations (`validator.py`)
`validate(output, structured_input) -> list[Violation]` checks:
1. JSON parses and matches schema.
2. **Numeric fidelity:** every number in text appears in the input within tolerance (±0.1 or 1%).
3. **Level consistency:** mentioned risk level equals the input level; no contradictory words (e.g., "safe" when level ≥ HIGH).
4. **Forbidden content:** control verbs/phrases list in config (e.g., "shut down the vehicle", "disable the BMS", "ignore").
5. **Grounding for RAG:** every factual claim paragraph has ≥ 1 valid chunk-id citation; cited ids exist in retrieved set.
6. Length limits.

Loop: `generate → validate → if violations: regenerate with violation list appended as feedback (max 2 retries) → else fallback to deterministic template`. **Every iteration is logged** into `reports/autoheal/log.jsonl` with `source="runtime"`.

### 8.4 Deterministic fallback
`explain.py::template_explanation(structured_input)` builds a faithful explanation from the factor list with no LLM. It is also the ground-truth style reference for LoRA data.

### 8.5 Chat agent
Tool-calling loop (max 5 tool calls per turn). Tool results are structured and passed back; final answer must pass the same validator (numbers/citations). Conversation memory: last 6 turns.

---

## 9. Code execution & auto-heal sandbox (`codeexec.py`)

Purpose: let users ask free-form data questions ("What was the max temperature in cycle 120 and when?") answered by LLM-written pandas code.

**Pipeline**
1. LLM gets: dataframe schema (columns, dtypes, 3 sample rows), the question, and rules. Returns Python code that sets `result`.
2. **Static check (AST):** allow only `pandas`, `numpy` (pre-imported; `import` statements forbidden), no `eval/exec/open/__import__/getattr` on dunder, no attribute names starting with `_`, no file/network/OS/subprocess access. Reject → heal attempt with the rejection reason.
3. **Execute** in a **subprocess** with CPU/time limit (5 s), memory limit, read-only dataframe copy passed via parquet/pickle, no env vars.
4. **Auto-heal:** on exception, empty/`None` result, or timeout → send `{code, traceback (last 15 lines), error class}` to `heal_codegen_v1`, regenerate, retry. **Max 3 attempts.**
5. On success return `{answer, code, attempts, healed: bool}`. On failure return a polite failure with the last error; never crash the app.
6. Log each attempt (Section 10.2 schema, `source="runtime_codeexec"`).

---

## 10. Auto-heal loop (development-time + runtime)

### 10.1 `tools/autoheal.py` (development-time harness)
CLI: `python tools/autoheal.py --milestone M3 [--max-attempts 5] -- <shell command>`.
Behaviour:
1. Runs command, captures stdout/stderr, exit code, duration.
2. On success → append a `PASS` record and exit 0.
3. On failure → append a `FAIL` record with an **error excerpt (last 40 lines)**, classify error (`ImportError, SyntaxError, AssertionError, ShapeMismatch, KeyError, NetworkError, OOM, Timeout, Other`), print a banner: *"HEAL REQUIRED: diagnose, patch, then re-run"*, and exit non-zero.
4. The agent must then, **before re-running**, call `python tools/autoheal.py --record-fix --milestone M3 --hypothesis "<root cause>" --files "<changed files>" --summary "<what changed>"`, so the log links *failure → diagnosis → fix → re-run result*.
5. Attempt counter per `(milestone, command)`; at the cap, write `docs/BLOCKED.md` and exit with code 3.
6. `--selftest`: creates a temp script with a deliberate bug, runs the full fail→fix→pass cycle and verifies the log (used in M0 acceptance).

### 10.2 Log schema (`reports/autoheal/log.jsonl`, one JSON per line)
```json
{"id": "uuid", "ts": "ISO8601", "source": "dev|runtime|runtime_codeexec",
 "milestone": "M3", "attempt": 2, "command": "...", "status": "FAIL|FIX|PASS",
 "exit_code": 1, "error_class": "ShapeMismatch", "error_excerpt": "...",
 "hypothesis": "...", "files_changed": ["src/..."], "fix_summary": "...",
 "duration_s": 12.3, "linked_to": "id-of-failed-record"}
```

### 10.3 Fallback ladder when blocked
BERT/LoRA/GAN failures → disable flag, keep going. LLM API failure → `template` provider. Dataset download failure → manual-instructions stop + ask user. Never delete tests to make them pass.

### 10.4 `reports/AUTOHEAL_REPORT.md` — generated by `make_autoheal_report.py`
Sections (all computed from the log):
1. **Executive summary** — total runs, failures, heal success rate, mean attempts-to-heal, by source (dev / runtime / codeexec).
2. **Method** — the loop diagram (Run → Detect → Diagnose → Patch → Re-run → Verify), caps, safeguards (AST sandbox, validators).
3. **Heal-case table** — id, milestone, error class, root cause, fix, attempts, outcome.
4. **Three deep-dive case studies** — pick the most instructive failures (before/after snippet, why the fix worked).
5. **Runtime self-correction results** — explanation validator: % first-pass valid vs after-retry vs fallback used; code-exec: first-try success vs healed vs failed.
6. **Prompt-variant comparison** (from M10).
7. **Failure taxonomy chart** (counts by error class).
8. **Limitations & residual risks** — what auto-heal cannot fix (wrong-but-running code, data issues), why validators and tests are still required.
9. **Reproducibility** — commands to regenerate.

---

## 11. AI Model Card (`reports/MODEL_CARD.md`)

One card per model + one system-level card, following the Mitchell et al. structure. `make_model_card.py` fills metrics from JSON; narrative sections are written once and committed.

**System card:** overview, intended use (decision support for engineers/students), out-of-scope uses (vehicle control, safety certification), architecture diagram, human-in-the-loop design, data lineage, known failure modes, ethical/safety considerations, maintenance.

**Per model (VAE, Temp predictor, GAN, BERT classifier, LoRA explainer, RAG pipeline):**
1. Model details (type, version, framework, date, seed, hyperparameters)
2. Intended use / out-of-scope
3. Training data (source, size, split, preprocessing, **synthetic components flagged**)
4. Evaluation data & metrics (with baselines, per-subgroup: per battery, charge vs discharge, ambient temperature, fault type)
5. Quantitative results (tables/figures from artifacts)
6. Limitations (small public dataset, lab cycling ≠ road conditions, injected faults ≠ real faults, synthetic incident text, ambient range limits, no real thermal-runaway events)
7. Risks & mitigations (false negatives, over-reliance, LLM hallucination → validators, prompt-injection via incident text → input sanitisation + no tool access from BERT path)
8. Explainability (VAE per-feature error, risk-factor breakdown, LightGBM importance/SHAP optional, RAG citations)
9. Responsible-AI notes (human-in-the-loop, no autonomous actions, data licence/attribution)
10. Reproducibility (commands, commit hash, artifact paths)

---

## 12. Final demo script (5–7 min; build a `docs/DEMO_SCRIPT.md`)
1. Select held-out battery → replay a **normal** cycle: green, low anomaly.
2. Switch to **injected thermal_ramp** session: anomaly rises → events appear → risk goes WARNING → HIGH.
3. Click event → factors, per-feature VAE error, validated LLM explanation, rule-based actions.
4. Ask Guardian: "Why can high current raise temperature?" → RAG answer with citations; then "Max temperature in this cycle and when?" → code-exec with a visible **auto-heal** attempt.
5. What-if: current 10A→15A, ambient +10 °C → predicted temp/risk delta; show OOD badge if applicable.
6. Incidents page: paste a note → BERT category; generate GAN scenario.
7. Open Model Card and Auto-Heal Report; show metrics table (VAE vs baselines, temp model vs persistence, prompt variants, LoRA vs base).

---

## 13. Definition of Done (checklist)

- [ ] M0–M16 acceptance commands all PASS in `reports/autoheal/log.jsonl`
- [ ] `pytest` green, coverage ≥ 70% on `src/guardian/risk`, `genai/validator.py`, `genai/codeexec.py`
- [ ] App runs fully offline (template provider) and with an LLM key
- [ ] All metrics in docs come from `artifacts/metrics/*.json`
- [ ] Both reports generated, zero placeholders
- [ ] README reproduces from a clean clone
- [ ] Every optional module either works or is cleanly disabled with the reason in BUILD_LOG
- [ ] Footer disclaimer on every page; no control commands anywhere in outputs (tested)

---

## 14. Suggested agent execution order & parallelism
Critical path: **M0 → M1 → M2 → M3 → M4 → M5 → M6 → M13**. Parallelisable after M2: M9 (RAG KB), M8 (incident data), M7 (GAN). M10 needs M5/M6 outputs. M11 needs M10. M12 needs M9/M10. M14 needs M4/M5. M15–M16 last.

Start now with: **M0 scaffold + autoheal harness**, then report back with the `--selftest` log before proceeding.


---

## 15. Compute plan: laptop (CPU) vs Colab T4

Only **LoRA/QLoRA** truly needs a GPU. Everything else runs on a CPU laptop.

| Component | Where | Notes |
|---|---|---|
| Data pipeline, features, fault injection | Laptop | pandas |
| VAE (small MLP) | Laptop CPU | minutes |
| LightGBM temp model | Laptop CPU | minutes |
| GAN (small MLP/1D-conv on windows) | Laptop CPU | cap epochs in config; fine for ~10k windows |
| DistilBERT incident classifier (~600 samples, 3 epochs, max_len 64) | Laptop CPU (~5–15 min) or Colab | run on Colab if CPU is too slow |
| **LoRA/QLoRA explainer** | **Colab T4** | use `notebooks/colab_train_lora.ipynb` |
| RAG index (MiniLM + FAISS) | Laptop CPU | |
| LLM calls | Groq API | |

### Colab workflow (the "save the model" part)
1. Push the repo to GitHub (or upload a zip). In Colab: `Runtime → Change runtime type → T4 GPU`.
2. Notebook cells: mount Google Drive → clone repo → `pip install -r requirements-colab.txt` → upload/copy `data/lora/train.jsonl` and `val.jsonl` (generated locally by `scripts/build_lora_dataset.py`) → run `python scripts/train_lora.py --out /content/drive/MyDrive/guardian/lora_explainer`.
3. **Save to Drive every N steps** (`save_steps=50`, `save_total_limit=2`) so a Colab disconnect doesn't lose progress; script supports `--resume`.
4. Only the **LoRA adapter** (a few MB) is saved, not the base model. Download the adapter folder and place it at `artifacts/models/lora_explainer/`. Optionally push the adapter to the Hugging Face Hub (`push_to_hub`) as a private repo.
5. At inference the app loads base model + adapter with `peft` (`local_hf` provider). On a CPU laptop a 0.5B model with adapter is slow but workable for a few demo calls; for the live demo, **default to Groq**, and show the LoRA model in a separate "Compare explainers" panel using pre-generated outputs saved to `artifacts/metrics/lora_samples.json`.
6. Also save `train_log.json` (loss curve) and `lora.json` metrics from Colab into the repo for the model card.
7. Use `Qwen2.5-0.5B-Instruct` for the first run (fits T4 easily without 4-bit); try QLoRA 4-bit on a 1.5B model only if time remains.

---

## 16. Deadline mode: build in one day (tiered scope)

**Principle:** a working golden path first, extras as honest, reduced-scale runs. All reduced runs must be labelled as such in the model card ("smoke-scale training: N samples, E epochs").

### Tier A — must be working (hours 0–~9)
M0 scaffold + autoheal harness → M1 data → M2 features/faults → M3 VAE → M4 temp model → M5 risk engine → M6 events/recommend → M10 LLM layer on Groq (with validator + template fallback) → M13 dashboard (Live Monitor + Events pages).
**Reduced settings for speed:** 3 batteries (B0005, B0006, B0007 train/val, B0018 test), 1 horizon (H=10), VAE ≤ 30 epochs, LightGBM defaults, prompt study with 3 variants on 15 assessments.

### Tier B — adds the scope and the story (hours ~9–16)
M9 RAG (12–15 short KB docs, 15-question eval) → M12 chat agent + code-exec with auto-heal → M14 What-If → M15 Model Card + Auto-Heal Report generation.

### Tier C — extra experiments at reduced scale (hours ~16–22, parallel where possible)
M7 GAN (conditional, 3 fault types, small) · M8 BERT (400 synthetic notes, DistilBERT, 3 epochs) · M11 LoRA (Colab, ~300 pairs, 1–2 epochs, 0.5B model).
If any Tier C item is not finished, set its feature flag to false, record the reason in BUILD_LOG, and keep its section in the Model Card with status "planned / partially run" — **never present unrun work as done.**

### Final 2 hours — M16 hardening
Pre-baked golden session, run the demo script (Section 12) twice, regenerate reports, rehearse the viva answers: why VAE trained on normal only, why the LLM doesn't compute risk, what the auto-heal loop does and cannot do, limits of NASA data.

### Parallelism tip for the agent
Run the Tier C items in separate agent threads/workspaces once M2 and M5 are done, so the critical path (Tier A) is never blocked.