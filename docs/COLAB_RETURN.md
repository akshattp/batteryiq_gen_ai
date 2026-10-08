# Google Colab Handoff & Return Runbook

> **Target:** NVIDIA T4 GPU runtime on Google Colab (Free tier).  
> **Modules Covered:** **M8 (DistilBERT Incident Classifier)** & **M11 (Qwen-2.5-0.5B LoRA Explainer)**.  
> **Key Principle:** Notebooks are 100% self-contained and require **ZERO API keys** (all data is pre-generated locally).

---

## 1. Quick Step-by-Step Instructions

### Step 1: Build the Colab Pack locally
Run this single command in your local project root:
```bash
make colab-pack
# or on Windows without make:
python scripts/make_colab_pack.py
```
This generates `colab_pack.zip` (~90 KB), containing only what Colab requires:
- `scripts/train_bert.py` & `scripts/train_lora.py` (self-contained, no `src/guardian` dependency)
- `data/incidents/incidents.csv` & `data/lora/*.jsonl`
- `requirements-colab.txt`
- `notebooks/colab_train_bert.ipynb` & `notebooks/colab_train_lora.ipynb`

---

### Step 2: Open and Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Go to **File $\rightarrow$ Upload Notebook** and upload either:
   - `notebooks/colab_train_bert.ipynb` (for DistilBERT)
   - `notebooks/colab_train_lora.ipynb` (for LoRA Qwen)
3. **Change Runtime Type to T4 GPU**:
   - Navigate to **Runtime $\rightarrow$ Change runtime type**.
   - Under *Hardware accelerator*, select **T4 GPU**.
   - Click **Save**.
4. Run all cells (**Runtime $\rightarrow$ Run all**):
   - **Cell 1**: Asserts T4 GPU is active.
   - **Cell 2**: Prompts you to upload `colab_pack.zip` from your local machine, extracts it, and installs pinned dependencies.
   - **Cell 3**: Runs a quick smoke test to verify execution integrity.
   - **Cell 4**: Executes the full training run with progress logging (DistilBERT: 3 epochs; LoRA: 2 epochs, fp16, checkpointing every 50 steps, resume support).
   - **Cell 5**: Automatically packages the results into a ZIP file and triggers a browser download.

---

## 2. Downloaded Files & Destination Mappings

When the notebooks finish running Cell 5, your browser will download:

| Colab Output Bundle | Destination Artifact Paths in Repository |
| :--- | :--- |
| **`results_bert.zip`** | - Model weights & tokenizer: `artifacts/models/incident_classifier/`<br>- Baseline & BERT metrics: `artifacts/metrics/bert.json`<br>- Confusion matrix plot: `artifacts/figures/bert_confusion_matrix.png` |
| **`results_lora.zip`** | - PEFT Adapter weights & config: `artifacts/models/lora_explainer/`<br>- Full metrics JSON: `artifacts/metrics/lora.json`<br>- Training loss curve: `artifacts/metrics/train_log.json`<br>- Base vs Adapter sample outputs: `artifacts/metrics/lora_samples.jsonl` |

---

## 3. Ingesting the Downloaded Files Locally

Once you have downloaded `results_bert.zip` and/or `results_lora.zip`:

1. **Move the ZIP file(s) into your project root directory:**
   ```
   GEN AI LAB/
   ├── results_bert.zip   <-- place here
   ├── results_lora.zip   <-- place here
   ├── artifacts/
   ...
   ```

2. **Run the automated ingestion script:**
   ```bash
   python scripts/ingest_colab_results.py
   ```
   This script will:
   - Extract the bundles into their target directories.
   - Validate that all metrics JSON files and model checkpoints exist and are well-formed.
   - Fail loudly with a clear error message if any expected file is missing.

---

## 4. Post-Ingestion Integration & Verification

After running `ingest_colab_results.py`:

1. **Regenerate the Model Card:**
   ```bash
   python tools/make_model_card.py
   ```
   The Model Card will immediately update with the real T4 training metrics and GPU specifications.

2. **Verify tests:**
   ```bash
   pytest tests/test_bert.py tests/test_lora.py
   ```

3. **Launch the Dashboard:**
   ```bash
   streamlit run app/Home.py
   ```
   - **Page 5 (`5_Incidents.py`)**: Features the Colab-trained DistilBERT incident triage model.
   - **Page 6 (`6_Model_Card.py`)**: Renders the complete updated benchmarks.
