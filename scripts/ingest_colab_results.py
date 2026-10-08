"""scripts/ingest_colab_results.py - Ingest and validate returned Colab training bundles.

Unzips results_bert.zip and results_lora.zip into target artifact directories,
validates that expected files exist and metrics are well-formed, and fails loudly if not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile


def ingest_bert_bundle(zip_path: Path) -> bool:
    """Ingest results_bert.zip into artifacts directories."""
    print(f"\n[INGESTING BERT] Unpacking: {zip_path.resolve()}...")
    if not zip_path.exists():
        raise FileNotFoundError(f"BERT results bundle not found at: {zip_path}")

    extract_tmp = Path("artifacts/.tmp_bert_extract")
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_tmp)

    # Search for bert.json
    found_metrics = list(extract_tmp.rglob("bert.json"))
    if not found_metrics:
        raise FileNotFoundError(f"Malformed bundle: bert.json not found in {zip_path}")

    metrics_src = found_metrics[0]
    metrics_dest = Path("artifacts/metrics/bert.json")
    metrics_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_src, metrics_dest)
    print(f"  -> Metrics installed: {metrics_dest}")

    # Validate JSON
    with open(metrics_dest, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    if "distilbert" not in m_data or "macro_f1" not in m_data["distilbert"]:
        raise ValueError("bert.json is missing required 'distilbert.macro_f1' field!")
    print(f"  -> Validated DistilBERT Macro-F1: {m_data['distilbert']['macro_f1']:.4f}")

    # Search for model weights / config
    model_dest = Path("artifacts/models/incident_classifier")
    model_dest.mkdir(parents=True, exist_ok=True)

    # Search for saved model files
    for weight_file in ["distilbert_weights.pt", "model.safetensors", "pytorch_model.bin", "config.json"]:
        found = list(extract_tmp.rglob(weight_file))
        if found:
            shutil.copy2(found[0], model_dest / weight_file)
            print(f"  -> Installed model file: {weight_file}")

    # Baseline files
    for baseline_file in ["logistic_classifier.joblib", "tfidf_vectorizer.joblib"]:
        found = list(extract_tmp.rglob(baseline_file))
        if found:
            shutil.copy2(found[0], model_dest / baseline_file)
            print(f"  -> Installed baseline artifact: {baseline_file}")

    # Confusion matrix figure
    found_cm = list(extract_tmp.rglob("*.png"))
    if found_cm:
        cm_dest = Path("artifacts/figures/bert_confusion_matrix.png")
        cm_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found_cm[0], cm_dest)
        print(f"  -> Installed confusion matrix figure: {cm_dest}")

    # Cleanup tmp
    shutil.rmtree(extract_tmp)
    print("✅ BERT Colab bundle ingested and validated successfully.")
    return True


def ingest_lora_bundle(zip_path: Path) -> bool:
    """Ingest results_lora.zip into artifacts directories."""
    print(f"\n[INGESTING LORA] Unpacking: {zip_path.resolve()}...")
    if not zip_path.exists():
        raise FileNotFoundError(f"LoRA results bundle not found at: {zip_path}")

    extract_tmp = Path("artifacts/.tmp_lora_extract")
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_tmp)

    # 1. Search for lora.json
    found_metrics = list(extract_tmp.rglob("lora.json"))
    if not found_metrics:
        raise FileNotFoundError(f"Malformed bundle: lora.json not found in {zip_path}")

    metrics_src = found_metrics[0]
    metrics_dest = Path("artifacts/metrics/lora.json")
    metrics_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_src, metrics_dest)
    print(f"  -> Metrics installed: {metrics_dest}")

    with open(metrics_dest, "r", encoding="utf-8") as f:
        l_data = json.load(f)
    if "evaluation" not in l_data:
        raise ValueError("lora.json is missing required 'evaluation' section!")
    print(f"  -> Validated LoRA metrics JSON (eval samples: {l_data.get('train_samples')})")

    # 2. Adapter files
    adapter_dest = Path("artifacts/models/lora_explainer")
    adapter_dest.mkdir(parents=True, exist_ok=True)
    found_adapters = list(extract_tmp.rglob("adapter_config.json"))
    if not found_adapters:
        raise FileNotFoundError("Malformed bundle: adapter_config.json not found in results_lora.zip!")

    adapter_src_dir = found_adapters[0].parent
    for item in adapter_src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, adapter_dest / item.name)
            print(f"  -> Installed adapter artifact: {item.name}")

    # 3. train_log.json
    found_tlog = list(extract_tmp.rglob("train_log.json"))
    if found_tlog:
        tlog_dest = Path("artifacts/metrics/train_log.json")
        shutil.copy2(found_tlog[0], tlog_dest)
        print(f"  -> Installed loss curve: {tlog_dest}")

    # 4. lora_samples.jsonl
    found_samples = list(extract_tmp.rglob("lora_samples.json*"))
    if found_samples:
        for s_file in found_samples:
            target = Path("artifacts/metrics") / s_file.name
            shutil.copy2(s_file, target)
            print(f"  -> Installed evaluation samples: {target}")

    # Cleanup tmp
    shutil.rmtree(extract_tmp)
    print("✅ LoRA Colab bundle ingested and validated successfully.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Google Colab training result bundles.")
    parser.add_argument("--bert-zip", type=str, default="results_bert.zip", help="Path to results_bert.zip")
    parser.add_argument("--lora-zip", type=str, default="results_lora.zip", help="Path to results_lora.zip")
    args = parser.parse_args()

    # Configure UTF-8 stdout if possible
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    bert_p = Path(args.bert_zip)
    lora_p = Path(args.lora_zip)

    if not bert_p.exists() and not lora_p.exists():
        print(f"[!] Neither '{bert_p}' nor '{lora_p}' found in workspace root.")
        print("Please place the downloaded Colab zip files in the project root and re-run:")
        print("  python scripts/ingest_colab_results.py")
        sys.exit(1)

    success_count = 0
    if bert_p.exists():
        try:
            ingest_bert_bundle(bert_p)
            success_count += 1
        except Exception as exc:
            print(f"[ERROR] Failed to ingest BERT bundle: {exc}")
            sys.exit(1)

    if lora_p.exists():
        try:
            ingest_lora_bundle(lora_p)
            success_count += 1
        except Exception as exc:
            print(f"[ERROR] Failed to ingest LoRA bundle: {exc}")
            sys.exit(1)

    print(f"\n[SUCCESS] Successfully ingested {success_count} Colab bundle(s).")
    print("Next steps:")
    print("  1. Regenerate model card: python tools/make_model_card.py")
    print("  2. Verify test suite:     pytest tests/test_bert.py tests/test_lora.py")
    print("  3. Launch dashboard:      streamlit run app/Home.py")


if __name__ == "__main__":
    main()
