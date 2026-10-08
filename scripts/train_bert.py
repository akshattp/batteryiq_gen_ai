"""scripts/train_bert.py - Train and evaluate incident report classifier (Milestone M8 / Exp 5).

Self-contained script suitable for local execution and Google Colab (T4 GPU).
Trains TF-IDF + Logistic Regression baseline and fine-tunes DistilBERT on synthetic
battery incident notes across 5 categories with stratified 70/15/15 splits.
Outputs comparative metrics, confusion matrix, sample predictions, and saved model weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_bert")

CATEGORIES: List[str] = [
    "thermal_overheat",
    "overcurrent",
    "overvoltage_or_undervoltage",
    "sensor_fault",
    "normal_observation",
]
CAT_TO_IDX: Dict[str, int] = {cat: i for i, cat in enumerate(CATEGORIES)}
IDX_TO_CAT: Dict[int, str] = {i: cat for i, cat in enumerate(CATEGORIES)}


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IncidentDataset(Dataset):
    """PyTorch Dataset for tokenized incident notes."""

    def __init__(self, encodings: Dict[str, Any], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


class IncidentBaselineClassifier:
    """TF-IDF + LogisticRegression baseline classifier."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=500, random_state=42)

    def fit(self, texts: List[str], labels: List[int]) -> None:
        x_tfidf = self.vectorizer.fit_transform(texts)
        self.clf.fit(x_tfidf, labels)

    def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        x_tfidf = self.vectorizer.transform(texts)
        probs = self.clf.predict_proba(x_tfidf)
        preds = np.argmax(probs, axis=1).tolist()
        confs = [float(np.max(p)) for p in probs]
        return preds, confs

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path / "tfidf_vectorizer.joblib")
        joblib.dump(self.clf, path / "logistic_classifier.joblib")


def plot_confusion_matrix(cm: List[List[int]], classes: List[str], save_path: Path) -> None:
    """Save confusion matrix as a PNG figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.matshow(cm, cmap="Blues")
        fig.colorbar(cax)

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="left", fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)

        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black" if cm[i][j] < (np.max(cm) / 2) else "white")

        plt.title("DistilBERT Incident Confusion Matrix", pad=20)
        plt.xlabel("Predicted Category")
        plt.ylabel("True Category")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {save_path}")
    except Exception as exc:
        logger.warning(f"Could not save confusion matrix plot: {exc}")


def train_distilbert(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 64,
    device: Optional[str] = None,
) -> Tuple[Any, Any, List[Dict[str, Any]]]:
    """Fine-tune DistilBERT on CPU or GPU for sequence classification."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading tokenizer and model: {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(CATEGORIES))
    model.to(device)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

    train_dataset = IncidentDataset(train_encodings, train_labels)
    val_dataset = IncidentDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    logger.info(f"Starting DistilBERT fine-tuning ({epochs} epochs, {len(train_texts)} train samples, batch_size={batch_size})...")
    t0 = time.time()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation pass
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().tolist())

        mean_train_loss = running_loss / max(1, len(train_loader))
        mean_val_loss = val_loss / max(1, len(val_loader))
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        epoch_stat = {
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": round(mean_val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
        }
        history.append(epoch_stat)
        logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | Val F1: {val_f1:.4f}")

    logger.info(f"DistilBERT fine-tuning completed in {time.time() - t0:.1f}s")
    return model, tokenizer, history


def evaluate_predictions(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Compute standard classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    per_class_f1 = f1_score(y_true, y_pred, average=None).tolist()
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_f1": {cat: round(score, 4) for cat, score in zip(CATEGORIES, per_class_f1)},
        "confusion_matrix": cm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train incident classifier (M8 / Colab).")
    parser.add_argument("--data", type=str, default="data/incidents/incidents.csv", help="Input incidents CSV")
    parser.add_argument("--epochs", type=int, default=3, help="DistilBERT epochs (default 3)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on subset")
    parser.add_argument("--out-dir", type=str, default="artifacts/models/incident_classifier", help="Model output dir")
    parser.add_argument("--metrics-out", type=str, default="artifacts/metrics/bert.json", help="Metrics JSON path")
    parser.add_argument("--cm-out", type=str, default="artifacts/figures/bert_confusion_matrix.png", help="Confusion matrix PNG")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'})")

    data_path = Path(args.data)
    if not data_path.exists():
        alt_path = data_path.parent / "incident_reports.csv"
        if alt_path.exists():
            data_path = alt_path

    logger.info(f"Loading incident dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Resolve text and category columns
    text_col = "text" if "text" in df.columns else "note"
    cat_col = "label" if "label" in df.columns else "category"

    if args.smoke:
        df = df.groupby(cat_col).head(20).reset_index(drop=True)
        epochs = 1
        logger.info(f"Running in smoke mode with {len(df)} samples, epochs=1.")
    else:
        epochs = args.epochs

    texts = df[text_col].tolist()
    labels = [CAT_TO_IDX[cat] for cat in df[cat_col]]

    # Stratified 70 / 15 / 15 split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        test_texts, test_labels, test_size=0.50, stratify=test_labels, random_state=42
    )

    logger.info(f"Split sizes: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    # 1. Train and evaluate TF-IDF baseline
    logger.info("Training TF-IDF + Logistic Regression baseline...")
    baseline = IncidentBaselineClassifier()
    baseline.fit(train_texts, train_labels)

    base_pred_indices, _ = baseline.predict(test_texts)
    baseline_metrics = evaluate_predictions(test_labels, base_pred_indices)
    logger.info(f"Baseline Test Accuracy: {baseline_metrics['accuracy']:.1%}, Macro-F1: {baseline_metrics['macro_f1']:.4f}")

    # 2. Fine-tune DistilBERT
    logger.info(f"Fine-tuning DistilBERT transformer for {epochs} epochs...")
    model, tokenizer, history = train_distilbert(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=args.batch_size,
        device=device,
    )

    # Evaluate DistilBERT on Test Split
    model.eval()
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        test_input_ids = test_encodings["input_ids"].to(device)
        test_mask = test_encodings["attention_mask"].to(device)
        outputs = model(input_ids=test_input_ids, attention_mask=test_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        bert_pred_indices = np.argmax(probs, axis=1).tolist()
        bert_confs = np.max(probs, axis=1).tolist()

    bert_metrics = evaluate_predictions(test_labels, bert_pred_indices)
    logger.info(f"DistilBERT Test Accuracy: {bert_metrics['accuracy']:.1%}, Macro-F1: {bert_metrics['macro_f1']:.4f}")

    # Sample test predictions table
    sample_preds = []
    for i in range(min(15, len(test_texts))):
        sample_preds.append({
            "text": test_texts[i],
            "true_label": IDX_TO_CAT[test_labels[i]],
            "baseline_pred": IDX_TO_CAT[base_pred_indices[i]],
            "bert_pred": IDX_TO_CAT[bert_pred_indices[i]],
            "bert_confidence": round(float(bert_confs[i]), 4),
        })

    # 3. Save Model Artifacts
    model_dir = Path(args.out_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    baseline.save(model_dir)

    # Save transformer weights and tokenizer
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    torch.save(model.state_dict(), model_dir / "distilbert_weights.pt")
    logger.info(f"Saved classifier artifacts to {model_dir}")

    # Plot confusion matrix
    cm_path = Path(args.cm_out)
    plot_confusion_matrix(bert_metrics["confusion_matrix"], CATEGORIES, cm_path)

    # 4. Save Comprehensive Metrics
    out_payload = {
        "milestone": "M8",
        "experiment": "Exp 5 (BERT Incident Classifier)",
        "dataset": {
            "total_samples": len(df),
            "is_synthetic": True,
            "categories": CATEGORIES,
            "split": {
                "train": len(train_texts),
                "val": len(val_texts),
                "test": len(test_texts),
            },
        },
        "training_config": {
            "model_name": "distilbert-base-uncased",
            "epochs": epochs,
            "batch_size": args.batch_size,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "history": history,
        },
        "baseline_tfidf": baseline_metrics,
        "distilbert": bert_metrics,
        "sample_predictions": sample_preds,
        "model_dir": str(model_dir),
        "caveat": (
            "Model trained on synthetic technician reports generated via domain templates; "
            "high macro-F1 is expected due to consistent synthetic phrasing patterns."
        ),
    }

    out_metrics = Path(args.metrics_out)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    logger.info(f"Saved incident metrics to {out_metrics}")
    print(f"\n=======================================================")
    print(f"[BERT INCIDENT CLASSIFIER COMPLETE]")
    print(f"Device: {device}")
    print(f"Baseline Macro-F1: {baseline_metrics['macro_f1']:.4f}")
    print(f"DistilBERT Macro-F1: {bert_metrics['macro_f1']:.4f}")
    print(f"Metrics saved to: {out_metrics}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
