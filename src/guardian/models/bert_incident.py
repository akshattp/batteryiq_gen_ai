"""Incident Report Classifier (Experiment 5).

Classifies unstructured technician/driver incident notes into failure taxonomies:
- thermal_overheat
- overcurrent
- overvoltage_or_undervoltage
- sensor_fault
- normal_observation

Supports both TF-IDF baseline and fine-tuned DistilBERT transformer models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn

CATEGORIES: List[str] = [
    "thermal_overheat",
    "overcurrent",
    "overvoltage_or_undervoltage",
    "sensor_fault",
    "normal_observation",
]
CAT_TO_IDX: Dict[str, int] = {cat: i for i, cat in enumerate(CATEGORIES)}
IDX_TO_CAT: Dict[int, str] = {i: cat for i, cat in enumerate(CATEGORIES)}


class IncidentClassifier:
    """Incident report text classifier combining baseline and transformer options."""

    CATEGORIES = CATEGORIES

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Baseline TF-IDF Pipeline
        self.baseline_pipeline: Optional[Pipeline] = None

        # Transformer objects
        self.tokenizer = None
        self.transformer_model = None

    def fit_baseline(self, texts: List[str], labels: List[int]) -> Pipeline:
        """Fit TF-IDF + LogisticRegression baseline model."""
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ])
        pipeline.fit(texts, labels)
        self.baseline_pipeline = pipeline
        return pipeline

    def predict(self, text: str, use_transformer: bool = False) -> Dict[str, Any]:
        """Predict incident category and confidence for an input note."""
        if use_transformer and self.transformer_model is not None and self.tokenizer is not None:
            self.transformer_model.eval()
            inputs = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.transformer_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_cat = IDX_TO_CAT[pred_idx]
            conf = float(probs[pred_idx])

            return {
                "category": pred_cat,
                "confidence": round(conf, 4),
                "probabilities": {cat: round(float(probs[i]), 4) for i, cat in enumerate(CATEGORIES)},
                "model_type": "distilbert",
            }

        # Baseline fallback / default
        if self.baseline_pipeline is None:
            raise RuntimeError("Classifier has not been fitted or loaded.")

        probs = self.baseline_pipeline.predict_proba([text])[0]
        pred_idx = int(np.argmax(probs))
        pred_cat = IDX_TO_CAT[pred_idx]
        conf = float(probs[pred_idx])

        return {
            "category": pred_cat,
            "confidence": round(conf, 4),
            "probabilities": {cat: round(float(probs[i]), 4) for i, cat in enumerate(CATEGORIES)},
            "model_type": "tfidf_logistic_regression",
        }

    def predict_batch(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        """Predict categories and probabilities for a batch of texts using baseline."""
        if self.baseline_pipeline is None:
            raise RuntimeError("Baseline pipeline not fitted.")
        probs = self.baseline_pipeline.predict_proba(texts)
        pred_indices = np.argmax(probs, axis=1)
        pred_cats = [IDX_TO_CAT[idx] for idx in pred_indices]
        return pred_cats, probs

    def save(self, model_dir: str | Path) -> None:
        """Save classifier artifacts to disk."""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        if self.baseline_pipeline is not None:
            joblib.dump(self.baseline_pipeline, path / "baseline_pipeline.joblib")

        meta = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "categories": CATEGORIES,
        }
        with open(path / "classifier_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, model_dir: str | Path) -> None:
        """Load classifier artifacts from disk."""
        path = Path(model_dir)
        baseline_file = path / "baseline_pipeline.joblib"
        if baseline_file.exists():
            self.baseline_pipeline = joblib.load(baseline_file)

        meta_file = path / "classifier_meta.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.model_name = meta.get("model_name", self.model_name)
            self.max_length = meta.get("max_length", self.max_length)
