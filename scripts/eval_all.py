"""scripts/eval_all.py - System-wide evaluation script.

Implements rigorous evaluation of VAE anomaly detection against Isolation Forest
and PCA baselines across synthetic fault modes on held-out battery B0018.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from guardian.utils.config import load_config
from guardian.utils.seed import set_seed
from guardian.data.features import (
    FEATURE_COLUMNS,
    create_splits,
    create_sliding_windows,
    load_scaler,
)
from guardian.data.faults import FAULT_TYPES, inject_fault
from guardian.models.vae import VAEAnomalyDetector
from guardian.utils.logging import setup_logger

logger = setup_logger("scripts.eval_all")


def compute_metrics_for_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute AUROC, PR-AUC, and Recall at 5% FPR."""
    auroc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Find operating point where FPR <= 0.05
    idx_5pct = np.where(fpr <= 0.05)[0]
    recall_at_5pct = float(tpr[idx_5pct[-1]]) if len(idx_5pct) > 0 else 0.0

    return {
        "auroc": round(auroc, 4),
        "pr_auc": round(pr_auc, 4),
        "recall_at_5pct_fpr": round(recall_at_5pct, 4),
    }


def evaluate_vae(config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate VAE anomaly detector and baselines on held-out test battery."""
    logger.info("Starting VAE Anomaly Detection evaluation on held-out battery B0018...")
    set_seed(config.get("seed", 42))

    # Load data and scaler
    features_parquet = Path("data/processed/features.parquet")
    scaler_path = Path("artifacts/models/scaler.joblib")
    vae_model_path = Path("artifacts/models/vae_model.pt")
    vae_meta_path = Path("artifacts/models/vae_meta.json")
    iforest_path = Path("artifacts/models/iforest_baseline.joblib")
    pca_path = Path("artifacts/models/pca_baseline.joblib")
    output_metrics = Path("artifacts/metrics/vae.json")

    df = pd.read_parquet(features_parquet)
    scaler = load_scaler(scaler_path)
    detector = VAEAnomalyDetector.load(vae_model_path, vae_meta_path)
    iforest = joblib.load(iforest_path) if iforest_path.exists() else None
    pca = joblib.load(pca_path) if pca_path.exists() else None

    # Step 1: Extract test battery normal cycles
    test_battery = config.get("data", {}).get("test_battery", "B0018")
    test_df = df[df["battery_id"] == test_battery].copy()

    window_size = config.get("data", {}).get("window_size", 30)

    # Normal test windows
    X_test_normal, meta_normal = create_sliding_windows(
        test_df,
        scaler=scaler,
        feature_cols=FEATURE_COLUMNS,
        window_size=window_size,
        stride=10,
        flatten=True,
    )
    y_normal = np.zeros(len(X_test_normal), dtype=int)

    # Step 2: Generate injected fault test windows for each mode
    fault_windows_dict: Dict[str, np.ndarray] = {}
    all_fault_windows: List[np.ndarray] = []
    fault_labels: List[str] = []

    # Select representative test cycles to inject faults
    test_cycles = test_df["cycle"].unique()
    sample_cycles = test_cycles[:: max(1, len(test_cycles) // 20)]

    for f_type in FAULT_TYPES:
        f_windows_for_type: List[np.ndarray] = []
        for c in sample_cycles:
            cycle_subset = test_df[test_df["cycle"] == c].copy()
            if len(cycle_subset) < window_size:
                continue
            faulty_cycle = inject_fault(cycle_subset, fault_type=f_type, severity=1.0)
            X_win, _ = create_sliding_windows(
                faulty_cycle,
                scaler=scaler,
                feature_cols=FEATURE_COLUMNS,
                window_size=window_size,
                stride=10,
                flatten=True,
            )
            if len(X_win) > 0:
                f_windows_for_type.append(X_win)

        if f_windows_for_type:
            combined_type = np.vstack(f_windows_for_type)
            fault_windows_dict[f_type] = combined_type
            all_fault_windows.append(combined_type)

    X_fault_all = np.vstack(all_fault_windows)
    y_fault_all = np.ones(len(X_fault_all), dtype=int)

    # Full test evaluation matrix
    X_eval = np.vstack([X_test_normal, X_fault_all])
    y_eval = np.concatenate([y_normal, y_fault_all])

    logger.info(
        f"Test evaluation dataset: {len(X_test_normal):,} normal windows, {len(X_fault_all):,} anomaly windows."
    )

    # Step 3: Score with VAE
    vae_scores, _, _ = detector.score_windows(X_eval)
    vae_overall = compute_metrics_for_scores(y_eval, vae_scores)

    # Per-fault breakdown
    per_fault_metrics: Dict[str, Dict[str, float]] = {}
    for f_type, X_f in fault_windows_dict.items():
        X_sub = np.vstack([X_test_normal, X_f])
        y_sub = np.concatenate([y_normal, np.ones(len(X_f), dtype=int)])
        sub_scores, _, _ = detector.score_windows(X_sub)
        per_fault_metrics[f_type] = compute_metrics_for_scores(y_sub, sub_scores)

    # Step 4: Score with Baselines
    baseline_metrics: Dict[str, Any] = {}
    if iforest:
        # Lower score_samples = more anomalous -> negate for ranking
        if_scores = -iforest.score_samples(X_eval)
        baseline_metrics["isolation_forest"] = compute_metrics_for_scores(y_eval, if_scores)

    if pca:
        X_pca_recon = pca.inverse_transform(pca.transform(X_eval))
        pca_errors = np.mean(np.square(X_eval - X_pca_recon), axis=1)
        baseline_metrics["pca"] = compute_metrics_for_scores(y_eval, pca_errors)

    # Step 5: Save metrics to artifacts/metrics/vae.json
    results: Dict[str, Any] = {
        "vae": {
            "auroc": vae_overall["auroc"],
            "pr_auc": vae_overall["pr_auc"],
            "recall_at_5pct_fpr": vae_overall["recall_at_5pct_fpr"],
            "target_met": bool(vae_overall["auroc"] >= 0.85),
            "per_fault_type": per_fault_metrics,
        },
        "baselines": baseline_metrics,
        "eval_dataset": {
            "test_battery": test_battery,
            "normal_windows": int(len(X_test_normal)),
            "anomaly_windows": int(len(X_fault_all)),
            "total_windows": int(len(X_eval)),
        },
    }

    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation report written to {output_metrics}")

    print("\n" + "=" * 65)
    print("VAE ANOMALY DETECTION EVALUATION (HELD-OUT BATTERY B0018)")
    print(f"Overall AUROC:          {vae_overall['auroc']:.4f} (Target: >= 0.85 | Met: {results['vae']['target_met']})")
    print(f"Overall PR-AUC:          {vae_overall['pr_auc']:.4f}")
    print(f"Recall @ 5% FPR:         {vae_overall['recall_at_5pct_fpr']:.4f}")
    print("-" * 65)
    print("Baseline Comparison (AUROC / Recall@5%FPR):")
    print(f"  Beta-VAE:              {vae_overall['auroc']:.4f} / {vae_overall['recall_at_5pct_fpr']:.4f}")
    if "isolation_forest" in baseline_metrics:
        print(
            f"  Isolation Forest:      {baseline_metrics['isolation_forest']['auroc']:.4f} / {baseline_metrics['isolation_forest']['recall_at_5pct_fpr']:.4f}"
        )
    if "pca" in baseline_metrics:
        print(
            f"  PCA (8 comps):         {baseline_metrics['pca']['auroc']:.4f} / {baseline_metrics['pca']['recall_at_5pct_fpr']:.4f}"
        )
    print("-" * 65)
    print("Breakdown Per Fault Mode (AUROC):")
    for f_type, m in per_fault_metrics.items():
        print(f"  {f_type:<25}: AUROC {m['auroc']:.4f} (Recall@5%FPR: {m['recall_at_5pct_fpr']:.4f})")
    print("=" * 65 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardian System Evaluation Runner")
    parser.add_argument("--only", type=str, default="all", choices=["all", "vae", "temp", "rag", "bert", "lora"])
    args = parser.parse_args()

    config = load_config()
    if args.only in ("all", "vae"):
        evaluate_vae(config)


if __name__ == "__main__":
    main()
