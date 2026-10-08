"""scripts/train_gan.py - Train Conditional GAN on synthetic fault windows (Milestone M7 / Exp 2).

Generates synthetic abnormal telemetry windows conditioned on fault category
(thermal_ramp, current_spike, voltage_sag), trains cGAN, evaluates fidelity,
utility against the trained VAE, mode-collapse diversity, and outputs overlay figures.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from guardian.data.faults import inject_current_spike, inject_thermal_ramp, inject_voltage_sag
from guardian.data.features import FEATURE_COLUMNS, create_sliding_windows
from guardian.models.gan import FAULT_IDX_TO_NAME, FAULT_NAME_TO_IDX, ConditionalTelemetryGAN
from guardian.models.vae import VAEAnomalyDetector
from guardian.utils.config import get_config
from guardian.utils.logging import get_logger
from guardian.utils.seed import set_seed

logger = get_logger(__name__)


def prepare_gan_training_data(
    features_path: str = "data/processed/features.parquet",
    scaler_path: str = "artifacts/models/scaler.joblib",
    max_cycles: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate labeled synthetic abnormal windows from normal train battery data."""
    logger.info("Loading feature dataset for GAN training...")
    df = pd.read_parquet(features_path)
    scaler = joblib.load(scaler_path)

    # Filter to normal training batteries
    train_batteries = ["B0005", "B0006", "B0007"]
    train_df = df[(df["battery_id"].isin(train_batteries)) & (df["label"] == "normal")].copy()

    # Select representative subset of cycles to keep training light and fast
    available_cycles = sorted(train_df["cycle"].unique())[:max_cycles]
    sub_df = train_df[train_df["cycle"].isin(available_cycles)].copy()

    all_windows: List[np.ndarray] = []
    all_labels: List[int] = []

    # Inject each fault type
    fault_injectors = [
        ("thermal_ramp", inject_thermal_ramp),
        ("current_spike", inject_current_spike),
        ("voltage_sag", inject_voltage_sag),
    ]

    for fault_name, injector in fault_injectors:
        class_idx = FAULT_NAME_TO_IDX[fault_name]
        logger.info(f"Injecting {fault_name} into {len(available_cycles)} training cycles...")
        injected_segments: List[pd.DataFrame] = []

        for _, cycle_df in sub_df.groupby(["battery_id", "cycle"]):
            if len(cycle_df) >= 30:
                inj = injector(cycle_df)
                injected_segments.append(inj)

        if injected_segments:
            combined_fault = pd.concat(injected_segments, ignore_index=True)
            # Generate sliding scaled windows
            X_scaled, _ = create_sliding_windows(
                combined_fault,
                scaler=scaler,
                window_size=30,
                stride=10,
                feature_cols=FEATURE_COLUMNS,
                flatten=True,
            )

            all_windows.append(X_scaled)
            all_labels.append(np.full(len(X_scaled), class_idx, dtype=np.int64))

    X = np.vstack(all_windows).astype(np.float32)
    y = np.concatenate(all_labels).astype(np.int64)
    logger.info(f"Prepared GAN dataset: {X.shape[0]} windows, input dimension: {X.shape[1]}")
    return X, y


def train_cgan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.0002,
    latent_dim: int = 32,
) -> ConditionalTelemetryGAN:
    """Train Conditional Telemetry GAN on CPU."""
    gan = ConditionalTelemetryGAN(latent_dim=latent_dim, num_classes=3, output_dim=X_train.shape[1], device="cpu")

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer_G = torch.optim.Adam(gan.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(gan.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    logger.info(f"Starting cGAN training for {epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        gan.generator.train()
        gan.discriminator.train()
        d_losses = []
        g_losses = []

        for real_x, labels in loader:
            b_size = real_x.size(0)

            # Labels for discriminator with label smoothing
            real_targets = torch.full((b_size, 1), 0.9)
            fake_targets = torch.zeros((b_size, 1))

            # Train Discriminator
            optimizer_D.zero_grad()
            d_real = gan.discriminator(real_x, labels)
            loss_d_real = criterion(d_real, real_targets)

            z = torch.randn(b_size, latent_dim)
            fake_x = gan.generator(z, labels)
            d_fake = gan.discriminator(fake_x.detach(), labels)
            loss_d_fake = criterion(d_fake, fake_targets)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_D.step()
            d_losses.append(loss_d.item())

            # Train Generator
            optimizer_G.zero_grad()
            gen_out = gan.discriminator(fake_x, labels)
            loss_g = criterion(gen_out, torch.ones((b_size, 1)))
            loss_g.backward()
            optimizer_G.step()
            g_losses.append(loss_g.item())

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch [{epoch}/{epochs}] - Loss D: {np.mean(d_losses):.4f}, Loss G: {np.mean(g_losses):.4f}")

    logger.info(f"cGAN training complete in {time.time() - t0:.1f}s")
    return gan


def evaluate_gan(
    gan: ConditionalTelemetryGAN,
    X_real: np.ndarray,
    y_real: np.ndarray,
    scaler_path: str = "artifacts/models/scaler.joblib",
    vae_path: str = "artifacts/models/vae_model.pt",
    vae_meta_path: str = "artifacts/models/vae_meta.json",
    out_fig: str = "artifacts/figures/gan_synthetic_overlay.png",
) -> Dict[str, Any]:
    """Perform fidelity, utility (VAE detection), diversity, and visualization evaluation."""
    scaler = joblib.load(scaler_path)

    # 1. Generate synthetic samples for each fault class
    samples_per_class = 200
    gen_by_class: Dict[str, np.ndarray] = {}
    diversity_scores: Dict[str, float] = {}
    mean_diffs: Dict[str, float] = {}
    std_diffs: Dict[str, float] = {}

    for fault_name, class_idx in FAULT_NAME_TO_IDX.items():
        gen = gan.generate(samples_per_class, class_idx, seed=42 + class_idx)
        gen_by_class[fault_name] = gen

        # Diversity score: mean pairwise distance among first 50 generated samples
        sub = gen[:50]
        dists = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        # Average non-diagonal distance
        mask = ~np.eye(50, dtype=bool)
        div = float(np.mean(dists[mask]))
        diversity_scores[fault_name] = round(div, 4)

        # Fidelity comparison against real windows of this class
        real_cls = X_real[y_real == class_idx]
        if len(real_cls) > 0:
            diff_mean = float(np.mean(np.abs(np.mean(real_cls, axis=0) - np.mean(gen, axis=0))))
            diff_std = float(np.mean(np.abs(np.std(real_cls, axis=0) - np.std(gen, axis=0))))
        else:
            diff_mean, diff_std = 0.0, 0.0
        mean_diffs[fault_name] = round(diff_mean, 4)
        std_diffs[fault_name] = round(diff_std, 4)

    # 2. Two-sample classifier AUC (distinguish real vs GAN across all classes)
    X_gen_all = np.vstack([gen_by_class[f] for f in FAULT_NAME_TO_IDX.keys()])
    y_two_sample = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_gen_all))])
    X_two_sample = np.vstack([X_real, X_gen_all])

    clf = LogisticRegression(max_iter=300, random_state=42)
    clf.fit(X_two_sample, y_two_sample)
    two_sample_auc = float(roc_auc_score(y_two_sample, clf.predict_proba(X_two_sample)[:, 1]))

    # 3. Utility Test: pass GAN abnormal windows into VAE
    utility_metrics: Dict[str, Any] = {}
    if os.path.exists(vae_path) and os.path.exists(vae_meta_path):
        detector = VAEAnomalyDetector.load(vae_path, vae_meta_path)
        threshold_warn = detector.p95_threshold

        detection_rates: Dict[str, float] = {}
        for fault_name, gen_data in gen_by_class.items():
            scores, raw_mses, _ = detector.score_windows(gen_data)
            # Detection rate = fraction with raw_mse >= warning threshold (or score >= 0.95)
            detected = np.mean(raw_mses >= threshold_warn)
            detection_rates[fault_name] = round(float(detected), 4)

        utility_metrics = {
            "vae_warning_threshold": round(threshold_warn, 4),
            "detection_rates": detection_rates,
            "overall_detection_rate": round(float(np.mean(list(detection_rates.values()))), 4),
        }

    # 4. Generate Visual Overlay Figure
    fig_path = Path(out_fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    temp_idx = 0  # temp_c is index 0 in feature set

    for ax, (fault_name, class_idx) in zip(axes, FAULT_NAME_TO_IDX.items()):
        real_cls = X_real[y_real == class_idx]
        gen_cls = gen_by_class[fault_name]

        # Reshape to (N, 30, 12) to plot temperature time steps
        W, F = 30, 12
        real_t = real_cls[:5].reshape(-1, W, F)[:, :, temp_idx]
        gen_t = gen_cls[:5].reshape(-1, W, F)[:, :, temp_idx]

        for i in range(min(5, len(real_t))):
            ax.plot(real_t[i], "k--", alpha=0.6, label="Real Injected" if i == 0 else "")
            ax.plot(gen_t[i], "r-", alpha=0.7, label="GAN Synthetic" if i == 0 else "")

        ax.set_title(f"Fault: {fault_name}")
        ax.set_xlabel("Window Step (0..29)")
        ax.set_ylabel("Standardized Temp Signal" if ax == axes[0] else "")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved GAN overlay figure to {fig_path}")

    return {
        "dataset_size_real": len(X_real),
        "synthetic_samples_per_class": samples_per_class,
        "two_sample_classifier_auc": round(two_sample_auc, 4),
        "fidelity_mean_abs_diff": mean_diffs,
        "fidelity_std_abs_diff": std_diffs,
        "diversity_scores": diversity_scores,
        "utility": utility_metrics,
        "figure_path": str(fig_path),
        "note": "All GAN data is synthetically generated for threshold stress-testing and anomaly evaluation.",
    }


def main() -> None:
    set_seed(42)
    cfg = get_config()
    gan_cfg = cfg.get("gan", {})

    epochs = gan_cfg.get("epochs", 15)
    batch_size = gan_cfg.get("batch_size", 64)
    lr = gan_cfg.get("lr", 0.0002)
    latent_dim = gan_cfg.get("latent_dim", 32)

    X_train, y_train = prepare_gan_training_data()
    gan = train_cgan(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr, latent_dim=latent_dim)

    # Save generator weights
    model_path = Path("artifacts/models/gan_generator.pt")
    gan.save(model_path)
    logger.info(f"Saved GAN generator to {model_path}")

    # Evaluate
    metrics = evaluate_gan(gan, X_train, y_train)

    out_metrics_path = Path("artifacts/metrics/gan.json")
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved GAN metrics to {out_metrics_path}")
    print("\n[GAN TRAINING COMPLETE]")
    print(f"Metrics saved to {out_metrics_path}")
    print(f"Overlay plot saved to {metrics['figure_path']}")


if __name__ == "__main__":
    main()
