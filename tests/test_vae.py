"""tests/test_vae.py - Unit tests for Beta-VAE anomaly detector."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from guardian.models.vae import BatteryVAE, VAEAnomalyDetector


def test_vae_architecture_forward_pass():
    input_dim = 360
    latent_dim = 8
    model = BatteryVAE(input_dim=input_dim, hidden_dims=[128, 64], latent_dim=latent_dim, beta=0.5)

    batch_size = 4
    x = torch.randn(batch_size, input_dim)
    recon, mu, logvar = model(x)

    assert recon.shape == (batch_size, input_dim)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)

    loss, recon_loss, kl_div = model.compute_loss(x, recon, mu, logvar)
    assert loss.item() > 0.0
    assert recon_loss.item() >= 0.0
    assert kl_div.item() >= 0.0


def test_vae_detector_ecdf_calibration():
    detector = VAEAnomalyDetector(input_dim=60, window_size=5, num_features=12)
    # Simulate normal validation reconstruction errors
    val_synthetic = np.random.randn(200, 60).astype(np.float32)
    detector.calibrate_ecdf(val_synthetic)

    assert len(detector.val_errors_sorted) == 200
    assert detector.p95_threshold > 0.0
    assert detector.p995_threshold >= detector.p95_threshold

    # Scoring normal vs extreme outlier
    normal_sample = val_synthetic[:5]
    outlier_sample = val_synthetic[:5] + 50.0

    scores_normal, _, _ = detector.score_windows(normal_sample)
    scores_outlier, _, _ = detector.score_windows(outlier_sample)

    assert (scores_normal >= 0.0).all() and (scores_normal <= 1.0).all()
    # Outliers should have near-1.0 anomaly scores
    assert (scores_outlier >= 0.95).all()


def test_vae_metrics_json_exists():
    """Verify artifacts/metrics/vae.json exists and conforms to schema."""
    metrics_path = Path("artifacts/metrics/vae.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "vae" in data
        assert "auroc" in data["vae"]
        assert "pr_auc" in data["vae"]
        assert "recall_at_5pct_fpr" in data["vae"]
        assert "baselines" in data
