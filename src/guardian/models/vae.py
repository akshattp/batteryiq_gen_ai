"""MLP Beta-VAE Anomaly Detector (Experiment 3).

Trained exclusively on normal telemetry windows.
Computes reconstruction error, maps to [0, 1] empirical CDF anomaly scores,
and breaks down feature-wise contributions for explainability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from guardian.utils.logging import setup_logger

logger = setup_logger("guardian.models.vae")


class BatteryVAE(nn.Module):
    """MLP Beta-Variational Autoencoder for battery telemetry windows."""

    def __init__(
        self,
        input_dim: int = 360,
        hidden_dims: List[int] | None = None,
        latent_dim: int = 8,
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        hidden = hidden_dims or [128, 64]

        # Encoder: input -> h1 -> h2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden[1], latent_dim)

        # Decoder: latent -> h2 -> h1 -> input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input into latent distribution parameters (mu, logvar)."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vector z using reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z back into reconstructed input space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder, sampler, and decoder."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def compute_loss(
        self, x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Beta-VAE loss = MSE reconstruction + beta * KL divergence."""
        # Mean squared error per window (averaged over features)
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        total_loss = recon_loss + (self.beta * kl_div)
        return total_loss, recon_loss, kl_div


class VAEAnomalyDetector:
    """End-to-end wrapper managing VAE training, inference, and ECDF calibration."""

    def __init__(
        self,
        input_dim: int = 360,
        hidden_dims: List[int] | None = None,
        latent_dim: int = 8,
        beta: float = 0.5,
        lr: float = 1e-3,
        window_size: int = 30,
        num_features: int = 12,
    ) -> None:
        self.input_dim = input_dim
        self.window_size = window_size
        self.num_features = num_features
        self.model = BatteryVAE(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, beta=beta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Calibration parameters
        self.val_errors_sorted: np.ndarray = np.array([])
        self.p95_threshold: float = 0.0
        self.p995_threshold: float = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        epochs: int = 25,
        batch_size: int = 128,
        patience: int = 5,
    ) -> Dict[str, List[float]]:
        """Fit VAE on normal training windows with validation early stopping."""
        self.model.train()
        device = torch.device("cpu")
        self.model.to(device)

        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        val_tensor = torch.tensor(X_val, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_recon": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(1, epochs + 1):
            # Training epoch
            self.model.train()
            train_loss_epoch = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self.model(batch_x)
                loss, _, _ = self.model.compute_loss(batch_x, recon, mu, logvar)
                loss.backward()
                self.optimizer.step()
                train_loss_epoch += loss.item() * len(batch_x)
            train_loss_epoch /= len(train_tensor)

            # Validation evaluation
            self.model.eval()
            val_loss_epoch = 0.0
            val_recon_epoch = 0.0
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(device)
                    recon, mu, logvar = self.model(batch_x)
                    loss, recon_l, _ = self.model.compute_loss(batch_x, recon, mu, logvar)
                    val_loss_epoch += loss.item() * len(batch_x)
                    val_recon_epoch += recon_l.item() * len(batch_x)
            val_loss_epoch /= len(val_tensor)
            val_recon_epoch /= len(val_tensor)

            history["train_loss"].append(round(train_loss_epoch, 5))
            history["val_loss"].append(round(val_loss_epoch, 5))
            history["val_recon"].append(round(val_recon_epoch, 5))

            if epoch % 5 == 0 or epoch == epochs:
                logger.info(
                    f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} (Recon: {val_recon_epoch:.4f})"
                )

            # Early stopping check
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        if best_weights:
            self.model.load_state_dict(best_weights)

        # Calibrate empirical CDF on normal validation reconstruction errors
        self.calibrate_ecdf(X_val)
        return history

    def compute_reconstruction_errors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute window MSE error and per-feature error breakdown.

        Args:
            X: Array of flattened windows (N, W * F).

        Returns:
            Tuple of:
              - window_errors: (N,) MSE error per window
              - feature_errors: (N, F) Mean error per feature over time steps W
        """
        self.model.eval()
        x_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            x_recon, _, _ = self.model(x_tensor)

        diff = (x_tensor - x_recon).numpy()  # (N, W * F)
        squared_diff = np.square(diff)

        # Window MSE
        window_errors = np.mean(squared_diff, axis=1)

        # Per-feature breakdown: reshape to (N, W, F) and average over W
        reshaped = squared_diff.reshape(-1, self.window_size, self.num_features)
        feature_errors = np.mean(reshaped, axis=1)  # (N, F)

        return window_errors, feature_errors

    def calibrate_ecdf(self, X_val_normal: np.ndarray) -> None:
        """Calibrate empirical CDF and percentile thresholds on normal validation windows."""
        val_errors, _ = self.compute_reconstruction_errors(X_val_normal)
        self.val_errors_sorted = np.sort(val_errors)
        self.p95_threshold = float(np.percentile(self.val_errors_sorted, 95.0))
        self.p995_threshold = float(np.percentile(self.val_errors_sorted, 99.5))
        logger.info(
            f"ECDF calibrated on {len(self.val_errors_sorted)} val windows: p95={self.p95_threshold:.5f}, p99.5={self.p995_threshold:.5f}"
        )

    def score_windows(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute raw reconstruction errors, ECDF anomaly scores, and per-feature error.

        Args:
            X: Array of flattened windows (N, W * F).

        Returns:
            Tuple of:
              - anomaly_scores: (N,) float scores in [0.0, 1.0]
              - window_errors: (N,) raw MSE errors
              - feature_errors: (N, F) per-feature error contributions
        """
        win_errors, feat_errors = self.compute_reconstruction_errors(X)
        if len(self.val_errors_sorted) == 0:
            # Fallback if uncalibrated: min-max normalization
            scores = np.clip(win_errors / (win_errors.max() + 1e-6), 0.0, 1.0)
            return scores, win_errors, feat_errors

        # Empirical CDF mapping: score = fraction of val errors <= current error
        ranks = np.searchsorted(self.val_errors_sorted, win_errors, side="right")
        scores = np.clip(ranks / len(self.val_errors_sorted), 0.0, 1.0)
        return scores, win_errors, feat_errors

    def save(self, model_path: str | Path, meta_path: str | Path) -> None:
        """Persist model weights and ECDF calibration metadata to disk."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

        meta = {
            "input_dim": self.input_dim,
            "window_size": self.window_size,
            "num_features": self.num_features,
            "latent_dim": self.model.latent_dim,
            "beta": self.model.beta,
            "p95_threshold": self.p95_threshold,
            "p995_threshold": self.p995_threshold,
            "val_errors_sample": self.val_errors_sorted[:: max(1, len(self.val_errors_sorted) // 1000)].tolist(),
        }
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved VAE model to {model_path} and metadata to {meta_path}.")

    @classmethod
    def load(cls, model_path: str | Path, meta_path: str | Path) -> VAEAnomalyDetector:
        """Load trained VAE and calibration state from disk."""
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        detector = cls(
            input_dim=meta["input_dim"],
            latent_dim=meta["latent_dim"],
            beta=meta["beta"],
            window_size=meta["window_size"],
            num_features=meta["num_features"],
        )
        detector.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        detector.model.eval()
        detector.p95_threshold = meta["p95_threshold"]
        detector.p995_threshold = meta["p995_threshold"]
        detector.val_errors_sorted = np.array(meta.get("val_errors_sample", []))
        return detector
