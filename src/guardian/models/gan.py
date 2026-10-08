"""Conditional Generative Adversarial Network (cGAN) for synthetic telemetry windows (Exp 2).

Generates synthetic abnormal windows conditioned on fault category for
stress-testing anomaly detectors and risk assessment boundaries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

FAULT_NAME_TO_IDX: Dict[str, int] = {
    "thermal_ramp": 0,
    "current_spike": 1,
    "voltage_sag": 2,
}
FAULT_IDX_TO_NAME: Dict[int, str] = {v: k for k, v in FAULT_NAME_TO_IDX.items()}


class TelemetryGenerator(nn.Module):
    """Generator network for synthetic telemetry windows conditioned on fault category."""

    def __init__(
        self,
        latent_dim: int = 32,
        num_classes: int = 3,
        class_emb_dim: int = 16,
        output_dim: int = 360,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim

        self.class_emb = nn.Embedding(num_classes, class_emb_dim)

        in_dim = latent_dim + class_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_dim),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Latent noise vector [batch_size, latent_dim].
            labels: Integer fault class indices [batch_size].

        Returns:
            Generated flattened windows [batch_size, output_dim].
        """
        c = self.class_emb(labels)
        x = torch.cat([z, c], dim=-1)
        return self.net(x)


class TelemetryDiscriminator(nn.Module):
    """Discriminator network classifying real vs generated windows conditioned on fault category."""

    def __init__(
        self,
        input_dim: int = 360,
        num_classes: int = 3,
        class_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.class_emb = nn.Embedding(num_classes, class_emb_dim)

        in_dim = input_dim + class_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass returning scalar validity logits [batch_size, 1]."""
        c = self.class_emb(labels)
        inp = torch.cat([x, c], dim=-1)
        return self.net(inp)


class ConditionalTelemetryGAN:
    """High-level interface for training, saving, loading, and generating synthetic telemetry."""

    def __init__(
        self,
        latent_dim: int = 32,
        num_classes: int = 3,
        output_dim: int = 360,
        device: Optional[str] = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.generator = TelemetryGenerator(latent_dim, num_classes, output_dim=output_dim).to(self.device)
        self.discriminator = TelemetryDiscriminator(output_dim, num_classes).to(self.device)

    def generate(
        self,
        n_samples: int,
        fault_type: Union[str, int],
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate synthetic scaled telemetry windows.

        Args:
            n_samples: Number of windows to generate.
            fault_type: Class name ('thermal_ramp', 'current_spike', 'voltage_sag') or int index.
            seed: Optional random seed.

        Returns:
            Numpy array of shape [n_samples, output_dim].
        """
        if isinstance(fault_type, str):
            if fault_type not in FAULT_NAME_TO_IDX:
                raise ValueError(f"Unknown fault type '{fault_type}'. Expected one of {list(FAULT_NAME_TO_IDX.keys())}")
            class_idx = FAULT_NAME_TO_IDX[fault_type]
        else:
            class_idx = int(fault_type)

        self.generator.eval()
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            labels = torch.full((n_samples,), class_idx, dtype=torch.long, device=self.device)
            generated = self.generator(z, labels).cpu().numpy()

        return generated

    def save(self, model_path: str | Path) -> None:
        """Save generator weights to disk."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator_state_dict": self.generator.state_dict(),
                "latent_dim": self.latent_dim,
                "num_classes": self.num_classes,
                "output_dim": self.output_dim,
            },
            path,
        )

    def load(self, model_path: str | Path) -> None:
        """Load generator weights from disk."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.generator.eval()
