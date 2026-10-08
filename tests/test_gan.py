"""Unit tests for Conditional GAN architecture (Milestone M7 / Exp 2)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from guardian.models.gan import (
    FAULT_NAME_TO_IDX,
    ConditionalTelemetryGAN,
    TelemetryDiscriminator,
    TelemetryGenerator,
)


def test_telemetry_generator_output_shape() -> None:
    batch_size = 8
    latent_dim = 32
    output_dim = 360
    num_classes = 3

    gen = TelemetryGenerator(latent_dim=latent_dim, num_classes=num_classes, output_dim=output_dim)
    z = torch.randn(batch_size, latent_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    out = gen(z, labels)
    assert out.shape == (batch_size, output_dim)
    assert not torch.isnan(out).any()


def test_telemetry_discriminator_output_shape() -> None:
    batch_size = 8
    input_dim = 360
    num_classes = 3

    disc = TelemetryDiscriminator(input_dim=input_dim, num_classes=num_classes)
    x = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    out = disc(x, labels)
    assert out.shape == (batch_size, 1)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("fault_type", ["thermal_ramp", "current_spike", "voltage_sag"])
def test_conditional_gan_generate(fault_type: str) -> None:
    gan = ConditionalTelemetryGAN(latent_dim=32, num_classes=3, output_dim=360, device="cpu")
    n_samples = 6

    samples = gan.generate(n_samples=n_samples, fault_type=fault_type, seed=42)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (n_samples, 360)
    assert np.all(np.isfinite(samples))


def test_conditional_gan_save_load() -> None:
    gan1 = ConditionalTelemetryGAN(latent_dim=32, num_classes=3, output_dim=360, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "gan_gen.pt"
        gan1.save(save_path)

        gan2 = ConditionalTelemetryGAN(latent_dim=32, num_classes=3, output_dim=360, device="cpu")
        gan2.load(save_path)

        samples1 = gan1.generate(4, "thermal_ramp", seed=123)
        samples2 = gan2.generate(4, "thermal_ramp", seed=123)

        np.testing.assert_allclose(samples1, samples2, rtol=1e-5, atol=1e-5)
