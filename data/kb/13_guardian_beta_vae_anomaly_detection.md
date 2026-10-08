# Guardian Beta-VAE Anomaly Detection Architecture

## Unsupervised Normality Modeling
The Guardian Beta-VAE (src/guardian/models/vae.py) is trained strictly on normal telemetry windows (label == 'normal' from batteries B0005, B0006, and B0007). It never observes synthetic fault data during training.

## Network Hyperparameters
- **Input Dimension:** 360 features (Sliding window of W=30 steps across F=12 physical and interaction features).
- **Hidden Layers:** Encoder [360 -> 128 -> 64], Latent Bottleneck [dim = 8], Decoder [8 -> 64 -> 128 -> 360].
- **Beta Regularization:** beta = 0.5 balancing reconstruction fidelity with latent space disentanglement.
- **Loss Function:** MSE Reconstruction Loss + beta * KL Divergence.

## Empirical CDF Calibration and Percentile Thresholds
Raw reconstruction error is mapped to an anomaly score in [0.0, 1.0] using the empirical cumulative distribution function (ECDF) of normal validation window errors:
score = ecdf(raw_error)
Calibrated thresholds from validation distribution:
- **Warning Threshold (p95):** 1.5470 (95th percentile of normal validation errors).
- **Critical Threshold (p99.5):** 8.2344 (99.5th percentile of normal validation errors).
Scores exceeding p95 indicate statistically significant deviation from learned normal battery dynamics.
