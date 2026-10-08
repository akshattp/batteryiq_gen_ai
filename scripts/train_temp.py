"""scripts/train_temp.py - Train LightGBM temperature predictor and baselines (Milestone 4).

Predicts future cell temperature T(t + H) (H=10 steps).
Compares against Persistence (T(t+H)=T(t)) and Linear Regression baselines on held-out battery B0018.
Saves model, metrics JSON, and predicted-vs-actual figures.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from guardian.utils.config import load_config
from guardian.utils.seed import set_seed
from guardian.data.features import create_splits
from guardian.models.temp_predictor import (
    build_lagged_features,
    TemperaturePredictor,
    evaluate_forecast,
)
from guardian.utils.logging import setup_logger

logger = setup_logger("scripts.train_temp")


def save_diagnostic_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importances: dict[str, float],
    figures_dir: Path,
) -> None:
    """Generate and save predicted-vs-actual and feature importance charts."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Predicted vs Actual scatter plot
    plt.figure(figsize=(8, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Downsample points for clear visualization
    subsample_idx = np.random.choice(len(y_true), size=min(2000, len(y_true)), replace=False)
    plt.scatter(y_true[subsample_idx], y_pred[subsample_idx], alpha=0.35, color="#1f77b4", s=15, label="Predictions")

    # Ideal 1:1 diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Forecast (1:1)")

    plt.xlabel("Actual Temperature (°C)", fontsize=11)
    plt.ylabel("Predicted Temperature at t+10 (°C)", fontsize=11)
    plt.title("Temperature Forecast: Predicted vs Actual (Held-out B0018)", fontsize=13, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()

    pred_actual_path = figures_dir / "temp_pred_vs_actual.png"
    plt.savefig(pred_actual_path)
    plt.close()
    logger.info(f"Saved figure to {pred_actual_path}")

    # 2. Feature Importance horizontal bar chart
    top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
    names, values = zip(*reversed(top_features))

    plt.figure(figsize=(9, 5), dpi=150)
    plt.barh(names, values, color="#2ca02c")
    plt.xlabel("Normalized Importance", fontsize=11)
    plt.title("LightGBM Temperature Predictor - Top 10 Features", fontsize=13, fontweight="bold")
    plt.tight_layout()

    imp_path = figures_dir / "temp_feature_importance.png"
    plt.savefig(imp_path)
    plt.close()
    logger.info(f"Saved figure to {imp_path}")


def main() -> None:
    logger.info("Initializing Temperature Predictor training pipeline...")
    config = load_config()
    set_seed(config.get("seed", 42))

    features_parquet = Path("data/processed/features.parquet")
    models_dir = Path("artifacts/models")
    metrics_dir = Path("artifacts/metrics")
    figures_dir = Path("artifacts/figures")

    if not features_parquet.exists():
        raise FileNotFoundError(f"Features dataset missing at {features_parquet}. Run 'make data' first.")

    # Load data and splits
    logger.info("Loading feature dataset...")
    df = pd.read_parquet(features_parquet)
    train_df, val_df, test_df, _ = create_splits(df, config=config, train_cycle_ratio=0.75)

    temp_cfg = config.get("temp_predictor", {})
    horizon_steps = temp_cfg.get("horizon_steps", 10)
    lags = temp_cfg.get("lags", [1, 5, 10])

    # Build lagged feature matrices
    logger.info(f"Constructing lagged feature matrices (horizon={horizon_steps}, lags={lags})...")
    X_train, y_train, meta_train = build_lagged_features(train_df, horizon_steps=horizon_steps, lags=lags)
    X_test, y_test, meta_test = build_lagged_features(test_df, horizon_steps=horizon_steps, lags=lags)

    logger.info(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    # Step 1: Train LightGBM predictor
    predictor = TemperaturePredictor(
        horizon_steps=horizon_steps,
        n_estimators=temp_cfg.get("n_estimators", 150),
        learning_rate=temp_cfg.get("learning_rate", 0.05),
        max_depth=temp_cfg.get("max_depth", 6),
        random_state=config.get("seed", 42),
    )
    predictor.fit(X_train, y_train)

    model_path = models_dir / "temp_model.joblib"
    predictor.save(model_path)

    # Step 2: Fit Linear Regression baseline
    logger.info("Fitting Linear Regression baseline...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_path = models_dir / "temp_linear_baseline.joblib"
    joblib.dump(linear_model, linear_path)

    # Step 3: Evaluate on held-out test battery B0018
    logger.info("Evaluating models on held-out battery B0018...")
    y_test_arr = y_test.values
    y_pred_lgbm = predictor.predict(X_test)
    y_pred_linear = linear_model.predict(X_test)
    persistence_pred = meta_test["current_temp"].values

    eval_results = evaluate_forecast(y_test_arr, y_pred_lgbm, persistence_pred, meta_test)

    # Add linear baseline metrics
    linear_mae = float(mean_absolute_error(y_test_arr, y_pred_linear))
    linear_rmse = float(np.sqrt(mean_squared_error(y_test_arr, y_pred_linear)))
    linear_r2 = float(r2_score(y_test_arr, y_pred_linear))

    eval_results["linear_regression"] = {
        "mae": round(linear_mae, 4),
        "rmse": round(linear_rmse, 4),
        "r2": round(linear_r2, 4),
    }

    eval_results["feature_importances"] = predictor.get_feature_importances()
    eval_results["metadata"] = {
        "test_battery": config.get("data", {}).get("test_battery", "B0018"),
        "test_samples": len(X_test),
        "horizon_steps": horizon_steps,
        "lags": lags,
    }

    # Step 4: Save metrics JSON
    metrics_path = metrics_dir / "temp.json"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    # Step 5: Save diagnostic plots
    save_diagnostic_figures(y_test_arr, y_pred_lgbm, eval_results["feature_importances"], figures_dir)

    print("\n" + "=" * 65)
    print("TEMPERATURE PREDICTOR EVALUATION (HELD-OUT BATTERY B0018)")
    print(f"Horizon H:                {horizon_steps} steps ahead")
    print(f"LightGBM MAE:             {eval_results['lgbm']['mae']:.4f} °C")
    print(f"LightGBM RMSE:            {eval_results['lgbm']['rmse']:.4f} °C")
    print(f"LightGBM R²:              {eval_results['lgbm']['r2']:.4f}")
    print("-" * 65)
    print("Baseline Comparison:")
    print(f"  LightGBM MAE:           {eval_results['lgbm']['mae']:.4f} °C")
    print(f"  Persistence MAE:        {eval_results['persistence']['mae']:.4f} °C")
    print(f"  Linear Regr MAE:        {eval_results['linear_regression']['mae']:.4f} °C")
    print(f"  Beats Persistence:      {eval_results['lgbm']['beats_persistence']}")
    print(f"  Improvement:            {eval_results['lgbm']['improvement_pct']:.2f}%")
    print("-" * 65)
    print("Subgroup Error by Mode:")
    for mode, m in eval_results["by_mode"].items():
        print(f"  {mode:<12}: MAE {m['mae']:.4f} °C | RMSE {m['rmse']:.4f} °C | R² {m['r2']:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
