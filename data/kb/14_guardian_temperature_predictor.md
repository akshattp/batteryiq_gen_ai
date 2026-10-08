# Guardian Supervised Temperature Predictor Architecture

## Forecasting Formulation
The Guardian temperature predictor (src/guardian/models/temp_predictor.py) forecasts future battery temperature H=10 steps ahead (T_{t+10}) using a trained LightGBM regressor.

## Feature Representation
The model takes 12 instantaneous physical features at time step t:
`temp_c, dT_dt, voltage_v, dV_dt, current_a, abs_current_a, soc_pct, cycle, is_charge, temp_minus_ambient, temp_x_abs_current, soh_pct`
augmented with historical temporal lag features at lags [1, 5, 10]:
`temp_c_lag_1, temp_c_lag_5, temp_c_lag_10, current_a_lag_1, current_a_lag_5, current_a_lag_10`.

## Empirical Evaluation Performance
Evaluated across 309,306 held-out test steps from battery B0018:
- **LightGBM MAE:** 0.0941 °C (R^2 = 0.9968)
- **Persistence Baseline MAE (T_{t+10} = T_t):** 0.1127 °C
The LightGBM predictor outperforms the persistence baseline by 16.50% overall. In the deterministic risk engine, predicted temperature is fed directly into max(T_actual, T_pred) to preemptively detect impending thermal breaches.
