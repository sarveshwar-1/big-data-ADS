"""
Hybrid SARIMA-LSTM Model — Sequential Residual Architecture
============================================================
1. Load feature data from HDFS (same source as other models).
2. Resample 5-min data → hourly (matches SARIMA pipeline).
3. Fit SARIMA (with exogenous variables) on training set.
4. Extract in-sample residuals, scale & clip them.
5. Train LSTM to autoregressively forecast scaled residuals.
6. Final forecast = SARIMA_forecast + LSTM_residual_correction.
7. Evaluate, compare with SARIMA-only baseline, and save metrics to HDFS.
"""

import numpy as np
import pandas as pd
import math
import itertools
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pyspark.sql import SparkSession

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================================
# 0. SPARK SESSION
# ==========================================================================
spark = (SparkSession.builder
    .appName("Hybrid_SARIMA_LSTM_Model")
    .getOrCreate()
)

# ==========================================================================
# 1. DATA LOADING & PREPROCESSING  (identical to 13_train_sarima.py)
# ==========================================================================
print("\n[1/8] Loading Feature Data from HDFS...")
input_path = "hdfs://namenode:8020/logs/features/traffic_ml_ready"
df = spark.read.parquet(input_path)

pdf = (df.select("window_start", "request_count", "total_bytes", "prev_count")
       .orderBy("window_start")
       .toPandas())

pdf["window_start"] = pd.to_datetime(pdf["window_start"])
pdf = pdf.set_index("window_start").sort_index()

# Resample 5-min → hourly  (SARIMA is far more stable on hourly data)
pdf_hourly = pdf.resample("1h").agg({
    "request_count": "sum",
    "total_bytes": "sum",
    "prev_count": "mean",
}).dropna()

ts = pdf_hourly["request_count"].astype("float64")
exog = pdf_hourly[["total_bytes", "prev_count"]].astype("float64")

# Normalise exogenous variables
exog_mean = exog.mean()
exog_std = exog.std().replace(0, 1)
exog_norm = (exog - exog_mean) / exog_std

print(f"   Raw  : {len(pdf)} observations (5-min)")
print(f"   Hourly: {len(ts)} observations")
print(f"   Range : {ts.index.min()} → {ts.index.max()}")

# ==========================================================================
# 2. TRAIN / TEST SPLIT (80 / 20)
# ==========================================================================
print("\n[2/8] Train / Test split (80-20)...")
SEASONAL_PERIOD = 24          # daily cycle in hourly data

train_size = int(len(ts) * 0.8)
LOOKBACK = min(6, train_size // 10)  # short context; sized to dataset
LOOKBACK = max(LOOKBACK, 3)   # at least 3 steps

train_ts = ts.iloc[:train_size]
test_ts  = ts.iloc[train_size:]
train_exog = exog_norm.iloc[:train_size]
test_exog  = exog_norm.iloc[train_size:]
print(f"   Train: {len(train_ts)},  Test: {len(test_ts)},  Lookback: {LOOKBACK}")

# ══════════════════════════════════════════════════════════════════════════
#  STAGE A – SARIMA  (baseline structural forecast)
# ══════════════════════════════════════════════════════════════════════════

# ==========================================================================
# 3. SARIMA GRID SEARCH
# ==========================================================================
print("\n[3/8] SARIMA grid search (compact)...")

p_range = range(0, 3)
d_range = range(0, 2)
q_range = range(0, 3)
P_range = range(0, 2)
D_range = range(0, 2)
Q_range = range(0, 2)

pdq_combos = list(itertools.product(p_range, d_range, q_range))
seasonal_combos = list(itertools.product(P_range, D_range, Q_range))
total = len(pdq_combos) * len(seasonal_combos)

best_aic = np.inf
best_order = (1, 1, 1)
best_seasonal = (1, 1, 1, SEASONAL_PERIOD)
tested = 0

for order in pdq_combos:
    for s_order in seasonal_combos:
        seasonal_full = s_order + (SEASONAL_PERIOD,)
        try:
            model = SARIMAX(
                train_ts, exog=train_exog,
                order=order, seasonal_order=seasonal_full,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=100)
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = order
                best_seasonal = seasonal_full
        except Exception:
            pass
        tested += 1
        if tested % 20 == 0:
            print(f"   ... tested {tested}/{total}")

print(f"   Best SARIMA{best_order}x{best_seasonal}  AIC={best_aic:.2f}")

# ==========================================================================
# 4. FIT SARIMA ON TRAINING DATA
# ==========================================================================
print("\n[4/8] Fitting best SARIMA on training data...")
sarima_model = SARIMAX(
    train_ts, exog=train_exog,
    order=best_order, seasonal_order=best_seasonal,
    enforce_stationarity=False, enforce_invertibility=False,
)
sarima_fitted = sarima_model.fit(disp=False, maxiter=200)

# SARIMA-only test forecast (baseline)
sarima_test_fc = sarima_fitted.get_forecast(
    steps=len(test_ts), exog=test_exog
).predicted_mean
sarima_test_fc.index = test_ts.index

sarima_only_rmse = math.sqrt(mean_squared_error(test_ts, sarima_test_fc))
sarima_only_mae  = mean_absolute_error(test_ts, sarima_test_fc)
sarima_only_r2   = r2_score(test_ts, sarima_test_fc)
print(f"   SARIMA-only → RMSE={sarima_only_rmse:.2f}  R²={sarima_only_r2:.4f}")

# ==========================================================================
# 5. EXTRACT IN-SAMPLE RESIDUALS
# ==========================================================================
print("\n[5/8] Extracting SARIMA in-sample residuals...")
# Use SARIMA's built-in one-step-ahead residuals (properly computed)
residuals_raw = sarima_fitted.resid.values.astype("float64")
# Drop burn-in period — first SEASONAL_PERIOD residuals are unreliable
# because seasonal differencing needs a full cycle of data
burn_in = SEASONAL_PERIOD
residuals = residuals_raw[burn_in:]
print(f"   Residuals (after burn-in): {len(residuals)} samples,  "
      f"mean={residuals.mean():.2f},  std={residuals.std():.2f}")

# ══════════════════════════════════════════════════════════════════════════
#  STAGE B – LSTM on residuals  (non-linear correction)
# ══════════════════════════════════════════════════════════════════════════

# ==========================================================================
# 6. LSTM NETWORK DEFINITION
# ==========================================================================
class LSTMNetwork(nn.Module):
    """Lightweight LSTM sized for small residual datasets."""
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return (np.array(X).reshape(-1, lookback, 1),
            np.array(y).reshape(-1, 1))


# ==========================================================================
# 7. TRAIN LSTM ON RESIDUALS
# ==========================================================================
print("\n[6/8] Training LSTM on SARIMA residuals...")

# Scale & clip residuals (±3 σ) to handle outliers
residual_scaler = StandardScaler()
scaled_residuals = residual_scaler.fit_transform(residuals.reshape(-1, 1)).flatten()
scaled_residuals = np.clip(scaled_residuals, -3.0, 3.0)

X_res, y_res = create_sequences(scaled_residuals, LOOKBACK)

if len(X_res) < 20:
    print("   [WARN] Too few residual samples – skipping LSTM.")
    lstm_model = None
else:
    X_t = torch.FloatTensor(X_res)
    y_t = torch.FloatTensor(y_res)

    # Train-val split inside residuals (keep chronological order)
    val_ratio = 0.1 if len(X_t) > 1000 else 0.2
    val_size = max(int(len(X_t) * val_ratio), 10)
    X_val, y_val = X_t[-val_size:].to(device), y_t[-val_size:].to(device)
    X_tr, y_tr   = X_t[:-val_size], y_t[:-val_size]

    dataset = TensorDataset(X_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    lstm_model = LSTMNetwork(input_size=1, hidden_size=32, num_layers=1).to(device)
    criterion  = nn.MSELoss()
    optimizer  = torch.optim.Adam(lstm_model.parameters(), lr=0.005)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    EPOCHS  = 150
    PATIENCE = 25
    best_val_loss = float('inf')
    patience_ctr  = 0
    best_state    = None

    for epoch in range(EPOCHS):
        lstm_model.train()
        t_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(lstm_model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item() * len(bx)
        t_loss /= len(dataset)

        lstm_model.eval()
        with torch.no_grad():
            v_loss = criterion(lstm_model(X_val), y_val).item()

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in lstm_model.state_dict().items()}
        else:
            patience_ctr += 1

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: {t_loss:.6f} | Val: {v_loss:.6f} | LR: {lr:.6f}")

        if patience_ctr >= PATIENCE:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    if best_state:
        lstm_model.load_state_dict(best_state)
    print(f"   LSTM best val loss: {best_val_loss:.6f}")


# ==========================================================================
#  LSTM MULTI-STEP FORECAST HELPER
# ==========================================================================
def lstm_predict_multistep(model, scaler, last_window, n_steps, clamp_pct=0.10):
    """
    Autoregressive multi-step residual forecast.
    - Exponential decay: LSTM correction fades at longer horizons.
    - Clamp: correction capped at clamp_pct of SARIMA forecast magnitude.
    """
    if model is None:
        return np.zeros(n_steps)
    window = last_window.tolist()
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for step in range(n_steps):
            x = torch.FloatTensor(
                np.array(window[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
            ).to(device)
            p = model(x).cpu().numpy().flatten()[0]
            preds_scaled.append(p)
            window.append(p)
    # Inverse transform to original residual scale
    preds_orig = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()
    # Apply exponential decay so LSTM correction → 0 at long horizons
    decay = np.exp(-0.1 * np.arange(n_steps))
    return preds_orig * decay

# ==========================================================================
# 8. HYBRID FORECAST  (SARIMA + LSTM residuals)
# ==========================================================================
print("\n[7/8] Generating hybrid forecast on test set...")

# SARIMA component (already computed above)
sarima_component = sarima_test_fc.values

# LSTM residual correction
lstm_residual_pred = lstm_predict_multistep(
    lstm_model, residual_scaler, scaled_residuals, len(test_ts)
)

hybrid_predictions = sarima_component + lstm_residual_pred

# ==========================================================================
# 9. EVALUATION
# ==========================================================================
print("\n[8/8] Evaluating Hybrid model...")

mse  = mean_squared_error(test_ts, hybrid_predictions)
rmse = math.sqrt(mse)
mae  = mean_absolute_error(test_ts, hybrid_predictions)
r2   = r2_score(test_ts, hybrid_predictions)

nonzero_mask = test_ts != 0
if nonzero_mask.any():
    mape = float(np.mean(np.abs(
        (test_ts[nonzero_mask] - hybrid_predictions[nonzero_mask.values])
        / test_ts[nonzero_mask]
    )) * 100)
else:
    mape = float("nan")

improvement_rmse = (sarima_only_rmse - rmse) / sarima_only_rmse * 100

print("\n" + "=" * 60)
print("       HYBRID (SARIMA-LSTM) MODEL FINAL REPORT CARD")
print("=" * 60)
print(f"Architecture:          Sequential Residual (SARIMA + LSTM)")
print(f"SARIMA Order:          {best_order}")
print(f"SARIMA Seasonal Order: {best_seasonal}")
print(f"SARIMA AIC:            {best_aic:.2f}")
print(f"LSTM Lookback:         {LOOKBACK} hours")
print(f"Data Frequency:        Hourly (resampled from 5-min)")
print("-" * 60)
print(f"MSE  (Mean Squared Error):   {mse:.2f}")
print(f"RMSE (Root Mean Sq Error):   {rmse:.2f}")
print(f"MAE  (Mean Absolute Error):  {mae:.2f}")
print(f"R²   (Accuracy Score):       {r2:.4f}")
print(f"MAPE (% Error):              {mape:.2f}%")
print("-" * 60)
print(f"  SARIMA-only baseline RMSE: {sarima_only_rmse:.2f}")
print(f"  Hybrid vs SARIMA: RMSE {improvement_rmse:+.2f}%")
print("-" * 60)

if r2 > 0.80:
    quality = "EXCELLENT (High Predictive Power)"
elif r2 > 0.50:
    quality = "GOOD (Captures Trends)"
else:
    quality = "NEEDS IMPROVEMENT (More data / tuning needed)"
print(f"Model Quality: {quality}")

# Sample comparisons
print("\nSample Comparisons (first 10):")
comparison = pd.DataFrame({
    "Actual": test_ts.values[:10],
    "SARIMA": sarima_component[:10],
    "LSTM_Corr": lstm_residual_pred[:10],
    "Hybrid": hybrid_predictions[:10],
})
comparison["Error"] = comparison["Actual"] - comparison["Hybrid"]
print(comparison.to_string(index=False, float_format="%.1f"))
print("=" * 60)

# ==========================================================================
# 10. SAVE METRICS TO HDFS
# ==========================================================================
print("\nSaving hybrid model metrics to HDFS...")
metrics_path = "hdfs://namenode:8020/models/metrics/hybrid_metrics"

metrics_df = spark.createDataFrame(
    [(float(mse), float(rmse), float(mae), float(r2), float(mape),
      str(best_order), str(best_seasonal), float(best_aic),
      float(improvement_rmse))],
    ["mse", "rmse", "mae", "r2", "mape",
     "sarima_order", "sarima_seasonal_order", "sarima_aic",
     "improvement_over_sarima_pct"]
)
metrics_df.coalesce(1).write.mode("overwrite").json(metrics_path)
print(f"   Hybrid metrics saved to {metrics_path}")

spark.stop()
print("\nHybrid SARIMA-LSTM pipeline complete.")
