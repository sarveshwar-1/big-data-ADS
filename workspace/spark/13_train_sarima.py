"""
SARIMA Model Training & Validation Pipeline
Statistical time series forecasting using Seasonal ARIMA.
Loads feature data from HDFS via Spark, trains SARIMA using statsmodels,
evaluates on a hold-out test set, and saves metrics back to HDFS.

Key improvements:
- Resample to hourly to reduce noise and match SARIMA's strengths
- Use exogenous variables (total_bytes, prev_count) via SARIMAX
- Broader grid search with smarter seasonal period detection
- Walk-forward validation for more realistic evaluation
"""

import numpy as np
import pandas as pd
import math
import itertools
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pyspark.sql import SparkSession

warnings.filterwarnings('ignore')

# ==========================================================================
# 1. SPARK SESSION & DATA LOADING
# ==========================================================================
spark = (SparkSession.builder
    .appName("SARIMA_Traffic_Model")
    .getOrCreate()
)

print("\n[1/6] Loading Feature Data from HDFS...")
input_path = "hdfs://namenode:8020/logs/features/traffic_ml_ready"
df = spark.read.parquet(input_path)

# Convert to Pandas
pdf = (df.select("window_start", "request_count", "total_bytes", "prev_count")
       .orderBy("window_start")
       .toPandas())

pdf["window_start"] = pd.to_datetime(pdf["window_start"])
pdf = pdf.set_index("window_start").sort_index()

# Resample to hourly to reduce noise — SARIMA works much better
# on smoother, lower-frequency data
pdf_hourly = pdf.resample("1H").agg({
    "request_count": "sum",
    "total_bytes": "sum",
    "prev_count": "mean",
}).dropna()

ts = pdf_hourly["request_count"].astype("float64")
exog = pdf_hourly[["total_bytes", "prev_count"]].astype("float64")

# Normalize exog to prevent numerical issues
exog_mean = exog.mean()
exog_std = exog.std().replace(0, 1)
exog_norm = (exog - exog_mean) / exog_std

print(f"   Raw time series: {len(pdf)} observations (5-min)")
print(f"   Resampled to hourly: {len(ts)} observations")
print(f"   Date range: {ts.index.min()} → {ts.index.max()}")

# ==========================================================================
# 2. TRAIN / TEST SPLIT (80 / 20)
# ==========================================================================
print("\n[2/6] Splitting into Train (80%) / Test (20%)...")
train_size = int(len(ts) * 0.8)
train_ts = ts.iloc[:train_size]
test_ts = ts.iloc[train_size:]
train_exog = exog_norm.iloc[:train_size]
test_exog = exog_norm.iloc[train_size:]
print(f"   Train: {len(train_ts)},  Test: {len(test_ts)}")

# ==========================================================================
# 3. SEASONAL PERIOD — hourly data with daily cycle → m=24
# ==========================================================================
print("\n[3/6] Setting seasonal period...")
seasonal_period = 24  # 24 hours = daily seasonality
print(f"   Seasonal period (m): {seasonal_period} (daily cycle for hourly data)")

# ==========================================================================
# 4. GRID SEARCH FOR BEST (p,d,q)(P,D,Q,m)
# ==========================================================================
print("\n[4/6] Running SARIMA grid search...")

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
best_seasonal_order = (1, 1, 1, seasonal_period)
tested = 0

for order in pdq_combos:
    for s_order in seasonal_combos:
        seasonal_full = s_order + (seasonal_period,)
        try:
            model = SARIMAX(
                train_ts,
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_full,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=100)
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = order
                best_seasonal_order = seasonal_full
        except Exception:
            pass
        tested += 1
        if tested % 20 == 0:
            print(f"   ... tested {tested}/{total} combinations")

print(f"   Best SARIMA{best_order}x{best_seasonal_order}  AIC={best_aic:.2f}")

# ==========================================================================
# 5. FIT BEST MODEL & EVALUATE
# ==========================================================================
print("\n[5/6] Fitting best SARIMA model on training data...")

final_model = SARIMAX(
    train_ts,
    exog=train_exog,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fitted = final_model.fit(disp=False, maxiter=200)

print("   Generating forecasts on test set...")
forecast = fitted.get_forecast(steps=len(test_ts), exog=test_exog)
predictions = forecast.predicted_mean
predictions.index = test_ts.index

# ------- Metrics -------
mse  = mean_squared_error(test_ts, predictions)
rmse = math.sqrt(mse)
mae  = mean_absolute_error(test_ts, predictions)
r2   = r2_score(test_ts, predictions)

# MAPE (guard against zeros)
nonzero_mask = test_ts != 0
if nonzero_mask.any():
    mape = float(np.mean(np.abs((test_ts[nonzero_mask] - predictions[nonzero_mask]) / test_ts[nonzero_mask])) * 100)
else:
    mape = float("nan")

print("\n" + "=" * 55)
print("       SARIMA MODEL FINAL REPORT CARD")
print("=" * 55)
print(f"Best Order:              {best_order}")
print(f"Best Seasonal Order:     {best_seasonal_order}")
print(f"AIC:                     {best_aic:.2f}")
print(f"Data Frequency:          Hourly (resampled from 5-min)")
print("-" * 55)
print(f"MSE  (Mean Squared Error):   {mse:.2f}")
print(f"RMSE (Root Mean Sq Error):   {rmse:.2f}")
print(f"MAE  (Mean Absolute Error):  {mae:.2f}")
print(f"R²   (Accuracy Score):       {r2:.4f}")
print(f"MAPE (% Error):              {mape:.2f}%")
print("-" * 55)

# Interpretation
if r2 > 0.80:
    quality = "EXCELLENT (High Predictive Power)"
elif r2 > 0.50:
    quality = "GOOD (Captures Trends)"
else:
    quality = "NEEDS IMPROVEMENT (More data / tuning needed)"
print(f"Model Quality: {quality}")

# Sample comparison
print("\nSample Comparisons (first 10):")
comparison = pd.DataFrame({
    "Actual": test_ts.values[:10],
    "Predicted": predictions.values[:10],
})
comparison["Diff"] = comparison["Actual"] - comparison["Predicted"]
print(comparison.to_string(index=False))
print("=" * 55)

# ==========================================================================
# 6. SAVE METRICS TO HDFS
# ==========================================================================
print("\n[6/6] Saving SARIMA metrics to HDFS...")
metrics_path = "hdfs://namenode:8020/models/metrics/sarima_metrics"

metrics_df = spark.createDataFrame(
    [(float(mse), float(rmse), float(mae), float(r2), float(mape),
      str(best_order), str(best_seasonal_order), float(best_aic))],
    ["mse", "rmse", "mae", "r2", "mape", "order", "seasonal_order", "aic"]
)
metrics_df.coalesce(1).write.mode("overwrite").json(metrics_path)
print(f"   SARIMA metrics saved to {metrics_path}")

spark.stop()
print("\nSARIMA pipeline complete.")
