from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .appName("ModelComparison")
    .getOrCreate()
)

# ==========================================
# 1. LOAD METRICS
# ==========================================
print("\nLoading saved metrics from HDFS...")

lr_df     = spark.read.json("hdfs://namenode:8020/models/metrics/lr_metrics")
lstm_df   = spark.read.json("hdfs://namenode:8020/models/metrics/lstm_metrics")
sarima_df = spark.read.json("hdfs://namenode:8020/models/metrics/sarima_metrics")
hybrid_df = spark.read.json("hdfs://namenode:8020/models/metrics/hybrid_metrics")

lr     = lr_df.first()
lstm   = lstm_df.first()
sarima = sarima_df.first()
hybrid = hybrid_df.first()

lr_rmse = float(lr["rmse"])
lr_mae  = float(lr["mae"])
lr_r2   = float(lr["r2"])

lstm_mse  = float(lstm["mse"])
lstm_rmse = float(lstm["rmse"])
lstm_mae  = float(lstm["mae"])
lstm_r2   = float(lstm["r2"])

sarima_mse  = float(sarima["mse"])
sarima_rmse = float(sarima["rmse"])
sarima_mae  = float(sarima["mae"])
sarima_r2   = float(sarima["r2"])

hybrid_mse  = float(hybrid["mse"])
hybrid_rmse = float(hybrid["rmse"])
hybrid_mae  = float(hybrid["mae"])
hybrid_r2   = float(hybrid["r2"])

# ==========================================
# 2. COMPARISON TABLE
# ==========================================
print("\n" + "=" * 90)
print("                       FINAL MODEL COMPARISON REPORT")
print("=" * 90)
print(f"{'Metric':<30} {'Linear Regression':>17} {'LSTM':>10} {'SARIMA':>12} {'Hybrid':>12}")
print("-" * 90)
print(f"{'RMSE (lower is better)':<30} {lr_rmse:>17.4f} {lstm_rmse:>10.4f} {sarima_rmse:>12.4f} {hybrid_rmse:>12.4f}")
print(f"{'MAE  (lower is better)':<30} {lr_mae:>17.4f} {lstm_mae:>10.4f} {sarima_mae:>12.4f} {hybrid_mae:>12.4f}")
print(f"{'R²   (higher is better)':<30} {lr_r2:>17.4f} {lstm_r2:>10.4f} {sarima_r2:>12.4f} {hybrid_r2:>12.4f}")
print("=" * 90)

# ==========================================
# 3. WINNER DETERMINATION
# ==========================================
models = {
    "Linear Regression": {"rmse": lr_rmse, "mae": lr_mae, "r2": lr_r2},
    "LSTM":              {"rmse": lstm_rmse, "mae": lstm_mae, "r2": lstm_r2},
    "SARIMA":            {"rmse": sarima_rmse, "mae": sarima_mae, "r2": sarima_r2},
    "Hybrid (SARIMA+LSTM)": {"rmse": hybrid_rmse, "mae": hybrid_mae, "r2": hybrid_r2},
}

rmse_winner = min(models, key=lambda m: models[m]["rmse"])
mae_winner  = min(models, key=lambda m: models[m]["mae"])
r2_winner   = max(models, key=lambda m: models[m]["r2"])

print("\nVerdict:")
print(f"  Best RMSE : {rmse_winner} ({models[rmse_winner]['rmse']:.4f})")
print(f"  Best MAE  : {mae_winner}  ({models[mae_winner]['mae']:.4f})")
print(f"  Best R²   : {r2_winner}  ({models[r2_winner]['r2']:.4f})")

scores = {name: 0 for name in models}
scores[rmse_winner] += 1
scores[mae_winner]  += 1
scores[r2_winner]   += 1

overall = max(scores, key=scores.get)
print(f"\n  Overall Winner: {overall} ({scores[overall]}/3 metrics)")

# Hybrid improvement summary
hybrid_imp = float(hybrid["improvement_over_sarima_pct"])
print(f"\n  Hybrid RMSE improvement over SARIMA-only: {hybrid_imp:+.2f}%")

# R² interpretation
print("\nModel Quality:")
for name, vals in models.items():
    r2 = vals["r2"]
    if r2 > 0.80:
        quality = "EXCELLENT"
    elif r2 > 0.50:
        quality = "GOOD"
    else:
        quality = "NEEDS IMPROVEMENT"
    print(f"  {name:<25}: R²={r2:.4f} -> {quality}")

print("=" * 90 + "\n")

spark.stop()
