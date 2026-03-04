from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .appName("ModelComparison")
    .getOrCreate()
)

# ==========================================
# 1. LOAD METRICS
# ==========================================
print("\nLoading saved metrics from HDFS...")

lr_df   = spark.read.json("hdfs://namenode:8020/models/metrics/lr_metrics")
lstm_df = spark.read.json("hdfs://namenode:8020/models/metrics/lstm_metrics")

lr   = lr_df.first()
lstm = lstm_df.first()

lr_rmse = float(lr["rmse"])
lr_mae  = float(lr["mae"])
lr_r2   = float(lr["r2"])

lstm_mse  = float(lstm["mse"])
lstm_rmse = float(lstm["rmse"])
lstm_mae  = float(lstm["mae"])
lstm_r2   = float(lstm["r2"])

# ==========================================
# 2. COMPARISON TABLE
# ==========================================
print("\n" + "=" * 60)
print("        FINAL MODEL COMPARISON REPORT")
print("=" * 60)
print(f"{'Metric':<30} {'Linear Regression':>15} {'LSTM':>10}")
print("-" * 60)
print(f"{'RMSE (lower is better)':<30} {lr_rmse:>15.4f} {lstm_rmse:>10.4f}")
print(f"{'MAE  (lower is better)':<30} {lr_mae:>15.4f} {lstm_mae:>10.4f}")
print(f"{'R²   (higher is better)':<30} {lr_r2:>15.4f} {lstm_r2:>10.4f}")
print("=" * 60)

# ==========================================
# 3. WINNER DETERMINATION
# ==========================================
print("\nVerdict:")
rmse_winner = "Linear Regression" if lr_rmse < lstm_rmse else "LSTM"
mae_winner  = "Linear Regression" if lr_mae  < lstm_mae  else "LSTM"
r2_winner   = "Linear Regression" if lr_r2   > lstm_r2   else "LSTM"

print(f"  Best RMSE : {rmse_winner} ({min(lr_rmse, lstm_rmse):.4f})")
print(f"  Best MAE  : {mae_winner}  ({min(lr_mae,  lstm_mae):.4f})")
print(f"  Best R²   : {r2_winner}  ({max(lr_r2,   lstm_r2):.4f})")

scores = {"Linear Regression": 0, "LSTM": 0}
scores[rmse_winner] += 1
scores[mae_winner]  += 1
scores[r2_winner]   += 1

overall = max(scores, key=scores.get)
print(f"\n  Overall Winner: {overall} ({scores[overall]}/3 metrics)")

# R² interpretation
print("\nModel Quality:")
for name, r2 in [("Linear Regression", lr_r2), ("LSTM", lstm_r2)]:
    if r2 > 0.80:
        quality = "EXCELLENT"
    elif r2 > 0.50:
        quality = "GOOD"
    else:
        quality = "NEEDS IMPROVEMENT"
    print(f"  {name:<20}: R²={r2:.4f} -> {quality}")

print("=" * 60 + "\n")

spark.stop()
