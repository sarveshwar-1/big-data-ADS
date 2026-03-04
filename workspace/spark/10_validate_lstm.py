import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Initialize Spark
spark = (SparkSession.builder 
    .appName("LSTM_Validation_Metrics") 
    .getOrCreate()
)

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
print("\n[1/5] Loading Feature Data from HDFS...")
input_path = "hdfs://namenode:8020/logs/features/traffic_ml_ready"
df = spark.read.parquet(input_path)

# Use multiple features: request_count, total_bytes, prev_count + time features
pdf = df.select("window_start", "request_count", "total_bytes", "prev_count") \
        .orderBy("window_start").toPandas()

# Engineer time-based features
pdf["window_start"] = pd.to_datetime(pdf["window_start"])
pdf["hour"] = pdf["window_start"].dt.hour
pdf["minute"] = pdf["window_start"].dt.minute
pdf["day_of_week"] = pdf["window_start"].dt.dayofweek
# Cyclical encoding for hour
pdf["hour_sin"] = np.sin(2 * np.pi * pdf["hour"] / 24)
pdf["hour_cos"] = np.cos(2 * np.pi * pdf["hour"] / 24)

# Features to use for input
feature_cols = ["request_count", "total_bytes", "prev_count", "hour_sin", "hour_cos"]
target_col = "request_count"

data_features = pdf[feature_cols].values.astype("float32")
data_target = pdf[[target_col]].values.astype("float32")

# Normalize each feature independently
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

data_features_norm = feature_scaler.fit_transform(data_features)
data_target_norm = target_scaler.fit_transform(data_target)

# Create Sequences with longer lookback
SEQ_LENGTH = 12  # 12 × 5min = 1 hour of history
N_FEATURES = len(feature_cols)

def create_sequences(features, target, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x = features[i:i+seq_length]
        y = target[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(data_features_norm, data_target_norm, SEQ_LENGTH)

# Convert to PyTorch Tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Split Training/Test (80% Train, 20% Test)
train_size = int(len(X) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

print(f"Data Prepared. Training Samples: {len(X_train)}, Test Samples: {len(X_test)}")
print(f"Features: {feature_cols}, Sequence Length: {SEQ_LENGTH}")

# ==========================================
# 2. DEFINE IMPROVED LSTM MODEL
# ==========================================
class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_out))
        return self.fc2(out)

model = TrafficLSTM(input_size=N_FEATURES)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ==========================================
# 3. TRAIN MODEL WITH MINI-BATCHES
# ==========================================
print("\n[2/5] Training LSTM Model (50 Epochs)...")
epochs = 50
batch_size = 32
best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    # Mini-batch training
    indices = torch.randperm(len(X_train))
    for start in range(0, len(X_train), batch_size):
        end = min(start + batch_size, len(X_train))
        batch_idx = indices[start:end]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_function(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test)
        val_loss = loss_function(val_pred, y_test).item()

    scheduler.step(val_loss)

    # Early stopping with best model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if patience_counter >= 10:
        print(f"  Early stopping at epoch {epoch}")
        break

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)

# ==========================================
# 4. GENERATE PREDICTIONS
# ==========================================
print("\n[3/5] Generating Predictions on Test Set...")
model.eval()
with torch.no_grad():
    test_pred_norm = model(X_test).numpy()

# Invert normalization
predicted_traffic = target_scaler.inverse_transform(test_pred_norm)
actual_traffic = target_scaler.inverse_transform(y_test.numpy())

# ==========================================
# 5. CALCULATE & PRINT METRICS
# ==========================================
print("\n[4/5] Calculating Performance Metrics...")

mse = mean_squared_error(actual_traffic, predicted_traffic)
rmse = math.sqrt(mse)
mae = mean_absolute_error(actual_traffic, predicted_traffic)
r2 = r2_score(actual_traffic, predicted_traffic)

print("\n" + "="*50)
print("       LSTM MODEL FINAL REPORT CARD       ")
print("="*50)
print(f"Dataset Size:    {len(X)} windows")
print(f"Test Set Size:   {len(actual_traffic)} windows")
print(f"Seq Length:      {SEQ_LENGTH} ({SEQ_LENGTH * 5} min lookback)")
print(f"Features:        {N_FEATURES} ({', '.join(feature_cols)})")
print("-" * 50)
print(f"MSE (Mean Squared Error):   {mse:.2f}")
print(f"RMSE (Root Mean Sq Error):  {rmse:.2f}")
print(f"MAE (Mean Absolute Error):  {mae:.2f}")
print(f"R² (Accuracy Score):        {r2:.4f}")
print("-" * 50)

# Interpretation
print("\n[5/5] Interpretation:")
print(f" -> On average, the prediction is off by {mae:.0f} requests.")
if r2 > 0.80:
    print(" -> Model Quality: EXCELLENT (High Predictive Power)")
elif r2 > 0.50:
    print(" -> Model Quality: GOOD (Captures Trends)")
else:
    print(" -> Model Quality: NEEDS IMPROVEMENT (More data needed)")

# Save metrics
print("\n[5/5] Saving LSTM metrics to HDFS...")
metrics_path = "hdfs://namenode:8020/models/metrics/lstm_metrics"

metrics_df = spark.createDataFrame(
    [(float(mse), float(rmse), float(mae), float(r2))],
    ["mse", "rmse", "mae", "r2"]
)
metrics_df.coalesce(1).write.mode("overwrite").json(metrics_path)
print(f"   Metrics saved to {metrics_path}")

spark.stop()
print("\nLSTM pipeline complete.")
