import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Initialize Spark
spark = (SparkSession.builder 
    .appName("LSTM_Traffic_Model") 
    .getOrCreate()
)

# 1. LOAD DATA (Hourly Feature Engineering Output)
print("Loading hourly feature data from HDFS...")
df = spark.read.parquet("hdfs://namenode:8020/logs/features/traffic_ml_ready_hourly")

# Sort by time strictly
pdf = df.select("hour_timestamp", "request_count", "total_bytes", "prev_count") \
        .orderBy("hour_timestamp").toPandas()

pdf["hour_timestamp"] = pd.to_datetime(pdf["hour_timestamp"])

# Add cyclical hour features
pdf["hour_sin"] = np.sin(2 * np.pi * pdf["hour_timestamp"].dt.hour / 24)
pdf["hour_cos"] = np.cos(2 * np.pi * pdf["hour_timestamp"].dt.hour / 24)

# Features for LSTM input
feature_cols = ["request_count", "total_bytes", "prev_count", "hour_sin", "hour_cos"]
target_col = "request_count"

data_features = pdf[feature_cols].values.astype('float32')
data_target = pdf[[target_col]].values.astype('float32')

# 2. PREPROCESS: Normalize Data (LSTMs rely on 0-1 scaling)
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

data_features_norm = feature_scaler.fit_transform(data_features)
data_target_norm = target_scaler.fit_transform(data_target)

# 3. CREATE SEQUENCES (Sliding Window)
# Use past 6 hours to predict the next 1 hour
SEQ_LENGTH = 6
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

# Chronological Split Training/Test (80/20) — no sequence crossing
train_size = int(len(X) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

print(f"Dataset: {len(X)} sequences, Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Features: {feature_cols}, Sequence Length: {SEQ_LENGTH} hours")

# 4. DEFINE LSTM MODEL
class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_layer_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_out))
        return self.fc2(out)

model = TrafficLSTM(input_size=N_FEATURES)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. TRAIN LOOP
print(f"Training LSTM for 30 epochs (Dataset size: {len(X)})...")
epochs = 30
batch_size = 32

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, len(X_train), batch_size):
        end = min(start + batch_size, len(X_train))
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_function(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch+1} loss: {avg_loss:.6f}')

# 6. EVALUATE
print("Evaluating on Test Data...")
model.eval()
with torch.no_grad():
    test_pred_norm = model(X_test).numpy()

# Invert Normalization to get real request counts
actual_predictions = target_scaler.inverse_transform(test_pred_norm)
actual_targets = target_scaler.inverse_transform(y_test.numpy())

# Metrics
rmse = np.sqrt(mean_squared_error(actual_targets, actual_predictions))
r2 = r2_score(actual_targets, actual_predictions)

print("\n" + "="*40)
print("      LSTM PERFORMANCE REPORT      ")
print("="*40)
print(f"Algorithm:       PyTorch LSTM (Deep Learning)")
print(f"Data Frequency:  Hourly")
print(f"Split:           Chronological (80/20)")
print("-" * 40)
print(f"RMSE (Error):    {rmse:.2f}")
print(f"R2 (Accuracy):   {r2:.2%}")
print("="*40 + "\n")

# Show Sample
print("Sample Comparison (Actual vs Predicted):")
comparison = pd.DataFrame({'Actual': actual_targets.flatten(), 'Predicted': actual_predictions.flatten()})
print(comparison.head(10))

spark.stop()
