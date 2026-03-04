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

# Convert to Pandas for PyTorch
pdf = df.select("request_count").orderBy("window_start").toPandas()
data = pdf.values.astype('float32')

# Normalize (LSTMs need 0-1 range)
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Create Sequences (Lookback Window = 3)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 3
X, y = create_sequences(data_normalized, SEQ_LENGTH)

# Convert to PyTorch Tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Split Training/Test (80% Train, 20% Test)
train_size = int(len(X) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

print(f"Data Prepared. Training Samples: {len(X_train)}, Test Samples: {len(X_test)}")

# ==========================================
# 2. DEFINE LSTM MODEL
# ==========================================
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = TrafficLSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 3. TRAIN MODEL (Retraining for Validation)
# ==========================================
print("\n[2/5] Training LSTM Model (15 Epochs)...")
epochs = 15

for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%5 == 0:
        print(f'Epoch {i} Loss: {single_loss.item():.6f}')

# ==========================================
# 4. GENERATE PREDICTIONS
# ==========================================
print("\n[3/5] Generating Predictions on Test Set...")
model.eval()
test_predictions = []

with torch.no_grad():
    for seq in X_test:
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_predictions.append(model(seq).item())

# Invert Normalization to get Real Request Counts
predicted_traffic = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
actual_traffic = scaler.inverse_transform(y_test.numpy())

# ==========================================
# 5. CALCULATE & PRINT METRICS
# ==========================================
print("\n[4/5] Calculating Performance Metrics...")

# 1. MSE (Mean Squared Error)
mse = mean_squared_error(actual_traffic, predicted_traffic)

# 2. RMSE (Root Mean Squared Error)
rmse = math.sqrt(mse)

# 3. MAE (Mean Absolute Error)
mae = mean_absolute_error(actual_traffic, predicted_traffic)

# 4. R2 Score (Accuracy Coefficient)
r2 = r2_score(actual_traffic, predicted_traffic)

print("\n" + "="*50)
print("       LSTM MODEL FINAL REPORT CARD       ")
print("="*50)
print(f"Dataset Size:    {len(X)} windows")
print(f"Test Set Size:   {len(actual_traffic)} windows")
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

# Show side-by-side sample
print("\nSample Comparisons:")
comparison = pd.DataFrame({'Actual': actual_traffic.flatten(), 'Predicted': predicted_traffic.flatten()})
comparison['Diff'] = comparison['Actual'] - comparison['Predicted']
print(comparison.head(10))

# ==========================================
# 6. SAVE METRICS TO HDFS for comparison
# ==========================================
print("\n[5/5] Saving LSTM metrics to HDFS...")
metrics_path = "hdfs://namenode:8020/models/metrics/lstm_metrics"
metrics_df = spark.createDataFrame(
    [(float(mse), float(rmse), float(mae), float(r2))],
    ["mse", "rmse", "mae", "r2"]
)
metrics_df.coalesce(1).write.mode("overwrite").json(metrics_path)
print(f"LSTM metrics saved to {metrics_path}")

spark.stop()
