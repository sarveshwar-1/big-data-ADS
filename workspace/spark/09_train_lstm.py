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

# 1. LOAD DATA (Feature Engineering Output)
print("Loading feature data from HDFS...")
df = spark.read.parquet("hdfs://namenode:8020/logs/features/traffic_ml_ready")

# Select only the target variable for time-series forecasting
# We need to sort by time strictly
pdf = df.select("request_count").orderBy("window_start").toPandas()
data = pdf.values.astype('float32')

# 2. PREPROCESS: Normalize Data (LSTMs rely on 0-1 scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# 3. CREATE SEQUENCES (Sliding Window)
# We will use the past 3 time steps (15 mins) to predict the next 1
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

# Split Training/Test (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# 4. DEFINE LSTM MODEL
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

# 5. TRAIN LOOP
print(f"Training LSTM for 10 epochs (Dataset size: {len(X)})...")
epochs = 10

for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    print(f'Epoch {i+1} loss: {single_loss.item():.6f}')

# 6. EVALUATE
print("Evaluating on Test Data...")
model.eval()
test_predictions = []

with torch.no_grad():
    for seq in X_test:
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_predictions.append(model(seq).item())

# Invert Normalization to get real request counts
actual_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
actual_targets = scaler.inverse_transform(y_test.numpy())

# Metrics
rmse = np.sqrt(mean_squared_error(actual_targets, actual_predictions))
r2 = r2_score(actual_targets, actual_predictions)

print("\n" + "="*40)
print("      LSTM PERFORMANCE REPORT      ")
print("="*40)
print(f"Algorithm:       PyTorch LSTM (Deep Learning)")
print("-" * 40)
print(f"RMSE (Error):    {rmse:.2f}")
print(f"R2 (Accuracy):   {r2:.2%}")
print("="*40 + "\n")

# Show Sample
print("Sample Comparison (Actual vs Predicted):")
comparison = pd.DataFrame({'Actual': actual_targets.flatten(), 'Predicted': actual_predictions.flatten()})
print(comparison.head(10))

spark.stop()
