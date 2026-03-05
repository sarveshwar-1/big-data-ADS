from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import math

spark = (SparkSession.builder
    .appName("TrafficInference")
    .getOrCreate()
)

model_path = "hdfs://namenode:8020/models/traffic_prediction_v1"
print(f"Loading model from {model_path}...")
model = PipelineModel.load(model_path)

print("Generating 'live' data for inference...")
# Model now expects prev_count, hour_sin, hour_cos
pi = math.pi
live_data = spark.createDataFrame([
    (2000, math.sin(2*pi*10/24), math.cos(2*pi*10/24)),
    (5000, math.sin(2*pi*14/24), math.cos(2*pi*14/24)),
    (100,  math.sin(2*pi*3/24),  math.cos(2*pi*3/24)),
    (8000, math.sin(2*pi*18/24), math.cos(2*pi*18/24)),
], ["prev_count", "hour_sin", "hour_cos"])

print("Predicting future traffic...")
predictions = model.transform(live_data)

print("Traffic Forecast:")
predictions.select("prev_count", "prediction").show()

print("Inference Job Complete.")
spark.stop()
