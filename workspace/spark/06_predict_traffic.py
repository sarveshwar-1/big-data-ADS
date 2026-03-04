from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = (SparkSession.builder
    .appName("TrafficInference")
    .getOrCreate()
)

model_path = "hdfs://namenode:8020/models/traffic_prediction_v1"
print(f"Loading model from {model_path}...")
model = PipelineModel.load(model_path)

print("Generating 'live' data for inference...")
live_data = spark.createDataFrame([
    (2000,), (5000,), (100,), (8000,)
], ["prev_count"])

print("Predicting future traffic...")
predictions = model.transform(live_data)

print("Traffic Forecast:")
predictions.select("prev_count", "prediction").show()

print("Inference Job Complete.")
spark.stop()
