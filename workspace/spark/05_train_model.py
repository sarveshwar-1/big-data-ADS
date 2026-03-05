from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, hour, sin, cos, lit, abs as _abs, avg
from pyspark.sql.expressions import Window as W
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
import math

spark = (SparkSession.builder
    .appName("TrafficPredictionModel")
    .getOrCreate()
)

input_path = "hdfs://namenode:8020/logs/features/traffic_ml_ready_hourly"
print(f"Loading hourly features from {input_path}...")
data = spark.read.parquet(input_path)

# Sort by timestamp for chronological split
sorted_data = data.orderBy("hour_timestamp")
indexed_data = sorted_data.withColumn("_row_idx",
    row_number().over(Window.orderBy("hour_timestamp")) - 1)

total_rows = indexed_data.count()
train_size = int(total_rows * 0.8)

print(f"Chronological split: train={train_size}, test={total_rows - train_size} (total={total_rows})")
train_data = indexed_data.filter(col("_row_idx") < train_size).drop("_row_idx")
test_data = indexed_data.filter(col("_row_idx") >= train_size).drop("_row_idx")

# Add cyclical hour features
pi_val = math.pi
train_data = (train_data
    .withColumn("hour_of_day", hour(col("hour_timestamp")))
    .withColumn("hour_sin", sin(lit(2.0 * pi_val) * col("hour_of_day") / lit(24.0)))
    .withColumn("hour_cos", cos(lit(2.0 * pi_val) * col("hour_of_day") / lit(24.0)))
)
test_data = (test_data
    .withColumn("hour_of_day", hour(col("hour_timestamp")))
    .withColumn("hour_sin", sin(lit(2.0 * pi_val) * col("hour_of_day") / lit(24.0)))
    .withColumn("hour_cos", cos(lit(2.0 * pi_val) * col("hour_of_day") / lit(24.0)))
)

assembler = VectorAssembler(
    inputCols=["prev_count", "hour_sin", "hour_cos"],
    outputCol="features"
)

print("Training Linear Regression Model...")
lr = LinearRegression(featuresCol="features", labelCol="request_count")

pipeline = Pipeline(stages=[assembler, lr])
model = pipeline.fit(train_data)

print("Evaluating model on Test Data...")
predictions = model.transform(test_data)

evaluator = RegressionEvaluator(
    labelCol="request_count",
    predictionCol="prediction",
    metricName="rmse"
)
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

print("Sample Predictions:")
predictions.select("hour_timestamp", "prev_count", "request_count", "prediction").show(5)

model_path = "hdfs://namenode:8020/models/traffic_prediction_v1"
print(f"Saving trained model to {model_path}...")
model.write().overwrite().save(model_path)

print("Model Training Complete.")
spark.stop()
