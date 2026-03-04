from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

spark = (SparkSession.builder
    .appName("TrafficPredictionModel")
    .getOrCreate()
)

input_path = "hdfs://namenode:8020/logs/features/traffic_ml_ready"
print(f"Loading features from {input_path}...")
data = spark.read.parquet(input_path)

assembler = VectorAssembler(
    inputCols=["prev_count"],
    outputCol="features"
)

print("Splitting data into Training (80%) and Testing (20%)...")
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

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
predictions.select("window_start", "prev_count", "request_count", "prediction").show(5)

model_path = "hdfs://namenode:8020/models/traffic_prediction_v1"
print(f"Saving trained model to {model_path}...")
model.write().overwrite().save(model_path)

print("Model Training Complete.")
spark.stop()
