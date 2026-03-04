from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, to_timestamp, sum as _sum, count, lag
from pyspark.sql.window import Window

spark = (SparkSession.builder
    .appName("FeatureEngineering")
    .getOrCreate()
)

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

print("Loading all parsed logs from HDFS...")
df = spark.read.parquet("hdfs://namenode:8020/logs/parsed")

print("Converting timestamps...")
df_clean = df.withColumn("event_ts", to_timestamp(col("event_time"), "dd/MMM/yyyy:HH:mm:ss"))
df_clean = df_clean.filter(col("event_ts").isNotNull())

print("Aggregating into 5-minute windows (This processes ALL data)...")
traffic_features = (df_clean
    .groupBy(window(col("event_ts"), "5 minutes").alias("time_window"))
    .agg(
        count("*").alias("request_count"),
        _sum("bytes").alias("total_bytes")
    )
    .select(
        col("time_window.start").alias("window_start"),
        col("time_window.end").alias("window_end"),
        col("request_count"),
        col("total_bytes")
    )
    .orderBy("window_start")
)

print("Creating Lag Features for ML training...")
w = Window.orderBy("window_start")
ml_ready_df = traffic_features.withColumn("prev_count", lag("request_count", 1).over(w))
ml_ready_df = ml_ready_df.na.drop()

output_path = "hdfs://namenode:8020/logs/features/traffic_ml_ready"
print(f"Saving ML-ready features to {output_path}...")

(ml_ready_df.write
    .mode("overwrite")
    .parquet(output_path)
)

print("Feature Engineering Complete. Data is ready for Model Training.")
spark.stop()
