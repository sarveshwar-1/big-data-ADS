package com.example.bigdata

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object FeatureEngineering {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("FeatureEngineering")
      .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
      .getOrCreate()

    import spark.implicits._

    println("Loading all parsed logs from HDFS...")
    val df = spark.read.parquet("hdfs://namenode:8020/logs/parsed")

    println("Converting timestamps...")
    val dfClean = df
      .withColumn("event_ts", to_timestamp(col("event_time"), "dd/MMM/yyyy:HH:mm:ss"))
      .filter(col("event_ts").isNotNull)

    println("Aggregating into 1-hour windows...")
    val trafficFeatures = dfClean
      .groupBy(window(col("event_ts"), "1 hour").alias("time_window"))
      .agg(
        count("*").alias("request_count"),
        sum("bytes").alias("total_bytes")
      )
      .select(
        col("time_window.start").alias("hour_timestamp"),
        col("request_count"),
        col("total_bytes")
      )
      .orderBy("hour_timestamp")

    println("Creating Lag Features for ML training...")
    val w = Window.orderBy("hour_timestamp")
    val mlReadyDf = trafficFeatures.withColumn("prev_count", lag("request_count", 1).over(w))
      .na.drop()

    val outputPath = "hdfs://namenode:8020/logs/features/traffic_ml_ready_hourly"
    println(s"Saving ML-ready hourly features to $outputPath...")
    mlReadyDf.write
      .mode("overwrite")
      .parquet(outputPath)

    println("Feature Engineering Complete. Data is ready for Model Training.")
    spark.stop()
  }
}
