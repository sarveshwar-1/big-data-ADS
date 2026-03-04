package com.example.bigdata

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object AggregateLogs {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("LogAnalyticsToMongo")
      .config("spark.mongodb.write.connection.uri", "mongodb://mongodb:27017/logs_db.metrics")
      .config("spark.mongodb.output.uri", "mongodb://mongodb:27017/logs_db.metrics")
      .getOrCreate()

    import spark.implicits._

    println("Reading Parquet data...")
    val df = spark.read.parquet("hdfs://namenode:8020/logs/parsed")

    println("Calculating Top 10 IPs...")
    val topIps = df.groupBy("ip")
      .agg(count("*").alias("request_count"))
      .orderBy(desc("request_count"))
      .limit(10)

    topIps.show()

    println("Calculating Status Code Distribution...")
    val statusCounts = df.groupBy("status")
      .agg(count("*").alias("count"))
      .orderBy(desc("count"))

    statusCounts.show()

    println("Writing Top IPs to MongoDB...")
    topIps.write
      .format("mongodb")
      .mode("overwrite")
      .option("database", "logs_db")
      .option("collection", "top_ips")
      .save()

    println("Writing Status Counts to MongoDB...")
    statusCounts.write
      .format("mongodb")
      .mode("overwrite")
      .option("database", "logs_db")
      .option("collection", "status_codes")
      .save()

    println("Analytics pipeline finished successfully!")
    spark.stop()
  }
}
