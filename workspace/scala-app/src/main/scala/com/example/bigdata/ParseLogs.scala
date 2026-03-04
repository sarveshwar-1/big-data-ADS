package com.example.bigdata

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object ParseLogs {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("ParseNginxLogs")
      .getOrCreate()

    import spark.implicits._

    val logs = spark.read.text("hdfs://namenode:8020/logs/raw/access.log")

    val pattern = """^(\S+) - - \[([^\]]+)\] "(\S+) ([^"]+) HTTP/[0-9.]+" (\d{3}) (\d+) "([^"]*)" "([^"]*)" "([^"]*)""""

    val parsed = logs.select(
      regexp_extract($"value", pattern, 1).alias("ip"),
      regexp_extract($"value", pattern, 2).alias("time_raw"),
      regexp_extract($"value", pattern, 3).alias("method"),
      regexp_extract($"value", pattern, 4).alias("url"),
      regexp_extract($"value", pattern, 5).cast("int").alias("status"),
      regexp_extract($"value", pattern, 6).cast("int").alias("bytes"),
      regexp_extract($"value", pattern, 7).alias("referrer"),
      regexp_extract($"value", pattern, 8).alias("user_agent")
    )

    val finalDf = parsed.withColumn(
      "event_time",
      to_timestamp(col("time_raw"), "dd/MMM/yyyy:HH:mm:ss Z")
    ).drop("time_raw")

    finalDf.write.mode("overwrite").parquet("hdfs://namenode:8020/logs/parsed")

    spark.stop()
  }
}
