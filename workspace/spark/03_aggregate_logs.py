from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, count

spark = (SparkSession.builder
    .appName("LogAnalyticsToMongo")
    .config("spark.mongodb.write.connection.uri", "mongodb://mongodb:27017/logs_db.metrics")
    .config("spark.mongodb.output.uri", "mongodb://mongodb:27017/logs_db.metrics")
    .getOrCreate()
)

print("Reading Parquet data...")
df = spark.read.parquet("hdfs://namenode:8020/logs/parsed")

print("Calculating Top 10 IPs...")
top_ips = (df.groupBy("ip")
    .agg(count("*").alias("request_count"))
    .orderBy(desc("request_count"))
    .limit(10)
)
top_ips.show()

print("Calculating Status Code Distribution...")
status_counts = (df.groupBy("status")
    .agg(count("*").alias("count"))
    .orderBy(desc("count"))
)
status_counts.show()

print("Writing Top IPs to MongoDB...")
(top_ips.write
    .format("mongodb")
    .mode("overwrite")
    .option("database", "logs_db")
    .option("collection", "top_ips")
    .save()
)

print("Writing Status Counts to MongoDB...")
(status_counts.write
    .format("mongodb")
    .mode("overwrite")
    .option("database", "logs_db")
    .option("collection", "status_codes")
    .save()
)

print("Analytics pipeline finished successfully!")
spark.stop()
