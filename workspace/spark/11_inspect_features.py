from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark
spark = (SparkSession.builder 
    .appName("InspectFeatures") 
    .getOrCreate()
)

# 1. READ: Load the feature data from HDFS
input_path = "hdfs://namenode:8020/logs/features/traffic_ml_ready"
print(f"Reading data from: {input_path}")
df = spark.read.parquet(input_path)

# 2. SCHEMA: Show the structure (Columns & Types)
print("\n" + "="*40)
print("      DATASET SCHEMA      ")
print("="*40)
df.printSchema()

# 3. DATA: Show the first 20 rows (Sorted by Time)
print("\n" + "="*40)
print("      SAMPLE DATA (First 20 Rows)      ")
print("="*40)
df.orderBy("window_start").show(20, truncate=False)

# 4. STATS: Show summary statistics (Count, Mean, Min, Max)
# This helps you spot outliers (e.g., if Min request_count is 0)
print("\n" + "="*40)
print("      STATISTICAL SUMMARY      ")
print("="*40)
df.select("request_count", "prev_count", "total_bytes").describe().show()

spark.stop()
