#!/bin/bash

# Ensure JAR exists
JAR_PATH="/opt/spark/jobs/jars/traffic-analysis.jar"
CONFIG_FILE="/opt/spark/jobs/spark-defaults.conf"
MASTER="spark://spark:7077"

echo "Submitting Scala jobs to Spark..."

# Create necessary directories in HDFS for logging
echo "Creating HDFS logs directory..."
docker exec namenode hdfs dfs -mkdir -p /spark-logs
docker exec namenode hdfs dfs -chmod 777 /spark-logs

# Function to submit and wait
submit_and_wait() {
    echo "Submitting: $3"
    # Submit job and capture ID
    # Use --properties-file to load config
    OUTPUT=$(docker exec spark /opt/spark/bin/spark-submit \
      --properties-file $CONFIG_FILE \
      --master $MASTER \
      $1 $2 \
      --class $3 \
      $JAR_PATH 2>&1)
    
    echo "$OUTPUT"
    
    # Extract Driver ID
    DRIVER_ID=$(echo "$OUTPUT" | grep "Driver successfully submitted as" | head -n 1 | awk '{print $NF}' | tr -d '",')
    
    if [ -z "$DRIVER_ID" ]; then
        echo "Failed to retrieve Driver ID."
        return 1
    fi
    
    echo "Job submitted with ID: $DRIVER_ID. Waiting for completion..."
    
    # Poll status
    while true; do
        STATUS=$(docker exec spark /opt/spark/bin/spark-submit --master $MASTER --status $DRIVER_ID 2>&1 | grep "State of $DRIVER_ID" | awk '{print $NF}')
        
        if [ "$STATUS" == "FINISHED" ]; then
            echo "Job $DRIVER_ID finished successfully."
            break
        elif [ "$STATUS" == "FAILED" ] || [ "$STATUS" == "ERROR" ] || [ "$STATUS" == "KILLED" ]; then
            echo "Job $DRIVER_ID failed with status: $STATUS"
            exit 1
        fi
        
        sleep 5
    done
}

# 1. Parse Logs
echo "Step 1: Parse Logs"
submit_and_wait "" "" "com.example.bigdata.ParseLogs"

# 2. Aggregate Logs
echo "Step 2: Aggregate Logs"
submit_and_wait "--conf" "spark.jars.packages=org.mongodb.spark:mongo-spark-connector_2.12:10.2.1" "com.example.bigdata.AggregateLogs"

# 3. Feature Engineering
echo "Step 3: Feature Engineering"
submit_and_wait "" "" "com.example.bigdata.FeatureEngineering"

# 4. Train Traffic Model (Linear Regression)
echo "Step 4: Train Model (Spark ML)"
submit_and_wait "" "" "com.example.bigdata.TrainModel"

# 5. Predict Traffic
echo "Step 5: Predict Traffic"
submit_and_wait "" "" "com.example.bigdata.PredictTraffic"

echo "All Scala jobs completed."
