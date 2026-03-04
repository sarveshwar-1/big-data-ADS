# Scala Traffic Analysis Project

This project has been converted from PySpark to Scala for improved performance and type safety.
The deep learning components (LSTM) remain in Python (PyTorch) as they rely on specific libraries not easily available in pure Scala.

## Architecture

1.  **Ingestion**: Express.js server (Node.js) receives logs and writes to HDFS.
2.  **ETL**: Scala Spark jobs parse logs and aggregate metrics.
3.  **Feature Engineering**: Scala Spark processes data for ML.
4.  **Machine Learning**:
    -   **Linear Regression**: Implemented in Scala (Spark MLlib).
    -   **Deep Learning (LSTM)**: Implemented in Python (PyTorch).

## Prerequisites
-   Docker and Docker Compose
-   (Optional) sbt for local development

## Run Instructions

### 1. Build the Scala JAR
Run the build script to compile the Scala project and package it into a JAR.
```bash
cd workspace/scala-app
./build_jar.sh
cd ../..
```
This uses a Docker container to build the project, so you don't need `sbt` installed locally.

### 2. Run the Scala Pipeline
Execute all Scala jobs (ETL, Feature Engineering, Training, Prediction) inside the Spark container:
```bash
./workspace/run_scala_jobs.sh
```

### 3. Run the Deep Learning Model (Python)
Since PyTorch is Python-native, the LSTM model training remains in Python.
You can run it after the Scala Feature Engineering step:
```bash
docker exec -it spark /opt/spark/bin/spark-submit \
  --master spark://spark:7077 \
  /opt/spark/jobs/09_train_lstm.py
```

## Project Structure
-   `workspace/scala-app/`: Scala source code and build verification.
-   `workspace/spark/`: Original Python scripts (reference) and location of running scripts.
-   `express-server/`: Ingestion server.
