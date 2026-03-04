#!/bin/bash
# Builds the Scala project using a custom Docker image for sbt.

echo "Building sbt Docker image..."
docker build -t local/scala-sbt -f ./workspace/scala-app/Dockerfile ./workspace/scala-app

echo "Building Scala JAR..."
docker run --rm \
  -v "$(pwd)/workspace/scala-app":/app \
  local/scala-sbt \
  sbt clean assembly

echo "Build complete. Copying JAR to spark jobs directory..."
mkdir -p workspace/spark/jars
cp workspace/scala-app/target/scala-2.12/TrafficAnalysis-assembly-0.1.jar workspace/spark/jars/traffic-analysis.jar

echo "JAR copied to workspace/spark/jars/traffic-analysis.jar"
