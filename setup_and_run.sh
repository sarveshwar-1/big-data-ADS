#!/bin/bash
set -e

echo "Waiting for NameNode..."
sleep 15
docker exec namenode hdfs dfsadmin -safemode leave || true

echo "Setting up HDFS data..."
docker exec namenode hdfs dfs -mkdir -p /logs/raw
docker cp workspace/data/access.log namenode:/tmp/access.log
docker exec namenode hdfs dfs -put -f /tmp/access.log /logs/raw/access.log

echo "Running Scala Jobs in Cluster Mode..."
bash workspace/run_scala_jobs.sh
