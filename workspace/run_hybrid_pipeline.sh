#!/bin/bash
# =============================================================================
# HYBRID PIPELINE: Scala (preprocessing + LR) → Python (LSTM) → Comparison
#
# Execution topology (Spark Standalone Cluster):
#   Master  : spark://spark:7077
#   Workers : spark-worker, spark-worker-2
#   Deploy  : cluster mode (driver runs on a worker node)
#
# Stage 1 – Scala  : ParseLogs → AggregateLogs → FeatureEngineering → TrainModel (LR)
# Stage 2 – Python : 09_train_lstm.py → 10_validate_lstm.py
# Stage 3 – Python : 12_compare_models.py  (LR vs LSTM)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MASTER="spark://spark:7077"
JAR_PATH="/opt/spark/jobs/jars/traffic-analysis.jar"
JOBS_DIR="/opt/spark/jobs"
CONFIG_FILE="$JOBS_DIR/spark-defaults.conf"

SCALA_CONTAINER="spark"   # Spark master container – used to call spark-submit
LOCAL_JAR="$(dirname "$0")/scala-app/target/scala-2.12/TrafficAnalysis-assembly-0.1.jar"
LOCAL_JAR_DEST="$(dirname "$0")/spark/jars/traffic-analysis.jar"

DIVIDER="================================================================"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Helper: submit a Spark job in CLUSTER mode and poll until it finishes.
#
# Usage: submit_cluster [extra spark-submit args...] --class <ClassName>
#   The last two arguments must always be --class <ClassName>.
# ---------------------------------------------------------------------------
submit_scala() {
    local STEP="$1"; shift          # human-readable step name
    local CLASS="$1"; shift         # main class
    local EXTRA_ARGS=("$@")         # any extra --conf / --packages flags

    log "[$STEP] Submitting Scala class: $CLASS"

    local OUTPUT
    OUTPUT=$(docker exec "$SCALA_CONTAINER" /opt/spark/bin/spark-submit \
        --master "$MASTER" \
        --deploy-mode cluster \
        --properties-file "$CONFIG_FILE" \
        "${EXTRA_ARGS[@]}" \
        --class "$CLASS" \
        "$JAR_PATH" 2>&1)

    echo "$OUTPUT"

    local DRIVER_ID
    DRIVER_ID=$(echo "$OUTPUT" | grep -oP 'driver-\d+-\d+' | head -n1)

    if [ -z "$DRIVER_ID" ]; then
        die "[$STEP] Could not extract Driver ID from submission output."
    fi

    log "[$STEP] Driver ID: $DRIVER_ID – polling for completion..."
    _poll_driver "$STEP" "$DRIVER_ID"
}

# ---------------------------------------------------------------------------
# Helper: submit a PySpark job in CLUSTER mode and poll until it finishes.
# In standalone cluster mode the Python file must be reachable on the worker;
# since all spark containers mount ./workspace/spark → /opt/spark/jobs, the
# path is identical on master and workers.
# ---------------------------------------------------------------------------
submit_python() {
    local STEP="$1"; shift          # human-readable step name
    local SCRIPT="$1"; shift        # path inside the container
    local EXTRA_ARGS=("$@")

    log "[$STEP] Submitting Python script: $SCRIPT"

    local OUTPUT
    OUTPUT=$(docker exec "$SCALA_CONTAINER" /opt/spark/bin/spark-submit \
        --master "$MASTER" \
        --deploy-mode cluster \
        --properties-file "$CONFIG_FILE" \
        "${EXTRA_ARGS[@]}" \
        "$SCRIPT" 2>&1)

    echo "$OUTPUT"

    local DRIVER_ID
    DRIVER_ID=$(echo "$OUTPUT" | grep -oP 'driver-\d+-\d+' | head -n1)

    if [ -z "$DRIVER_ID" ]; then
        die "[$STEP] Could not extract Driver ID from submission output."
    fi

    log "[$STEP] Driver ID: $DRIVER_ID – polling for completion..."
    _poll_driver "$STEP" "$DRIVER_ID"
}

# ---------------------------------------------------------------------------
# Internal: poll spark-submit --status until FINISHED / FAILED
# ---------------------------------------------------------------------------
_poll_driver() {
    local STEP="$1"
    local DRIVER_ID="$2"

    while true; do
        local STATE
        STATE=$(docker exec "$SCALA_CONTAINER" /opt/spark/bin/spark-submit \
            --master "$MASTER" \
            --status "$DRIVER_ID" 2>&1 \
            | grep -oP '(?<=State of '"$DRIVER_ID"' is )\S+' | head -n1)

        case "$STATE" in
            FINISHED)
                log "[$STEP] Driver $DRIVER_ID finished successfully."
                return 0 ;;
            FAILED|KILLED|ERROR)
                die "[$STEP] Driver $DRIVER_ID ended with state: $STATE" ;;
            *)
                sleep 5 ;;
        esac
    done
}

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================
log "Checking Docker containers are running..."
for c in namenode "$SCALA_CONTAINER" spark-worker spark-worker-2; do
    docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null | grep -q true \
        || die "Container '$c' is not running. Start the cluster first: docker compose up -d"
done

# =============================================================================
# STEP 0 – INITIALISE HDFS DIRECTORIES
# =============================================================================
log "$DIVIDER"
log "STEP 0 – Initialising HDFS directories"
log "$DIVIDER"
docker exec namenode hdfs dfs -mkdir -p /spark-logs
docker exec namenode hdfs dfs -chmod 777 /spark-logs
docker exec namenode hdfs dfs -mkdir -p /models/metrics
docker exec namenode hdfs dfs -mkdir -p /logs/raw
docker exec namenode hdfs dfs -mkdir -p /logs/parsed
docker exec namenode hdfs dfs -mkdir -p /logs/features

# =============================================================================
# STEP 0b – ENSURE JAR IS BUILT AND DEPLOYED
# =============================================================================
log "$DIVIDER"
log "STEP 0b – Building Scala JAR"
log "$DIVIDER"

SCALA_APP_DIR="$(dirname "$0")/scala-app"
if [ ! -f "$LOCAL_JAR" ]; then
    log "JAR not found – building via sbt assembly inside Docker..."
    docker run --rm \
        -v "$(realpath "$SCALA_APP_DIR")":/workspace \
        -w /workspace \
        hseeberger/scala-sbt:17.0.2_1.6.2_2.12.15 \
        sbt assembly \
        || die "sbt assembly failed"
fi

mkdir -p "$(dirname "$LOCAL_JAR_DEST")"
cp "$LOCAL_JAR" "$LOCAL_JAR_DEST"
log "JAR available at $LOCAL_JAR_DEST (→ $JAR_PATH inside containers)"

# =============================================================================
# STAGE 1 – SCALA JOBS (Preprocessing + Linear Regression)
# =============================================================================
log "$DIVIDER"
log "STAGE 1 – Scala: Parse Logs"
log "$DIVIDER"
submit_scala "1-ParseLogs" "com.example.bigdata.ParseLogs"

log "$DIVIDER"
log "STAGE 1 – Scala: Aggregate Logs → MongoDB"
log "$DIVIDER"
submit_scala "1-AggregateLogs" "com.example.bigdata.AggregateLogs" \
    --packages "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1"

log "$DIVIDER"
log "STAGE 1 – Scala: Feature Engineering"
log "$DIVIDER"
submit_scala "1-FeatureEngineering" "com.example.bigdata.FeatureEngineering"

log "$DIVIDER"
log "STAGE 1 – Scala: Train Linear Regression model"
log "$DIVIDER"
submit_scala "1-TrainModel" "com.example.bigdata.TrainModel"

# =============================================================================
# STAGE 2 – PYTHON JOBS (LSTM Training + Validation)
# Note: PyTorch runs on the driver node. In standalone cluster mode the driver
# is placed on a worker container. Ensure torch, scikit-learn, numpy, and
# pandas are installed on every spark-worker image, or pre-install them:
#   docker exec spark-worker   pip install torch scikit-learn pandas numpy
#   docker exec spark-worker-2 pip install torch scikit-learn pandas numpy
# =============================================================================
log "$DIVIDER"
log "STAGE 2 – Python: Train LSTM (09_train_lstm.py)"
log "$DIVIDER"
submit_python "2-TrainLSTM" "$JOBS_DIR/09_train_lstm.py"

log "$DIVIDER"
log "STAGE 2 – Python: Validate LSTM + save metrics (10_validate_lstm.py)"
log "$DIVIDER"
submit_python "2-ValidateLSTM" "$JOBS_DIR/10_validate_lstm.py"

# =============================================================================
# STAGE 3 – COMPARISON (Linear Regression vs LSTM)
# =============================================================================
log "$DIVIDER"
log "STAGE 3 – Python: Compare LR vs LSTM (12_compare_models.py)"
log "$DIVIDER"
submit_python "3-Compare" "$JOBS_DIR/12_compare_models.py"

log "$DIVIDER"
log "ALL STAGES COMPLETE. Check HDFS paths:"
log "  LR  metrics : hdfs://namenode:8020/models/metrics/lr_metrics"
log "  LSTM metrics: hdfs://namenode:8020/models/metrics/lstm_metrics"
log "  LR  model   : hdfs://namenode:8020/models/traffic_prediction_v1"
log "$DIVIDER"
