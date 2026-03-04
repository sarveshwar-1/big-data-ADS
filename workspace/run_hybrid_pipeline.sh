
set -euo pipefail

MASTER="spark://spark:7077"
JAR_PATH="/opt/spark/jobs/jars/traffic-analysis.jar"
JOBS_DIR="/opt/spark/jobs"
CONFIG_FILE="$JOBS_DIR/spark-defaults.conf"

SCALA_CONTAINER="spark"   # Spark master container – used to call spark-submit
LOCAL_JAR="$(dirname "$0")/scala-app/target/scala-2.12/TrafficAnalysis-assembly-0.1.jar"
LOCAL_JAR_DEST="$(dirname "$0")/spark/jars/traffic-analysis.jar"

DIVIDER="================================================================"
FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
    shift
fi

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Helper: check if an HDFS path already exists.  Returns 0 if it does.
# ---------------------------------------------------------------------------
hdfs_exists() {
    docker exec namenode hdfs dfs -test -e "$1" 2>/dev/null
}

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
#
# NOTE: Spark standalone does NOT support cluster deploy mode for Python.
# Python jobs run in client mode on the master container.
# ---------------------------------------------------------------------------
submit_python() {
    local STEP="$1"; shift          # human-readable step name
    local SCRIPT="$1"; shift        # path inside the container
    local EXTRA_ARGS=("$@")

    log "[$STEP] Submitting Python script (client mode): $SCRIPT"

    docker exec "$SCALA_CONTAINER" /opt/spark/bin/spark-submit \
        --master "$MASTER" \
        --deploy-mode client \
        --properties-file "$CONFIG_FILE" \
        "${EXTRA_ARGS[@]}" \
        "$SCRIPT" 2>&1 | tee /dev/stderr | tail -1 > /dev/null

    local RC=${PIPESTATUS[0]}
    if [ "$RC" -ne 0 ]; then
        die "[$STEP] spark-submit exited with code $RC"
    fi

    log "[$STEP] Completed successfully."
}

# ---------------------------------------------------------------------------
# Internal: poll Spark Master JSON API until driver reaches a terminal state.
# The REST --status endpoint is unreliable; the web-UI JSON is always available.
# ---------------------------------------------------------------------------
_poll_driver() {
    local STEP="$1"
    local DRIVER_ID="$2"
    local MAX_POLLS=120          # 120 × 5 s = 10 min timeout
    local POLLS=0

    while true; do
        local STATE
        STATE=$(docker exec "$SCALA_CONTAINER" \
            curl -sf http://localhost:8080/json/ 2>/dev/null \
            | python3 -c "
import json, sys
data = json.load(sys.stdin)
for d in data.get('completeddrivers', []) + data.get('activedrivers', []):
    if d.get('id') == '$DRIVER_ID':
        print(d.get('state', 'UNKNOWN'))
        break
" 2>/dev/null || true)

        case "$STATE" in
            FINISHED)
                log "[$STEP] Driver $DRIVER_ID finished successfully."
                return 0 ;;
            FAILED|KILLED|ERROR)
                die "[$STEP] Driver $DRIVER_ID ended with state: $STATE" ;;
            RUNNING|"")
                POLLS=$((POLLS + 1))
                if (( POLLS % 6 == 0 )); then
                    log "[$STEP] Still waiting… ($((POLLS * 5))s elapsed, state: ${STATE:-pending})"
                fi
                if (( POLLS >= MAX_POLLS )); then
                    die "[$STEP] Timed out after $((MAX_POLLS * 5))s waiting for $DRIVER_ID"
                fi
                sleep 5 ;;
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
docker exec namenode hdfs dfs -chmod -R 777 /models
docker exec namenode hdfs dfs -mkdir -p /logs/raw
docker exec namenode hdfs dfs -mkdir -p /logs/parsed
docker exec namenode hdfs dfs -mkdir -p /logs/features
docker exec namenode hdfs dfs -chmod -R 777 /logs

# Ensure Ivy cache dirs exist on workers (needed for --packages resolution)
# Containers run as 'spark' user – must use root to create dirs under /home
for w in spark-worker spark-worker-2; do
    docker exec --user root "$w" sh -c 'mkdir -p /home/spark/.ivy2/cache /home/spark/.ivy2/jars && chown -R spark:spark /home/spark' 2>/dev/null || true
done

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
if ! $FORCE && hdfs_exists /logs/parsed; then
    log "[1-ParseLogs] SKIP – /logs/parsed already exists (use --force to re-run)"
else
    submit_scala "1-ParseLogs" "com.example.bigdata.ParseLogs"
fi

log "$DIVIDER"
log "STAGE 1 – Scala: Aggregate Logs → MongoDB"
log "$DIVIDER"
# AggregateLogs writes to MongoDB only – always re-run (no cheap HDFS check)
# The mongo-spark-connector is bundled in the fat JAR (sbt assembly), no --packages needed.
submit_scala "1-AggregateLogs" "com.example.bigdata.AggregateLogs"

log "$DIVIDER"
log "STAGE 1 – Scala: Feature Engineering"
log "$DIVIDER"
if ! $FORCE && hdfs_exists /logs/features/traffic_ml_ready; then
    log "[1-FeatureEngineering] SKIP – /logs/features/traffic_ml_ready already exists (use --force to re-run)"
else
    submit_scala "1-FeatureEngineering" "com.example.bigdata.FeatureEngineering"
fi

log "$DIVIDER"
log "STAGE 1 – Scala: Train Linear Regression model"
log "$DIVIDER"
if ! $FORCE && hdfs_exists /models/metrics/lr_metrics; then
    log "[1-TrainModel] SKIP – /models/metrics/lr_metrics already exists (use --force to re-run)"
else
    submit_scala "1-TrainModel" "com.example.bigdata.TrainModel"
fi

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
# 09_train_lstm.py has no HDFS output – always re-run
submit_python "2-TrainLSTM" "$JOBS_DIR/09_train_lstm.py"

log "$DIVIDER"
log "STAGE 2 – Python: Validate LSTM + save metrics (10_validate_lstm.py)"
log "$DIVIDER"
if ! $FORCE && hdfs_exists /models/metrics/lstm_metrics; then
    log "[2-ValidateLSTM] SKIP – /models/metrics/lstm_metrics already exists (use --force to re-run)"
else
    submit_python "2-ValidateLSTM" "$JOBS_DIR/10_validate_lstm.py"
fi

# =============================================================================
# STAGE 2b – PYTHON JOB (SARIMA Training + Validation)
# Note: statsmodels runs on the driver node. Ensure statsmodels, scikit-learn,
# numpy, and pandas are installed on every spark-worker image:
#   docker exec spark-worker   pip install statsmodels scikit-learn pandas numpy
#   docker exec spark-worker-2 pip install statsmodels scikit-learn pandas numpy
# =============================================================================
log "$DIVIDER"
log "STAGE 2b – Python: Train & Validate SARIMA (13_train_sarima.py)"
log "$DIVIDER"
if ! $FORCE && hdfs_exists /models/metrics/sarima_metrics; then
    log "[2b-TrainSARIMA] SKIP – /models/metrics/sarima_metrics already exists (use --force to re-run)"
else
    submit_python "2b-TrainSARIMA" "$JOBS_DIR/13_train_sarima.py"
fi

# =============================================================================
# STAGE 2c – PYTHON JOB (Hybrid SARIMA-LSTM Training + Validation)
# Note: Requires both torch and statsmodels on driver. Uses Sequential
# Residual Architecture: SARIMA baseline + LSTM residual correction.
# =============================================================================
log "$DIVIDER"
log "STAGE 2c – Python: Train & Validate Hybrid SARIMA-LSTM (14_train_hybrid.py)"
log "$DIVIDER"
if ! $FORCE && hdfs_exists /models/metrics/hybrid_metrics; then
    log "[2c-TrainHybrid] SKIP – /models/metrics/hybrid_metrics already exists (use --force to re-run)"
else
    submit_python "2c-TrainHybrid" "$JOBS_DIR/14_train_hybrid.py"
fi

# =============================================================================
# STAGE 3 – COMPARISON (Linear Regression vs LSTM vs SARIMA vs Hybrid)
# =============================================================================
log "$DIVIDER"
log "STAGE 3 – Python: Compare LR vs LSTM vs SARIMA vs Hybrid (12_compare_models.py)"
log "$DIVIDER"
submit_python "3-Compare" "$JOBS_DIR/12_compare_models.py"

log "$DIVIDER"
log "ALL STAGES COMPLETE. Check HDFS paths:"
log "  LR     metrics : hdfs://namenode:8020/models/metrics/lr_metrics"
log "  LSTM   metrics : hdfs://namenode:8020/models/metrics/lstm_metrics"
log "  SARIMA metrics : hdfs://namenode:8020/models/metrics/sarima_metrics"
log "  Hybrid metrics : hdfs://namenode:8020/models/metrics/hybrid_metrics"
log "  LR     model   : hdfs://namenode:8020/models/traffic_prediction_v1"
log "$DIVIDER"
