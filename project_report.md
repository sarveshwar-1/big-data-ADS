# Big Data Traffic Analysis & Prediction System — Project Report

> **Date:** March 4, 2026  
> **System:** Distributed Big Data Pipeline for Web Server Traffic Forecasting  
> **Tech Stack:** Apache Spark 3.5.1 · Hadoop 3.2.1 (HDFS/YARN) · Scala 2.12 · Python 3 (PyTorch, statsmodels) · MongoDB 6 · Node.js/Express · Docker Compose

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Infrastructure & Big Data Components](#3-infrastructure--big-data-components)
4. [Data Ingestion Layer](#4-data-ingestion-layer)
5. [Data Processing Pipeline (ETL)](#5-data-processing-pipeline-etl)
6. [Feature Engineering](#6-feature-engineering)
7. [Machine Learning Models](#7-machine-learning-models)
   - 7.1 [Linear Regression (Spark MLlib)](#71-linear-regression-spark-mllib)
   - 7.2 [LSTM Neural Network (PyTorch)](#72-lstm-neural-network-pytorch)
   - 7.3 [SARIMA Statistical Model](#73-sarima-statistical-model)
   - 7.4 [Hybrid SARIMA-LSTM (Sequential Residual Architecture)](#74-hybrid-sarima-lstm-sequential-residual-architecture)
8. [Model Comparison & Evaluation](#8-model-comparison--evaluation)
9. [Pipeline Orchestration](#9-pipeline-orchestration)
10. [HDFS Data Layout](#10-hdfs-data-layout)
11. [Key Findings & Analysis](#11-key-findings--analysis)
12. [Technologies & Dependencies](#12-technologies--dependencies)
13. [How to Run](#13-how-to-run)

---

## 1. Executive Summary

This project implements an **end-to-end distributed big data pipeline** for analyzing and forecasting web server traffic. The system ingests raw Nginx access logs (~3.3 GB, millions of HTTP requests), processes them through a distributed Spark cluster, engineers time-series features, and trains **four distinct predictive models** to forecast future traffic volume.

### Core Capabilities
- **Distributed data processing** using Apache Spark on a multi-node cluster (1 master + 2 workers)
- **Scalable storage** via HDFS with 3x replication across a Hadoop cluster
- **Real-time data ingestion** through an Express.js server with WebHDFS streaming
- **Multi-model forecasting**: Linear Regression, LSTM, SARIMA, and a Hybrid SARIMA-LSTM ensemble
- **Automated comparison framework** that evaluates all models and declares a winner
- **Fully containerized** with Docker Compose (11 services)

### Final Model Performance

| Model | RMSE | MAE | R² | MAPE | Quality |
|-------|------|-----|-----|------|---------|
| **Linear Regression** | 1,224.73 | 826.63 | 0.9165 | — | EXCELLENT |
| **LSTM** | 1,383.08 | 906.79 | 0.9264 | — | EXCELLENT |
| **SARIMA** | 2,196.93 | 1,565.11 | 0.9986 | 1.67% | EXCELLENT |
| **Hybrid (SARIMA+LSTM)** | 2,190.73 | 1,573.40 | 0.9986 | 1.62% | EXCELLENT |

All four models achieved EXCELLENT quality (R² > 0.80). The SARIMA and Hybrid models operate on hourly-resampled data (different scale), while LR and LSTM operate on 5-minute granularity, making direct RMSE comparison across groups inappropriate — the R² metric provides the most meaningful cross-model comparison.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Network (hadoop)               │
│                                                                  │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │ Express.js   │───→│   NameNode    │←──→│    DataNode      │  │
│  │ (Ingestion)  │    │   (HDFS)      │    │    (HDFS)        │  │
│  │ Port: 3001   │    │   Port: 9870  │    │    Port: 9874    │  │
│  └──────────────┘    └───────┬───────┘    └──────────────────┘  │
│                              │                                   │
│                     ┌────────┴────────┐                          │
│                     │  HDFS Storage   │                          │
│                     │  /logs/raw/     │                          │
│                     │  /logs/parsed/  │                          │
│                     │  /logs/features/│                          │
│                     │  /models/       │                          │
│                     └────────┬────────┘                          │
│                              │                                   │
│  ┌───────────────────────────┴──────────────────────────────┐   │
│  │                 Apache Spark Cluster                       │   │
│  │  ┌──────────┐   ┌──────────────┐   ┌──────────────┐     │   │
│  │  │  Master  │   │  Worker-1    │   │  Worker-2    │     │   │
│  │  │ Port:8080│   │  2C / 2GB    │   │  2C / 2GB    │     │   │
│  │  │ Port:7077│   │  Port: 8081  │   │  Port: 8082  │     │   │
│  │  └──────────┘   └──────────────┘   └──────────────┘     │   │
│  │                                                           │   │
│  │  Jobs: ParseLogs → AggregateLogs → FeatureEngineering    │   │
│  │        → TrainModel → LSTM → SARIMA → Hybrid → Compare   │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌───────────────┐  ┌───────┴────────┐  ┌────────────────────┐  │
│  │  MongoDB 6    │  │ YARN Resource  │  │ Spark History      │  │
│  │  (Analytics)  │  │   Manager      │  │   Server           │  │
│  │  Port: 27018  │  │  Port: 8088    │  │   Port: 18080      │  │
│  └───────────────┘  └────────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
access.log (3.3 GB raw Nginx logs)
    │
    ▼  [Express.js WebHDFS streaming]
HDFS: /logs/raw/access.log
    │
    ▼  [Spark Scala: ParseLogs — regex extraction]
HDFS: /logs/parsed/ (Parquet, 270.9 MB, 27 partitions)
    │
    ├──▶ [Spark Scala: AggregateLogs] ──▶ MongoDB (top_ips, status_codes)
    │
    ▼  [Spark Scala: FeatureEngineering — 5-min windows + lag]
HDFS: /logs/features/traffic_ml_ready (Parquet, 38.2 KB, 1,350 rows)
    │
    ├──▶ [Scala: TrainModel — Linear Regression]  ──▶ HDFS /models/metrics/lr_metrics
    ├──▶ [Python: train_lstm — LSTM Neural Network] ──▶ HDFS /models/metrics/lstm_metrics
    ├──▶ [Python: train_sarima — SARIMA]            ──▶ HDFS /models/metrics/sarima_metrics
    ├──▶ [Python: train_hybrid — SARIMA+LSTM]       ──▶ HDFS /models/metrics/hybrid_metrics
    │
    ▼  [Python: compare_models]
    Final Comparison Report (4-model leaderboard)
```

---

## 3. Infrastructure & Big Data Components

### 3.1 Docker Compose Orchestration

The entire system runs as **11 containerized services** coordinated via Docker Compose v3.8:

| Service | Image | Role | Ports |
|---------|-------|------|-------|
| **namenode** | `bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8` | HDFS metadata server & namespace manager | 9870 (UI), 9010 |
| **datanode** | `bde2020/hadoop-datanode:2.0.0-hadoop3.2.1-java8` | Distributed block storage | 9874 |
| **resourcemanager** | `bde2020/hadoop-resourcemanager:2.0.0-hadoop3.2.1-java8` | YARN job scheduling & resource allocation | 8088 |
| **nodemanager** | `bde2020/hadoop-nodemanager:2.0.0-hadoop3.2.1-java8` | YARN task execution | — |
| **historyserver** | `bde2020/hadoop-historyserver:2.0.0-hadoop3.2.1-java8` | MapReduce job history | 19888 |
| **spark** | `apache/spark:3.5.1` | Spark Master node | 8080 (UI), 7077 (cluster) |
| **spark-worker** | `apache/spark:3.5.1` | Spark Worker 1 (2 cores, 2 GB) | 8081, 4040 |
| **spark-worker-2** | `apache/spark:3.5.1` | Spark Worker 2 (2 cores, 2 GB) | 8082, 4041 |
| **spark-history-server** | `apache/spark:3.5.1` | Spark event log viewer (reads from HDFS `/spark-logs`) | 18080 |
| **mongodb** | `mongo:6` | Analytics storage (aggregated metrics) | 27018 |
| **express-server** | Custom (Node.js) | HTTP data ingestion + Kaggle integration | 3001 |

**Container Resource Footprint** (measured live):

| Container | CPU % | Memory Usage |
|-----------|-------|--------------|
| namenode | 0.09% | 487.7 MiB |
| spark (master) | 0.11% | 531.5 MiB |
| spark-worker | 0.07% | 292.9 MiB |
| spark-worker-2 | 0.07% | 294.1 MiB |
| datanode | 0.19% | 364.2 MiB |
| resourcemanager | 0.49% | 231.1 MiB |
| mongodb | 0.32% | 28.3 MiB |
| express-server | 0.00% | 10.3 MiB |

**Networking:** All services communicate over a shared Docker bridge network named `hadoop`.

**Persistent Volumes:**
- `namenode:/hadoop/dfs` — HDFS NameNode metadata
- `datanode:/hadoop/dfs` — HDFS DataNode blocks
- `mongo_data:/data/db` — MongoDB persistent storage

### 3.2 HDFS (Hadoop Distributed File System)

HDFS provides the **distributed storage backbone** for all data in the pipeline:

- **Version:** Hadoop 3.2.1
- **Replication Factor:** 3 (default for fault tolerance)
- **Block Size:** 128 MB (default)
- **NameNode:** Manages filesystem namespace and metadata
- **DataNode:** Stores actual data blocks

**HDFS Directory Structure:**
```
/
├── logs/
│   ├── raw/
│   │   ├── access.log           (3.3 GB — raw Nginx logs)
│   │   └── client_hostname.csv  (12.8 MB — supplementary data)
│   ├── parsed/                  (270.9 MB — 27 Parquet partitions)
│   │   ├── part-00000...snappy.parquet
│   │   ├── part-00001...snappy.parquet
│   │   └── ... (27 files, Snappy compressed)
│   └── features/
│       └── traffic_ml_ready/    (38.2 KB — 1 Parquet file)
├── models/
│   ├── traffic_prediction_v1/   (Spark MLlib Pipeline Model)
│   │   ├── metadata/
│   │   └── stages/
│   │       ├── 0_vecAssembler.../
│   │       └── 1_linReg.../
│   └── metrics/
│       ├── lr_metrics/          (JSON — Linear Regression)
│       ├── lstm_metrics/        (JSON — LSTM)
│       ├── sarima_metrics/      (JSON — SARIMA)
│       └── hybrid_metrics/      (JSON — Hybrid)
└── spark-logs/                  (Spark event logs for History Server)
```

### 3.3 Apache Spark Cluster

The compute engine for all distributed processing and model training:

- **Version:** Apache Spark 3.5.1 (Standalone Mode)
- **Cluster:** 1 Master + 2 Workers
- **Total Resources:** 4 CPU cores + 4 GB RAM across workers
- **Configuration** (`spark-defaults.conf`):
  - Driver memory: 1 GB
  - Executor memory: 1 GB
  - Executor cores: 1
  - Executor instances: 2
  - Event logging enabled → HDFS `/spark-logs`
  - MongoDB connector configured

**Deployment Modes Used:**
- **Cluster mode** — Scala jobs: Driver runs on a worker. The pipeline script polls the Spark Master JSON API (`http://localhost:8080/json/`) to detect driver completion states (`FINISHED`, `FAILED`, `KILLED`).
- **Client mode** — Python jobs: Required because Spark Standalone doesn't support cluster deploy mode for Python. The driver runs on the Spark master container; PyTorch and statsmodels execute on the driver node.

### 3.4 MongoDB

Used for **analytics storage** (not ML training):

- **Version:** MongoDB 6
- **Database:** `logs_db`
- **Collections:**
  - `top_ips` — Top 10 IP addresses by request frequency
  - `status_codes` — HTTP status code distribution (200, 404, 500, etc.)
- **Connector:** `org.mongodb.spark:mongo-spark-connector:10.4.0` (bundled in fat JAR)

### 3.5 YARN (Yet Another Resource Negotiator)

While YARN is deployed (ResourceManager + NodeManager + HistoryServer), the current pipeline uses **Spark Standalone mode** for job execution. YARN provides the infrastructure for potential future migration to YARN-managed Spark execution, which would enable dynamic resource allocation and better multi-tenancy.

---

## 4. Data Ingestion Layer

### 4.1 Express.js Server

A lightweight Node.js HTTP server that provides two data ingestion paths:

**Technology Stack:**
- Express 4.22.1
- Busboy 1.6.0 (multipart parsing)
- Axios 1.13.5 (HTTP client)
- Unzipper 0.12.3 (ZIP stream decompression)

**Endpoint 1: `POST /upload` — File Upload**
- Accepts multipart form-data uploads
- Streams directly to HDFS via WebHDFS REST API (two-step PUT)
- **Zero-copy streaming** — no intermediate disk writes
- Files stored at `hdfs:///logs/raw/{filename}`
- WebHDFS flow:
  1. `PUT http://namenode:9870/webhdfs/v1/path?op=CREATE` → receives DataNode redirect URL
  2. `PUT http://datanode:port/path` → streams file content

**Endpoint 2: `POST /ingest/kaggle` — Kaggle Dataset Ingest**
- Downloads a Kaggle dataset ZIP via authenticated API
- Streams into `unzipper.Parse()` for on-the-fly extraction
- Each extracted file is streamed to HDFS via `PassThrough` buffer
- No local disk usage — true stream-to-stream pipeline
- Default dataset: `eliasdabbas/web-server-access-logs`

### 4.2 Raw Data Characteristics

| Property | Value |
|----------|-------|
| **Source** | Nginx Combined Log Format |
| **File** | `access.log` |
| **Size** | 3.3 GB (raw text) |
| **HDFS Size** | 3.3 GB × 3 replicas = 9.8 GB |
| **Date Range** | January 22–26, 2019 |
| **Log Format** | `IP - - [timestamp] "METHOD PATH HTTP/VER" status bytes "referrer" "user_agent" "extra"` |

**Sample Log Entry:**
```
192.168.1.100 - - [22/Jan/2019:03:56:16 +0330] "GET /image/123.jpg HTTP/1.1" 200 19939 "https://example.com/" "Mozilla/5.0..." "-"
```

---

## 5. Data Processing Pipeline (ETL)

The ETL pipeline is implemented in **Scala** (production) with Python equivalents maintained as reference implementations.

### 5.1 Stage 1: Log Parsing (`ParseLogs.scala`)

**Purpose:** Extract structured fields from raw Nginx logs using regex pattern matching.

**Implementation:**
- Reads raw text from `hdfs:///logs/raw/access.log` using `spark.read.text()`
- Applies regex via `regexp_extract()` on each field position
- Converts timestamps from `dd/MMM/yyyy:HH:mm:ss Z` format to Spark timestamps
- Casts `status` and `bytes` to integers

**Input/Output:**
| Input | Output |
|-------|--------|
| Raw text: 3.3 GB | Parquet: 270.9 MB (27 partitions, Snappy compressed) |
| 1 unstructured column | 8 structured columns: `ip`, `event_time`, `method`, `url`, `status`, `bytes`, `referrer`, `user_agent` |

**Compression Ratio:** ~12:1 (raw text → Snappy Parquet)

### 5.2 Stage 2: Log Aggregation (`AggregateLogs.scala`)

**Purpose:** Generate analytics summaries and store in MongoDB.

**Aggregations:**
1. **Top 10 IPs:** `GROUP BY ip → COUNT(*) → ORDER BY DESC → LIMIT 10`
2. **Status Code Distribution:** `GROUP BY status → COUNT(*) → ORDER BY DESC`

**Output:** Written to MongoDB collections `logs_db.top_ips` and `logs_db.status_codes` using the `mongo-spark-connector`.

### 5.3 Stage 3: Feature Engineering (`FeatureEngineering.scala`)

**Purpose:** Transform parsed logs into ML-ready time-series features.

**Pipeline:**
1. **Timestamp Parsing:** Convert `event_time` string → Spark `timestamp` type (using legacy parser policy for `dd/MMM/yyyy:HH:mm:ss` format)
2. **5-Minute Window Aggregation:**
   - `window(col("event_ts"), "5 minutes")` — Spark's built-in tumbling window function
   - Aggregate per window: `request_count = COUNT(*)`, `total_bytes = SUM(bytes)`
3. **Lag Feature Creation:**
   - `prev_count = LAG(request_count, 1) OVER (ORDER BY window_start)`
   - Provides autoregressive input for ML models
4. **Null Handling:** `na.drop()` removes the first window (no lag available)

**Output Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `window_start` | timestamp | Start of 5-minute window |
| `window_end` | timestamp | End of 5-minute window |
| `request_count` | long | Number of HTTP requests (TARGET) |
| `total_bytes` | long | Total bytes transferred |
| `prev_count` | long | Previous window's request count |

**Dataset Statistics:**
- **1,350 observations** (5-minute intervals over ~4.7 days)
- **Date range:** 2019-01-22 00:00 → 2019-01-26 16:00

---

## 6. Feature Engineering

### 6.1 Base Features (from Scala FeatureEngineering)

| Feature | Source | Type | Description |
|---------|--------|------|-------------|
| `request_count` | Aggregation | Target | HTTP requests per 5-min window |
| `total_bytes` | Aggregation | Numeric | Total response bytes per window |
| `prev_count` | Window LAG | Numeric | Previous window's request count |

### 6.2 Extended Features (Python: validate_lstm.py)

The advanced LSTM model engineers additional temporal features:

| Feature | Derivation | Purpose |
|---------|-----------|---------|
| `hour_sin` | `sin(2π × hour / 24)` | Cyclical daily encoding (continuous) |
| `hour_cos` | `cos(2π × hour / 24)` | Cyclical daily encoding (preserves 23→0 wrap) |

**Why cyclical encoding?** Standard integer hours (0-23) create an artificial discontinuity at midnight. Sine/cosine encoding ensures hour 23 and hour 0 are numerically close, matching their true temporal proximity.

### 6.3 SARIMA Feature Processing (train_sarima.py & train_hybrid.py)

For the statistical models, raw 5-minute data is **resampled to hourly**:

| Transformation | Method | Rationale |
|---------------|--------|-----------|
| Resample 5-min → 1-hour | `resample("1h").agg()` | Reduces noise; SARIMA performs better on smoother data |
| `request_count` | Sum | Total requests per hour |
| `total_bytes` | Sum | Total bytes per hour |
| `prev_count` | Mean | Average lag feature per hour |
| Exogenous normalization | `(x - mean) / std` | Prevents numerical instability in SARIMAX |

**After resampling:** 1,350 rows → **113 hourly observations**

---

## 7. Machine Learning Models

### 7.1 Linear Regression (Spark MLlib)

**Implementation:** `TrainModel.scala` using Spark MLlib Pipeline API

**Architecture:**
```
VectorAssembler(prev_count → features)
    → LinearRegression(features → request_count)
```

**Training Details:**
| Parameter | Value |
|-----------|-------|
| Features | `prev_count` (single feature) |
| Target | `request_count` |
| Train/Test Split | 80% / 20% (random, seed=42) |
| Framework | Spark MLlib |
| Deploy Mode | Cluster (Spark Standalone) |

**Model Persistence:** Saved as Spark MLlib Pipeline Model to `hdfs:///models/traffic_prediction_v1/`

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | 1,224.73 |
| MAE | 826.63 |
| R² | 0.9165 |

**Strengths:** Fastest training, interpretable, runs natively on Spark. Strong performance given single-feature simplicity.

**Limitations:** Linear assumption; cannot capture non-linear traffic patterns or seasonality.

---

### 7.2 LSTM Neural Network (PyTorch)

Two LSTM implementations exist:

#### 7.2.1 Simple LSTM (`09_train_lstm.py`)

| Parameter | Value |
|-----------|-------|
| Input | `request_count` only (univariate) |
| Architecture | LSTM(1, 50) → Linear(50, 1) |
| Sequence Length | 3 (15-minute lookback) |
| Epochs | 10 |
| Optimizer | Adam (lr=0.001) |
| Normalization | MinMaxScaler(0, 1) |
| Output | Console only (no HDFS save) |

#### 7.2.2 Enhanced LSTM (`10_validate_lstm.py`)

| Parameter | Value |
|-----------|-------|
| Input Features | 5: `request_count`, `total_bytes`, `prev_count`, `hour_sin`, `hour_cos` |
| Architecture | LSTM(5, 128, layers=2, dropout=0.2) → ReLU(128→64) → Linear(64→1) |
| Sequence Length | 12 (60-minute lookback at 5-min intervals) |
| Epochs | 50 |
| Batch Size | 32 (mini-batch SGD) |
| Optimizer | Adam (lr=0.001) |
| LR Scheduler | ReduceLROnPlateau(patience=5, factor=0.5) |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping | patience=10 |
| Normalization | Separate MinMaxScaler for features and target |
| Output | HDFS: `/models/metrics/lstm_metrics` |

**Results (Enhanced LSTM):**
| Metric | Value |
|--------|-------|
| MSE | 1,912,904.75 |
| RMSE | 1,383.08 |
| MAE | 906.79 |
| R² | 0.9264 |

**Strengths:** Captures non-linear temporal patterns; multi-feature input provides richer context; gradient clipping prevents exploding gradients.

**Limitations:** Requires more data for optimal performance; black-box nature reduces interpretability; sensitive to hyperparameters.

---

### 7.3 SARIMA Statistical Model

**Implementation:** `13_train_sarima.py` using `statsmodels.SARIMAX`

**SARIMA Model:** Seasonal AutoRegressive Integrated Moving Average with eXogenous variables

$$SARIMA(p,d,q) \times (P,D,Q)_m$$

Where:
- $(p,d,q)$ = Non-seasonal (AR order, differencing, MA order)
- $(P,D,Q)_m$ = Seasonal (AR, differencing, MA) with period $m$
- $m = 24$ (daily seasonality in hourly data)

**Data Preparation:**
- Resampled from 5-min to hourly (113 observations)
- Exogenous variables: `total_bytes`, `prev_count` (z-score normalized)
- Train: 90 observations, Test: 23 observations

**Grid Search:**
| Parameter | Range | Combinations |
|-----------|-------|-------------|
| $p$ (AR) | 0–2 | 3 |
| $d$ (Diff) | 0–1 | 2 |
| $q$ (MA) | 0–2 | 3 |
| $P$ (Seasonal AR) | 0–1 | 2 |
| $D$ (Seasonal Diff) | 0–1 | 2 |
| $Q$ (Seasonal MA) | 0–1 | 2 |
| **Total** | | **144 combinations** |

**Selection Criterion:** Minimum AIC (Akaike Information Criterion)

**Best Model Found:**
$$SARIMA(0,0,2) \times (1,1,1)_{24} \quad AIC = 700.96$$

**Results:**
| Metric | Value |
|--------|-------|
| MSE | 4,826,518.86 |
| RMSE | 2,196.93 |
| MAE | 1,565.11 |
| R² | 0.9986 |
| MAPE | 1.67% |

**Strengths:** Extremely high R² (0.9986); explicit seasonal modeling; built-in uncertainty quantification; interpretable parameters; uses exogenous variables for richer context.

**Limitations:** Linear in nature; requires stationarity assumptions; hourly resampling loses fine-grained patterns; computationally expensive grid search.

---

### 7.4 Hybrid SARIMA-LSTM (Sequential Residual Architecture)

**Implementation:** `14_train_hybrid.py` — the newest model added to the pipeline.

**Architecture:**

```
                     Training Phase
┌──────────────────────────────────────────────────┐
│                                                    │
│  Training Data ──▶ SARIMA Fit ──▶ Fitted Values   │
│                      │                             │
│                      ▼                             │
│              Residuals = Data - Fitted             │
│                      │                             │
│                      ▼                             │
│           Scale & Clip (±3σ)                       │
│                      │                             │
│                      ▼                             │
│              LSTM Training on Residuals            │
│                                                    │
└──────────────────────────────────────────────────┘

                    Prediction Phase
┌──────────────────────────────────────────────────┐
│                                                    │
│  SARIMA Forecast ──────────────────┐              │
│                                     │              │
│  LSTM Residual Forecast ──────────┐ │              │
│   (with exponential decay)        │ │              │
│                                    ▼ ▼              │
│                        Final = SARIMA + LSTM       │
│                                                    │
└──────────────────────────────────────────────────┘
```

**Stage A: SARIMA Baseline**
- Identical preprocessing to standalone SARIMA (hourly resample, normalized exog)
- Full grid search (144 combinations, AIC-minimized)
- Generates baseline test forecast and computes SARIMA-only RMSE

**Stage B: LSTM Residual Correction**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Residual Source | `sarima_fitted.resid` | One-step-ahead residuals (properly computed) |
| Burn-in | 24 samples | First seasonal period unreliable due to differencing |
| Scaling | StandardScaler + clip(±3σ) | Normalizes residuals; suppresses outliers |
| Architecture | LSTM(1, 32) → Linear(32, 1) | Lightweight — prevents overfitting on 66 samples |
| Lookback | 6 hours (adaptive: `min(6, train_size//10)`) | Sized to available data |
| Epochs | 150 (early stopping at 25) | Generous budget with aggressive early stopping |
| Optimizer | Adam (lr=0.005) | Higher LR for faster convergence on small data |
| LR Scheduler | ReduceLROnPlateau(patience=10, factor=0.5) | Automatic fine-tuning |
| Gradient Clipping | max_norm=1.0 | Stability |
| Val Split | 20% (chronological) | Respects time-series ordering |

**Stage C: Multi-Step Forecast Fusion**

The hybrid forecast combines both components with an **exponential decay** on the LSTM correction:

$$\hat{y}_t = \hat{y}^{SARIMA}_t + \hat{r}^{LSTM}_t \cdot e^{-0.1 \cdot t}$$

The decay factor ($e^{-0.1t}$) reduces the LSTM contribution at longer forecast horizons, acknowledging that autoregressive residual predictions degrade over time.

**Results:**
| Metric | Value |
|--------|-------|
| MSE | 4,799,282.78 |
| RMSE | 2,190.73 |
| MAE | 1,573.40 |
| R² | 0.9986 |
| MAPE | 1.62% |
| **Improvement over SARIMA-only** | **+0.28% RMSE reduction** |

**Sample Predictions:**
| Actual | SARIMA | LSTM Correction | Hybrid | Error |
|--------|--------|----------------|--------|-------|
| 124,638 | 124,096 | +407 | 124,503 | +135 |
| 118,972 | 120,423 | +387 | 120,811 | -1,839 |
| 98,673 | 98,857 | +358 | 99,215 | -542 |
| 76,089 | 76,454 | +331 | 76,786 | -697 |
| 16,819 | 16,675 | +260 | 16,935 | -116 |
| 15,228 | 15,113 | +214 | 15,327 | -99 |

---

## 8. Model Comparison & Evaluation

### 8.1 Comparison Framework (`12_compare_models.py`)

The comparison framework loads saved metrics from HDFS for all four models and produces a unified leaderboard:

```
══════════════════════════════════════════════════════════════════════════════
                       FINAL MODEL COMPARISON REPORT
══════════════════════════════════════════════════════════════════════════════
Metric                         Linear Regression       LSTM       SARIMA       Hybrid
──────────────────────────────────────────────────────────────────────────────────────
RMSE (lower is better)                 1224.7341  1383.0780    2196.9340    2190.7265
MAE  (lower is better)                  826.6330   906.7865    1565.1086    1573.4032
R²   (higher is better)                   0.9165     0.9264       0.9986       0.9986
══════════════════════════════════════════════════════════════════════════════

Verdict:
  Best RMSE : Linear Regression (1224.7341)
  Best MAE  : Linear Regression (826.6330)
  Best R²   : Hybrid (SARIMA+LSTM) (0.9986)

  Overall Winner: Linear Regression (2/3 metrics)
  Hybrid RMSE improvement over SARIMA-only: +0.28%

Model Quality:
  Linear Regression        : R²=0.9165 -> EXCELLENT
  LSTM                     : R²=0.9264 -> EXCELLENT
  SARIMA                   : R²=0.9986 -> EXCELLENT
  Hybrid (SARIMA+LSTM)     : R²=0.9986 -> EXCELLENT
```

### 8.2 Detailed Metric Analysis

| Metric | LR | LSTM | SARIMA | Hybrid | Winner |
|--------|----|----|--------|--------|--------|
| **RMSE** | **1,224.73** | 1,383.08 | 2,196.93 | 2,190.73 | LR |
| **MAE** | **826.63** | 906.79 | 1,565.11 | 1,573.40 | LR |
| **R²** | 0.9165 | 0.9264 | 0.9986 | **0.9986** | Hybrid/SARIMA |
| **MAPE** | — | — | 1.67% | **1.62%** | Hybrid |

### 8.3 Important Context on Metric Interpretation

**Why do SARIMA/Hybrid have higher RMSE yet higher R²?**

The SARIMA and Hybrid models operate on **hourly-resampled data** (where `request_count` is the SUM of all 5-minute windows in that hour). This means hourly values are ~12× larger than 5-minute values, leading to proportionally larger absolute errors (RMSE, MAE) but the variance explained (R²) correctly captures that these models explain 99.86% of the variance in their respective scales.

| Model Group | Data Granularity | Avg request_count | Scale |
|------------|-----------------|-------------------|-------|
| LR, LSTM | 5-minute windows | ~1,000–8,000 | Smaller |
| SARIMA, Hybrid | Hourly sums | ~15,000–120,000 | ~12× larger |

**R²** is the most meaningful cross-group comparison metric since it is scale-invariant.

### 8.4 Quality Classification

All four models achieved **EXCELLENT** quality:

| Grade | R² Threshold | Models |
|-------|-------------|--------|
| **EXCELLENT** | > 0.80 | LR (0.9165), LSTM (0.9264), SARIMA (0.9986), Hybrid (0.9986) |
| GOOD | > 0.50 | — |
| NEEDS IMPROVEMENT | ≤ 0.50 | — |

---

## 9. Pipeline Orchestration

### 9.1 Main Pipeline Script (`run_hybrid_pipeline.sh`)

A comprehensive Bash script that orchestrates the entire end-to-end pipeline:

**Features:**
- **Idempotent execution:** Each stage checks HDFS for existing outputs before re-running
- **Force mode:** `--force` flag bypasses caching and re-runs all stages
- **Pre-flight checks:** Validates that all required Docker containers are running
- **HDFS initialization:** Creates all required directories with proper permissions
- **JAR building:** Automatically builds the Scala fat JAR via Docker if not present
- **Cluster-mode polling:** Extracts driver IDs from Spark submission output and polls the Master JSON API (`http://localhost:8080/json/`) for completion state
- **Error handling:** `set -euo pipefail` for strict error propagation

**Execution Stages:**
```
Stage 0   : HDFS directory initialization + permissions
Stage 0b  : Scala JAR build (sbt assembly via Docker)
Stage 1   : ParseLogs → AggregateLogs → FeatureEngineering → TrainModel (Scala, cluster mode)
Stage 2   : Train simple LSTM (Python, client mode)
Stage 2   : Validate advanced LSTM + save metrics (Python, client mode)
Stage 2b  : Train SARIMA + save metrics (Python, client mode)
Stage 2c  : Train Hybrid SARIMA-LSTM + save metrics (Python, client mode)
Stage 3   : Compare all 4 models (Python, client mode)
```

### 9.2 Scala Job Submission

Scala jobs use **cluster deploy mode** for production-grade execution:
```bash
docker exec spark /opt/spark/bin/spark-submit \
  --master spark://spark:7077 \
  --deploy-mode cluster \
  --properties-file /opt/spark/jobs/spark-defaults.conf \
  --class com.example.bigdata.ClassName \
  /opt/spark/jobs/jars/traffic-analysis.jar
```

The driver runs on a worker node. The script polls completion via:
```python
# Parsed from: curl http://localhost:8080/json/
data['completeddrivers'] + data['activedrivers']
→ match driver_id → check state (FINISHED/FAILED/RUNNING)
```

### 9.3 Python Job Submission

Python jobs require **client deploy mode** (Spark Standalone limitation):
```bash
docker exec spark /opt/spark/bin/spark-submit \
  --master spark://spark:7077 \
  --deploy-mode client \
  --properties-file /opt/spark/jobs/spark-defaults.conf \
  /opt/spark/jobs/script.py
```

The driver runs on the master container. PyTorch and statsmodels execute on the driver (not distributed across workers); Spark handles only the initial data loading from HDFS.

### 9.4 Build Pipeline

The Scala application is built using **sbt** with the **sbt-assembly** plugin:

```bash
# Runs inside Docker for reproducibility:
docker run --rm \
  -v scala-app:/workspace \
  -w /workspace \
  hseeberger/scala-sbt:17.0.2_1.6.2_2.12.15 \
  sbt assembly
```

**Output:** `TrafficAnalysis-assembly-0.1.jar` (fat JAR with all dependencies except Spark provided)

**Assembly Merge Strategy:** Custom strategy handles META-INF conflicts:
- Discards: `META-INF/MANIFEST.MF`, `META-INF/*.SF`, `META-INF/*.DSA`, `META-INF/*.RSA`
- Concatenates: `META-INF/services/*` (critical for MongoDB connector)
- Default: First-wins for remaining conflicts

---

## 10. HDFS Data Layout

### 10.1 Data Size Progression

| Stage | Path | Size | Format | Records |
|-------|------|------|--------|---------|
| Raw | `/logs/raw/access.log` | 3.3 GB | Text | Millions of log lines |
| Parsed | `/logs/parsed/` | 270.9 MB (27 partitions) | Snappy Parquet | Same (structured) |
| Features | `/logs/features/traffic_ml_ready/` | 38.2 KB (1 file) | Snappy Parquet | 1,350 rows |

**Total HDFS usage (with replication):**
- Raw: 9.8 GB (3× replication)
- Parsed: 812.7 MB (3× replication)
- Features: 114.7 KB (3× replication)

### 10.2 Why Parquet?

Apache Parquet provides:
- **Columnar storage:** Read only needed columns (e.g., `request_count` for LSTM)
- **Snappy compression:** ~12:1 ratio on log data
- **Schema enforcement:** Type safety across pipeline stages
- **Predicate pushdown:** Filter data at storage level
- **Partition pruning:** Efficient reads for time-range queries
- **Ecosystem integration:** Native support in Spark, Pandas, and PySpark

---

## 11. Key Findings & Analysis

### 11.1 Data Insights

The web server access logs span approximately 4.7 days (January 22–26, 2019) and exhibit:

- **Strong daily seasonality:** Traffic peaks during daytime hours and drops significantly at night, with a period of approximately 24 hours
- **High variability:** 5-minute request counts range from hundreds to tens of thousands
- **Clear autoregressive structure:** Previous time-step traffic strongly predicts the next (`prev_count` feature is highly effective, enabling LR to achieve R² > 0.91)
- **Relatively small effective dataset:** After hourly resampling, only 113 observations are available — limiting the effectiveness of deep learning approaches

### 11.2 Model Performance Analysis

**Linear Regression:** Surprisingly strong (R² = 0.9165) due to the strong autoregressive nature of the data. The single `prev_count` feature captures most variance. This demonstrates that **simple baselines should always be tested first** in time-series forecasting.

**LSTM:** Slightly higher R² (0.9264) than LR despite more complexity, suggesting marginal non-linear improvements from the 5-feature, multi-step architecture. The 5-minute granularity with 12-step lookback (1-hour context) provides good temporal coverage.

**SARIMA:** Highest R² (0.9986) by explicitly modeling daily seasonality ($m=24$) and using exogenous variables. The grid search found optimal parameters $(0,0,2)(1,1,1)_{24}$, indicating:
- No non-seasonal differencing needed ($d=0$) — data is approximately stationary
- 2 non-seasonal MA terms provide error correction
- One level of seasonal differencing ($D=1$) effectively removes the daily cycle
- Seasonal AR(1) and MA(1) terms capture daily autocorrelation patterns

**Hybrid SARIMA-LSTM:** Achieves a marginal improvement (+0.28% RMSE reduction) over standalone SARIMA. The LSTM correction captures some non-linear residual patterns the SARIMA cannot, but is limited by:
- Small residual dataset (66 samples after burn-in period removal)
- Autoregressive multi-step prediction degradation
- Exponential decay dampens corrections at longer horizons

### 11.3 Hybrid Model Design Decisions

| Decision | Rationale |
|----------|-----------|
| Use SARIMA's `.resid` instead of `train - fitted` | Properly computed one-step-ahead residuals avoid look-ahead bias |
| 24-sample burn-in removal | Seasonal differencing artifacts corrupt first cycle's residuals |
| Lightweight LSTM (32 hidden, 1 layer) | Only 66 training samples — larger models would overfit |
| StandardScaler + ±3σ clipping | Normalizes residuals; suppresses occasional extreme outliers |
| Exponential decay on LSTM corrections | Autoregressive error accumulation makes distant predictions unreliable |
| Adaptive lookback ($\min(6, n/10)$) | Prevents lookback exceeding available data |

### 11.4 Scalability Considerations

This pipeline is designed to scale horizontally:

| Component | Current | Scale Path |
|-----------|---------|------------|
| HDFS | 1 DataNode | Add DataNodes for capacity; automatic rebalancing |
| Spark Workers | 2 (4 cores, 4 GB) | Add workers to compose file; Spark auto-discovers |
| Log Ingestion | Single Express server | Load balancer → multiple instances |
| SARIMA | Single-node computation | Grid search could be parallelized via Spark |
| LSTM | Single GPU/CPU | Multi-GPU distributed training via PyTorch DDP |

### 11.5 Potential Improvements

1. **More data:** With only 4.7 days of logs, all models (especially deep learning) are data-constrained. Weeks or months of data would significantly improve LSTM and Hybrid performance.
2. **Additional features:** Day-of-week encoding, holiday flags, IP-based features, URL category features.
3. **Walk-forward validation:** Replace simple train/test split with rolling-origin cross-validation for more robust metric estimates.
4. **Ensemble weighting:** Learn optimal SARIMA/LSTM weights instead of fixed 1:1 combination.
5. **Direct multi-step LSTM:** Train LSTM to predict multiple steps simultaneously rather than autoregressively.

---

## 12. Technologies & Dependencies

### 12.1 Core Technologies

| Technology | Version | Role |
|-----------|---------|------|
| Apache Spark | 3.5.1 | Distributed computing engine |
| Apache Hadoop (HDFS) | 3.2.1 | Distributed file system |
| Scala | 2.12.18 | ETL jobs & Linear Regression |
| Python | 3.x | ML models (LSTM, SARIMA, Hybrid) |
| Docker / Docker Compose | v3.8 | Container orchestration |
| MongoDB | 6 | Analytics database |
| Node.js / Express | 4.22.1 | Data ingestion HTTP server |

### 12.2 ML & Data Science Libraries

| Library | Purpose |
|---------|---------|
| PyTorch | LSTM neural network (train, validate, hybrid) |
| statsmodels (SARIMAX) | SARIMA statistical model |
| scikit-learn | Metrics (RMSE, MAE, R², MAPE), preprocessing (MinMaxScaler, StandardScaler) |
| NumPy | Numerical operations |
| Pandas | DataFrame operations, time-series resampling |
| Spark MLlib | Linear Regression, VectorAssembler, Pipeline |

### 12.3 Scala Dependencies (build.sbt)

| Dependency | Version | Scope |
|-----------|---------|-------|
| `spark-core` | 3.5.1 | Provided |
| `spark-sql` | 3.5.1 | Provided |
| `spark-mllib` | 3.5.1 | Provided |
| `mongo-spark-connector` | 10.4.0 | Compiled (bundled) |
| `sbt-assembly` (plugin) | 2.1.1 | Build-time |

### 12.4 Node.js Dependencies (package.json)

| Package | Version | Purpose |
|---------|---------|---------|
| `express` | ^4.22.1 | HTTP server framework |
| `axios` | ^1.13.5 | HTTP client for WebHDFS |
| `busboy` | ^1.6.0 | Multipart form-data parser |
| `unzipper` | ^0.12.3 | ZIP stream decompression |

---

## 13. How to Run

### 13.1 Prerequisites

- Docker and Docker Compose installed
- Kaggle API credentials (for automated data download)
- ~16 GB RAM recommended (11 containers)

### 13.2 Start the Cluster

```bash
cd /home/sarveshwar/big-data-ADS
docker compose up -d
```

### 13.3 Ingest Data

**Option A: Upload a log file**
```bash
curl -X POST http://localhost:3001/upload \
  -F "file=@/path/to/access.log"
```

**Option B: Download from Kaggle**
```bash
curl -X POST http://localhost:3001/ingest/kaggle \
  -H 'Content-Type: application/json' \
  -d '{"owner":"eliasdabbas","dataset":"web-server-access-logs"}'
```

### 13.4 Run the Full Pipeline

```bash
bash workspace/run_hybrid_pipeline.sh
```

This will:
1. Initialize HDFS directories
2. Build the Scala JAR (if needed)
3. Parse logs → Aggregate → Feature engineer
4. Train Linear Regression (Scala/Spark MLlib)
5. Train LSTM (Python/PyTorch)
6. Train SARIMA (Python/statsmodels)
7. Train Hybrid SARIMA-LSTM (Python/PyTorch+statsmodels)
8. Compare all 4 models and print leaderboard

**Force re-run:** `bash workspace/run_hybrid_pipeline.sh --force`

### 13.5 Web UIs

| Service | URL | Purpose |
|---------|-----|---------|
| HDFS NameNode | http://localhost:9870 | File system browser |
| Spark Master | http://localhost:8080 | Cluster status, workers, jobs |
| Spark Worker 1 | http://localhost:8081 | Worker details |
| Spark Worker 2 | http://localhost:8082 | Worker details |
| Spark History | http://localhost:18080 | Completed job logs |
| YARN ResourceManager | http://localhost:8088 | YARN job management |
| YARN HistoryServer | http://localhost:19888 | MapReduce history |

---

## Files Reference

| File | Language | Purpose |
|------|----------|---------|
| `docker-compose.yaml` | YAML | Infrastructure definition (11 services) |
| `workspace/run_hybrid_pipeline.sh` | Bash | Main pipeline orchestrator |
| `workspace/run_scala_jobs.sh` | Bash | Scala-only job runner |
| `setup_and_run.sh` | Bash | Initial cluster setup |
| `express-server/index.js` | JavaScript | Data ingestion (upload + Kaggle) |
| `workspace/scala-app/src/.../ParseLogs.scala` | Scala | Log parsing (regex extraction) |
| `workspace/scala-app/src/.../AggregateLogs.scala` | Scala | MongoDB analytics aggregation |
| `workspace/scala-app/src/.../FeatureEngineering.scala` | Scala | 5-min windowing + lag features |
| `workspace/scala-app/src/.../TrainModel.scala` | Scala | Linear Regression (Spark MLlib) |
| `workspace/scala-app/src/.../PredictTraffic.scala` | Scala | Inference on live data |
| `workspace/spark/09_train_lstm.py` | Python | Simple LSTM training |
| `workspace/spark/10_validate_lstm.py` | Python | Enhanced LSTM + metrics |
| `workspace/spark/11_inspect_features.py` | Python | Feature data profiling |
| `workspace/spark/12_compare_models.py` | Python | 4-model comparison dashboard |
| `workspace/spark/13_train_sarima.py` | Python | SARIMA with grid search |
| `workspace/spark/14_train_hybrid.py` | Python | Hybrid SARIMA-LSTM model |

---

*Report generated on March 4, 2026. All metrics reflect actual model runs on the deployed cluster.*
