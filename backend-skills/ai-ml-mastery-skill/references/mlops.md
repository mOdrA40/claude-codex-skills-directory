# MLOps Reference

## Table of Contents
1. [Experiment Tracking](#experiment-tracking)
2. [Model Registry](#model-registry)
3. [Docker for ML](#docker-for-ml)
4. [Model Serving](#model-serving)
5. [Monitoring & Observability](#monitoring--observability)
6. [CI/CD for ML](#cicd-for-ml)
7. [Data Versioning](#data-versioning)

---

## Experiment Tracking

### MLflow

```python
import mlflow
from mlflow.tracking import MlflowClient

# Setup
mlflow.set_tracking_uri("http://localhost:5000")  # or "sqlite:///mlflow.db"
mlflow.set_experiment("my_experiment")

# Log experiment
with mlflow.start_run(run_name="xgboost_baseline") as run:
    # Log parameters
    mlflow.log_params({
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 100,
    })
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    metrics = evaluate(model, X_test, y_test)
    mlflow.log_metrics(metrics)
    
    # Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="churn_predictor",
    )
    
    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
    mlflow.log_dict(config, "config.json")
    
    # Log custom tags
    mlflow.set_tags({
        "model_type": "xgboost",
        "data_version": "v1.2",
    })

# Query runs
client = MlflowClient()
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.f1 > 0.8",
    order_by=["metrics.f1 DESC"],
    max_results=10,
)
```

### Weights & Biases

```python
import wandb

# Initialize
wandb.init(
    project="my_project",
    name="experiment_001",
    config={
        "learning_rate": 0.05,
        "epochs": 100,
        "batch_size": 32,
    },
)

# Log during training
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    })

# Log model
wandb.save("model.pt")

# Log artifacts
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pt")
wandb.log_artifact(artifact)

# Finish
wandb.finish()
```

### Experiment Decorator

```python
import functools
import mlflow
import time
from typing import Callable, Any

def track_experiment(
    experiment_name: str,
    params: dict | None = None,
) -> Callable:
    """Decorator to track ML experiments."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run():
                # Log parameters
                if params:
                    mlflow.log_params(params)
                mlflow.log_params(kwargs)
                
                # Track time
                start_time = time.time()
                
                # Run function
                result = func(*args, **kwargs)
                
                # Log duration
                duration = time.time() - start_time
                mlflow.log_metric("duration_seconds", duration)
                
                # Log results if dict
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
                
                return result
        return wrapper
    return decorator

# Usage
@track_experiment("classification", params={"model": "xgboost"})
def train_model(learning_rate: float = 0.05, max_depth: int = 6):
    # Training code
    return {"accuracy": 0.95, "f1": 0.92}
```

---

## Model Registry

### MLflow Model Registry

```python
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

client = MlflowClient()

# Register model
model_uri = f"runs:/{run_id}/model"
model_version = mlflow.register_model(
    model_uri=model_uri,
    name="churn_predictor",
)

# Transition to production
client.transition_model_version_stage(
    name="churn_predictor",
    version=model_version.version,
    stage="Production",
)

# Load production model
model = mlflow.pyfunc.load_model(
    model_uri="models:/churn_predictor/Production"
)

# Get model info
model_info = client.get_registered_model("churn_predictor")
for version in model_info.latest_versions:
    print(f"Version {version.version}: {version.current_stage}")
```

### Model Signature

```python
from mlflow.models import infer_signature
import pandas as pd

# Infer signature from data
signature = infer_signature(X_train, model.predict(X_train))

# Or define manually
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([
    ColSpec("double", "feature1"),
    ColSpec("double", "feature2"),
    ColSpec("string", "category"),
])

output_schema = Schema([ColSpec("double", "prediction")])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log with signature
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    signature=signature,
    input_example=X_train[:5],
)
```

---

## Docker for ML

### Dockerfile for ML Training

```dockerfile
# training.Dockerfile
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install Python dependencies
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[train]"

# Copy source code
COPY src/ src/
COPY configs/ configs/

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.train"]
```

### Dockerfile for Model Serving

```dockerfile
# serving.Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy model and code
COPY model/ model/
COPY src/inference/ src/inference/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for ML Stack

```yaml
# docker-compose.yml
version: "3.8"

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.0
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@db:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts/
    depends_on:
      - db
    command: mlflow server --host 0.0.0.0

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflow
      - POSTGRES_DB=mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

  training:
    build:
      context: .
      dockerfile: training.Dockerfile
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  serving:
    build:
      context: .
      dockerfile: serving.Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model
    volumes:
      - ./model:/app/model:ro
    deploy:
      replicas: 2

volumes:
  postgres_data:
```

---

## Model Serving

### FastAPI Model Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

# Request/Response schemas
class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)
    
    model_config = {"json_schema_extra": {"example": {"features": [1.0, 2.0, 3.0]}}}


class PredictionResponse(BaseModel):
    prediction: float
    probability: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Model loading
class ModelManager:
    def __init__(self):
        self.model = None
    
    def load(self, path: str):
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
    
    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        prediction = self.model.predict(features)
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[:, 1]
        return prediction, proba


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    model_manager.load("model/model.joblib")
    yield
    # Shutdown
    logger.info("Shutting down")


app = FastAPI(title="ML Model API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction, probability = model_manager.predict(features)
        
        return PredictionResponse(
            prediction=float(prediction[0]),
            probability=float(probability[0]) if probability is not None else None,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(requests: list[PredictionRequest]):
    features = np.array([r.features for r in requests])
    predictions, probabilities = model_manager.predict(features)
    
    return [
        PredictionResponse(
            prediction=float(pred),
            probability=float(prob) if probabilities is not None else None,
        )
        for pred, prob in zip(
            predictions,
            probabilities if probabilities is not None else [None] * len(predictions),
        )
    ]
```

### TorchServe Configuration

```python
# model_handler.py
import torch
from ts.torch_handler.base_handler import BaseHandler
import json

class ModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_dir = ctx.system_properties.get("model_dir")
        
        self.model = torch.jit.load(f"{model_dir}/model.pt")
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.initialized = True
    
    def preprocess(self, data):
        inputs = []
        for row in data:
            input_data = row.get("data") or row.get("body")
            if isinstance(input_data, (bytes, bytearray)):
                input_data = json.loads(input_data.decode("utf-8"))
            inputs.append(torch.tensor(input_data["features"]))
        
        return torch.stack(inputs).to(self.device)
    
    def inference(self, data):
        with torch.no_grad():
            return self.model(data)
    
    def postprocess(self, data):
        return data.cpu().numpy().tolist()
```

---

## Monitoring & Observability

### Model Monitoring with Evidently

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

def create_monitoring_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_col: str,
    prediction_col: str,
) -> Report:
    """Generate model monitoring report."""
    column_mapping = ColumnMapping(
        target=target_col,
        prediction=prediction_col,
        numerical_features=reference_data.select_dtypes(include=np.number).columns.tolist(),
        categorical_features=reference_data.select_dtypes(include="object").columns.tolist(),
    )
    
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    
    return report


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.05,
) -> dict:
    """Detect data drift using statistical tests."""
    from scipy import stats
    
    drift_results = {}
    
    for col in reference_data.columns:
        if reference_data[col].dtype in [np.float64, np.int64]:
            # KS test for numerical
            stat, p_value = stats.ks_2samp(reference_data[col], current_data[col])
            drift_results[col] = {
                "test": "ks",
                "statistic": stat,
                "p_value": p_value,
                "drift_detected": p_value < threshold,
            }
        else:
            # Chi-square for categorical
            ref_counts = reference_data[col].value_counts()
            cur_counts = current_data[col].value_counts()
            
            all_categories = set(ref_counts.index) | set(cur_counts.index)
            ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
            cur_freq = [cur_counts.get(cat, 0) for cat in all_categories]
            
            stat, p_value = stats.chisquare(cur_freq, ref_freq)
            drift_results[col] = {
                "test": "chi2",
                "statistic": stat,
                "p_value": p_value,
                "drift_detected": p_value < threshold,
            }
    
    return drift_results
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total predictions made",
    ["model_name", "model_version"],
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_name"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_VALUE = Histogram(
    "model_prediction_value",
    "Distribution of prediction values",
    ["model_name"],
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether model is loaded",
    ["model_name", "model_version"],
)


def track_prediction(model_name: str, model_version: str):
    """Decorator to track predictions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            
            result = func(*args, **kwargs)
            
            # Record metrics
            PREDICTION_COUNTER.labels(model_name, model_version).inc()
            PREDICTION_LATENCY.labels(model_name).observe(time.time() - start)
            
            if isinstance(result, (int, float)):
                PREDICTION_VALUE.labels(model_name).observe(result)
            
            return result
        return wrapper
    return decorator


# Start metrics server
start_http_server(8001)
```

---

## CI/CD for ML

### GitHub Actions Workflow

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests
        run: pytest tests/ -v --cov=src
      
      - name: Lint
        run: ruff check src/

  train:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Train model
        run: |
          python -m src.train \
            --config configs/production.yaml \
            --mlflow-tracking-uri ${{ secrets.MLFLOW_URI }}
      
      - name: Register model
        run: |
          python -m src.register_model \
            --model-name my_model \
            --stage Production

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster ml-cluster \
            --service model-serving \
            --force-new-deployment
```

---

## Data Versioning

### DVC (Data Version Control)

```bash
# Initialize
dvc init

# Track data
dvc add data/raw/train.csv
git add data/raw/train.csv.dvc data/raw/.gitignore

# Configure remote
dvc remote add -d s3remote s3://my-bucket/dvc-storage
dvc remote modify s3remote access_key_id ${AWS_ACCESS_KEY_ID}
dvc remote modify s3remote secret_access_key ${AWS_SECRET_ACCESS_KEY}

# Push data
dvc push

# Pull data
dvc pull
```

### DVC Pipeline

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare_data.py
    deps:
      - src/prepare_data.py
      - data/raw/
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet

  train:
    cmd: python src/train.py --config configs/train.yaml
    deps:
      - src/train.py
      - data/processed/train.parquet
      - configs/train.yaml
    params:
      - train.yaml:
          - learning_rate
          - max_depth
    outs:
      - model/model.joblib
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - model/model.joblib
      - data/processed/test.parquet
    metrics:
      - evaluation/metrics.json:
          cache: false
    plots:
      - evaluation/confusion_matrix.csv
```

```bash
# Run pipeline
dvc repro

# Compare experiments
dvc metrics diff

# Show pipeline DAG
dvc dag
```
