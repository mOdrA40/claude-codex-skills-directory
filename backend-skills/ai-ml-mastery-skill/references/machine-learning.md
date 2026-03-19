# Machine Learning Reference

## Table of Contents
1. [Scikit-learn Patterns](#scikit-learn-patterns)
2. [Feature Engineering](#feature-engineering)
3. [Gradient Boosting (XGBoost, LightGBM, CatBoost)](#gradient-boosting)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Model Evaluation](#model-evaluation)
6. [Pipeline Best Practices](#pipeline-best-practices)
7. [Common Pitfalls](#common-pitfalls)

---

## Scikit-learn Patterns

### Complete Training Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def create_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create feature preprocessing pipeline."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def create_full_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model,
) -> Pipeline:
    """Create complete ML pipeline."""
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


# Usage
numeric_cols = ["age", "income", "tenure"]
categorical_cols = ["gender", "region", "product_type"]

pipeline = create_full_pipeline(
    numeric_cols,
    categorical_cols,
    RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
print(f"AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
```

### Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables by frequency."""
    
    def __init__(self, columns: list[str] | None = None):
        self.columns = columns
        self.freq_maps_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        cols = self.columns or X.select_dtypes(include=["object", "category"]).columns
        
        for col in cols:
            self.freq_maps_[col] = X[col].value_counts(normalize=True).to_dict()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        
        for col, freq_map in self.freq_maps_.items():
            default = min(freq_map.values()) / 2  # Rare category fallback
            X[col] = X[col].map(lambda x: freq_map.get(x, default))
        
        return X


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding with regularization."""
    
    def __init__(
        self,
        columns: list[str] | None = None,
        smoothing: float = 10.0,
    ):
        self.columns = columns
        self.smoothing = smoothing
        self.target_maps_ = {}
        self.global_mean_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = pd.DataFrame(X)
        self.global_mean_ = y.mean()
        cols = self.columns or X.select_dtypes(include=["object", "category"]).columns
        
        for col in cols:
            stats = y.groupby(X[col]).agg(["mean", "count"])
            
            # Smoothed target encoding
            smoothed = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
            ) / (stats["count"] + self.smoothing)
            
            self.target_maps_[col] = smoothed.to_dict()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        
        for col, target_map in self.target_maps_.items():
            X[col] = X[col].map(lambda x: target_map.get(x, self.global_mean_))
        
        return X
```

---

## Feature Engineering

### Datetime Features

```python
def extract_datetime_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Extract comprehensive datetime features."""
    df = df.copy()
    dt = pd.to_datetime(df[col])
    
    df[f"{col}_year"] = dt.dt.year
    df[f"{col}_month"] = dt.dt.month
    df[f"{col}_day"] = dt.dt.day
    df[f"{col}_dayofweek"] = dt.dt.dayofweek
    df[f"{col}_hour"] = dt.dt.hour
    df[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df[f"{col}_is_month_start"] = dt.dt.is_month_start.astype(int)
    df[f"{col}_is_month_end"] = dt.dt.is_month_end.astype(int)
    df[f"{col}_quarter"] = dt.dt.quarter
    
    # Cyclical encoding for periodic features
    df[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    df[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    
    return df
```

### Aggregation Features

```python
def create_aggregation_features(
    df: pd.DataFrame,
    group_col: str,
    agg_col: str,
    aggs: list[str] = ["mean", "std", "min", "max", "count"],
) -> pd.DataFrame:
    """Create aggregation features grouped by category."""
    agg_df = df.groupby(group_col)[agg_col].agg(aggs)
    agg_df.columns = [f"{agg_col}_{agg}_by_{group_col}" for agg in aggs]
    return df.merge(agg_df, on=group_col, how="left")


def create_lag_features(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    lags: list[int] = [1, 7, 14, 30],
    date_col: str = "date",
) -> pd.DataFrame:
    """Create lag features for time series."""
    df = df.sort_values([group_col, date_col]).copy()
    
    for lag in lags:
        df[f"{value_col}_lag_{lag}"] = df.groupby(group_col)[value_col].shift(lag)
    
    # Rolling features
    for window in [7, 14, 30]:
        df[f"{value_col}_rolling_mean_{window}"] = (
            df.groupby(group_col)[value_col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    
    return df
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.inspection import permutation_importance

def select_features_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 20,
) -> list[str]:
    """Select top k features using mutual information."""
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    
    scores = pd.Series(selector.scores_, index=X.columns)
    return scores.nlargest(k).index.tolist()


def select_features_permutation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.01,
) -> list[str]:
    """Select features based on permutation importance."""
    model.fit(X, y)
    
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importance = pd.Series(result.importances_mean, index=X.columns)
    
    return importance[importance > threshold].index.tolist()


def remove_multicollinear_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> list[str]:
    """Remove highly correlated features."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [col for col in df.columns if col not in to_drop]
```

---

## Gradient Boosting

### XGBoost

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Optimal parameters for most cases
xgb_params = {
    "objective": "binary:logistic",  # or "reg:squarederror", "multi:softprob"
    "eval_metric": "auc",
    "tree_method": "hist",           # Fast histogram-based
    "device": "cuda",                # GPU if available
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,                # L1 regularization
    "reg_lambda": 1.0,               # L2 regularization
    "random_state": 42,
    "n_jobs": -1,
}

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Train with early stopping
model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=100,
)

# Feature importance
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.get_score(importance_type="gain").values(),
}).sort_values("importance", ascending=False)
```

### LightGBM

```python
import lightgbm as lgb

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,              # Main complexity control
    "max_depth": -1,               # Unlimited
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100),
    ],
)

# SHAP values for interpretability
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)
```

### CatBoost

```python
from catboost import CatBoostClassifier, Pool

cat_params = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "task_type": "GPU",              # Use GPU if available
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
    "verbose": 100,
}

# CatBoost handles categoricals natively
cat_features = [X_train.columns.get_loc(col) for col in categorical_cols]

model = CatBoostClassifier(**cat_params)
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    plot=False,
)
```

---

## Hyperparameter Tuning

### Optuna (Recommended)

```python
import optuna
from optuna.integration import XGBoostPruningCallback

def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    
    pruning_callback = XGBoostPruningCallback(trial, "val-auc")
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        callbacks=[pruning_callback],
        verbose_eval=False,
    )
    
    return model.best_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## Model Evaluation

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Comprehensive classification evaluation."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find optimal classification threshold."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        if metric == "f1":
            scores.append(f1_score(y_true, y_pred))
        elif metric == "precision":
            scores.append(precision_score(y_true, y_pred))
        elif metric == "recall":
            scores.append(recall_score(y_true, y_pred))
    
    return thresholds[np.argmax(scores)]
```

### Regression Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Comprehensive regression evaluation."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        "r2": r2_score(y_true, y_pred),
    }
```

---

## Pipeline Best Practices

### Reproducibility

```python
import random
import numpy as np

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
    # If using other libraries
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# Always at start of script
set_seed(42)
```

### Data Validation

```python
def validate_data(df: pd.DataFrame, schema: dict) -> list[str]:
    """Validate dataframe against expected schema."""
    errors = []
    
    # Check required columns
    missing = set(schema.keys()) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
    
    # Check dtypes
    for col, expected_dtype in schema.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if not actual_dtype.startswith(expected_dtype):
                errors.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")
    
    # Check for nulls in required columns
    null_cols = df.isnull().sum()
    high_null = null_cols[null_cols > len(df) * 0.5].index.tolist()
    if high_null:
        errors.append(f"High null columns (>50%): {high_null}")
    
    return errors
```

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Data leakage in preprocessing | Fit transformers on train only, transform both |
| Target leakage | Check features don't contain future information |
| Class imbalance ignored | Use stratified sampling, class weights, or SMOTE |
| Overfitting to validation | Use proper nested CV for hyperparameter tuning |
| Using mean imputation for test | Impute with train statistics |
| Not scaling for distance-based | Scale features for KNN, SVM, neural networks |
| Encoding test categories not in train | Use handle_unknown="ignore" |
| Evaluating on imbalanced with accuracy | Use F1, AUC-PR, or balanced accuracy |
