#!/usr/bin/env python3
"""
modelling.py (final, fixed)
Usage:
  python modelling.py [csv_path] [target_col] [task]
task: auto (default) | regression | classification
"""
import os
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import mlflow
import mlflow.sklearn

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# MLflow config
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Project_Akhir_Rehan")
mlflow.sklearn.autolog()
logger.info("MLflow tracking URI set and autolog enabled. Tracking UI: http://127.0.0.1:5000")

def choose_target_column(df: pd.DataFrame) -> str:
    n_rows = len(df)
    candidates = []
    for col in df.columns:
        col_low = col.lower()
        nunique = df[col].nunique(dropna=False)
        if "id" in col_low or nunique == n_rows:
            continue
        candidates.append((col, nunique))
    if candidates:
        candidates_sorted = sorted(candidates, key=lambda x: x[1])
        chosen = candidates_sorted[0][0]
        logger.info(f"Auto-chosen target column by heuristic: {chosen}")
        return chosen
    fallback = df.columns[-1]
    logger.warning(f"No safe target column found by heuristic. Falling back to last column: {fallback}")
    return fallback

def safe_stratify_check(y: pd.Series) -> bool:
    vc = y.value_counts()
    min_count = vc.min() if not vc.empty else 0
    logger.info(f"Target value counts (top few):\n{vc.head().to_dict()}")
    if min_count >= 2:
        return True
    logger.warning(f"Stratify disabled: smallest class has {min_count} members (<2).")
    return False

def detect_task_auto(y_series: pd.Series) -> str:
    """Auto-detect: numeric -> regression, else classification."""
    if pd.api.types.is_numeric_dtype(y_series):
        return "regression"
    else:
        return "classification"

def main(csv_path="namadataset_preprocessing.csv", target_col=None, task_mode="auto",
         test_size=0.2, random_state=42):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"CSV not found at {csv_path.resolve()}.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset with shape {df.shape}")

    if target_col is None:
        target_col = choose_target_column(df)
    else:
        if target_col not in df.columns:
            logger.error(f"Provided target_col '{target_col}' not found. Columns: {list(df.columns)}")
            sys.exit(1)
        logger.info(f"Using provided target column: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if task_mode == "auto":
        task = detect_task_auto(y)
    elif task_mode in ("regression", "classification"):
        task = task_mode
    else:
        logger.error("task must be one of: auto, regression, classification")
        sys.exit(1)

    logger.info(f"Detected/selected task: {task}")

    stratify_arg = y if (task == "classification" and safe_stratify_check(y)) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    # Preprocessing
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    logger.info(f"Numeric cols: {num_cols}")
    logger.info(f"Categorical cols: {cat_cols}")

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Model choice
    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    pipeline = Pipeline(steps=[("preproc", preprocessor), ("model", model)])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        if task == "classification":
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            logger.info(f"Test Accuracy: {acc:.4f}")
            logger.info(f"Test F1 (weighted): {f1:.4f}")
            mlflow.log_metric("test_accuracy", float(acc))
            mlflow.log_metric("test_f1_weighted", float(f1))
        else:
            # compute RMSE in a backward-compatible way
            mse = mean_squared_error(y_test, preds)
            rmse = float(np.sqrt(mse))
            logger.info(f"Test RMSE: {rmse:.4f}")
            mlflow.log_metric("test_rmse", rmse)

        artifact_dir = Path("artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        model_file = artifact_dir / "pipeline.pkl"
        joblib.dump(pipeline, model_file)
        mlflow.log_artifact(str(model_file), artifact_path="pipeline_files")
        logger.info(f"Saved pipeline to {model_file} and logged as artifact.")

    logger.info("Done. Open MLflow UI at http://127.0.0.1:5000 to inspect runs/artifacts.")

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "namadataset_preprocessing.csv"
    target = sys.argv[2] if len(sys.argv) > 2 else None
    task_arg = sys.argv[3].lower() if len(sys.argv) > 3 else "auto"
    main(csv_path=csv, target_col=target, task_mode=task_arg)
