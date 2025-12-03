#!/usr/bin/env python3

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

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------- MLflow configuration ----------------
# If the user or environment set a tracking URI (e.g., CI or DagsHub), prefer that.
env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
if env_tracking_uri:
    mlflow.set_tracking_uri(env_tracking_uri)
    logger.info(f"Using MLFLOW_TRACKING_URI from env: {env_tracking_uri}")
else:
    # Use local file-based tracking to avoid network dependency (safe for CI and local)
    mlflow.set_tracking_uri("file:./mlruns")
    logger.info("No MLFLOW_TRACKING_URI provided. Using local mlruns folder (file:./mlruns).")

# Experiment name
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "Project_Akhir_Rehan")
mlflow.set_experiment(EXPERIMENT_NAME)
# Enable autolog (Basic requirement)
mlflow.sklearn.autolog()
logger.info(f"MLflow autolog enabled. Experiment: {EXPERIMENT_NAME}")

# -----------------------------------------------------

def make_onehot_encoder():
    """
    Return a OneHotEncoder instance compatible across sklearn versions.
    Older versions expect `sparse=False`, newer use `sparse_output=False`.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            # Fallback: create without specifying sparse option (may default to sparse matrix)
            return OneHotEncoder(handle_unknown="ignore")

def choose_target_column(df: pd.DataFrame) -> str:
    """
    Heuristic: skip columns that look like IDs or are unique per row; prefer a low-cardinality column.
    """
    n_rows = len(df)
    candidates = []
    for col in df.columns:
        col_low = col.lower()
        nunique = df[col].nunique(dropna=False)
        if "id" in col_low or nunique == n_rows:
            continue
        candidates.append((col, nunique))

    if candidates:
        # prefer low-cardinality column (categorical-like)
        candidates_sorted = sorted(candidates, key=lambda x: x[1])
        chosen = candidates_sorted[0][0]
        logger.info(f"Auto-chosen target column by heuristic: {chosen}")
        return chosen

    fallback = df.columns[-1]
    logger.warning(f"No safe target column found by heuristic. Falling back to last column: {fallback}")
    return fallback

def safe_stratify_check(y: pd.Series) -> bool:
    """Return True if stratify is safe (every class count >= 2)."""
    vc = y.value_counts()
    if vc.empty:
        return False
    min_count = int(vc.min())
    logger.info(f"Target counts (top): {vc.head().to_dict()}")
    if min_count >= 2:
        return True
    logger.warning(f"Stratify disabled: smallest class has {min_count} members (<2).")
    return False

def detect_task_auto(y_series: pd.Series) -> str:
    """Auto-detect task: numeric -> regression, else classification."""
    if pd.api.types.is_numeric_dtype(y_series):
        return "regression"
    else:
        return "classification"

def build_pipeline(num_cols, cat_cols, random_state=42, task="regression"):
    enc = make_onehot_encoder() if cat_cols else None
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", enc, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    pipeline = Pipeline(steps=[("preproc", preprocessor), ("model", model)])
    return pipeline

def main(csv_path="namadataset_preprocessing.csv", target_col=None, task_mode="auto",
         test_size=0.2, random_state=42):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"CSV not found at {csv_path.resolve()}. Exiting.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset with shape {df.shape}")

    # choose or validate target column
    if target_col is None:
        target_col = choose_target_column(df)
    else:
        if target_col not in df.columns:
            logger.error(f"Provided target_col '{target_col}' not found in dataset columns.")
            logger.error(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        logger.info(f"Using provided target column: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # determine task
    if task_mode == "auto":
        task = detect_task_auto(y)
    elif task_mode in ("regression", "classification"):
        task = task_mode
    else:
        logger.error("task_mode must be one of: auto, regression, classification")
        sys.exit(1)

    logger.info(f"Selected task: {task}")

    # stratify if classification and safe
    stratify_arg = y if (task == "classification" and safe_stratify_check(y)) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    logger.info(f"Train/test split: {X_train.shape} / {X_test.shape}")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    logger.info(f"Numeric cols ({len(num_cols)}): {num_cols}")
    logger.info(f"Categorical cols ({len(cat_cols)}): {cat_cols}")

    pipeline = build_pipeline(num_cols=num_cols, cat_cols=cat_cols, random_state=random_state, task=task)

    # Start an MLflow run (autolog will capture params/metrics/model)
    with mlflow.start_run():
        logger.info("Starting model training...")
        pipeline.fit(X_train, y_train)
        logger.info("Training finished, evaluating...")

        preds = pipeline.predict(X_test)

        if task == "classification":
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            logger.info(f"Test Accuracy: {acc:.4f}")
            logger.info(f"Test F1 (weighted): {f1:.4f}")
            # explicit logging (in addition to autolog) to ensure visibility in UI
            mlflow.log_metric("test_accuracy", float(acc))
            mlflow.log_metric("test_f1_weighted", float(f1))
        else:
            mse = mean_squared_error(y_test, preds)
            rmse = float(np.sqrt(mse))
            logger.info(f"Test RMSE: {rmse:.4f}")
            mlflow.log_metric("test_rmse", rmse)

        # Save artifact(s) to a stable folder (suitable for CI upload)
        # prefer MLProject/artifacts if running inside project, else artifacts/
        artifact_base = Path("MLProject/artifacts") if Path("MLProject").exists() else Path("artifacts")
        artifact_base.mkdir(parents=True, exist_ok=True)

        model_file = artifact_base / "model.pkl"
        import joblib
        joblib.dump(pipeline, model_file)
        # Log artifact into mlflow (so it also appears in mlruns)
        try:
            mlflow.log_artifact(str(model_file), artifact_path="pipeline_files")
            logger.info(f"Saved and logged model artifact to {model_file}")
        except Exception as ex:
            logger.warning(f"Failed to mlflow.log_artifact (non-fatal): {ex}")
            logger.info(f"Model saved locally at {model_file}")

    logger.info("Run complete. If MLflow UI is available, open it to inspect the run.")

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "MLProject/namadataset_preprocessing.csv"
    target = sys.argv[2] if len(sys.argv) > 2 else None
    task_arg = sys.argv[3].lower() if len(sys.argv) > 3 else "auto"
    main(csv_path=csv, target_col=target, task_mode=task_arg)
