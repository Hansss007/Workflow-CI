#!/usr/bin/env python3
"""
modelling.py (CI-friendly final)
Supports:
 - positional args: python modelling.py <csv_path> <target_col> <task>
 - runs fine under `mlflow run MLProject` (params passed from MLproject)
 - logs artifacts into MLflow + also writes files to MLProject/artifacts/
 - explicitly logs MLflow Model (artifact_path="model") to create model/ in Artifacts UI
"""
import os
import sys
import logging
from pathlib import Path
import json
import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# MLflow tracking config: prefer env var (used in CI), fallback to local mlruns/ file store
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file:{Path.cwd() / 'mlruns'}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "Project_Akhir_Rehan")
mlflow.set_experiment(EXPERIMENT_NAME)
# autolog helps capture params/metrics, but we will also explicitly log the model (for model/ folder)
mlflow.sklearn.autolog()
logger.info(f"MLflow set to {mlflow.get_tracking_uri()}, experiment={EXPERIMENT_NAME}, autolog enabled")


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore")


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
        chosen = sorted(candidates, key=lambda x: x[1])[0][0]
        logger.info(f"Auto-chosen target column: {chosen}")
        return chosen
    fallback = df.columns[-1]
    logger.warning(f"No safe target; fallback to last column: {fallback}")
    return fallback


def safe_stratify_check(y: pd.Series) -> bool:
    vc = y.value_counts()
    if vc.empty:
        return False
    min_count = int(vc.min())
    logger.info(f"Target counts (top): {vc.head().to_dict()}")
    return min_count >= 2


def detect_task_auto(y_series: pd.Series) -> str:
    return "regression" if pd.api.types.is_numeric_dtype(y_series) else "classification"


def write_estimator_html(path: Path, pipeline: Pipeline, X_train: pd.DataFrame):
    try:
        params = {}
        model = pipeline.named_steps.get("model", None)
        if model is not None:
            params = model.get_params()
    except Exception:
        params = {"error": "could not inspect params"}

    schema_rows = []
    for col in X_train.columns:
        schema_rows.append(f"<tr><td>{col}</td><td>{str(X_train[col].dtype)}</td></tr>")

    now = datetime.datetime.utcnow().isoformat() + "Z"
    html = f"""
    <html><head><meta charset="utf-8"><title>Estimator</title></head><body>
    <h2>Pipeline summary</h2><pre>{pipeline}</pre>
    <h3>Input schema</h3><table border="1"><tr><th>feature</th><th>dtype</th></tr>
    {''.join(schema_rows[:200])}
    </table>
    <h3>Estimator params</h3><pre>{json.dumps(params, default=str, indent=2)}</pre>
    <hr/><small>generated at {now}</small>
    </body></html>
    """
    path.write_text(html, encoding="utf-8")
    logger.info(f"Wrote estimator summary to {path}")


def build_pipeline(num_cols, cat_cols, task="regression", random_state=42):
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", make_onehot_encoder(), cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestClassifier(n_estimators=100, random_state=random_state) if task == "classification" \
        else RandomForestRegressor(n_estimators=100, random_state=random_state)

    return Pipeline([("preproc", preprocessor), ("model", model)])


def main(csv_path="namadataset_preprocessing.csv", target_col=None, task_mode="auto",
         test_size=0.2, random_state=42):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path.resolve()}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset: {df.shape}")

    if target_col is None or str(target_col).strip() == "":
        target_col = choose_target_column(df)
    else:
        if target_col not in df.columns:
            logger.error(f"Provided target_col '{target_col}' not in columns: {list(df.columns)}")
            sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    task = detect_task_auto(y) if task_mode == "auto" else task_mode
    logger.info(f"Task selected: {task}")

    stratify_arg = y if (task == "classification" and safe_stratify_check(y)) else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=stratify_arg)
    logger.info(f"Split shapes train={X_train.shape}, test={X_test.shape}")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    logger.info(f"num_cols={num_cols}, cat_cols={cat_cols}")

    pipeline = build_pipeline(num_cols=num_cols, cat_cols=cat_cols, task=task, random_state=random_state)

    # Ensure artifacts dir (relative to working dir). When executed by mlflow run MLProject,
    # working dir will be MLProject folder, so artifacts -> MLProject/artifacts
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        logger.info("Fitting pipeline...")
        pipeline.fit(X_train, y_train)
        logger.info("Predicting and logging metrics...")
        preds = pipeline.predict(X_test)

        if task == "classification":
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            mlflow.log_metric("test_accuracy", float(acc))
            mlflow.log_metric("test_f1_weighted", float(f1))
            logger.info(f"Acc={acc:.4f}, F1={f1:.4f}")
        else:
            mse = mean_squared_error(y_test, preds)
            rmse = float(np.sqrt(mse))
            mlflow.log_metric("test_rmse", rmse)
            logger.info(f"RMSE={rmse:.4f}")

        # save pipeline and estimator summary locally (and log to mlflow artifacts)
        model_file = artifact_dir / "pipeline.pkl"
        joblib.dump(pipeline, model_file)
        try:
            mlflow.log_artifact(str(model_file), artifact_path="pipeline_files")
            logger.info(f"Saved pipeline to {model_file} and logged as pipeline_files")
        except Exception as e:
            logger.warning(f"mlflow.log_artifact pipeline_files failed: {e}")

        # estimator.html
        estimator_file = artifact_dir / "estimator.html"
        try:
            write_estimator_html(estimator_file, pipeline, X_train)
            mlflow.log_artifact(str(estimator_file), artifact_path="artifact_files")
        except Exception as e:
            logger.warning(f"estimator.html logging failed: {e}")

        # Explicit log_model to create model/ artifact folder
        try:
            signature = None
            try:
                signature = infer_signature(X_train, pipeline.predict(X_train))
            except Exception:
                signature = None
            mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model", signature=signature)
            logger.info("mlflow.sklearn.log_model finished (artifact_path='model')")
        except Exception as e:
            logger.warning(f"mlflow.sklearn.log_model failed: {e}")

    logger.info("Run finished. Inspect MLflow UI for details.")


if __name__ == "__main__":
    # support both positional args (from MLproject) and explicit flags if desired
    import argparse
    parser = argparse.ArgumentParser(description="Train and log model with MLflow")
    parser.add_argument("csv_path", nargs="?", default="MLProject/namadataset_preprocessing.csv")
    parser.add_argument("target_col", nargs="?", default=None)
    parser.add_argument("task_mode", nargs="?", default="auto")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(csv_path=args.csv_path, target_col=args.target_col, task_mode=args.task_mode,
         test_size=args.test_size, random_state=args.random_state)
