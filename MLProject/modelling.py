#!/usr/bin/env python3
"""
modelling.py (CI-friendly final, robust start_run)

Perubahan penting:
- Gunakan MLFLOW_TRACKING_URI dari env (dengan fallback ke ./mlruns)
- Saat menjalankan mlflow.start_run(): coba attach ke run_id dari env (MLFLOW_RUN_ID),
  kalau gagal gunakan start_run() baru, dan kalau tetap gagal gunakan nested run.
- Pastikan selalu memanggil mlflow.end_run() di finally supaya run tidak menggantung.
- Simpan artifact di folder 'artifacts' relative ke working dir (MLProject/artifacts jika ada).
- Tangani infer_signature/log_model dalam try/except agar CI tidak gagal jika signature bermasalah.
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

# ---------- MLflow configuration ----------
# Prefer environment variable MLFLOW_TRACKING_URI if set (CI will set it).
# Fallback to sqlite file in current working directory (stable for local dev).
env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if env_tracking_uri:
    mlflow.set_tracking_uri(env_tracking_uri)
    logger.info(f"Using MLFLOW_TRACKING_URI from env: {env_tracking_uri}")
else:
    # fallback file DB (relative to cwd)
    fallback_db = Path.cwd() / "mlruns"
    mlflow.set_tracking_uri(f"file:{fallback_db}")
    logger.info(f"No MLFLOW_TRACKING_URI provided. Using fallback file store: file:{fallback_db}")

EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "Project_Akhir_Rehan")
mlflow.set_experiment(EXPERIMENT_NAME)
# enable autologging for sklearn (still do explicit model log to create model/ in UI)
mlflow.sklearn.autolog()
logger.info(f"MLflow configured: tracking_uri={mlflow.get_tracking_uri()}, experiment={EXPERIMENT_NAME}")

# ---------- helpers ----------
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

# ---------- main ----------
def main(csv_path="namadataset_preprocessing.csv", target_col=None, task_mode="auto",
         test_size=0.2, random_state=42):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path.resolve()}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset: {df.shape}")

    # choose/validate target column
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

    # artifacts dir: prefer MLProject/artifacts if that folder exists in repo root of this run,
    # otherwise use 'artifacts' in cwd.
    # NOTE: when mlflow run MLProject executes, working dir will normally be MLProject folder.
    artifact_dir = Path("artifacts")
    # if we're running from a parent repo folder and MLProject exists, keep artifacts inside MLProject
    if Path("MLProject").exists():
        artifact_dir = Path("MLProject") / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Robust MLflow run handling:
    # - try to attach to parent run if MLFLOW_RUN_ID env is present
    # - otherwise start a normal run
    # - if start_run fails, fallback to nested run
    run_started = False
    try:
        run_id_env = os.environ.get("MLFLOW_RUN_ID")
        if run_id_env:
            try:
                mlflow.start_run(run_id=run_id_env)
                logger.info(f"Attached to parent run id from env: {run_id_env}")
                run_started = True
            except Exception as e:
                logger.warning(f"Failed to attach to run_id from env ({run_id_env}): {e}. Trying start_run()")
        if not run_started:
            try:
                mlflow.start_run()
                logger.info("Started a new MLflow run")
                run_started = True
            except Exception as e:
                logger.warning(f"mlflow.start_run() failed: {e}. Trying nested run fallback.")
                try:
                    mlflow.start_run(nested=True)
                    logger.info("Started nested MLflow run as fallback")
                    run_started = True
                except Exception as e2:
                    logger.error(f"Failed to start any MLflow run: {e2}")
                    raise

        # ===== training & logging =====
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
        logger.info(f"Saved pipeline locally to {model_file}")

        # Log artifacts to mlflow (artifact_path keys chosen to match MLflow UI)
        try:
            mlflow.log_artifact(str(model_file), artifact_path="pipeline_files")
            logger.info("Logged pipeline.pkl to mlflow artifact_path='pipeline_files'")
        except Exception as e:
            logger.warning(f"mlflow.log_artifact pipeline_files failed: {e}")

        # estimator.html
        estimator_file = artifact_dir / "estimator.html"
        try:
            write_estimator_html(estimator_file, pipeline, X_train)
            mlflow.log_artifact(str(estimator_file), artifact_path="artifact_files")
            logger.info("Wrote and logged estimator.html")
        except Exception as e:
            logger.warning(f"estimator.html write/log failed: {e}")

        # Explicit log_model to create model/ artifact folder in UI (signature inference optional)
        try:
            signature = None
            try:
                # quick attempt to infer signature; guard against failures
                signature = infer_signature(X_train, pipeline.predict(X_train))
            except Exception as sig_ex:
                logger.warning(f"Signature inference failed (continuing without signature): {sig_ex}")
                signature = None
            mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model", signature=signature)
            logger.info("mlflow.sklearn.log_model finished (artifact_path='model')")
        except Exception as e:
            logger.warning(f"mlflow.sklearn.log_model failed: {e}")

    finally:
        # ensure run closed so DB state is consistent
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run (end_run called).")
        except Exception as e:
            logger.warning(f"mlflow.end_run() raised exception (ignored): {e}")

    logger.info("Run finished. Inspect MLflow UI for details and artifacts.")

if __name__ == "__main__":
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
