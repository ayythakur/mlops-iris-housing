# src/utils/tracking.py
import os
import mlflow
from contextlib import contextmanager
from urllib.parse import urlparse

def _portable_file_uri(dir_path: str) -> str:
    """Return a file: URI for an absolute, POSIX-safe path."""
    abs_dir = os.path.abspath(dir_path)
    return f"file:{abs_dir}"

def setup_mlflow(tracking_uri: str, experiment_name: str):
    """
    Normalize MLflow tracking URI so it works on Windows, Linux, and CI.
    """
    import logging
    logger = logging.getLogger("mlflow_setup")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    force_local = False

    if os.getenv("GITHUB_ACTIONS") == "true":
        force_local = True
    else:
        parsed = urlparse(tracking_uri)
        if not parsed.scheme:
            tracking_uri = _portable_file_uri(tracking_uri)
        elif parsed.scheme.lower() == "file":
            path = parsed.path or ""
            if len(path) >= 3 and path[0] == "/" and path[2] == ":":
                force_local = True

    if force_local:
        tracking_uri = _portable_file_uri("mlruns")

    logger.info(f"Using MLflow tracking URI: {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def run(run_name: str = None):
    with mlflow.start_run(run_name=run_name):
        yield

def log_param(k, v):
    mlflow.log_param(k, v)

def log_metric(k, v):
    mlflow.log_metric(k, v)

def log_model(model, artifact_path: str = "model"):
    import mlflow.sklearn
    mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path)
