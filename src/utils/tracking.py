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
    Normalize MLflow tracking URI so it works on Windows & Linux and in CI.
    - If running in GitHub Actions, always use a local ./mlruns store.
    - If a relative path is provided, convert to file:<abs path>.
    - If a file: URI contains a Windows drive on Linux (e.g. '/F:'), replace with ./mlruns.
    """
    if os.getenv("GITHUB_ACTIONS") == "true":
        # Force a local store in CI to avoid Windows drive letters
        tracking_uri = _portable_file_uri("mlruns")
    else:
        parsed = urlparse(tracking_uri)
        if not parsed.scheme:
            # Plain path like "mlruns" -> make it a portable file: URI
            tracking_uri = _portable_file_uri(tracking_uri)
        elif parsed.scheme.lower() == "file":
            # Normalize weird '/F:' paths that appear on Linux for Windows drives
            path = parsed.path or ""
            if len(path) >= 3 and path[0] == "/" and path[2] == ":":
                tracking_uri = _portable_file_uri("mlruns")

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
    try:
        # Prefer logging as a run artifact (portable)
        mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path)
    except TypeError:
        # Fallback for very old/new APIs if the signature differs
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
