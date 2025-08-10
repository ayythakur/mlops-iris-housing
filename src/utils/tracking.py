import mlflow
from contextlib import contextmanager

def setup_mlflow(tracking_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

@contextmanager
def run(run_name: str = None):
    with mlflow.start_run(run_name=run_name):
        yield

def log_param(k, v): mlflow.log_param(k, v)
def log_metric(k, v): mlflow.log_metric(k, v)

def log_model(model, artifact_path: str = "model"):
    import mlflow.sklearn
    # Use keyword to avoid API ambiguity across MLflow versions
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)
