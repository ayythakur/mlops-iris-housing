import argparse
import os
import yaml
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.data.load import load_iris
from src.data.preprocess import split_data, build_preprocessor
from src.utils.tracking import setup_mlflow, run, log_param, log_metric, log_model
from src.utils.logging import get_logger
from src.utils.io import save_joblib, save_json


def load_params(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Params file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if not params:
        raise ValueError(f"Params file is empty or invalid YAML: {path}")
    return params


def build_model(cfg: dict):
    model_type = cfg["type"]
    params = cfg.get("params", {})
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    raise ValueError(f"Unknown model type: {model_type}")


def main(params_path: str) -> None:
    logger = get_logger("train", "training.log")
    params = load_params(params_path)

    setup_mlflow(params["mlflow_tracking_uri"], params["experiment_name"])

    X, y = load_iris()
    Xtr, Xte, ytr, yte = split_data(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )
    preproc = build_preprocessor(params["features"]["scale_numeric"])

    best_score = -np.inf
    best_name = None
    best_pipe = None

    for m in params["models"]:
        name = m["name"]
        clf = build_model(m)

        steps = []
        if preproc is not None:
            steps.append(("preprocess", preproc))
        steps.append(("model", clf))
        pipe = Pipeline(steps)

        with run(run_name=name):
            log_param("model", name)
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            acc = float(accuracy_score(yte, pred))
            log_metric("accuracy", acc)
            log_model(pipe, artifact_path=f"{name}_model")
            logger.info(f"{name} accuracy={acc:.4f}")

            if acc > best_score:
                best_score = acc
                best_name = name
                best_pipe = pipe

    if best_pipe is not None:
        save_joblib(best_pipe, "models/registry/model.joblib")
        save_json(
            {"best_model": best_name, "metric": "accuracy", "score": best_score},
            "models/registry/model_meta.json",
        )
        logger.info(
            f"Saved best model '{best_name}' (accuracy={best_score:.4f})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    main(parser.parse_args().params)
