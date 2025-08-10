from typing import List, Tuple, Optional
import numpy as np
from src.utils.io import load_joblib

_model_cache = None

def load_production_model(path: str = "models/registry/model.joblib"):
    global _model_cache
    if _model_cache is None:
        _model_cache = load_joblib(path)
    return _model_cache

def predict_proba(model, features: List[float]) -> Tuple[int, Optional[list]]:
    X = np.array(features, dtype=float).reshape(1, -1)
    y_pred = model.predict(X)[0]
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X).tolist()[0]
    return int(y_pred), y_prob
