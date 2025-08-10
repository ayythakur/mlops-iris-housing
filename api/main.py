from fastapi import FastAPI, HTTPException, Response
from api.schemas import IrisRequest, PredictionResponse
from src.predict.service import load_production_model, predict_proba
from src.utils.logging import get_logger
import os
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from src.utils.audit import log_prediction

app = FastAPI(title="MLOps Model API", version="0.1.0")
logger = get_logger("api", "api.log")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
PREDICTIONS_TOTAL = Counter("predictions_total", "Total number of prediction requests")

@app.get("/health")
def health():
    exists = os.path.exists("models/registry/model.joblib")
    return {"status": "ok", "model_present": exists}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: IrisRequest):
    if not os.path.exists("models/registry/model.joblib"):
        raise HTTPException(status_code=503, detail="Model not available. Train first.")
    model = load_production_model()
    logger.info({"event": "predict_call", "input": req.model_dump()})
    y_pred, y_prob = predict_proba(model, req.features)

    label = CLASS_NAMES[y_pred]
    proba_by_class = {name: float(p) for name, p in zip(CLASS_NAMES, y_prob)} if y_prob else None

    out = {"prediction": y_pred, "label": label, "probabilities": y_prob, "proba_by_class": proba_by_class}
    logger.info({"event": "predict_response", "output": out})

    # monitoring + audit
    PREDICTIONS_TOTAL.inc()
    log_prediction(req.features, y_pred, y_prob)

    return out

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
