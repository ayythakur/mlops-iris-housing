from fastapi import FastAPI, HTTPException
from api.schemas import IrisRequest, PredictionResponse
from src.predict.service import load_production_model, predict_proba
from src.utils.logging import get_logger
import os

app = FastAPI(title="MLOps Model API", version="0.1.0")
logger = get_logger("api", "api.log")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

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
    proba_by_class = None
    if y_prob is not None:
        proba_by_class = {name: float(p) for name, p in zip(CLASS_NAMES, y_prob)}

    out = {
        "prediction": y_pred,            # keep the id
        "label": label,                  # human-readable
        "probabilities": y_prob,         # raw list if you still want it
        "proba_by_class": proba_by_class # nicer dict
    }
    logger.info({"event": "predict_response", "output": out})
    return out
