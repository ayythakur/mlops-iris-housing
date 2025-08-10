# api/schemas.py
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class IrisRequest(BaseModel):
    # Order: sepal_length, sepal_width, petal_length, petal_width
    features: List[float] = Field(..., min_length=4, max_length=4)

class PredictionResponse(BaseModel):
    prediction: int
    label: Optional[str] = None
    probabilities: Optional[List[float]] = None
    proba_by_class: Optional[Dict[str, float]] = None
