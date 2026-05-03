# src/churn_nn/api/app.py
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel

from churn_nn.models.mlp import ChurnMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

MODEL_VERSION = "mlp_best"
THRESHOLD = 0.4
MODELS_DIR = Path("models")

_state: dict[str, Any] = {}


def _load_artifacts() -> None:
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    dummy = pd.DataFrame(
        [
            {
                "SeniorCitizen": 0,
                "tenure": 1,
                "MonthlyCharges": 1.0,
                "TotalCharges": 1.0,
                "gender": "Male",
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
            }
        ]
    )
    input_dim = preprocessor.transform(dummy).shape[1]
    model = ChurnMLP(input_dim)
    model.load_state_dict(
        torch.load(MODELS_DIR / "mlp_best.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    _state["preprocessor"] = preprocessor
    _state["model"] = model
    logger.info("Artefatos carregados. input_dim=%d", input_dim)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "method=%s path=%s status=%d latency_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


class CustomerFeatures(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


class PredictionResponse(BaseModel):
    churn: bool
    probability: float
    threshold: float
    model_version: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    df = pd.DataFrame([customer.model_dump()])
    X = _state["preprocessor"].transform(df)
    tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logit = _state["model"](tensor)
        prob = float(torch.sigmoid(logit).item())
    return PredictionResponse(
        churn=prob >= THRESHOLD,
        probability=round(prob, 4),
        threshold=THRESHOLD,
        model_version=MODEL_VERSION,
    )
