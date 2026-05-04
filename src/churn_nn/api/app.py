import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn_nn.api.schemas import CustomerFeatures, PredictionResponse
from churn_nn.config import THRESHOLD
from churn_nn.models.mlp import ChurnMLP

logger = logging.getLogger(__name__)

MODEL_VERSION = "mlp_best"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

_state: dict[str, Any] = {}


def _load_artifacts() -> None:
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model = ChurnMLP(metadata["input_dim"])
    model.load_state_dict(
        torch.load(MODELS_DIR / "mlp_best.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    _state["preprocessor"] = preprocessor
    _state["model"] = model
    logger.info("Artefatos carregados. input_dim=%d", metadata["input_dim"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _load_artifacts()
    yield


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    missing, invalid = [], []
    for err in exc.errors():
        field = err["loc"][-1] if err["loc"] else "unknown"
        if err["type"] == "missing":
            missing.append(field)
        else:
            invalid.append({"field": field, "error": err["msg"]})
    return JSONResponse(
        status_code=422,
        content={"missing_fields": missing, "invalid_fields": invalid},
    )


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


def _run_inference(customer: CustomerFeatures) -> PredictionResponse:
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


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_inference, customer)
