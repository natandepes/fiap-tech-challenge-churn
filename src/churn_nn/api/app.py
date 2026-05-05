import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn_nn.api.schemas import CustomerFeatures, PredictionResponse
from churn_nn.config import MODELS_DIR, THRESHOLD
from churn_nn.models.mlp import ChurnMLP

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {}


def _load_artifacts() -> None:
    metadata_path = MODELS_DIR / "model_metadata.json"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    weights_path = MODELS_DIR / "mlp_best.pt"

    missing = [
        str(p) for p in (metadata_path, preprocessor_path, weights_path)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Artefatos ausentes em '{MODELS_DIR}': {missing}. Execute 'make train'."
        )

    metadata = json.loads(metadata_path.read_text())
    preprocessor = joblib.load(preprocessor_path)
    model = ChurnMLP(metadata["input_dim"])
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    _state["preprocessor"] = preprocessor
    _state["model"] = model
    _state["model_version"] = metadata.get("model_version", "unknown")
    logger.info(
        "Artefatos carregados. input_dim=%d version=%s",
        metadata["input_dim"],
        _state["model_version"],
    )


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
    return {"status": "ok", "model_version": _state["model_version"]}


def _run_inference(customer: CustomerFeatures) -> PredictionResponse:
    df = pd.DataFrame([customer.model_dump()])
    X = _state["preprocessor"].transform(df)
    tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logit = _state["model"](tensor)
        prob = float(torch.sigmoid(logit).item())
    prediction = PredictionResponse(
        churn=prob >= THRESHOLD,
        probability=round(prob, 4),
        threshold=THRESHOLD,
        model_version=_state["model_version"],
    )
    logger.info(
        "prediction churn=%s probability=%.4f model_version=%s",
        prediction.churn,
        prediction.probability,
        prediction.model_version,
    )
    return prediction


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_inference, customer)
