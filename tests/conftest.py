# tests/conftest.py
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SAMPLE_CUSTOMER = {
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "SeniorCitizen": 0,
    "gender": "Male",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
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


@pytest.fixture(scope="session")
def client():
    artifacts_missing = not (Path("models") / "preprocessor.pkl").exists() or not (
        Path("models") / "model_metadata.json"
    ).exists()
    if artifacts_missing:
        pytest.skip("Artefatos de modelo não encontrados. Execute 'make train'.")
    from churn_nn.api.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_customer():
    return SAMPLE_CUSTOMER.copy()
