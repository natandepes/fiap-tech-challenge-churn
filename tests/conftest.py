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
    if not (Path("models") / "preprocessor.pkl").exists():
        pytest.skip("Artefatos não encontrados. Execute 'make train' antes de 'make test'.")  # noqa: E501
    from churn_nn.api.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_customer():
    return SAMPLE_CUSTOMER.copy()
