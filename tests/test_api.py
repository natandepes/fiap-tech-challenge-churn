# tests/test_api.py
import time

import numpy as np


def test_health_retorna_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_retorna_campos_esperados(client, sample_customer):
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert "churn" in data
    assert "probability" in data
    assert "threshold" in data
    assert "model_version" in data
    assert isinstance(data["churn"], bool)
    assert 0.0 <= data["probability"] <= 1.0
    assert isinstance(data["model_version"], str) and data["model_version"]


def test_predict_campo_faltando_retorna_422(client):
    payload_incompleto = {"tenure": 12, "MonthlyCharges": 65.5}  # falta TotalCharges e outros  # noqa: E501
    response = client.post("/predict", json=payload_incompleto)
    assert response.status_code == 422


def test_predict_risco_alto_maior_que_risco_baixo(client):
    """Modelo deve atribuir probabilidade maior a cliente de alto risco vs baixo risco."""  # noqa: E501
    alto_risco = {
        "tenure": 1,
        "MonthlyCharges": 95.0,
        "TotalCharges": 95.0,
        "SeniorCitizen": 1,
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    baixo_risco = {
        "tenure": 72,
        "MonthlyCharges": 45.0,
        "TotalCharges": 3240.0,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
    }
    prob_alto = client.post("/predict", json=alto_risco).json()["probability"]
    prob_baixo = client.post("/predict", json=baixo_risco).json()["probability"]
    assert prob_alto > prob_baixo


def test_latencia_p95_abaixo_de_200ms(client, sample_customer):
    """p95 de 100 requisições via TestClient (sem overhead de rede/uvicorn)."""
    tempos = []
    for _ in range(100):
        start = time.perf_counter()
        client.post("/predict", json=sample_customer)
        tempos.append((time.perf_counter() - start) * 1000)
    p95 = float(np.percentile(tempos, 95))
    assert p95 < 200, f"p95 = {p95:.1f}ms, acima do limite de 200ms"
