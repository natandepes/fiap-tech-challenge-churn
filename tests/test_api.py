# tests/test_api.py


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


def test_predict_campo_faltando_retorna_422(client):
    payload_incompleto = {"tenure": 12, "MonthlyCharges": 65.5}  # falta TotalCharges e outros  # noqa: E501
    response = client.post("/predict", json=payload_incompleto)
    assert response.status_code == 422
