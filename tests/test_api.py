# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True
        assert response.json()["scaler_loaded"] is True


def test_predict_returns_valid_response():
    # A legitimate-looking transaction (all near-zero PCA components)
    payload = {
        "Time": 100.0,
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Amount": 50.0,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert isinstance(data["is_fraud"], bool)
