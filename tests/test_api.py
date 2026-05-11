from __future__ import annotations

from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from mlops_template.api import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    client = TestClient(app)
    sample = load_breast_cancer().data[0].tolist()
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in {0, 1}
    assert 0 <= payload["probability_positive"] <= 1
