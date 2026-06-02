from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

import mlops_template.api as api
import mlops_template.train as train
from mlops_template.config import Settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    test_settings = Settings(
        artifact_dir=tmp_path,
        model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
        version_path=tmp_path / "model_version.json",
    )
    monkeypatch.setattr(api, "settings", test_settings)
    monkeypatch.setattr(train, "settings", test_settings)
    monkeypatch.setattr(train, "_log_to_mlflow", lambda metrics, version: None)
    api._load_model.cache_clear()
    yield TestClient(api.app)
    api._load_model.cache_clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_available" in response.json()


def test_features(client):
    response = client.get("/features")
    assert response.status_code == 200
    payload = response.json()
    assert payload["feature_count"] == 30
    assert len(payload["feature_names"]) == 30


def test_predict(client):
    sample = load_breast_cancer().data[0].tolist()
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in {0, 1}
    assert 0 <= payload["probability_positive"] <= 1


def test_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_type"] == "logistic_regression"
    assert len(payload["feature_names"]) == 30
