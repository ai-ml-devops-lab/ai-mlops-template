from __future__ import annotations

import json

from mlops_template.train import train_model


def test_train_model_creates_version(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("METRICS_PATH", str(tmp_path / "metrics.json"))
    monkeypatch.setenv("VERSION_PATH", str(tmp_path / "model_version.json"))

    import mlops_template.config as config
    import mlops_template.train as train

    new_settings = config.Settings(
        artifact_dir=tmp_path,
        model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
        version_path=tmp_path / "model_version.json",
    )
    monkeypatch.setattr(train, "settings", new_settings)
    monkeypatch.setattr(train, "_log_to_mlflow", lambda metrics, version: None)

    version = train_model()

    assert version["metrics"]["roc_auc"] > 0.9
    assert version["model_type"] == "logistic_regression"
    assert len(version["feature_names"]) == 30
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "MODEL_CARD.md").exists()

    metadata = json.loads((tmp_path / "model_version.json").read_text(encoding="utf-8"))
    assert metadata["model_sha256"] == version["model_sha256"]
