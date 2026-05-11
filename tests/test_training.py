from __future__ import annotations

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

    version = train_model()

    assert version["metrics"]["roc_auc"] > 0.9
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "MODEL_CARD.md").exists()
