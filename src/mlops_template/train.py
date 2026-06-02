from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_template.config import settings

MODEL_TYPE = "logistic_regression"
TRAINING_DATASET = "sklearn.datasets.load_breast_cancer"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_feature_names() -> list[str]:
    """Return the ordered feature contract used by the demo model."""
    data = load_breast_cancer()
    return [str(name) for name in data.feature_names]


def build_pipeline() -> Pipeline:
    """Build the deterministic scikit-learn pipeline used for training."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def evaluate_model(pipeline: Pipeline, x_test: Any, y_test: Any) -> dict[str, float | int]:
    """Evaluate a fitted pipeline with the metrics exposed by the template."""
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "n_test": int(x_test.shape[0]),
        "n_features": int(x_test.shape[1]),
    }


def train_model() -> dict[str, Any]:
    """Train the demo classifier and write model, metrics, metadata and model card."""
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    metrics = evaluate_model(pipeline, x_test, y_test)
    metrics["n_train"] = int(x_train.shape[0])

    settings.ensure_artifact_dirs()
    joblib.dump(pipeline, settings.model_path)
    settings.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    version = {
        "model_version": settings.model_version,
        "created_at": datetime.now(UTC).isoformat(),
        "model_sha256": _sha256(settings.model_path),
        "training_dataset": TRAINING_DATASET,
        "model_type": MODEL_TYPE,
        "feature_names": get_feature_names(),
        "metrics": metrics,
    }
    settings.version_path.write_text(json.dumps(version, indent=2), encoding="utf-8")
    _write_model_card(settings.artifact_dir / "MODEL_CARD.md", version)
    _log_to_mlflow(metrics, version)
    return version


def _write_model_card(path: Path, version: dict[str, Any]) -> None:
    metrics = version["metrics"]
    path.write_text(
        f"""# Model Card: Breast Cancer Classifier Demo

## Intended use

Portfolio demonstration for a binary classification MLOps workflow. Not for medical use.

## Version

- Version: `{version['model_version']}`
- Artifact SHA-256: `{version['model_sha256']}`
- Model type: `{version['model_type']}`
- Training dataset: `{version['training_dataset']}`

## Metrics

- Accuracy: `{metrics['accuracy']:.4f}`
- F1: `{metrics['f1']:.4f}`
- ROC AUC: `{metrics['roc_auc']:.4f}`

## Limitations

This model uses a small built-in dataset and is only intended to 
demonstrate engineering practices.
""",
        encoding="utf-8",
    )


def _log_to_mlflow(metrics: dict[str, float | int], version: dict[str, Any]) -> None:
    try:
        import mlflow
    except Exception:
        return

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("ai-mlops-template")
    with mlflow.start_run(run_name=f"model-{settings.model_version}"):
        mlflow.log_params(
            {
                "model_type": MODEL_TYPE,
                "dataset": version["training_dataset"],
                "model_version": version["model_version"],
            }
        )
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        mlflow.log_artifact(str(settings.model_path))
        mlflow.log_artifact(str(settings.version_path))


def main() -> None:
    version = train_model()
    print(json.dumps(version, indent=2))


if __name__ == "__main__":
    main()
