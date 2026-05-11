from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
    model_path: Path = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    metrics_path: Path = Path(os.getenv("METRICS_PATH", "artifacts/metrics.json"))
    version_path: Path = Path(os.getenv("VERSION_PATH", "artifacts/model_version.json"))
    model_version: str = os.getenv("MODEL_VERSION", "0.1.0")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


settings = Settings()
