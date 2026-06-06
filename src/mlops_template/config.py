from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for training, serving, and tracking.

    The settings object stores artifact paths, the model version label, and the
    optional MLflow tracking URI. See the Sphinx reference page for a complete
    field-by-field description.
    """

    artifact_dir: Path = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
    model_path: Path = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    metrics_path: Path = Path(os.getenv("METRICS_PATH", "artifacts/metrics.json"))
    version_path: Path = Path(os.getenv("VERSION_PATH", "artifacts/model_version.json"))
    model_version: str = os.getenv("MODEL_VERSION", "0.1.0")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the current process environment.

        Returns:
            Settings populated from environment variables with local defaults.
        """
        return cls(
            artifact_dir=Path(os.getenv("ARTIFACT_DIR", "artifacts")),
            model_path=Path(os.getenv("MODEL_PATH", "artifacts/model.joblib")),
            metrics_path=Path(os.getenv("METRICS_PATH", "artifacts/metrics.json")),
            version_path=Path(os.getenv("VERSION_PATH", "artifacts/model_version.json")),
            model_version=os.getenv("MODEL_VERSION", "0.1.0"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        )

    def ensure_artifact_dirs(self) -> None:
        """Create all artifact parent directories required by the workflow.

        This method creates ``artifact_dir`` and the parent directories for the
        model, metrics, and version metadata files.
        """
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.version_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings.from_env()
