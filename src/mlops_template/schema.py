from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request payload for a single model prediction."""

    features: list[float] = Field(..., min_length=30, max_length=30)

    @field_validator("features")
    @classmethod
    def finite_values(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("features must contain only finite values")
        return values


class PredictionResponse(BaseModel):
    """Prediction output returned by the API."""

    prediction: int
    probability_positive: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class HealthResponse(BaseModel):
    """Health endpoint payload."""

    status: str
    model_path: str
    model_available: bool


class FeatureInfoResponse(BaseModel):
    """Feature contract used by the training pipeline and prediction API."""

    feature_count: int
    feature_names: list[str]


class MetadataResponse(BaseModel):
    """Versioned model metadata generated during training."""

    model_version: str
    created_at: str
    model_sha256: str
    training_dataset: str
    model_type: str
    feature_names: list[str]
    metrics: dict[str, float | int]
