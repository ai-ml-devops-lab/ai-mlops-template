from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, cast

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from mlops_template.config import settings
from mlops_template.schema import (
    FeatureInfoResponse,
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from mlops_template.train import get_feature_names, train_model

app = FastAPI(title="AI MLOps Template API", version="0.1.0")


@lru_cache(maxsize=1)
def _load_model() -> Any:
    if not settings.model_path.exists():
        train_model()
    return joblib.load(settings.model_path)


def _load_version() -> str:
    if settings.version_path.exists():
        payload = json.loads(settings.version_path.read_text(encoding="utf-8"))
        return str(payload.get("model_version", settings.model_version))
    return settings.model_version


def _load_metadata() -> dict[str, Any]:
    if not settings.version_path.exists():
        train_model()
    payload = json.loads(settings.version_path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], payload)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_path=str(settings.model_path),
        model_available=settings.model_path.exists(),
    )


@app.get("/features", response_model=FeatureInfoResponse)
def features() -> FeatureInfoResponse:
    feature_names = get_feature_names()
    return FeatureInfoResponse(feature_count=len(feature_names), feature_names=feature_names)


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    return MetadataResponse.model_validate(_load_metadata())


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = _load_model()
    try:
        features = np.asarray([request.features], dtype=float)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(
        prediction=prediction,
        probability_positive=probability,
        model_version=_load_version(),
    )
