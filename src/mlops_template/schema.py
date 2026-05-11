from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=30, max_length=30)

    @field_validator("features")
    @classmethod
    def finite_values(cls, values: list[float]) -> list[float]:
        if not all(value == value and abs(value) != float("inf") for value in values):
            raise ValueError("features must contain only finite values")
        return values


class PredictionResponse(BaseModel):
    prediction: int
    probability_positive: float
    model_version: str
