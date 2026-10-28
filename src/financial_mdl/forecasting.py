"""One aggregate output or independent direct models for each horizon/variable."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from financial_mdl.contracts import ForecastStrategy, ModelFactory
from financial_mdl.schemas import Forecast, ForecastCoordinates, Targets


def _check_rows(X: pd.DataFrame, coordinates: ForecastCoordinates) -> None:
    if not X.index.equals(coordinates.origins):
        raise ValueError("Feature rows must exactly match forecast origins in order")


class OneStepStrategy:
    """One forecast output, which may aggregate a multi-session target window."""

    name = "one_step"

    def fit(self, factory: ModelFactory, X: pd.DataFrame, targets: Targets) -> None:
        coordinates = targets.observations.coordinates
        _check_rows(X, coordinates)
        if coordinates.shape[1:] != (1, 1):
            raise ValueError("One-step strategy requires exactly one horizon and one variable")
        self.horizons_ = coordinates.horizons
        self.targets_ = coordinates.target_names
        self.model_ = factory()
        self.model_.fit(X, targets.observations.values[:, 0, 0])

    def predict(self, X: pd.DataFrame, coordinates: ForecastCoordinates) -> Forecast:
        _check_rows(X, coordinates)
        if not hasattr(self, "model_"):
            raise ValueError("Strategy has not been fitted")
        if coordinates.horizons != self.horizons_ or coordinates.target_names != self.targets_:
            raise ValueError("Prediction horizon/target labels differ from training")
        values = np.asarray(self.model_.predict(X))
        if values.shape != (len(X),):
            raise ValueError("Scalar estimator predictions must have shape (N,)")
        # Explicit scalar-to-contract mapping; never infer shapes from total size.
        return Forecast(values[:, None, None], coordinates)

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name, "version": "1"}


class DirectMultiHorizonStrategy:
    """A separate scalar estimator per (horizon, variable), never a joint model."""

    name = "direct_multi_horizon"

    def fit(self, factory: ModelFactory, X: pd.DataFrame, targets: Targets) -> None:
        coordinates = targets.observations.coordinates
        _check_rows(X, coordinates)
        self.horizons_ = coordinates.horizons
        self.targets_ = coordinates.target_names
        self.models_ = {}
        for h in range(coordinates.shape[1]):
            for k in range(coordinates.shape[2]):
                model = factory()
                if any(model is previous for previous in self.models_.values()):
                    raise ValueError("Estimator factory must return a fresh model per output")
                model.fit(X, targets.observations.values[:, h, k])
                self.models_[h, k] = model

    def predict(self, X: pd.DataFrame, coordinates: ForecastCoordinates) -> Forecast:
        _check_rows(X, coordinates)
        if not hasattr(self, "models_"):
            raise ValueError("Strategy has not been fitted")
        if coordinates.horizons != self.horizons_ or coordinates.target_names != self.targets_:
            raise ValueError("Prediction horizon/target labels differ from training")
        values = np.empty(coordinates.shape)
        for (h, k), model in self.models_.items():
            predictions = np.asarray(model.predict(X))
            if predictions.shape != (len(X),):
                raise ValueError("Direct scalar estimator predictions must have shape (N,)")
            values[:, h, k] = predictions
        return Forecast(values, coordinates)

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name, "version": "1", "joint": False}


class ForecastingModel:
    def __init__(self, estimator: ModelFactory, strategy: ForecastStrategy):
        self.estimator = estimator
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, targets: Targets) -> ForecastingModel:
        self.strategy.fit(self.estimator, X, targets)
        return self

    def predict(self, X: pd.DataFrame, coordinates: ForecastCoordinates) -> Forecast:
        return self.strategy.predict(X, coordinates)
