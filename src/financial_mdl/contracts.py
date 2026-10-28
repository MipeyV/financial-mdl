"""Small, independent contracts. Forecast strategies compose estimator factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from financial_mdl.schemas import Forecast, ForecastCoordinates, Targets


class BaseModel(Protocol):
    """Scalar estimator contract; adapters must return one value per input row."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> BaseModel: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def get_params(self) -> dict[str, Any]: ...


ModelFactory = Callable[[], BaseModel]


class FeatureSelector(Protocol):
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> FeatureSelector: ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    def get_selected_features(self) -> tuple[str, ...]: ...

    def get_params(self) -> dict[str, Any]: ...


class FeatureTransformer(Protocol):
    """Fitted transformations must preserve row index and column labels."""

    def fit(self, X: pd.DataFrame) -> FeatureTransformer: ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    def get_params(self) -> dict[str, Any]: ...


class ForecastStrategy(Protocol):
    """Joint/recursive extensions implement this without changing the benchmark.

    Joint strategies must declare native joint output capability. Independent
    per-output wrappers are direct strategies, not genuinely joint estimators.
    Recursive strategies must supply an explicit causal lag/exogenous update policy.
    """

    name: str

    def fit(self, factory: ModelFactory, X: pd.DataFrame, targets: Targets) -> None: ...

    def predict(self, X: pd.DataFrame, coordinates: ForecastCoordinates) -> Forecast: ...

    def get_params(self) -> dict[str, Any]: ...


class Tracker(Protocol):
    def record(
        self, metadata: dict[str, Any], metrics: dict[str, float], directory: str
    ) -> None: ...
