"""Simple scalar estimators. No forecasting, selection or tracking dependencies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


class SklearnAdapter:
    """Clone an external estimator; require explicitly scalar (N,) predictions."""

    def __init__(self, estimator: Any):
        self.estimator = clone(estimator)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SklearnAdapter:
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError("Scalar estimators require finite training targets of shape (N,)")
        self.columns_ = tuple(X.columns)
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "columns_") or tuple(X.columns) != self.columns_:
            raise ValueError("Estimator must be fitted with matching ordered columns")
        values = np.asarray(self.estimator.predict(X))
        if values.shape != (len(X),):
            raise ValueError("Scalar estimator must return exactly shape (N,)")
        if not np.isfinite(values).all():
            raise ValueError("Estimator produced nonfinite predictions")
        return values

    def get_params(self) -> dict[str, Any]:
        return {
            "adapter": "sklearn_scalar_v1",
            "estimator": f"{type(self.estimator).__module__}.{type(self.estimator).__qualname__}",
            "parameters": self.estimator.get_params(deep=True),
        }

    def get_feature_importance(self) -> dict[str, Any] | None:
        """Optional fitted diagnostics, with their meaning preserved explicitly."""
        if not hasattr(self, "columns_"):
            raise ValueError("Estimator has not been fitted")
        if hasattr(self.estimator, "feature_importances_"):
            kind, values = "impurity_importance", self.estimator.feature_importances_
        elif hasattr(self.estimator, "coef_"):
            kind, values = "coefficient_in_transformed_feature_space", self.estimator.coef_
        else:
            return None
        if np.asarray(values).shape != (len(self.columns_),):
            raise ValueError("Feature diagnostic shape does not match fitted columns")
        return {"kind": kind, "features": dict(zip(self.columns_, values.tolist(), strict=True))}


def make_estimator(name: str, seed: int = 42) -> SklearnAdapter:
    estimators = {
        "naive": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: Ridge(alpha=1.0),
        "tree": lambda: RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=1,
        ),
    }
    if name not in estimators:
        raise ValueError(f"Unknown estimator {name!r}; choose from {tuple(estimators)}")
    return SklearnAdapter(estimators[name]())
