"""Simple scalar estimators. No forecasting, selection or tracking dependencies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from financial_mdl.contracts import BaseModel

PERSISTENCE_WINDOWS = {"persistence-5": 5, "persistence-20": 20}
ESTIMATOR_NAMES = ("naive", "ridge", "tree", *PERSISTENCE_WINDOWS)


class VolatilityPersistence:
    """Copy a trailing RMS volatility feature at each origin, without learning from y.

    Source features use annualization=252. Convert explicitly to the target's
    annualization. Every future horizon receives the same current volatility estimate.
    Inputs must retain original volatility units; use IdentityFeatures in the runner.
    """

    def __init__(self, window: int, annualization: int = 252):
        if type(window) is not int or window not in (5, 20):
            raise ValueError("Persistence window must be 5 or 20 sessions")
        if type(annualization) is not int or annualization < 1:
            raise ValueError("annualization must be a positive integer")
        self.window = window
        self.annualization = annualization
        self.feature = f"realized_volatility_{window}"

    def _values(self, X: pd.DataFrame) -> np.ndarray:
        if X.attrs.get("feature_transform") == "standard_scaler":
            raise ValueError("Persistence requires original feature units; use IdentityFeatures")
        if not X.columns.is_unique or self.feature not in X.columns:
            raise ValueError(f"Persistence requires feature {self.feature!r}")
        values = X[self.feature].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Persistence requires finite nonnegative volatility features")
        return values * np.sqrt(self.annualization / 252)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> VolatilityPersistence:
        if X.empty or y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError("Persistence training targets must have finite shape (N,)")
        self._values(X)
        self.columns_ = tuple(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "columns_") or tuple(X.columns) != self.columns_:
            raise ValueError("Estimator must be fitted with matching ordered columns")
        return self._values(X)

    def get_params(self) -> dict[str, Any]:
        return {
            "estimator": "volatility_persistence",
            "version": "1",
            "parameters": {
                "window_sessions": self.window,
                "feature": self.feature,
                "source_annualization": 252,
                "target_annualization": self.annualization,
                "learns_from_targets": False,
            },
        }


class SklearnAdapter:
    """Clone an external estimator; require explicitly scalar (N,) predictions."""

    def __init__(self, estimator: Any):
        self.estimator = clone(estimator)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SklearnAdapter:
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError("Scalar estimators require finite training targets of shape (N,)")
        self.columns_ = tuple(X.columns)
        if hasattr(self, "constant_"):
            del self.constant_
        if not self.columns_:
            if not isinstance(self.estimator, Ridge):
                raise ValueError("This adapter supports empty feature subsets only for Ridge")
            self.constant_ = float(y.mean())
            return self
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "columns_") or tuple(X.columns) != self.columns_:
            raise ValueError("Estimator must be fitted with matching ordered columns")
        if hasattr(self, "constant_"):
            return np.full(len(X), self.constant_)
        values = np.asarray(self.estimator.predict(X))
        if values.shape != (len(X),):
            raise ValueError("Scalar estimator must return exactly shape (N,)")
        if not np.isfinite(values).all():
            raise ValueError("Estimator produced nonfinite predictions")
        return values

    def get_params(self) -> dict[str, Any]:
        return {
            "adapter": "sklearn_scalar_v1",
            "effective_model": (
                "intercept_only_training_mean"
                if hasattr(self, "constant_")
                else "configured_estimator"
            ),
            "estimator": f"{type(self.estimator).__module__}.{type(self.estimator).__qualname__}",
            "parameters": self.estimator.get_params(deep=True),
        }

    def get_feature_importance(self) -> dict[str, Any] | None:
        """Optional fitted diagnostics, with their meaning preserved explicitly."""
        if not hasattr(self, "columns_"):
            raise ValueError("Estimator has not been fitted")
        if hasattr(self, "constant_"):
            return None
        if hasattr(self.estimator, "feature_importances_"):
            kind, values = "impurity_importance", self.estimator.feature_importances_
        elif hasattr(self.estimator, "coef_"):
            kind, values = "coefficient_in_transformed_feature_space", self.estimator.coef_
        else:
            return None
        if np.asarray(values).shape != (len(self.columns_),):
            raise ValueError("Feature diagnostic shape does not match fitted columns")
        return {"kind": kind, "features": dict(zip(self.columns_, values.tolist(), strict=True))}


def make_estimator(name: str, seed: int = 42, annualization: int = 252) -> BaseModel:
    if name in PERSISTENCE_WINDOWS:
        return VolatilityPersistence(PERSISTENCE_WINDOWS[name], annualization=annualization)
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
        raise ValueError(f"Unknown estimator {name!r}; choose from {ESTIMATOR_NAMES}")
    return SklearnAdapter(estimators[name]())
