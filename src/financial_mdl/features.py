"""Causal, unfitted daily feature definitions and train-only fitted components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureFamily(StrEnum):
    MARKET = "MarketFeatures"
    FUNDAMENTAL = "FundamentalFeatures"
    STATISTICAL = "StatisticalFeatures"
    ALGORITHMIC_COMPLEXITY = "AlgorithmicComplexityFeatures"


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    family: FeatureFamily
    definition: str
    required_source_columns: tuple[str, ...]
    lookback_sessions: int
    availability: str = "after_origin_session_close"
    output_type: str = "float64"
    missing_value_policy: str = "drop_origin_for_all_models_in_protocol"
    version: str = "1"


FEATURES = (
    FeatureDefinition(
        "log_return_1", FeatureFamily.MARKET, "log(close[t]/close[t-1])", ("close",), 1
    ),
    FeatureDefinition(
        "log_return_5", FeatureFamily.MARKET, "log(close[t]/close[t-5])", ("close",), 5
    ),
    FeatureDefinition(
        "volume_change_1", FeatureFamily.MARKET, "volume[t]/volume[t-1]-1", ("volume",), 1
    ),
    FeatureDefinition(
        "realized_volatility_5",
        FeatureFamily.STATISTICAL,
        "sqrt(252 * mean(last 5 squared log returns, including t))",
        ("close",),
        5,
    ),
    FeatureDefinition(
        "realized_volatility_20",
        FeatureFamily.STATISTICAL,
        "sqrt(252 * mean(last 20 squared log returns, including t))",
        ("close",),
        20,
    ),
)
DEFAULT_FEATURES = tuple(f.name for f in FEATURES)
FEATURE_VERSION = "1"


def feature_manifest(names: tuple[str, ...]) -> list[dict]:
    definitions = {f.name: f for f in FEATURES}
    if any(name not in definitions for name in names):
        raise ValueError("Requested features are unavailable in this implementation")
    return [asdict(definitions[name]) for name in names]


def market_features(frame: pd.DataFrame, names: tuple[str, ...] = DEFAULT_FEATURES) -> pd.DataFrame:
    feature_manifest(names)
    returns = np.log(frame.close).diff()
    features = pd.DataFrame(
        {
            "log_return_1": returns,
            "log_return_5": np.log(frame.close).diff(5),
            "volume_change_1": frame.volume.pct_change(fill_method=None),
            "realized_volatility_5": np.sqrt(252 * returns.pow(2).rolling(5).mean()),
            "realized_volatility_20": np.sqrt(252 * returns.pow(2).rolling(20).mean()),
        },
        index=frame.index,
    )
    return features.loc[:, list(names)].replace([np.inf, -np.inf], np.nan)


class IdentityFeatureSelector:
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> IdentityFeatureSelector:
        if X_train.empty or len(X_train) != len(y_train) or not X_train.columns.is_unique:
            raise ValueError("Selector training inputs must have aligned rows and unique columns")
        self.columns_ = tuple(X_train.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "columns_"):
            raise ValueError("Selector has not been fitted")
        if tuple(X.columns) != self.columns_:
            raise ValueError("Selector input columns/order differ from training")
        return X.copy()

    def get_selected_features(self) -> tuple[str, ...]:
        if not hasattr(self, "columns_"):
            raise ValueError("Selector has not been fitted")
        return self.columns_

    def get_params(self) -> dict[str, Any]:
        return {"name": "identity", "version": "1"}


class StandardizeFeatures:
    """Example fitted transform. The runner fits a fresh instance on training only."""

    def fit(self, X: pd.DataFrame) -> StandardizeFeatures:
        self.columns_ = tuple(X.columns)
        self.scaler_ = StandardScaler().fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "scaler_") or tuple(X.columns) != self.columns_:
            raise ValueError("Transformer must be fitted with matching ordered columns")
        return pd.DataFrame(self.scaler_.transform(X), index=X.index, columns=X.columns)

    def get_params(self) -> dict[str, Any]:
        return {"name": "standard_scaler", "version": "1", "with_mean": True, "with_std": True}
