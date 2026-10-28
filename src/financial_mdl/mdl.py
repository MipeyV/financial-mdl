"""Explicit two-part reference code for exhaustive scalar Ridge feature selection.

X, its ordered feature universe, n and the coding specification are shared side
information. Model parameters are actual IEEE-754 binary64 words. Target symbols
use a rounded grid and a factorized two-sided geometric residual distribution.
Target bits are Shannon lengths, not a measured compressed file size or NML.
"""

from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MDLCodingSpec:
    target_precision: float = 0.0001
    ridge_alpha: float = 1.0
    max_features: int = 12
    version: str = "ridge_binary64_geometric_v1"

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_precision) or self.target_precision <= 0:
            raise ValueError("target_precision must be finite and positive")
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive")
        if type(self.max_features) is not int or not 1 <= self.max_features <= 16:
            raise ValueError("max_features must be an integer between 1 and 16")
        if self.version != "ridge_binary64_geometric_v1":
            raise ValueError("Unsupported MDL coding version")


def subset_code_bits(universe_size: int, subset_size: int) -> int:
    """Fixed-width k, then fixed-width lexicographic combination rank; Kraft-valid."""
    if (
        type(universe_size) is not int
        or type(subset_size) is not int
        or not 0 <= subset_size <= universe_size
    ):
        raise ValueError("Require integer 0 <= subset_size <= universe_size")
    return universe_size.bit_length() + (math.comb(universe_size, subset_size) - 1).bit_length()


def quantize(values: np.ndarray, precision: float) -> np.ndarray:
    """Nearest grid symbol, ties to even, with an explicit safe integer range."""
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("precision must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        symbols = np.rint(np.asarray(values, dtype=float) / precision)
    if not np.isfinite(symbols).all() or (np.abs(symbols) > 2**50).any():
        raise ValueError("Values exceed the coding grid's safe integer range")
    return symbols.astype(np.int64)


def geometric_target_bits(residuals: np.ndarray, q: float) -> tuple[float, int]:
    """P(r)=(1-q)/(1+q)*q**abs(r), r in Z; ceil joint -log2 P is a Shannon length."""
    if not np.isfinite(q) or not 0 <= q < 1:
        raise ValueError("Geometric q must satisfy 0 <= q < 1")
    residuals = np.asarray(residuals)
    if residuals.ndim != 1 or not np.issubdtype(residuals.dtype, np.integer):
        raise ValueError("Residual symbols must be a one-dimensional integer array")
    absolute_sum = float(np.abs(residuals.astype(float)).sum())
    if q == 0:
        if absolute_sum != 0:
            raise ValueError("q=0 cannot encode nonzero residuals")
        return 0.0, 0
    ideal = (
        len(residuals) * (math.log1p(q) - math.log1p(-q)) - absolute_sum * math.log(q)
    ) / math.log(2)
    if not math.isfinite(ideal) or ideal < 0:
        raise ValueError("Nonfinite or negative target code length")
    return ideal, math.ceil(ideal)


def _reference_score(
    standardized: np.ndarray,
    y: np.ndarray,
    names: tuple[str, ...],
    positions: tuple[int, ...],
    spec: MDLCodingSpec,
) -> dict[str, Any]:
    if positions:
        model = Ridge(alpha=spec.ridge_alpha).fit(standardized[:, positions], y)
        coefficients = np.asarray(model.coef_, dtype=float)
        intercept = float(model.intercept_)
        predicted = model.predict(standardized[:, positions])
    else:
        coefficients = np.empty(0)
        intercept = float(y.mean())
        predicted = np.full(len(y), intercept)
    residuals = quantize(y, spec.target_precision) - quantize(predicted, spec.target_precision)
    average_absolute = float(np.abs(residuals).mean())
    q = average_absolute / (math.hypot(average_absolute, 1) + 1)
    ideal, target_bits = geometric_target_bits(residuals, q)
    # k is already known from the subset code; the decoder reads k+2 binary64 words.
    parameters = struct.pack(f">{len(positions) + 2}d", intercept, *coefficients, q)
    model_bits = len(parameters) * 8
    subset_bits = subset_code_bits(len(names), len(positions))
    return {
        "features": [names[i] for i in positions],
        "n_features": len(positions),
        "subset_bits": subset_bits,
        "model_bits": model_bits,
        "target_bits": target_bits,
        "target_ideal_bits": ideal,
        "total_bits": subset_bits + model_bits + target_bits,
        "training_mae": float(np.abs(predicted - y).mean()),
        "training_rmse": float(np.sqrt(np.mean((predicted - y) ** 2))),
        "parameter_hex": parameters.hex(),
        "residual_q": q,
    }


class ExhaustiveMDLSelector:
    """Score every subset, including the intercept-only model, on training rows only.

    The reference estimator is standardized Ridge with a fixed alpha, independently
    of a downstream estimator. Code lengths describe this training reference code.
    A decoder derives scaler statistics from the shared training X, so these do not
    require parameter words. Selection ties prefer fewer features, then input order.
    """

    def __init__(self, spec: MDLCodingSpec | None = None):
        self.spec = spec if spec is not None else MDLCodingSpec()

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> ExhaustiveMDLSelector:
        if (
            not isinstance(X_train, pd.DataFrame)
            or len(X_train) < 2
            or not X_train.columns.is_unique
            or not len(X_train.columns)
            or any(not isinstance(c, str) or not c for c in X_train.columns)
        ):
            raise ValueError("MDL requires at least two rows and a unique named feature universe")
        if len(X_train.columns) > self.spec.max_features:
            raise ValueError("Feature universe exceeds the exhaustive-search max_features limit")
        y = np.asarray(y_train)
        if y.shape == (len(X_train), 1, 1):
            y = y[:, 0, 0]
        if y.shape != (len(X_train),):
            raise ValueError("MDL v1 requires one scalar target: (N,) or (N, 1, 1)")
        values = X_train.to_numpy()
        if not np.isrealobj(values) or not np.isrealobj(y):
            raise ValueError("MDL inputs must be finite real numbers")
        values, y = values.astype(float), y.astype(float)
        if not np.isfinite(values).all() or not np.isfinite(y).all():
            raise ValueError("MDL inputs must be finite real numbers")
        quantize(y, self.spec.target_precision)
        # Column-wise statistics depend only on the shared training design matrix.
        standardized = StandardScaler().fit_transform(values)
        names = tuple(X_train.columns)
        candidates = []
        best = None
        best_positions = None
        for size in range(len(names) + 1):
            for positions in combinations(range(len(names)), size):
                candidate = _reference_score(standardized, y, names, positions, self.spec)
                candidates.append(candidate)
                key = candidate["total_bits"], size, positions
                if best is None or key < best:
                    best, best_positions = key, positions
        self.columns_ = names
        self.selected_ = tuple(names[i] for i in best_positions)
        self.candidates_ = candidates
        self.training_rows_ = len(X_train)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_"):
            raise ValueError("MDL selector has not been fitted")
        if tuple(X.columns) != self.columns_:
            raise ValueError("Selector input columns/order differ from training")
        return X.loc[:, list(self.selected_)].copy()

    def get_selected_features(self) -> tuple[str, ...]:
        if not hasattr(self, "selected_"):
            raise ValueError("MDL selector has not been fitted")
        return self.selected_

    def get_params(self) -> dict[str, Any]:
        return {"name": "exhaustive_mdl", **asdict(self.spec)}

    def get_diagnostics(self) -> dict[str, Any]:
        selected = self.get_selected_features()
        # Rebuild containers to keep callers from mutating fitted search results.
        candidates = [{**row, "features": list(row["features"])} for row in self.candidates_]
        return {
            "coding_spec": self.get_params(),
            "training_rows": self.training_rows_,
            "feature_universe": list(self.columns_),
            "selected_features": list(selected),
            "selected": next(row for row in candidates if tuple(row["features"]) == selected),
            "full": candidates[-1],
            "constant": candidates[0],
            "candidates": candidates,
        }
