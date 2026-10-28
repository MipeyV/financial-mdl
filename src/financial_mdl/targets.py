"""Target construction is independent of models, tracking and orchestration."""

import numpy as np
import pandas as pd

from financial_mdl.schemas import (
    Forecast,
    ForecastCoordinates,
    Targets,
    TargetSpec,
    validate_daily_index,
)


def future_volatility(close: pd.Series, spec: TargetSpec, asset: str) -> Targets:
    """y[t,h] = sqrt(A/h * sum(log(C[t+j]/C[t+j-1])**2, j=1..h)).

    This is RMS realized volatility (no demeaning), not a sample standard deviation.
    Only complete future windows are returned; no partial tail labels or filling.
    """
    validate_daily_index(close.index)
    prices = close.to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Close prices must be finite and positive")
    n = len(close) - max(spec.horizons)
    if n < 1:
        raise ValueError("Not enough observations for target horizon")
    returns = np.diff(np.log(prices))
    values = np.empty((n, len(spec.horizons), 1))
    stamps = np.empty((n, len(spec.horizons)), dtype="datetime64[ns]")
    starts = np.empty_like(stamps)
    dates = close.index.to_numpy(dtype="datetime64[ns]")
    for j, horizon in enumerate(spec.horizons):
        squared = np.lib.stride_tricks.sliding_window_view(returns, horizon)[:n] ** 2
        values[:, j, 0] = np.sqrt(spec.annualization * squared.mean(axis=1))
        stamps[:, j] = dates[horizon : horizon + n]
        starts[:, j] = dates[1 : n + 1]
    coordinates = ForecastCoordinates(
        close.index[:n],
        spec.horizons,
        spec.target_names,
        stamps,
        (asset,) * n,
    )
    return Targets(Forecast(values, coordinates), starts)
