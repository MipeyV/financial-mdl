"""Offline smoke fixtures, not substitutes for the real-market research experiment."""

import numpy as np
import pandas as pd

from financial_mdl.data import DatasetMetadata, MarketDataset


def synthetic_market(n: int = 360, seed: int = 42) -> MarketDataset:
    if n < 80:
        raise ValueError("Synthetic example needs at least 80 sessions")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    scale = 0.007 + 0.006 * (1 + np.sin(np.arange(n) / 25))
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, scale)))
    opening = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame(
        {
            "open": opening,
            "high": np.maximum(opening, close) * 1.003,
            "low": np.minimum(opening, close) * 0.997,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )
    return MarketDataset(
        frame,
        DatasetMetadata(
            "SYNTHETIC",
            f"synthetic-v1-n{n}-s{seed}",
            "seeded_numpy_generator_v1",
            "synthetic",
            "2020-01-01T00:00:00+00:00",
        ),
    )


def controlled_regression(scenario: str, seed: int, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Independent regression observations; a control experiment, not market forecasts.

    signal_a_copy is exactly redundant. The three noise columns are independent
    of both true signals. A fixed seed shares the design/noise across scenarios.
    """
    if scenario not in {"signal", "noise_only", "weak_signal"}:
        raise ValueError("Unknown controlled-regression scenario")
    if type(n) is not int or n < 4 or type(seed) is not int or seed < 0:
        raise ValueError("Require n >= 4 and a nonnegative integer seed")
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, 5))
    X = pd.DataFrame(
        {
            "signal_a": values[:, 0],
            "signal_b": values[:, 1],
            "signal_a_copy": values[:, 0],
            "noise_0": values[:, 2],
            "noise_1": values[:, 3],
            "noise_2": values[:, 4],
        }
    )
    y = 1.0 + 0.4 * rng.normal(size=n)
    if scenario == "signal":
        y += 2.0 * values[:, 0] - 1.5 * values[:, 1]
    elif scenario == "weak_signal":
        y += 0.05 * values[:, 0]
    return X, y
