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
