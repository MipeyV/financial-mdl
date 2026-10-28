"""Versioned daily OHLCV snapshots and optional data providers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from financial_mdl.schemas import validate_daily_index

OHLCV = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class DatasetMetadata:
    asset: str
    version: str
    source: str
    price_basis: str
    retrieved_at: str

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError("Dataset provenance fields must be nonempty strings")


@dataclass(frozen=True)
class MarketDataset:
    frame: pd.DataFrame
    metadata: DatasetMetadata

    def __post_init__(self) -> None:
        validate_daily_index(self.frame.index)
        if tuple(self.frame.columns) != OHLCV:
            raise ValueError(f"OHLCV columns must be exactly {OHLCV}, in that order")
        if any(not pd.api.types.is_numeric_dtype(d) for d in self.frame.dtypes):
            raise ValueError("OHLCV values must be numeric")
        values = self.frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("OHLCV values must be finite")
        if (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
            raise ValueError("Prices must be positive and volume nonnegative")
        f = self.frame
        if (f.high < f[["open", "close", "low"]].max(axis=1)).any() or (
            f.low > f[["open", "close", "high"]].min(axis=1)
        ).any():
            raise ValueError("Inconsistent OHLC price bounds")
        object.__setattr__(self, "frame", self.frame.astype(float).copy())

    @property
    def fingerprint(self) -> str:
        """Canonical SHA-256 over provenance, ordered dates, columns and float64 values."""
        digest = hashlib.sha256()
        digest.update(json.dumps(asdict(self.metadata), sort_keys=True).encode())
        digest.update(json.dumps(OHLCV).encode())
        digest.update(self.frame.index.to_numpy(dtype="datetime64[ns]").astype("<i8").tobytes())
        digest.update(self.frame.to_numpy(dtype="<f8").tobytes())
        return digest.hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(path)
        path.with_suffix(".metadata.json").write_text(
            json.dumps(asdict(self.metadata), indent=2), encoding="utf-8"
        )


class MarketDataProvider(Protocol):
    def load(self) -> MarketDataset: ...


@dataclass(frozen=True)
class ParquetMarketDataProvider:
    path: Path

    def load(self) -> MarketDataset:
        metadata = json.loads(self.path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        return MarketDataset(pd.read_parquet(self.path), DatasetMetadata(**metadata))


@dataclass(frozen=True)
class YFinanceMarketDataProvider:
    """Optional convenience source. Freeze each vendor snapshot before benchmarking.

    Unadjusted vendor OHLC is not a guarantee of point-in-time corporate-action data.
    This provider makes no fundamental-data or historical-universe guarantees.
    """

    asset: str
    start: str
    end: str
    version: str

    def load(self) -> MarketDataset:
        try:
            import yfinance as yf
        except ImportError as error:
            raise ImportError("Install the market extra: uv sync --extra market") from error

        frame = yf.Ticker(self.asset).history(
            start=self.start, end=self.end, interval="1d", auto_adjust=False, actions=False
        )
        if frame.empty:
            raise ValueError("Market provider returned no observations")
        # Vendor dates are exchange-local session labels, not UTC event timestamps.
        frame.index = frame.index.tz_localize(None).normalize()
        frame = frame.rename(columns=str.lower).loc[:, list(OHLCV)]
        return MarketDataset(
            frame,
            DatasetMetadata(
                self.asset,
                self.version,
                "yfinance",
                "vendor_unadjusted_ohlc",
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
