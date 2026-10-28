"""Strict timestamped (origin, horizon, variable) schemas; no inferred reshaping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def validate_daily_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError("Expected a nonempty DatetimeIndex")
    if index.tz is not None or index.hasnans or not index.equals(index.normalize()):
        raise ValueError("Daily session dates must be timezone-naive midnight dates without NaT")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Session dates must be unique and strictly chronological")


def validate_labels(labels: tuple, name: str) -> None:
    if not isinstance(labels, tuple) or not labels or len(set(labels)) != len(labels):
        raise ValueError(f"{name} must be a nonempty tuple of unique labels")


@dataclass(frozen=True)
class TargetSpec:
    """Each horizon aggregates future session returns into one scalar volatility.

    Five future sessions => horizons=(5,), outputs=1, variables=1, not five outputs.
    """

    horizons: tuple[int, ...] = (5,)
    target_names: tuple[str, ...] = ("realized_volatility",)
    aggregation: str = "sqrt_annualized_mean_squared_log_return"
    annualization: int = 252
    version: str = "1"

    def __post_init__(self) -> None:
        validate_labels(self.horizons, "horizons")
        if any(type(h) is not int or h < 1 for h in self.horizons):
            raise ValueError("Horizons must be positive session counts")
        if tuple(sorted(self.horizons)) != self.horizons:
            raise ValueError("Horizon labels must be strictly increasing")
        if self.target_names != ("realized_volatility",):
            raise ValueError("Version 1 target builder supports only realized_volatility")
        if self.aggregation != "sqrt_annualized_mean_squared_log_return" or self.version != "1":
            raise ValueError("Unsupported target aggregation/version")
        if type(self.annualization) is not int or self.annualization < 1:
            raise ValueError("annualization must be a positive integer")

    @property
    def number_of_outputs(self) -> int:
        return len(self.horizons) * len(self.target_names)

    @property
    def number_of_target_variables(self) -> int:
        return len(self.target_names)


@dataclass(frozen=True, eq=False)
class ForecastCoordinates:
    origins: pd.DatetimeIndex
    horizons: tuple[int, ...]
    target_names: tuple[str, ...]
    target_timestamps: np.ndarray
    asset_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_daily_index(self.origins)
        validate_labels(self.horizons, "horizons")
        validate_labels(self.target_names, "target_names")
        if any(type(h) is not int or h < 1 for h in self.horizons):
            raise ValueError("Horizon labels must be positive integers")
        if tuple(sorted(self.horizons)) != self.horizons:
            raise ValueError("Horizon labels must be increasing")
        if any(not isinstance(n, str) or not n for n in self.target_names):
            raise ValueError("Target names must be nonempty strings")
        stamps = np.asarray(self.target_timestamps)
        if stamps.shape != (len(self.origins), len(self.horizons)):
            raise ValueError("target_timestamps must have shape (N, H)")
        if stamps.dtype != np.dtype("datetime64[ns]") or np.isnat(stamps).any():
            raise ValueError("Target timestamps must be datetime64[ns] without NaT")
        if (stamps <= self.origins.to_numpy()[:, None]).any():
            raise ValueError("Target timestamps must follow forecast origins")
        if stamps.shape[1] > 1 and (stamps[:, 1:] <= stamps[:, :-1]).any():
            raise ValueError("Target timestamps must increase with horizon labels")
        if not isinstance(self.asset_ids, tuple) or len(self.asset_ids) != len(self.origins):
            raise ValueError("One asset identifier is required per origin")
        if any(not isinstance(a, str) or not a for a in self.asset_ids):
            raise ValueError("Asset identifiers must be nonempty strings")
        stamps = stamps.copy()
        stamps.setflags(write=False)
        object.__setattr__(self, "target_timestamps", stamps)
        object.__setattr__(self, "origins", self.origins.copy())

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.origins), len(self.horizons), len(self.target_names)

    def take(self, positions: np.ndarray) -> ForecastCoordinates:
        return ForecastCoordinates(
            self.origins[positions],
            self.horizons,
            self.target_names,
            self.target_timestamps[positions],
            tuple(self.asset_ids[i] for i in positions),
        )

    def assert_equal(self, other: ForecastCoordinates) -> None:
        if (
            not self.origins.equals(other.origins)
            or self.horizons != other.horizons
            or self.target_names != other.target_names
            or self.asset_ids != other.asset_ids
            or not np.array_equal(self.target_timestamps, other.target_timestamps)
        ):
            raise ValueError(
                "Forecast alignment differs: origins/horizons/targets/assets/timestamps"
            )


@dataclass(frozen=True, eq=False)
class Forecast:
    values: np.ndarray
    coordinates: ForecastCoordinates

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.shape != self.coordinates.shape:
            raise ValueError(f"Expected explicit forecast shape {self.coordinates.shape}")
        if not np.issubdtype(values.dtype, np.number) or not np.isrealobj(values):
            raise ValueError("Forecast values must be real numeric values")
        if not np.isfinite(values).all():
            raise ValueError("Forecast values must be finite")
        values = values.astype(float, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, eq=False)
class Targets:
    observations: Forecast
    observation_starts: np.ndarray

    def __post_init__(self) -> None:
        starts = np.asarray(self.observation_starts)
        coordinates = self.observations.coordinates
        if starts.shape != coordinates.target_timestamps.shape:
            raise ValueError("Observation starts must have shape (N, H)")
        if starts.dtype != np.dtype("datetime64[ns]") or np.isnat(starts).any():
            raise ValueError("Observation starts must be datetime64[ns] without NaT")
        if (starts <= coordinates.origins.to_numpy()[:, None]).any() or (
            starts > coordinates.target_timestamps
        ).any():
            raise ValueError("Target observation windows must follow their origins")
        starts = starts.copy()
        starts.setflags(write=False)
        object.__setattr__(self, "observation_starts", starts)

    def take(self, positions: np.ndarray) -> Targets:
        return Targets(
            Forecast(
                self.observations.values[positions], self.observations.coordinates.take(positions)
            ),
            self.observation_starts[positions],
        )
