"""Immutable, serializable scientific rules, independent of candidate models."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from financial_mdl.features import (
    DEFAULT_FEATURES,
    FEATURE_VERSION,
    FeatureFamily,
    feature_manifest,
)
from financial_mdl.schemas import TargetSpec, validate_labels

METRIC_VERSION = "1"


@dataclass(frozen=True)
class TemporalSplit:
    train_start: str
    train_end: str
    validation_end: str
    test_end: str
    gap_sessions: int = 0
    strategy: str = "fixed_chronological_holdout"
    purging: str = "label_end_at_or_before_partition_end_minus_gap"

    def __post_init__(self) -> None:
        dates = (self.train_start, self.train_end, self.validation_end, self.test_end)
        if any(not isinstance(d, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in dates):
            raise ValueError("Split boundaries must be ISO session dates (YYYY-MM-DD)")
        stamps = [pd.Timestamp(d) for d in dates]
        if not stamps[0] <= stamps[1] < stamps[2] < stamps[3]:
            raise ValueError("Require train_start <= train_end < validation_end < test_end")
        if type(self.gap_sessions) is not int or self.gap_sessions < 0:
            raise ValueError("gap_sessions must be a nonnegative integer")
        if self.strategy != "fixed_chronological_holdout":
            raise ValueError("Unsupported split strategy")
        if self.purging != "label_end_at_or_before_partition_end_minus_gap":
            raise ValueError("Unsupported purging rule")


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_id: str
    version: str
    dataset_version: str
    dataset_fingerprint: str
    assets: tuple[str, ...]
    target: TargetSpec
    split: TemporalSplit
    feature_names: tuple[str, ...] = DEFAULT_FEATURES
    feature_families: tuple[str, ...] = (
        FeatureFamily.MARKET.value,
        FeatureFamily.STATISTICAL.value,
    )
    feature_implementation_version: str = FEATURE_VERSION
    feature_eligibility_policy: str = "complete_cases_across_all_protocol_features"
    forecast_origin_policy: str = "after_session_close"
    metrics: tuple[str, ...] = ("mae", "rmse")
    metric_implementation_version: str = METRIC_VERSION

    def __post_init__(self) -> None:
        for name in ("benchmark_id", "version", "dataset_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not re.fullmatch(r"[a-f0-9]{64}", self.dataset_fingerprint):
            raise ValueError("dataset_fingerprint must be a canonical SHA-256 digest")
        validate_labels(self.assets, "assets")
        if len(self.assets) != 1 or any(not isinstance(a, str) or not a for a in self.assets):
            raise ValueError("Version 1 supports exactly one asset per benchmark")
        if not isinstance(self.target, TargetSpec) or not isinstance(self.split, TemporalSplit):
            raise ValueError("Benchmark requires immutable target and split specifications")
        validate_labels(self.feature_names, "feature_names")
        validate_labels(self.feature_families, "feature_families")
        definitions = feature_manifest(self.feature_names)
        if {f["family"] for f in definitions} != set(self.feature_families):
            raise ValueError("Feature families must exactly describe the configured feature names")
        if self.feature_implementation_version != FEATURE_VERSION:
            raise ValueError("Unsupported feature implementation version")
        if self.feature_eligibility_policy != "complete_cases_across_all_protocol_features":
            raise ValueError("Unsupported feature eligibility policy")
        if self.forecast_origin_policy != "after_session_close":
            raise ValueError("Only forecasts after the daily close are implemented")
        validate_labels(self.metrics, "metrics")
        if (
            set(self.metrics) - {"mae", "rmse"}
            or self.metric_implementation_version != METRIC_VERSION
        ):
            raise ValueError("Unsupported metric definition/version")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A protocol is never overwritten in place by this API.
        with path.open("x", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)

    @classmethod
    def load(cls, path: Path) -> BenchmarkConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        target = data.pop("target")
        target["horizons"] = tuple(target["horizons"])
        target["target_names"] = tuple(target["target_names"])
        data["target"] = TargetSpec(**target)
        data["split"] = TemporalSplit(**data["split"])
        for name in ("assets", "feature_names", "feature_families", "metrics"):
            data[name] = tuple(data[name])
        return cls(**data)
