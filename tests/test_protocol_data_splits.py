from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from financial_mdl.data import MarketDataset, ParquetMarketDataProvider
from financial_mdl.protocol import BenchmarkConfig, TemporalSplit
from financial_mdl.splitting import chronological_partitions
from financial_mdl.targets import future_volatility


def test_protocol_roundtrip_frozen_and_no_overwrite(protocol, tmp_path):
    path = tmp_path / "protocol.json"
    protocol.save(path)
    loaded = BenchmarkConfig.load(path)
    assert loaded == protocol
    assert loaded.fingerprint == protocol.fingerprint
    with pytest.raises(FrozenInstanceError):
        protocol.version = "2"
    with pytest.raises(FrozenInstanceError):
        protocol.split.gap_sessions = 10
    with pytest.raises(FileExistsError):
        protocol.save(path)
    assert replace(protocol, version="2").fingerprint != protocol.fingerprint
    assert (
        replace(protocol, split=replace(protocol.split, gap_sessions=1)).fingerprint
        != protocol.fingerprint
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metrics": ("r2",)},
        {"metric_implementation_version": "2"},
        {"feature_implementation_version": "2"},
        {"assets": ("SPY", "AAPL")},
        {"feature_names": ("current_pe_ratio",)},
        {"feature_families": ("FundamentalFeatures",)},
        {"feature_eligibility_policy": "candidate_specific_dropna"},
        {"metrics": ["mae"]},
        {"dataset_fingerprint": "wrong"},
        {"forecast_origin_policy": "before_open"},
    ],
)
def test_protocol_rejects_unsupported_or_mutable_rules(protocol, kwargs):
    with pytest.raises(ValueError):
        replace(protocol, **kwargs)


def test_dataset_roundtrip_fingerprint_tracks_content_and_provenance(dataset, tmp_path):
    path = tmp_path / "market.parquet"
    dataset.save(path)
    loaded = ParquetMarketDataProvider(path).load()
    assert loaded.fingerprint == dataset.fingerprint
    pd.testing.assert_frame_equal(loaded.frame, dataset.frame, check_freq=False)
    changed = dataset.frame.copy()
    changed.iloc[0, 4] += 1
    assert MarketDataset(changed, dataset.metadata).fingerprint != dataset.fingerprint
    assert (
        MarketDataset(dataset.frame, replace(dataset.metadata, source="other")).fingerprint
        != dataset.fingerprint
    )


@pytest.mark.parametrize("kind", ["duplicate", "unsorted", "negative", "nan", "ohlc", "columns"])
def test_invalid_ohlcv_rejected(dataset, kind):
    frame = dataset.frame.copy()
    if kind == "duplicate":
        frame.index = pd.DatetimeIndex([frame.index[0]] * len(frame))
    elif kind == "unsorted":
        frame = frame.iloc[::-1]
    elif kind == "negative":
        frame.iloc[0, 4] = -1
    elif kind == "nan":
        frame.iloc[0, 0] = np.nan
    elif kind == "ohlc":
        frame.iloc[0, 1] = frame.iloc[0, 2] / 2
    else:
        frame = frame[frame.columns[::-1]]
    with pytest.raises(ValueError):
        MarketDataset(frame, dataset.metadata)


@pytest.mark.parametrize("gap", [0, 3])
def test_purges_label_windows_at_both_boundaries_and_adds_gap(dataset, protocol, gap):
    split = replace(protocol.split, gap_sessions=gap)
    targets = future_volatility(dataset.frame.close, protocol.target, dataset.metadata.asset)
    partitions = chronological_partitions(targets, dataset.frame.index, split)
    coordinates = targets.observations.coordinates
    for part, end, nxt in (
        ("train", split.train_end, "validation"),
        ("validation", split.validation_end, "test"),
    ):
        sessions = dataset.frame.index[dataset.frame.index <= end]
        assert (
            coordinates.target_timestamps[partitions[part]].max()
            <= sessions[-gap - 1].to_datetime64()
        )
        assert (
            coordinates.target_timestamps[partitions[part]].max()
            < coordinates.origins[partitions[nxt][0]]
        )
    assert coordinates.target_timestamps[partitions["test"]].max() <= pd.Timestamp(split.test_end)
    assert not set(partitions["train"]) & set(partitions["validation"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gap_sessions": -1},
        {"gap_sessions": True},
        {"train_end": "1900-01-01"},
        {"test_end": "2020-01-01"},
        {"validation_end": "2021-01-01T12:00:00"},
    ],
)
def test_invalid_splits(protocol, kwargs):
    with pytest.raises(ValueError):
        replace(protocol.split, **kwargs)


def test_empty_purged_partition_fails(dataset, protocol):
    split = TemporalSplit("2020-01-01", "2020-01-02", "2020-01-03", "2020-02-01")
    targets = future_volatility(dataset.frame.close, protocol.target, dataset.metadata.asset)
    with pytest.raises(ValueError, match="Empty train"):
        chronological_partitions(targets, dataset.frame.index, split)
