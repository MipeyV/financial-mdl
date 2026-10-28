"""Persistence must copy causal trailing volatility in physical target units."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from financial_mdl.benchmark import BenchmarkRunner, compare_runs
from financial_mdl.candidates import make_candidate
from financial_mdl.cli import main
from financial_mdl.features import IdentityFeatures, StandardizeFeatures, market_features
from financial_mdl.forecasting import DirectMultiHorizonStrategy
from financial_mdl.models import VolatilityPersistence
from financial_mdl.pipelines.nodes import run_benchmark


@pytest.mark.parametrize("window", [5, 20])
def test_persistence_copies_current_feature_without_learning_from_targets(window):
    X = pd.DataFrame(
        {
            "realized_volatility_5": [0.1, 0.2, 0.3],
            "realized_volatility_20": [0.15, 0.25, 0.35],
        }
    )
    expected = X[f"realized_volatility_{window}"].to_numpy()
    model = VolatilityPersistence(window)
    with pytest.raises(ValueError, match="fitted"):
        model.predict(X)
    first = model.fit(X, np.array([0.4, 0.5, 0.6])).predict(X)
    second = model.fit(X, np.array([100.0, 200.0, 300.0])).predict(X)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert model.get_params()["parameters"]["learns_from_targets"] is False
    with pytest.raises(ValueError, match="ordered columns"):
        model.predict(X[X.columns[::-1]])


@pytest.mark.parametrize("window", [5, 20])
def test_standardized_volatility_is_rejected_even_for_constant_features(window):
    X = pd.DataFrame({f"realized_volatility_{window}": [0.25, 0.25, 0.25]})
    scaled = StandardizeFeatures().fit(X).transform(X)
    np.testing.assert_array_equal(scaled.to_numpy(), np.zeros((3, 1)))
    with pytest.raises(ValueError, match="original feature units"):
        VolatilityPersistence(window).fit(scaled, np.ones(3))


@pytest.mark.parametrize("values", [[0.1, np.nan], [0.1, np.inf], [0.1, -0.1]])
def test_persistence_rejects_invalid_volatility(values):
    with pytest.raises(ValueError, match="finite nonnegative"):
        VolatilityPersistence(5).fit(
            pd.DataFrame({"realized_volatility_5": values}),
            np.ones(2),
        )


def test_persistence_requires_named_source_feature_and_scalar_training_contract():
    model = VolatilityPersistence(5)
    with pytest.raises(ValueError, match="requires feature"):
        model.fit(pd.DataFrame({"other": [0.2]}), np.ones(1))
    X = pd.DataFrame({"realized_volatility_5": [0.2]})
    with pytest.raises(ValueError, match="shape"):
        model.fit(X, np.ones((1, 1)))
    with pytest.raises(ValueError, match="5 or 20"):
        VolatilityPersistence(10)
    with pytest.raises(ValueError, match="positive integer"):
        VolatilityPersistence(5, annualization=0)


@pytest.mark.parametrize("name,window", [("persistence-5", 5), ("persistence-20", 20)])
def test_benchmark_persistence_matches_raw_features_on_identical_origins(
    dataset,
    protocol,
    tmp_path,
    fast_reports,
    name,
    window,
):
    fingerprint = protocol.fingerprint
    estimator, transformer = make_candidate(name)
    result = BenchmarkRunner().run(
        dataset,
        protocol,
        estimator,
        tmp_path / name,
        estimator_name=name,
        transformer_factory=transformer,
    )
    ridge_estimator, ridge_transformer = make_candidate("ridge")
    ridge = BenchmarkRunner().run(
        dataset,
        protocol,
        ridge_estimator,
        tmp_path / "ridge",
        estimator_name="ridge",
        transformer_factory=ridge_transformer,
    )
    predicted = pd.read_parquet(result.directory / "predictions.parquet")
    expected = market_features(dataset.frame).loc[predicted.origin, f"realized_volatility_{window}"]
    np.testing.assert_array_equal(predicted.predicted, expected.to_numpy())
    assert result.metadata["transformer"]["name"] == "identity"
    assert result.metadata["selected_features"] == protocol.feature_names
    assert result.metadata["evaluation_fingerprint"] == ridge.metadata["evaluation_fingerprint"]
    assert protocol.fingerprint == fingerprint == result.metadata["benchmark_fingerprint"]
    leaderboard = compare_runs([result.directory, ridge.directory], tmp_path / "comparison")
    assert set(leaderboard.estimator) == {name, "ridge"}


def test_persistence_scales_to_target_annualization_and_supports_direct_horizons(
    dataset,
    multi_protocol,
    tmp_path,
    fast_reports,
):
    protocol = replace(multi_protocol, target=replace(multi_protocol.target, annualization=365))
    estimator, transformer = make_candidate("persistence-5", annualization=365)
    result = BenchmarkRunner().run(
        dataset,
        protocol,
        estimator,
        tmp_path / "direct",
        strategy_factory=DirectMultiHorizonStrategy,
        transformer_factory=transformer,
    )
    table = pd.read_parquet(result.directory / "predictions.parquet")
    expected = market_features(dataset.frame).loc[table.origin, "realized_volatility_5"]
    np.testing.assert_allclose(table.predicted, expected.to_numpy() * np.sqrt(365 / 252))
    by_horizon = table.pivot(index="origin", columns="horizon", values="predicted")
    np.testing.assert_array_equal(by_horizon[1], by_horizon[5])
    assert len(result.metadata["estimator"]) == 2
    assert result.metadata["estimator"][0]["parameters"]["target_annualization"] == 365


def test_persistence_is_causal_when_future_prices_change(dataset):
    frame = dataset.frame
    X = market_features(frame).iloc[20:70]
    model = VolatilityPersistence(20).fit(X, np.ones(len(X)))
    original = model.predict(X)
    changed = frame.copy()
    changed.iloc[70:, :4] *= 10
    changed.iloc[70:, 4] *= 100
    np.testing.assert_array_equal(model.predict(market_features(changed).loc[X.index]), original)


@pytest.mark.parametrize("name", ["persistence-5", "persistence-20"])
def test_cli_persistence_uses_original_units_and_common_protocol(
    dataset,
    protocol,
    tmp_path,
    fast_reports,
    name,
):
    data_path, protocol_path = tmp_path / "market.parquet", tmp_path / "protocol.json"
    dataset.save(data_path)
    protocol.save(protocol_path)
    output = tmp_path / "cli"
    main(
        [
            "run",
            "--data",
            str(data_path),
            "--protocol",
            str(protocol_path),
            "--estimator",
            name,
            "--output",
            str(output),
        ]
    )
    table = pd.read_parquet(output / "predictions.parquet")
    window = int(name.split("-")[-1])
    expected = market_features(dataset.frame).loc[table.origin, f"realized_volatility_{window}"]
    np.testing.assert_array_equal(table.predicted, expected.to_numpy())
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["phase"] == "validation"
    assert metadata["benchmark_fingerprint"] == protocol.fingerprint
    assert metadata["transformer"]["name"] == "identity"


def test_kedro_node_uses_same_persistence_preprocessing(dataset, protocol, tmp_path, fast_reports):
    result = run_benchmark(
        dataset,
        protocol,
        {
            "estimator": "persistence-20",
            "output": str(tmp_path / "node"),
        },
    )
    table = pd.read_parquet(result.directory / "predictions.parquet")
    expected = market_features(dataset.frame).loc[table.origin, "realized_volatility_20"]
    np.testing.assert_array_equal(table.predicted, expected.to_numpy())


def test_candidate_factories_are_fresh_and_preserve_existing_ridge_preprocessing():
    estimator, transformer = make_candidate("persistence-5")
    assert estimator() is not estimator()
    assert isinstance(transformer(), IdentityFeatures)
    _, ridge_transformer = make_candidate("ridge")
    assert isinstance(ridge_transformer(), StandardizeFeatures)
    with pytest.raises(ValueError, match="Unknown estimator"):
        make_candidate("missing")
