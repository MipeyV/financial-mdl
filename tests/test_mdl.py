"""Coding validity, controlled recovery and honest intercept-only predictions."""

import math
import struct
from dataclasses import replace

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from financial_mdl.experiments import controlled_regression
from financial_mdl.features import StandardizeFeatures
from financial_mdl.mdl import (
    ExhaustiveMDLSelector,
    MDLCodingSpec,
    geometric_target_bits,
    quantize,
    subset_code_bits,
)
from financial_mdl.models import SklearnAdapter, make_estimator


def test_subset_prefix_lengths_satisfy_kraft_including_empty_and_full():
    for p in range(9):
        mass = sum(math.comb(p, k) * 2 ** (-subset_code_bits(p, k)) for k in range(p + 1))
        assert mass <= 1
    assert subset_code_bits(5, 0) == subset_code_bits(5, 5) == 3
    assert subset_code_bits(5, 2) == 7
    with pytest.raises(ValueError):
        subset_code_bits(2, 3)


def test_target_code_uses_normalized_integer_probabilities_and_shannon_length():
    # At q=1/2: P(0)=1/3 and P(+1)=P(-1)=1/6; joint probability = 1/108.
    ideal, bits = geometric_target_bits(np.array([0, 1, -1]), 0.5)
    assert ideal == pytest.approx(math.log2(108))
    assert bits == 7
    assert geometric_target_bits(np.zeros(4, dtype=int), 0) == (0, 0)
    with pytest.raises(ValueError, match="cannot encode"):
        geometric_target_bits(np.array([1]), 0)
    with pytest.raises(ValueError, match="0 <= q < 1"):
        geometric_target_bits(np.array([0]), 1)


def test_quantization_is_explicit_and_rejects_unsafe_values():
    np.testing.assert_array_equal(quantize(np.array([0.125, 0.375, -0.375]), 0.25), [0, 2, -2])
    for values, precision in (([np.inf], 0.1), ([1e30], 1e-4), ([1.0], 0), ([1.0], -1)):
        with pytest.raises(ValueError):
            quantize(np.array(values), precision)


def test_reference_model_parameters_are_real_binary64_words_and_replayable():
    X, y = controlled_regression("signal", 1, 300)
    selector = ExhaustiveMDLSelector().fit(X, y)
    diagnostic = selector.get_diagnostics()
    assert len(diagnostic["candidates"]) == 2 ** len(X.columns)
    for row in (diagnostic["selected"], diagnostic["full"], diagnostic["constant"]):
        payload = bytes.fromhex(row["parameter_hex"])
        assert len(payload) * 8 == row["model_bits"] == 64 * (row["n_features"] + 2)
        parameters = struct.unpack(f">{row['n_features'] + 2}d", payload)
        assert parameters[-1] == row["residual_q"]
        if row["features"]:
            standardized = (
                StandardizeFeatures().fit(X[row["features"]]).transform(X[row["features"]])
            )
            replay = parameters[0] + standardized.to_numpy() @ np.array(parameters[1:-1])
        else:
            replay = np.full(len(y), parameters[0])
        assert np.sqrt(np.mean((replay - y) ** 2)) == pytest.approx(row["training_rmse"])
        residuals = quantize(y, 1e-4) - quantize(replay, 1e-4)
        assert geometric_target_bits(residuals, parameters[-1])[1] == row["target_bits"]
        assert row["total_bits"] == row["subset_bits"] + row["model_bits"] + row["target_bits"]


def test_strong_signals_are_retained_without_duplicate_or_noise_features():
    X, y = controlled_regression("signal", 42, 800)
    selector = ExhaustiveMDLSelector().fit(X, y[:, None, None])
    selected = set(selector.get_selected_features())
    assert len(selected) == 2
    assert "signal_b" in selected
    assert len(selected & {"signal_a", "signal_a_copy"}) == 1
    assert not selected & {"noise_0", "noise_1", "noise_2"}
    diagnostic = selector.get_diagnostics()
    assert diagnostic["selected"]["total_bits"] <= diagnostic["full"]["total_bits"]
    assert diagnostic["training_rows"] == 800
    transformed = selector.transform(X.iloc[700:])
    assert transformed.index.equals(X.iloc[700:].index)
    assert tuple(transformed.columns) == selector.get_selected_features()


def test_constant_target_selects_zero_features_and_predicts_training_mean():
    X, _ = controlled_regression("signal", 7, 200)
    y = np.full(len(X), 0.25)
    selector = ExhaustiveMDLSelector().fit(X, y)
    assert selector.get_selected_features() == ()
    empty = selector.transform(X)
    assert empty.shape == (200, 0) and empty.index.equals(X.index)
    transformer = StandardizeFeatures().fit(empty)
    model = make_estimator("ridge").fit(transformer.transform(empty), y)
    np.testing.assert_array_equal(model.predict(transformer.transform(empty.iloc[:3])), [0.25] * 3)
    assert model.get_params()["effective_model"] == "intercept_only_training_mean"
    assert model.get_feature_importance() is None
    assert selector.get_diagnostics()["selected"]["target_bits"] == 0
    # Refit the same adapter on features: its earlier constant must not survive.
    model.fit(X[["signal_a"]], np.arange(len(X), dtype=float))
    assert model.get_params()["effective_model"] == "configured_estimator"


def test_noise_only_control_favors_constant_over_spurious_predictors():
    X, y = controlled_regression("noise_only", 42, 600)
    assert ExhaustiveMDLSelector().fit(X, y).get_selected_features() == ()


def test_selector_fails_clearly_on_unsupported_shapes_schemas_and_search_sizes():
    X, y = controlled_regression("signal", 0, 20)
    selector = ExhaustiveMDLSelector()
    with pytest.raises(ValueError, match="not been fitted"):
        selector.transform(X)
    for invalid in (np.ones((20, 2)), np.ones((20, 2, 1)), np.ones((20, 1))):
        with pytest.raises(ValueError, match="one scalar target"):
            selector.fit(X, invalid)
    with pytest.raises(ValueError, match="max_features"):
        ExhaustiveMDLSelector(replace(MDLCodingSpec(), max_features=5)).fit(X, y)
    invalid_X = X.copy()
    invalid_X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite real"):
        selector.fit(invalid_X, y)
    selector.fit(X, y)
    with pytest.raises(ValueError, match="columns/order"):
        selector.transform(X[X.columns[::-1]])
    with pytest.raises(ValueError, match="empty feature subsets only for Ridge"):
        make_estimator("tree").fit(X.iloc[:, :0], y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_precision": 0},
        {"target_precision": np.nan},
        {"ridge_alpha": -1},
        {"max_features": 0},
        {"max_features": 17},
        {"version": "other"},
    ],
)
def test_invalid_coding_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MDLCodingSpec(**kwargs)


def test_diagnostics_cannot_mutate_the_fitted_search():
    X, y = controlled_regression("signal", 0, 100)
    selector = ExhaustiveMDLSelector().fit(X, y)
    expected = selector.get_diagnostics()
    changed = selector.get_diagnostics()
    changed["candidates"][0]["features"].append("injected")
    assert selector.get_diagnostics() == expected


def test_reference_alpha_matches_an_independently_fitted_ridge():
    X, y = controlled_regression("signal", 3, 100)
    spec = MDLCodingSpec(ridge_alpha=2)
    diagnostic = ExhaustiveMDLSelector(spec).fit(X, y).get_diagnostics()["full"]
    transformer = StandardizeFeatures().fit(X)
    model = SklearnAdapter(Ridge(alpha=2)).fit(transformer.transform(X), y)
    assert diagnostic["training_rmse"] == pytest.approx(
        np.sqrt(np.mean((model.predict(transformer.transform(X)) - y) ** 2))
    )
