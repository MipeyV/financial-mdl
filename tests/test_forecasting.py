from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from financial_mdl.features import IdentityFeatureSelector, market_features
from financial_mdl.forecasting import (
    DirectMultiHorizonStrategy,
    ForecastingModel,
    OneStepStrategy,
)
from financial_mdl.metrics import evaluate
from financial_mdl.models import SklearnAdapter, make_estimator
from financial_mdl.schemas import Forecast, TargetSpec
from financial_mdl.targets import future_volatility


def test_direct_models_are_independent_and_have_exact_labeled_shapes(dataset):
    targets = future_volatility(dataset.frame.close, TargetSpec(horizons=(1, 5)), "SPY")
    positions = np.arange(20, 70)
    targets = targets.take(positions)
    X = market_features(dataset.frame).loc[targets.observations.coordinates.origins]
    instances = []

    def factory():
        model = make_estimator("naive")
        instances.append(model)
        return model

    strategy = DirectMultiHorizonStrategy()
    model = ForecastingModel(factory, strategy).fit(X, targets)
    result = model.predict(X, targets.observations.coordinates)
    assert result.values.shape == (50, 2, 1)
    assert len(instances) == 2 and instances[0] is not instances[1]
    expected = targets.observations.values.mean(axis=0)
    np.testing.assert_allclose(result.values, np.broadcast_to(expected, result.values.shape))
    assert strategy.get_params()["joint"] is False
    assert "mae/h5/realized_volatility" in evaluate(targets.observations, result, ("mae",))
    with pytest.raises(ValueError, match="horizon/target"):
        model.predict(X, replace(targets.observations.coordinates, target_names=("other",)))


def test_direct_rejects_reused_estimator_factory(dataset):
    targets = future_volatility(dataset.frame.close, TargetSpec(horizons=(1, 5)), "SPY")
    targets = targets.take(np.arange(20, 70))
    X = market_features(dataset.frame).loc[targets.observations.coordinates.origins]
    shared = make_estimator("naive")
    with pytest.raises(ValueError, match="fresh model"):
        DirectMultiHorizonStrategy().fit(lambda: shared, X, targets)


def test_one_output_strategy_accepts_five_session_aggregation(dataset):
    targets = future_volatility(dataset.frame.close, TargetSpec(), "SPY").take(np.arange(20, 70))
    X = market_features(dataset.frame).loc[targets.observations.coordinates.origins]
    model = ForecastingModel(lambda: make_estimator("ridge"), OneStepStrategy()).fit(X, targets)
    assert model.predict(X, targets.observations.coordinates).values.shape == (50, 1, 1)
    with pytest.raises(ValueError, match="exactly match"):
        model.predict(X.iloc[::-1], targets.observations.coordinates)


def test_one_output_strategy_rejects_multiple_horizons(dataset):
    targets = future_volatility(dataset.frame.close, TargetSpec(horizons=(1, 5)), "SPY")
    targets = targets.take(np.arange(20, 70))
    X = market_features(dataset.frame).loc[targets.observations.coordinates.origins]
    with pytest.raises(ValueError, match="one horizon"):
        OneStepStrategy().fit(lambda: make_estimator("naive"), X, targets)


def test_adapter_clones_external_model_and_rejects_column_or_output_mismatch():
    external = DummyRegressor()
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]})
    adapter = SklearnAdapter(external).fit(X, np.array([1.0, 2.0, 3.0]))
    assert not hasattr(external, "constant_")
    with pytest.raises(ValueError, match="columns"):
        adapter.predict(X[["b", "a"]])
    with pytest.raises(ValueError, match="shape"):
        adapter.fit(X, np.ones((3, 1)))
    adapter.estimator.predict = lambda X: np.ones((len(X), 1))
    with pytest.raises(ValueError, match="exactly shape"):
        adapter.predict(X)


def test_metric_alignment_and_known_values(dataset):
    coordinates = future_volatility(
        dataset.frame.close, TargetSpec(), "SPY"
    ).observations.coordinates
    actual = Forecast(np.zeros(coordinates.shape), coordinates)
    predicted = Forecast(np.ones(coordinates.shape) * 2, coordinates)
    assert evaluate(actual, predicted, ("mae", "rmse"))["rmse"] == 2
    wrong = Forecast(
        predicted.values, replace(coordinates, asset_ids=("OTHER",) * len(coordinates.origins))
    )
    with pytest.raises(ValueError, match="alignment"):
        evaluate(actual, wrong, ("mae",))


def test_identity_selector_requires_fit_and_matching_schema():
    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    selector = IdentityFeatureSelector()
    with pytest.raises(ValueError, match="not been fitted"):
        selector.transform(X)
    selector.fit(X, np.zeros((1, 1, 1)))
    assert selector.get_selected_features() == ("a", "b")
    with pytest.raises(ValueError, match="order"):
        selector.transform(X[["b", "a"]])
