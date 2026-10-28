from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from financial_mdl.features import market_features
from financial_mdl.schemas import Forecast, ForecastCoordinates, Targets, TargetSpec
from financial_mdl.targets import future_volatility


def test_five_day_scalar_and_explicit_future_window():
    dates = pd.bdate_range("2024-01-01", periods=12)
    close = pd.Series(100 * np.exp(np.arange(12) * 0.01), index=dates)
    target = future_volatility(close, TargetSpec(), "SPY")
    assert target.observations.values.shape == (7, 1, 1)
    np.testing.assert_allclose(target.observations.values, 0.01 * np.sqrt(252))
    assert target.observation_starts[0, 0] == dates[1].to_datetime64()
    assert target.observations.coordinates.target_timestamps[0, 0] == dates[5].to_datetime64()
    assert target.observations.coordinates.horizons == (5,)
    spec = TargetSpec()
    assert spec.number_of_outputs == spec.number_of_target_variables == 1
    with pytest.raises(FrozenInstanceError):
        spec.annualization = 365


def test_target_matches_manual_returns_and_never_uses_partial_tail(dataset):
    target = future_volatility(dataset.frame.close, TargetSpec(horizons=(1, 5)), "SPY")
    returns = np.diff(np.log(dataset.frame.close.to_numpy()))
    for i in (0, 20, 174):
        for j, h in enumerate((1, 5)):
            expected = np.sqrt(252 * np.mean(returns[i : i + h] ** 2))
            assert target.observations.values[i, j, 0] == pytest.approx(expected)
    assert len(target.observations.values) == len(dataset.frame) - 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizons": [5]},
        {"horizons": (5, 1)},
        {"horizons": (1, 1)},
        {"horizons": (0,)},
        {"horizons": (True,)},
        {"annualization": 0},
        {"version": "future"},
        {"target_names": ("return",)},
        {"aggregation": "sample_standard_deviation"},
    ],
)
def test_reject_ambiguous_target_spec(kwargs):
    with pytest.raises(ValueError):
        TargetSpec(**kwargs)


def test_forecast_rejects_silent_reshape_nonfinite_and_misaligned_stamps(dataset):
    target = future_volatility(dataset.frame.close, TargetSpec(), "SPY")
    forecast = target.observations
    c = forecast.coordinates
    with pytest.raises(ValueError, match="explicit forecast shape"):
        Forecast(forecast.values[:, 0, 0], c)
    values = forecast.values.copy()
    values[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Forecast(values, c)
    with pytest.raises(ValueError, match="follow forecast origins"):
        replace(c, target_timestamps=c.origins.to_numpy()[:, None])
    with pytest.raises(ValueError, match="chronological"):
        replace(c, origins=c.origins[::-1])
    with pytest.raises(ValueError, match="asset identifier"):
        replace(c, asset_ids=("SPY",))
    with pytest.raises(ValueError, match="alignment differs"):
        c.assert_equal(replace(c, asset_ids=("QQQ",) * len(c.origins)))
    with pytest.raises(ValueError):
        forecast.values[0, 0, 0] = 0
    with pytest.raises(ValueError):
        Targets(forecast, np.full_like(target.observation_starts, np.datetime64("NaT", "ns")))


def test_coordinate_shape_and_label_order(dataset):
    c = future_volatility(
        dataset.frame.close, TargetSpec(horizons=(1, 5)), "SPY"
    ).observations.coordinates
    with pytest.raises(ValueError, match="increase"):
        replace(c, target_timestamps=c.target_timestamps[:, ::-1])
    with pytest.raises(ValueError, match="increasing"):
        replace(c, horizons=(5, 1))
    with pytest.raises(ValueError, match="shape"):
        replace(c, target_timestamps=c.target_timestamps[:, :1])
    with pytest.raises(ValueError, match="timezone-naive"):
        replace(c, origins=c.origins.tz_localize("UTC"))
    assert isinstance(c, ForecastCoordinates)


def test_features_are_causal_and_have_documented_warmup(dataset):
    original = market_features(dataset.frame)
    changed = dataset.frame.copy()
    changed.iloc[70:, :4] *= 10
    changed.iloc[70:, 4] *= 100
    pd.testing.assert_frame_equal(original.iloc[:70], market_features(changed).iloc[:70])
    assert original.realized_volatility_20.iloc[:20].isna().all()
    assert np.isfinite(original.iloc[20:].to_numpy()).all()


def test_invalid_target_prices_and_insufficient_history(dataset):
    with pytest.raises(ValueError, match="Not enough"):
        future_volatility(dataset.frame.close.iloc[:5], TargetSpec(), "SPY")
    prices = dataset.frame.close.copy()
    prices.iloc[2] = 0
    with pytest.raises(ValueError, match="positive"):
        future_volatility(prices, TargetSpec(), "SPY")
