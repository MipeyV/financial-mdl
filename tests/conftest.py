from dataclasses import replace

import pytest

from financial_mdl.experiments import synthetic_market
from financial_mdl.protocol import BenchmarkConfig, TemporalSplit
from financial_mdl.schemas import TargetSpec


@pytest.fixture
def dataset():
    return synthetic_market(n=180, seed=7)


@pytest.fixture
def protocol(dataset):
    dates = dataset.frame.index
    return BenchmarkConfig(
        "test-volatility",
        "1",
        dataset.metadata.version,
        dataset.fingerprint,
        (dataset.metadata.asset,),
        TargetSpec(),
        TemporalSplit(*(dates[i].strftime("%Y-%m-%d") for i in (0, 89, 134, 179))),
    )


@pytest.fixture
def multi_protocol(protocol):
    return replace(protocol, target=TargetSpec(horizons=(1, 5)))


@pytest.fixture
def fast_reports(monkeypatch):
    # Contract/leakage tests do not repeatedly embed Plotly's JS runtime.
    # Real report generation is exercised by the CLI smoke test.
    monkeypatch.setattr("financial_mdl.benchmark.forecast_figures", lambda *args: None)
    monkeypatch.setattr("financial_mdl.benchmark.training_figures", lambda *args: None)
    monkeypatch.setattr("financial_mdl.benchmark.mdl_figures", lambda *args: None)
