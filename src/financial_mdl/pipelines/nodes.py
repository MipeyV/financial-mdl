"""Functions usable directly as Kedro nodes, backed by the same research modules.

No pipeline registry is installed yet. See docs/architecture.md for catalog boundaries.
"""

from pathlib import Path

from financial_mdl.benchmark import BenchmarkResult, BenchmarkRunner
from financial_mdl.data import MarketDataset, ParquetMarketDataProvider
from financial_mdl.forecasting import DirectMultiHorizonStrategy, OneStepStrategy
from financial_mdl.models import make_estimator
from financial_mdl.protocol import BenchmarkConfig


def load_market(path: str) -> MarketDataset:
    return ParquetMarketDataProvider(Path(path)).load()


def load_protocol(path: str) -> BenchmarkConfig:
    return BenchmarkConfig.load(Path(path))


def run_benchmark(
    dataset: MarketDataset,
    protocol: BenchmarkConfig,
    candidate: dict,
) -> BenchmarkResult:
    """candidate is a Kedro parameters mapping, separate from the benchmark catalog entry."""
    name, seed = candidate["estimator"], candidate.get("seed", 42)
    strategies = {"one_step": OneStepStrategy, "direct": DirectMultiHorizonStrategy}
    return BenchmarkRunner().run(
        dataset,
        protocol,
        lambda: make_estimator(name, seed),
        Path(candidate["output"]),
        estimator_name=name,
        seed=seed,
        phase=candidate.get("phase", "validation"),
        strategy_factory=strategies[candidate.get("strategy", "one_step")],
    )
