"""Explicit snapshot -> frozen protocol -> candidate runs -> matched comparison."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from financial_mdl.benchmark import BenchmarkRunner, compare_runs
from financial_mdl.candidates import make_candidate
from financial_mdl.data import ParquetMarketDataProvider, YFinanceMarketDataProvider
from financial_mdl.experiments import synthetic_market
from financial_mdl.forecasting import DirectMultiHorizonStrategy, OneStepStrategy
from financial_mdl.mdl import ExhaustiveMDLSelector, MDLCodingSpec
from financial_mdl.mdl_experiment import run_synthetic_mdl
from financial_mdl.models import ESTIMATOR_NAMES
from financial_mdl.protocol import BenchmarkConfig, TemporalSplit
from financial_mdl.schemas import TargetSpec
from financial_mdl.tracking import LocalMLflowTracker


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Freeze a real daily OHLCV snapshot")
    fetch.add_argument("--asset", default="SPY")
    fetch.add_argument("--start", required=True)
    fetch.add_argument("--end", required=True, help="Exclusive vendor end date")
    fetch.add_argument("--dataset-version", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    protocol = commands.add_parser("protocol", help="Create a protocol; never overwrite one")
    protocol.add_argument("--data", type=Path, required=True)
    protocol.add_argument("--output", type=Path, required=True)
    protocol.add_argument("--id", default="spy-volatility")
    protocol.add_argument("--version", default="1")
    for name in ("train-start", "train-end", "validation-end", "test-end"):
        protocol.add_argument(f"--{name}", required=True)
    protocol.add_argument("--gap-sessions", type=int, default=0)
    protocol.add_argument("--horizons", type=int, nargs="+", default=[5])
    run = commands.add_parser("run", help="Evaluate a candidate under an existing protocol")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--estimator", choices=ESTIMATOR_NAMES, required=True)
    run.add_argument("--strategy", choices=["one-step", "direct"], default="one-step")
    run.add_argument("--phase", choices=["validation", "test"], default="validation")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--selector", choices=["identity", "mdl"], default="identity")
    run.add_argument("--mdl-target-precision", type=float, default=0.0001)
    run.add_argument("--tracking", action="store_true")
    run.add_argument("--tracking-root", type=Path, default=Path("artifacts/mlflow"))
    compare = commands.add_parser("compare", help="Reject incompatible run comparisons")
    compare.add_argument("runs", nargs="+", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--tracking", action="store_true")
    compare.add_argument("--tracking-root", type=Path, default=Path("artifacts/mlflow"))
    demo = commands.add_parser("demo", help="Offline synthetic smoke example only")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    synthetic = commands.add_parser("mdl-synthetic", help="Run independent synthetic MDL controls")
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return root


def _execute(args: argparse.Namespace) -> None:
    if args.command == "fetch":
        if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
            raise FileExistsError("Use a new snapshot path instead of overwriting research data")
        data = YFinanceMarketDataProvider(
            args.asset,
            args.start,
            args.end,
            args.dataset_version,
        ).load()
        data.save(args.output)
        print(f"Saved {len(data.frame)} sessions; SHA-256 {data.fingerprint}")
    elif args.command == "protocol":
        data = ParquetMarketDataProvider(args.data).load()
        protocol = BenchmarkConfig(
            args.id,
            args.version,
            data.metadata.version,
            data.fingerprint,
            (data.metadata.asset,),
            TargetSpec(horizons=tuple(args.horizons)),
            TemporalSplit(
                args.train_start,
                args.train_end,
                args.validation_end,
                args.test_end,
                args.gap_sessions,
            ),
        )
        protocol.save(args.output)
        print(f"Saved immutable protocol {protocol.fingerprint}")
    elif args.command == "run":
        tracker = LocalMLflowTracker(args.tracking_root) if args.tracking else None
        protocol = BenchmarkConfig.load(args.protocol)
        estimator_factory, transformer_factory = make_candidate(
            args.estimator,
            args.seed,
            annualization=protocol.target.annualization,
        )
        selector_kwargs = {}
        if args.selector == "mdl":
            if args.estimator != "ridge":
                raise ValueError("MDL CLI v1 evaluates the fixed Ridge reference estimator")
            selector_kwargs["selector_factory"] = partial(
                ExhaustiveMDLSelector,
                MDLCodingSpec(target_precision=args.mdl_target_precision),
            )
        result = BenchmarkRunner(tracker).run(
            ParquetMarketDataProvider(args.data).load(),
            protocol,
            estimator_factory,
            args.output,
            estimator_name=args.estimator,
            seed=args.seed,
            phase=args.phase,
            transformer_factory=transformer_factory,
            strategy_factory=(
                OneStepStrategy if args.strategy == "one-step" else DirectMultiHorizonStrategy
            ),
            **selector_kwargs,
        )
        print(f"{args.phase}: {result.metrics}; artifacts: {result.directory}")
    elif args.command == "compare":
        tracker = LocalMLflowTracker(args.tracking_root) if args.tracking else None
        print(compare_runs(args.runs, args.output, tracker).to_string(index=False))
    elif args.command == "mdl-synthetic":
        results = run_synthetic_mdl(args.output, seeds=tuple(args.seeds))
        print(results.groupby(["scenario", "method"])[["n_features", "rmse", "total_bits"]].mean())
    elif args.command == "demo":
        args.output.mkdir(parents=True, exist_ok=False)
        data = synthetic_market()
        dates = data.frame.index
        protocol = BenchmarkConfig(
            "synthetic-smoke",
            "1",
            data.metadata.version,
            data.fingerprint,
            (data.metadata.asset,),
            TargetSpec(),
            TemporalSplit(*(dates[i].strftime("%Y-%m-%d") for i in (0, 199, 279, 359))),
        )
        protocol.save(args.output / "protocol.json")
        runs = []
        for name in ("naive", "ridge"):
            directory = args.output / name
            estimator_factory, transformer_factory = make_candidate(name)
            BenchmarkRunner().run(
                data,
                protocol,
                estimator_factory,
                directory,
                estimator_name=name,
                transformer_factory=transformer_factory,
            )
            runs.append(directory)
        print(compare_runs(runs, args.output / "comparison").to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    root = parser()
    args = root.parse_args(argv)
    try:
        _execute(args)
    except (ValueError, FileExistsError, FileNotFoundError, ImportError) as error:
        root.error(str(error))


if __name__ == "__main__":
    main()
