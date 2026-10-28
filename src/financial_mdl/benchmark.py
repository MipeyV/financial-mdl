"""One runner owns eligibility, splits and metrics; candidates never override them."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from financial_mdl.contracts import (
    FeatureSelector,
    FeatureTransformer,
    ForecastStrategy,
    ModelFactory,
    Tracker,
)
from financial_mdl.data import MarketDataset
from financial_mdl.features import (
    IdentityFeatureSelector,
    StandardizeFeatures,
    feature_manifest,
    market_features,
)
from financial_mdl.forecasting import ForecastingModel, OneStepStrategy
from financial_mdl.metrics import evaluate
from financial_mdl.protocol import BenchmarkConfig
from financial_mdl.reporting import forecast_figures, mdl_figures, save_figure, training_figures
from financial_mdl.schemas import Forecast, Targets
from financial_mdl.splitting import chronological_partitions
from financial_mdl.targets import future_volatility
from financial_mdl.tracking import NoOpTracker


@dataclass(frozen=True)
class BenchmarkResult:
    directory: Path
    metrics: dict[str, float]
    metadata: dict[str, Any]


def _provenance() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        git_commit = commit.stdout.strip() if commit.returncode == 0 else None
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        git_commit, dirty = None, None
    package = Path(__file__).parent
    hashes = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    lock = package.parent.parent / "uv.lock"
    return {
        "git_commit": git_commit,
        "git_dirty": dirty,
        "source_sha256": hashes,
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "financial-mdl",
                "numpy",
                "pandas",
                "scikit-learn",
                "pyarrow",
                "plotly",
            )
        },
    }


def _validate_frame(frame: pd.DataFrame, expected: pd.DataFrame, columns: tuple) -> None:
    if not isinstance(frame, pd.DataFrame) or not frame.index.equals(expected.index):
        raise ValueError("Fitted components must preserve the exact feature row index")
    if tuple(frame.columns) != columns:
        raise ValueError("Fitted components must preserve declared feature columns and order")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Fitted components must produce finite features")


def _prediction_table(targets: Targets, prediction: Forecast) -> pd.DataFrame:
    actual = targets.observations
    c = actual.coordinates
    rows = []
    for h, horizon in enumerate(c.horizons):
        for k, target in enumerate(c.target_names):
            rows.append(
                pd.DataFrame(
                    {
                        "origin": c.origins,
                        "target_start": targets.observation_starts[:, h],
                        "target_timestamp": c.target_timestamps[:, h],
                        "asset": c.asset_ids,
                        "horizon": horizon,
                        "target": target,
                        "actual": actual.values[:, h, k],
                        "predicted": prediction.values[:, h, k],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _evaluation_fingerprint(table: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for column in ("origin", "target_start", "target_timestamp"):
        digest.update(table[column].to_numpy(dtype="datetime64[ns]").astype("<i8").tobytes())
    digest.update(table.horizon.to_numpy(dtype="<i8").tobytes())
    digest.update(table.actual.to_numpy(dtype="<f8").tobytes())
    digest.update(json.dumps(table[["asset", "target"]].values.tolist()).encode())
    return digest.hexdigest()


class BenchmarkRunner:
    def __init__(self, tracker: Tracker | None = None):
        self.tracker = tracker if tracker is not None else NoOpTracker()

    def run(
        self,
        dataset: MarketDataset,
        protocol: BenchmarkConfig,
        estimator_factory: ModelFactory,
        output: Path,
        *,
        estimator_name: str = "custom",
        strategy_factory: Callable[[], ForecastStrategy] = OneStepStrategy,
        selector_factory: Callable[[], FeatureSelector] = IdentityFeatureSelector,
        transformer_factory: Callable[[], FeatureTransformer] = StandardizeFeatures,
        phase: str = "validation",
        seed: int = 42,
    ) -> BenchmarkResult:
        output = Path(output)
        if output.exists():
            raise FileExistsError(f"Use a new output directory: {output}")
        if phase not in {"validation", "test"}:
            raise ValueError("phase must be validation or test")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        # Revalidate a defensive snapshot: mutable pandas inputs never bypass validation.
        dataset = MarketDataset(dataset.frame, dataset.metadata)
        if (
            dataset.fingerprint != protocol.dataset_fingerprint
            or dataset.metadata.version != protocol.dataset_version
            or (dataset.metadata.asset,) != protocol.assets
        ):
            raise ValueError("Dataset fingerprint/version/asset does not match authorized protocol")
        frame = dataset.frame
        X_all = market_features(frame, protocol.feature_names)
        targets = future_volatility(frame.close, protocol.target, dataset.metadata.asset)
        origins = targets.observations.coordinates.origins
        X_all = X_all.loc[origins]
        eligible = np.flatnonzero(np.isfinite(X_all.to_numpy()).all(axis=1))
        targets = targets.take(eligible)
        X_all = X_all.iloc[eligible]
        partitions = chronological_partitions(targets, frame.index, protocol.split)
        train, evaluation = partitions["train"], partitions[phase]
        X_train, X_eval = X_all.iloc[train].copy(), X_all.iloc[evaluation].copy()
        y_train, y_eval = targets.take(train), targets.take(evaluation)

        selector, transformer, strategy = (
            selector_factory(),
            transformer_factory(),
            strategy_factory(),
        )
        selector.fit(X_train.copy(), y_train.observations.values.copy())
        selected = selector.get_selected_features()
        if (
            not isinstance(selected, tuple)
            or len(set(selected)) != len(selected)
            or any(name not in protocol.feature_names for name in selected)
        ):
            raise ValueError("Selector must declare a unique subset of protocol features")
        selected_train, selected_eval = selector.transform(X_train), selector.transform(X_eval)
        _validate_frame(selected_train, X_train, selected)
        _validate_frame(selected_eval, X_eval, selected)
        transformer.fit(selected_train.copy())
        transformed_train = transformer.transform(selected_train)
        transformed_eval = transformer.transform(selected_eval)
        _validate_frame(transformed_train, X_train, selected)
        _validate_frame(transformed_eval, X_eval, selected)

        estimators = []

        def recorded_factory():
            model = estimator_factory()
            estimators.append(model)
            return model

        model = ForecastingModel(recorded_factory, strategy)
        model.fit(transformed_train, y_train)
        prediction = model.predict(transformed_eval, y_eval.observations.coordinates)
        metrics = evaluate(y_eval.observations, prediction, protocol.metrics)
        table = _prediction_table(y_eval, prediction)
        split_summary = {}
        for name, positions in partitions.items():
            c = targets.observations.coordinates
            split_summary[name] = {
                "count": len(positions),
                "origin_start": c.origins[positions[0]].isoformat(),
                "origin_end": c.origins[positions[-1]].isoformat(),
                "label_end": str(c.target_timestamps[positions].max()),
            }
        selector_params = selector.get_params()
        selection_suffix = (
            ""
            if selector_params.get("name") == "identity"
            else f"-{selector_params.get('name', 'custom')}"
        )
        metadata = {
            "run_name": (
                f"{protocol.benchmark_id}-v{protocol.version}-{estimator_name}"
                f"{selection_suffix}-{phase}-s{seed}"
            ),
            "benchmark_id": protocol.benchmark_id,
            "benchmark_version": protocol.version,
            "benchmark_fingerprint": protocol.fingerprint,
            "dataset_fingerprint": dataset.fingerprint,
            "dataset_metadata": vars(dataset.metadata),
            "evaluation_fingerprint": _evaluation_fingerprint(table),
            "phase": phase,
            "seed": seed,
            "estimator_name": estimator_name,
            "estimator": [estimator.get_params() for estimator in estimators],
            "strategy": strategy.get_params(),
            "selector": selector_params,
            "transformer": transformer.get_params(),
            "selected_features": selected,
            "feature_families": protocol.feature_families,
            "feature_definitions": feature_manifest(protocol.feature_names),
            "horizons": protocol.target.horizons,
            "target_names": protocol.target.target_names,
            "splits": split_summary,
            **_provenance(),
        }
        # Fail before writing artifacts if a candidate's metadata cannot be serialized faithfully.
        serialized = json.dumps(metadata, indent=2, allow_nan=False)
        output.mkdir(parents=True, exist_ok=False)
        protocol.save(output / "protocol.json")
        (output / "run.json").write_text(serialized, encoding="utf-8")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        table.to_parquet(output / "predictions.parquet", index=False)
        selection = None
        selection_method = getattr(selector, "get_diagnostics", None)
        if callable(selection_method):
            selection = selection_method()
            (output / "selection.json").write_text(
                json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8"
            )
            pd.DataFrame(selection["candidates"]).to_parquet(
                output / "mdl_candidates.parquet", index=False
            )
        diagnostics = []
        for index, estimator in enumerate(estimators):
            method = getattr(estimator, "get_feature_importance", None)
            diagnostic = method() if callable(method) else None
            if diagnostic is not None:
                diagnostics.append({"model_index": index, **diagnostic})
        if diagnostics:
            (output / "feature_importance.json").write_text(
                json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8"
            )
        figures = output / "figures"
        forecast_figures(table, figures)
        training_figures(frame.loc[X_train.index], X_train, figures)
        if selection is not None:
            mdl_figures(selection, figures)
        self.tracker.record(metadata, metrics, str(output))
        return BenchmarkResult(output, metrics, metadata)


def compare_runs(
    directories: list[Path],
    output: Path,
    tracker: Tracker | None = None,
) -> pd.DataFrame:
    """Refuse pooling incompatible targets, datasets, phases or evaluation origins."""
    if len(directories) < 2:
        raise ValueError("Comparison requires at least two runs")
    runs = [json.loads((Path(d) / "run.json").read_text(encoding="utf-8")) for d in directories]
    for key in ("benchmark_fingerprint", "evaluation_fingerprint", "phase"):
        if len({run[key] for run in runs}) != 1:
            raise ValueError(f"Cannot compare incompatible {key}")
    rows = []
    for directory, run in zip(directories, runs, strict=True):
        metrics = json.loads((Path(directory) / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "run": str(directory),
                "estimator": run["estimator_name"],
                "selector": run["selector"]["name"],
                **metrics,
            }
        )
    leaderboard = pd.DataFrame(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    metric = next(key for key in rows[0] if key not in {"run", "estimator", "selector"})
    save_figure(
        px.bar(
            leaderboard,
            x="run",
            y=metric,
            color="estimator",
            title=f"Matched {runs[0]['phase']} benchmark: {metric}",
        ),
        output,
        "leaderboard",
    )
    metadata = {
        **runs[0],
        "run_name": (
            f"{runs[0]['benchmark_id']}-v{runs[0]['benchmark_version']}"
            f"-comparison-{runs[0]['phase']}"
        ),
        "estimator_name": "comparison",
        "runs": [str(d) for d in directories],
        "candidates": runs,
    }
    for key in ("estimator", "strategy", "selector", "transformer", "selected_features"):
        metadata[key] = [run[key] for run in runs]
    (output / "comparison.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if tracker is not None:
        scores = {
            f"candidate_{i}/{name}": value
            for i, row in enumerate(rows)
            for name, value in row.items()
            if name not in {"run", "estimator", "selector"}
        }
        tracker.record(metadata, scores, str(output))
    return leaderboard
