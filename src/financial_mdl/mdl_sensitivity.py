"""Predeclared synthetic grid: nested training sizes, fixed holdout, paired precisions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from financial_mdl.benchmark import _provenance
from financial_mdl.experiments import controlled_regression
from financial_mdl.mdl import MDLCodingSpec
from financial_mdl.mdl_experiment import SYNTHETIC_SCENARIOS, _evaluate_pair
from financial_mdl.reporting import save_figure


@dataclass(frozen=True)
class SensitivityConfig:
    train_sizes: tuple[int, ...] = (150, 600, 2400, 9600)
    precisions: tuple[float, ...] = (0.0001, 0.01, 0.1)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_heldout: int = 300
    version: str = "controlled_regression_mdl_sensitivity_v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.train_sizes, tuple)
            or not self.train_sizes
            or any(type(n) is not int or n < 2 for n in self.train_sizes)
            or tuple(sorted(set(self.train_sizes))) != self.train_sizes
        ):
            raise ValueError("train_sizes must be distinct increasing integers >= 2")
        if (
            not isinstance(self.precisions, tuple)
            or not self.precisions
            or any(
                type(p) not in (float, int) or not math.isfinite(p) or p <= 0
                for p in self.precisions
            )
            or len(set(self.precisions)) != len(self.precisions)
        ):
            raise ValueError("precisions must be distinct finite positive numbers")
        if (
            not isinstance(self.seeds, tuple)
            or not self.seeds
            or any(type(s) is not int or s < 0 for s in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must be distinct nonnegative integers")
        if type(self.n_heldout) is not int or self.n_heldout < 2:
            raise ValueError("n_heldout must be an integer >= 2")
        if self.version != "controlled_regression_mdl_sensitivity_v1":
            raise ValueError("Unsupported sensitivity experiment version")

    @classmethod
    def load(cls, path: Path) -> SensitivityConfig:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {"train_sizes", "precisions", "seeds", "n_heldout", "version"}
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f"Sensitivity configuration requires exactly {sorted(expected)}")
        for name in ("train_sizes", "precisions", "seeds"):
            if not isinstance(values[name], list):
                raise ValueError(f"{name} must be a JSON array")
            values[name] = tuple(values[name])
        return cls(**values)


def _fingerprint(X: pd.DataFrame, y: np.ndarray) -> str:
    return hashlib.sha256(X.to_numpy(dtype="<f8").tobytes() + y.astype("<f8").tobytes()).hexdigest()


def run_mdl_sensitivity(output: Path, config: SensitivityConfig) -> pd.DataFrame:
    """Run the entire declared grid without ranking configurations by held-out error.

    All precisions and sizes share one untouched holdout per scenario/seed. Training
    blocks are prefixes of the same maximum-sized pool, disjoint from that holdout.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    specification = {
        **asdict(config),
        "generator": "controlled_regression_v1",
        "scenarios": SYNTHETIC_SCENARIOS,
        "design": "six columns: signal_a, signal_b, signal_a_copy, noise_0, noise_1, noise_2",
        "redundancy": "signal_a_copy equals signal_a exactly",
        "split": "nested training prefixes; fixed last n_heldout rows after max(train_sizes)",
        "reference_code": asdict(MDLCodingSpec()),
        "varied_code_field": "target_precision only; binary64 model cost and alpha stay fixed",
        "metrics": ["mae", "rmse"],
        "metric_version": "1",
        "rmse_tolerance": 0.02,
        "selection": "training code length only; no configuration chosen from held-out error",
    }
    fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    # Persist the complete plan and provenance before any data generation or fitting.
    (output / "experiment.json").write_text(
        json.dumps(
            {**specification, "experiment_fingerprint": fingerprint, **_provenance()}, indent=2
        ),
        encoding="utf-8",
    )
    maximum = max(config.train_sizes)
    rows, selections = [], []
    for scenario in SYNTHETIC_SCENARIOS:
        for seed in config.seeds:
            X, y = controlled_regression(scenario, seed, maximum + config.n_heldout)
            heldout_X, heldout_y = X.iloc[maximum:], y[maximum:]
            heldout_hash = _fingerprint(heldout_X, heldout_y)
            for n_train in config.train_sizes:
                train_X, train_y = X.iloc[:n_train], y[:n_train]
                train_hash = _fingerprint(train_X, train_y)
                for precision_index, precision in enumerate(config.precisions):
                    pair, diagnostic = _evaluate_pair(
                        train_X,
                        train_y,
                        heldout_X,
                        heldout_y,
                        MDLCodingSpec(target_precision=precision),
                    )
                    cell = {
                        "scenario": scenario,
                        "seed": seed,
                        "n_train": n_train,
                        "target_precision": precision,
                    }
                    stem = f"{scenario}-s{seed}-n{n_train}-p{precision_index}"
                    (output / f"{stem}-selection.json").write_text(
                        json.dumps({**cell, **diagnostic}, indent=2, allow_nan=False),
                        encoding="utf-8",
                    )
                    for row in pair:
                        row.update(
                            **cell,
                            training_fingerprint=train_hash,
                            heldout_fingerprint=heldout_hash,
                        )
                    rows.extend(pair)
                    selected = set(diagnostic["selected_features"])
                    selections.extend(
                        {**cell, "feature": name, "selected": name in selected}
                        for name in X.columns
                    )
    results = pd.DataFrame(rows)
    results.to_csv(output / "results.csv", index=False)
    groups = ["scenario", "n_train", "target_precision"]
    summary = results.groupby([*groups, "method"], as_index=False).agg(
        n_seeds=("seed", "size"),
        mean_features=("n_features", "mean"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_total_bits=("total_bits", "mean"),
        mean_rmse_ratio=("rmse_ratio_to_full", "mean"),
        max_rmse_ratio=("rmse_ratio_to_full", "max"),
        tolerance_fraction=("within_2pct_tolerance", "mean"),
    )
    summary.to_csv(output / "summary.csv", index=False)
    frequencies = (
        pd.DataFrame(selections)
        .groupby([*groups, "feature"], as_index=False)
        .agg(frequency=("selected", "mean"))
    )
    frequencies.to_csv(output / "selection_frequency.csv", index=False)
    mdl = summary.loc[summary.method == "mdl"].copy()
    mdl["precision"] = mdl.target_precision.astype(str)
    for name, column, title in (
        ("selected_features", "mean_features", "MDL mean selected features across seeds"),
        ("heldout_rmse_ratio", "mean_rmse_ratio", "Mean paired held-out RMSE ratio: MDL / full"),
    ):
        figure = px.line(
            mdl,
            x="n_train",
            y=column,
            color="precision",
            facet_col="scenario",
            markers=True,
            log_x=True,
            title=title,
        )
        if column == "mean_rmse_ratio":
            figure.add_hline(y=1.02, line_dash="dash", annotation_text="+2% tolerance")
        save_figure(figure, output, name)
    return results
