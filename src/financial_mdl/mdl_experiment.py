"""Reproducible synthetic controls with an explicit, separate regression task."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.linear_model import Ridge

from financial_mdl.benchmark import _provenance
from financial_mdl.experiments import controlled_regression
from financial_mdl.features import StandardizeFeatures
from financial_mdl.mdl import ExhaustiveMDLSelector, MDLCodingSpec
from financial_mdl.metrics import point_metrics
from financial_mdl.models import SklearnAdapter
from financial_mdl.reporting import save_figure


def run_synthetic_mdl(
    output: Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    n_train: int = 600,
    n_validation: int = 300,
    spec: MDLCodingSpec | None = None,
) -> pd.DataFrame:
    """Fit selection on a fixed training block and report one untouched held-out block.

    Results are not combined with the SPY benchmark: the target, data and task differ.
    Weak signals intentionally expose the fixed binary64 model code's sensitivity.
    """
    if (
        not isinstance(seeds, tuple)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("seeds must be a nonempty tuple of distinct nonnegative integers")
    if type(n_train) is not int or n_train < 2 or type(n_validation) is not int or n_validation < 2:
        raise ValueError("Require at least two training and validation observations")
    spec = spec if spec is not None else MDLCodingSpec()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configuration = {
        "experiment": "controlled_regression_mdl_v1",
        "seeds": seeds,
        "n_train": n_train,
        "n_validation": n_validation,
        "scenarios": {
            "signal": "y=1+2*signal_a-1.5*signal_b+Normal(0,0.4)",
            "noise_only": "y=1+Normal(0,0.4), independent of X",
            "weak_signal": "y=1+0.05*signal_a+Normal(0,0.4)",
        },
        "redundancy": "signal_a_copy equals signal_a exactly",
        "noise_features": ["noise_0", "noise_1", "noise_2"],
        "coding_spec": asdict(spec),
        "metrics": ["mae", "rmse"],
        "metric_version": "1",
        "split": "first n_train independent observations; remaining rows held out",
        "baseline": "same Ridge/scaler/alpha, all features",
        "tolerance": 0.02,
    }
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    (output / "experiment.json").write_text(
        json.dumps(
            {
                **configuration,
                "experiment_fingerprint": fingerprint,
                **_provenance(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rows, frequency = [], []
    for scenario in configuration["scenarios"]:
        for seed in seeds:
            X, y = controlled_regression(scenario, seed, n_train + n_validation)
            data_hash = hashlib.sha256(
                X.to_numpy(dtype="<f8").tobytes() + y.astype("<f8").tobytes()
            ).hexdigest()
            selector = ExhaustiveMDLSelector(spec).fit(X.iloc[:n_train], y[:n_train])
            diagnostic = selector.get_diagnostics()
            (output / f"{scenario}-s{seed}-selection.json").write_text(
                json.dumps(diagnostic, indent=2, allow_nan=False), encoding="utf-8"
            )
            seed_rows = []
            for method, names, code in (
                ("full", tuple(X.columns), diagnostic["full"]),
                ("mdl", selector.get_selected_features(), diagnostic["selected"]),
            ):
                train, validation = (
                    X.iloc[:n_train].loc[:, list(names)],
                    X.iloc[n_train:].loc[:, list(names)],
                )
                transformer = StandardizeFeatures().fit(train)
                # Match the reference alpha exactly; this experiment uses no hyperparameter search.
                estimator = SklearnAdapter(Ridge(alpha=spec.ridge_alpha)).fit(
                    transformer.transform(train),
                    y[:n_train],
                )
                predictions = estimator.predict(transformer.transform(validation))
                scores = point_metrics(y[n_train:], predictions)
                seed_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "n_features": len(names),
                        "features": ",".join(names),
                        **{
                            name: code[name]
                            for name in (
                                "subset_bits",
                                "model_bits",
                                "target_bits",
                                "total_bits",
                            )
                        },
                        "representation_bits": code["subset_bits"] + code["model_bits"],
                        "dataset_fingerprint": data_hash,
                        **scores,
                    }
                )
            reference_rmse = seed_rows[0]["rmse"]
            for row in seed_rows:
                row["rmse_ratio_to_full"] = row["rmse"] / reference_rmse
                row["within_2pct_tolerance"] = row["rmse_ratio_to_full"] <= 1.02
            rows.extend(seed_rows)
            frequency.extend(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "feature": name,
                    "selected": name in selector.get_selected_features(),
                }
                for name in X.columns
            )
    results = pd.DataFrame(rows)
    results.to_csv(output / "results.csv", index=False)
    counts = (
        pd.DataFrame(frequency).groupby(["scenario", "feature"], as_index=False).selected.mean()
    )
    counts.to_csv(output / "selection_frequency.csv", index=False)
    save_figure(
        px.box(
            results,
            x="scenario",
            y="rmse",
            color="method",
            points="all",
            title="Independent synthetic controls: held-out RMSE across seeds",
        ),
        output,
        "heldout_error",
    )
    frequencies = counts.pivot(index="scenario", columns="feature", values="selected")
    save_figure(
        px.imshow(frequencies, zmin=0, zmax=1, title="Synthetic MDL selection frequency"),
        output,
        "selection_frequency",
    )
    return results
