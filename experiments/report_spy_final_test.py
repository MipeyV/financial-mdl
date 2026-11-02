"""Audit frozen predictions and export final SPY tables/figures without fitting models."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from mlflow.tracking import MlflowClient

from financial_mdl.benchmark import _evaluation_fingerprint
from financial_mdl.metrics import point_metrics
from financial_mdl.protocol import BenchmarkConfig
from financial_mdl.reporting import save_figure

LABELS = {"full": "Ridge complet", "mdl": "Ridge + MDL", "persistence-20": "Persistance 20"}
COLORS = {"full": "#586e85", "mdl": "#cf2854", "persistence-20": "#208d93"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_panels(summary, panels, title, output, stem):
    with rc_context({"font.size": 11, "svg.fonttype": "none"}):
        figure = Figure(figsize=(11, 5.4), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2, sharey=panels[0][1] == panels[1][1])
        for axis, (phase, metric, label) in zip(axes, panels, strict=True):
            table = summary.loc[summary.phase == phase].set_index("candidate").loc[list(LABELS)]
            bars = axis.bar(
                ["Ridge\ncomplet", "Ridge\n+ MDL", "Persistance\n20 séances"],
                table[metric],
                color=list(COLORS.values()),
                width=0.65,
            )
            axis.bar_label(bars, labels=[f"{value:.6f}" for value in table[metric]], padding=4)
            axis.set_ylim(0, max(summary[metric]) * 1.18)
            axis.set_title(label, pad=12)
            axis.set_ylabel(f"{metric.upper()} (unités de volatilité annualisée)")
            axis.yaxis.grid(alpha=0.2)
            axis.set_axisbelow(True)
        figure.suptitle(title + "\nConfigurations figées ; ajustement sur 2015–2021 uniquement")
        figure.savefig(output / f"{stem}.png", dpi=200)
        figure.savefig(output / f"{stem}.svg")


def build_report(plan_path, output):
    plan = read_json(plan_path)
    root = Path(plan["output_root"])
    protocol = BenchmarkConfig.load(Path(plan["protocol_path"]))
    assert protocol.fingerprint == plan["protocol_fingerprint"]
    rows, predictions, metadata, fingerprints = [], {}, {}, {}
    for candidate in plan["candidates"]:
        name = candidate["id"]
        validation_metadata = read_json(Path(candidate["validation_run"]) / "run.json")
        for phase, directory in (
            ("validation", Path(candidate["validation_run"])),
            ("test", root / name),
        ):
            run = read_json(directory / "run.json")
            metrics = read_json(directory / "metrics.json")
            table = pd.read_parquet(directory / "predictions.parquet")
            assert run["phase"] == phase
            assert run["benchmark_fingerprint"] == plan["protocol_fingerprint"]
            assert run["dataset_fingerprint"] == plan["dataset_fingerprint"]
            assert _evaluation_fingerprint(table) == run["evaluation_fingerprint"]
            for metric, value in point_metrics(
                table.actual.to_numpy(), table.predicted.to_numpy()
            ).items():
                np.testing.assert_allclose(value, metrics[metric], rtol=1e-13, atol=1e-15)
            fingerprints.setdefault(phase, set()).add(run["evaluation_fingerprint"])
            if phase == "test":
                assert len(table) == plan["evaluation"]["expected_test_origins"]
                assert table.origin.min() > pd.Timestamp(protocol.split.validation_end)
                assert table.target_timestamp.max() <= pd.Timestamp(protocol.split.test_end)
                assert not run["git_dirty"]
                for field in (
                    "estimator",
                    "selector",
                    "transformer",
                    "strategy",
                    "seed",
                    "selected_features",
                    "feature_definitions",
                    "splits",
                    "packages",
                ):
                    assert run[field] == validation_metadata[field], field
                for source, digest in run["source_sha256"].items():
                    assert (
                        hashlib.sha256(
                            (Path("src/financial_mdl") / source).read_bytes()
                        ).hexdigest()
                        == digest
                    )
                assert (
                    hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest() == run["lock_sha256"]
                )
                predictions[name], metadata[name] = table, run
            rows.append(
                {
                    "phase": phase,
                    "candidate": name,
                    "features_used": 1
                    if name == "persistence-20"
                    else len(run["selected_features"]),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "n_origins": len(table),
                    "evaluation_fingerprint": run["evaluation_fingerprint"],
                    "git_commit": run["git_commit"],
                }
            )
    assert all(len(values) == 1 for values in fingerprints.values())
    assert fingerprints["validation"] != fingerprints["test"]
    for table in predictions.values():
        pd.testing.assert_frame_equal(
            table.drop(columns="predicted"), predictions["full"].drop(columns="predicted")
        )
    selection = read_json(root / "mdl/selection.json")
    mdl_plan = next(c for c in plan["candidates"] if c["id"] == "mdl")
    assert selection == read_json(Path(mdl_plan["validation_run"]) / "selection.json")
    assert selection["selected_features"] == mdl_plan["expected_selected_features"]
    assert selection["selected"]["total_bits"] == mdl_plan["expected_selected_training_bits"]
    assert selection["full"]["total_bits"] == mdl_plan["expected_full_training_bits"]
    commits = {run["git_commit"] for run in metadata.values()}
    assert len(commits) == 1
    client = MlflowClient(
        tracking_uri=f"sqlite:///{(Path(plan['tracking_root']).resolve() / 'mlflow.db').as_posix()}"
    )
    experiment = client.get_experiment_by_name("financial-mdl")
    tracking = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.git_commit = '{next(iter(commits))}' and tags.phase = 'test'",
    )
    assert len(tracking) == 4 and all(run.info.status == "FINISHED" for run in tracking)
    summary = pd.DataFrame(rows)
    for phase in ("validation", "test"):
        full_rmse = summary.loc[
            (summary.phase == phase) & (summary.candidate == "full"), "rmse"
        ].iloc[0]
        summary.loc[summary.phase == phase, "relative_rmse_change_vs_full"] = (
            summary.loc[summary.phase == phase, "rmse"] / full_rmse - 1
        )
    test = summary.loc[summary.phase == "test"].set_index("candidate")
    ratio = float(test.loc["mdl", "rmse"] / test.loc["full", "rmse"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "summary.csv", index=False)
    verification = {
        "plan_sha256": hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        "test_run_commit": next(iter(commits)),
        "protocol_fingerprint": protocol.fingerprint,
        "evaluation_fingerprints": {
            phase: next(iter(values)) for phase, values in fingerprints.items()
        },
        "matched_observations": len(predictions["full"]),
        "origin_start": str(predictions["full"].origin.min()),
        "origin_end": str(predictions["full"].origin.max()),
        "last_label_end": str(predictions["full"].target_timestamp.max()),
        "training_selection_identical_to_validation": True,
        "candidate_configuration_identical_to_validation": True,
        "exported_metrics_recomputed": True,
        "mdl_rmse_ratio_to_full": ratio,
        "within_declared_tolerance": ratio <= 1 + plan["evaluation"]["mdl_relative_rmse_tolerance"],
        "mlflow_runs": [
            {"id": run.info.run_id, "name": run.info.run_name, "status": run.info.status}
            for run in tracking
        ],
        "prediction_sha256": {
            name: hashlib.sha256((root / name / "predictions.parquet").read_bytes()).hexdigest()
            for name in LABELS
        },
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    metric_panels(
        summary,
        [("test", "mae", "MAE"), ("test", "rmse", "RMSE")],
        "Test SPY 2024–2025 : 497 prévisions communes",
        output,
        "test-metrics",
    )
    metric_panels(
        summary,
        [("validation", "rmse", "Validation 2022–2023"), ("test", "rmse", "Test final 2024–2025")],
        "Comparaisons distinctes sur deux périodes",
        output,
        "validation-test-rmse",
    )
    display = summary.assign(model=summary.candidate.map(LABELS))
    save_figure(
        px.bar(
            display,
            x="model",
            y="rmse",
            color="model",
            facet_col="phase",
            title="RMSE : validation et test, configurations figées",
        ),
        output,
        "validation-test-rmse",
    )
    forecast = pd.DataFrame(
        {
            "origin": predictions["full"].origin,
            "Observée (5 séances futures)": predictions["full"].actual,
        }
    )
    for name, table in predictions.items():
        forecast[LABELS[name]] = table.predicted
    save_figure(
        px.line(
            forecast,
            x="origin",
            y=list(forecast.columns[1:]),
            title="Test SPY : volatilité future observée et prévisions",
        ),
        output,
        "test-forecasts",
    )
    with rc_context({"font.size": 11, "svg.fonttype": "none"}):
        figure = Figure(figsize=(12, 5.7), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(
            forecast.origin,
            forecast.iloc[:, 1],
            color="#202020",
            label=forecast.columns[1],
            linewidth=1.7,
        )
        for name in LABELS:
            axis.plot(
                forecast.origin,
                forecast[LABELS[name]],
                color=COLORS[name],
                label=LABELS[name],
                linewidth=1.1,
                alpha=0.9,
            )
        axis.set_title("Test SPY 2024–2025 : prévisions aux dates d'origine")
        axis.set_ylabel("Volatilité annualisée (%)")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=0.18)
        figure.legend(loc="outside lower center", ncols=2, frameon=False)
        figure.savefig(output / "test-forecasts.png", dpi=200)
        figure.savefig(output / "test-forecasts.svg")
    print(summary.to_string(index=False))
    print(json.dumps(verification, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/spy-final-test-v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.plan, args.output)
