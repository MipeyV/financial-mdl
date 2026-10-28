import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from financial_mdl.benchmark import BenchmarkRunner, compare_runs
from financial_mdl.candidates import make_candidate
from financial_mdl.cli import main
from financial_mdl.mdl import ExhaustiveMDLSelector
from financial_mdl.models import make_estimator
from financial_mdl.tracking import LocalMLflowTracker


def test_cli_offline_demo_generates_real_plotly_and_comparison(tmp_path, capsys):
    output = tmp_path / "demo"
    main(["demo", "--output", str(output)])
    assert "ridge" in capsys.readouterr().out
    for name in ("naive", "ridge"):
        figure = json.loads((output / name / "figures" / "forecasts.json").read_text())
        assert len(figure["data"]) == 2
        assert "Actual versus predicted" in (
            output / name / "figures" / "forecasts.html"
        ).read_text(encoding="utf-8")
    assert (output / "comparison" / "leaderboard.csv").exists()


def test_core_imports_and_noop_work_without_mlflow():
    code = """
import sys
class BlockTracking:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlflow' or fullname.startswith('mlflow.'):
            raise ImportError('MLflow deliberately unavailable')
sys.meta_path.insert(0, BlockTracking())
from financial_mdl.benchmark import BenchmarkRunner
from financial_mdl.tracking import NoOpTracker
NoOpTracker().record({}, {}, '.')
assert 'mlflow' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30
    )


@pytest.mark.tracking
@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="optional tracking extra")
def test_real_local_mlflow_records_params_metrics_and_predictions(
    dataset, protocol, tmp_path, fast_reports
):
    # Disable telemetry in the optional integration test; no tracking server is needed.
    os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"
    from mlflow.tracking import MlflowClient

    root = tmp_path / "mlflow"
    tracker = LocalMLflowTracker(root, experiment_name="integration")
    result = BenchmarkRunner(tracker).run(
        dataset, protocol, lambda: make_estimator("naive"), tmp_path / "run"
    )
    client = MlflowClient(tracking_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    experiment = client.get_experiment_by_name("integration")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.metrics["validation/rmse"] == result.metrics["rmse"]
    assert json.loads(run.data.params["dataset_fingerprint"]) == dataset.fingerprint
    assert "predictions.parquet" in {
        Path(a.path).name for a in client.list_artifacts(run.info.run_id)
    }
    ridge = BenchmarkRunner().run(
        dataset,
        protocol,
        lambda: make_estimator("ridge"),
        tmp_path / "ridge",
        estimator_name="ridge",
    )
    compare_runs([result.directory, ridge.directory], tmp_path / "comparison", tracker)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 2
    comparison = next(r for r in runs if "comparison" in r.data.tags["mlflow.runName"])
    assert comparison.data.metrics["validation/candidate_1/rmse"] == ridge.metrics["rmse"]
    assert "leaderboard.csv" in {
        Path(a.path).name for a in client.list_artifacts(comparison.info.run_id)
    }
    estimator, transformer = make_candidate("persistence-20")
    persistence = BenchmarkRunner(tracker).run(
        dataset,
        protocol,
        estimator,
        tmp_path / "persistence",
        estimator_name="persistence-20",
        transformer_factory=transformer,
    )
    runs = client.search_runs([experiment.experiment_id])
    logged = next(r for r in runs if "persistence-20" in r.data.tags["mlflow.runName"])
    assert logged.data.metrics["validation/rmse"] == persistence.metrics["rmse"]
    assert json.loads(logged.data.params["transformer"])["name"] == "identity"
    mdl = BenchmarkRunner(tracker).run(
        dataset,
        protocol,
        lambda: make_estimator("ridge"),
        tmp_path / "mdl",
        estimator_name="ridge",
        selector_factory=ExhaustiveMDLSelector,
    )
    selection = json.loads((mdl.directory / "selection.json").read_text())
    logged = next(
        r
        for r in client.search_runs([experiment.experiment_id])
        if r.data.params.get("selector_name") == "exhaustive_mdl"
    )
    assert logged.data.params["n_selected_features"] == str(len(mdl.metadata["selected_features"]))
    assert (
        logged.data.metrics["training/mdl/selected/total_bits"]
        == selection["selected"]["total_bits"]
    )
    assert logged.data.metrics["validation/rmse"] == mdl.metrics["rmse"]
    assert "validation/total_bits" not in logged.data.metrics
    assert "mdl_candidates.parquet" in {
        Path(a.path).name for a in client.list_artifacts(logged.info.run_id)
    }
