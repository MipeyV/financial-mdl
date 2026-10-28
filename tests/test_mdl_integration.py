import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from financial_mdl.benchmark import BenchmarkRunner, compare_runs
from financial_mdl.cli import main
from financial_mdl.data import MarketDataset
from financial_mdl.mdl import ExhaustiveMDLSelector
from financial_mdl.mdl_experiment import run_synthetic_mdl
from financial_mdl.models import make_estimator


def test_mdl_selection_is_unchanged_when_all_heldout_prices_change(
    dataset,
    protocol,
    tmp_path,
    fast_reports,
):
    def execute(data, rules, path):
        return BenchmarkRunner().run(
            data,
            rules,
            lambda: make_estimator("ridge"),
            path,
            estimator_name="ridge",
            selector_factory=ExhaustiveMDLSelector,
        )

    reference = execute(dataset, protocol, tmp_path / "reference")
    changed = dataset.frame.copy()
    heldout = changed.index > protocol.split.train_end
    changed.loc[heldout, ["open", "high", "low", "close"]] *= 10
    changed.loc[heldout, "volume"] *= 100
    altered = MarketDataset(changed, dataset.metadata)
    second = execute(
        altered, replace(protocol, dataset_fingerprint=altered.fingerprint), tmp_path / "altered"
    )
    diagnostic = json.loads((reference.directory / "selection.json").read_text())
    assert diagnostic == json.loads((second.directory / "selection.json").read_text())
    assert diagnostic["training_rows"] == reference.metadata["splits"]["train"]["count"]
    assert len(pd.read_parquet(reference.directory / "mdl_candidates.parquet")) == 32
    assert reference.metadata["selected_features"] == tuple(diagnostic["selected_features"])
    assert reference.metadata["benchmark_fingerprint"] == protocol.fingerprint
    assert "exhaustive_mdl" in reference.metadata["run_name"]


def test_empty_mdl_subset_completes_same_benchmark_without_fake_features(
    dataset,
    protocol,
    tmp_path,
    fast_reports,
):
    class ConstantSelector(ExhaustiveMDLSelector):
        def fit(self, X_train, y_train):
            # Force the valid intercept-only output to exercise runner/model contracts.
            super().fit(X_train, np.full_like(y_train, 0.25))
            return self

    result = BenchmarkRunner().run(
        dataset,
        protocol,
        lambda: make_estimator("ridge"),
        tmp_path / "constant",
        selector_factory=ConstantSelector,
    )
    assert result.metadata["selected_features"] == ()
    table = pd.read_parquet(result.directory / "predictions.parquet")
    np.testing.assert_array_equal(table.predicted, np.full(len(table), table.predicted.iloc[0]))
    assert result.metadata["estimator"][0]["effective_model"] == "intercept_only_training_mean"
    assert not (result.directory / "feature_importance.json").exists()


def test_cli_mdl_exports_real_decomposition_and_compares_with_full_ridge(
    dataset,
    protocol,
    tmp_path,
):
    data, rules = tmp_path / "market.parquet", tmp_path / "protocol.json"
    dataset.save(data)
    protocol.save(rules)
    for name, selector in (("full", "identity"), ("mdl", "mdl")):
        main(
            [
                "run",
                "--data",
                str(data),
                "--protocol",
                str(rules),
                "--estimator",
                "ridge",
                "--selector",
                selector,
                "--output",
                str(tmp_path / name),
            ]
        )
    chart = json.loads((tmp_path / "mdl/figures/mdl_decomposition.json").read_text())
    assert len(chart["data"]) == 3
    diagnostic = json.loads((tmp_path / "mdl/selection.json").read_text())
    assert diagnostic["selected"]["total_bits"] <= diagnostic["full"]["total_bits"]
    leaderboard = compare_runs([tmp_path / "full", tmp_path / "mdl"], tmp_path / "comparison")
    assert set(leaderboard.selector) == {"identity", "exhaustive_mdl"}
    assert (tmp_path / "mdl/figures/mdl_complexity_error.html").exists()


def test_cli_rejects_nonreference_estimators_for_mdl(dataset, protocol, tmp_path, capsys):
    data, rules = tmp_path / "market.parquet", tmp_path / "protocol.json"
    dataset.save(data)
    protocol.save(rules)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "run",
                "--data",
                str(data),
                "--protocol",
                str(rules),
                "--estimator",
                "tree",
                "--selector",
                "mdl",
                "--output",
                str(tmp_path / "unused"),
            ]
        )
    assert error.value.code == 2
    assert "fixed Ridge reference" in capsys.readouterr().err
    assert not (tmp_path / "unused").exists()


def test_synthetic_experiment_is_reproducible_and_preserves_paired_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr("financial_mdl.mdl_experiment.save_figure", lambda *args: None)
    first = run_synthetic_mdl(tmp_path / "one", seeds=(0,), n_train=100, n_validation=60)
    second = run_synthetic_mdl(tmp_path / "two", seeds=(0,), n_train=100, n_validation=60)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 6
    assert set(first.scenario) == {"signal", "noise_only", "weak_signal"}
    for _, pair in first.groupby("scenario"):
        assert pair.dataset_fingerprint.nunique() == 1
        assert (
            pair.loc[pair.method == "mdl", "total_bits"].iloc[0]
            <= pair.loc[pair.method == "full", "total_bits"].iloc[0]
        )
    assert (tmp_path / "one/selection_frequency.csv").exists()
    with pytest.raises(FileExistsError):
        run_synthetic_mdl(tmp_path / "one", seeds=(0,))
