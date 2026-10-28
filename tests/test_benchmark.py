import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from financial_mdl.benchmark import BenchmarkRunner, compare_runs
from financial_mdl.data import MarketDataset
from financial_mdl.features import IdentityFeatureSelector, StandardizeFeatures
from financial_mdl.forecasting import DirectMultiHorizonStrategy
from financial_mdl.models import make_estimator


def run_candidate(dataset, protocol, path, name="ridge", **kwargs):
    return BenchmarkRunner().run(
        dataset, protocol, lambda: make_estimator(name), path, estimator_name=name, **kwargs
    )


def test_two_models_share_protocol_and_evaluation_origins(
    dataset, protocol, tmp_path, fast_reports
):
    baseline = run_candidate(dataset, protocol, tmp_path / "naive", name="naive")
    ridge = run_candidate(dataset, protocol, tmp_path / "ridge")
    for key in ("benchmark_fingerprint", "dataset_fingerprint", "evaluation_fingerprint", "splits"):
        assert baseline.metadata[key] == ridge.metadata[key]
    assert baseline.metrics != ridge.metrics
    assert baseline.metadata["phase"] == "validation"
    table = pd.read_parquet(ridge.directory / "predictions.parquet")
    assert table.origin.min() > pd.Timestamp(protocol.split.train_end)
    assert table.target_timestamp.max() <= pd.Timestamp(protocol.split.validation_end)
    assert table.horizon.unique().tolist() == [5]
    assert (ridge.directory / "protocol.json").exists()
    assert "source_sha256" in json.loads((ridge.directory / "run.json").read_text())
    assert not (baseline.directory / "feature_importance.json").exists()
    diagnostics = json.loads((ridge.directory / "feature_importance.json").read_text())
    assert diagnostics[0]["kind"] == "coefficient_in_transformed_feature_space"
    assert tuple(diagnostics[0]["features"]) == protocol.feature_names
    leaderboard = compare_runs([baseline.directory, ridge.directory], tmp_path / "comparison")
    assert set(leaderboard.estimator) == {"naive", "ridge"}


def test_selector_transformer_estimator_never_fit_on_heldout_data(
    dataset, protocol, tmp_path, fast_reports
):
    calls = {"selector": [], "transformer": [], "estimator": [], "y": []}

    class SpySelector(IdentityFeatureSelector):
        def fit(self, X_train, y_train):
            calls["selector"].append(X_train.copy())
            calls["y"].append(y_train.copy())
            return super().fit(X_train, y_train)

    class SpyTransformer(StandardizeFeatures):
        def fit(self, X):
            calls["transformer"].append(X.copy())
            return super().fit(X)

    def factory():
        adapter = make_estimator("naive")
        original_fit = adapter.fit

        def fit(X, y):
            calls["estimator"].append(X.copy())
            return original_fit(X, y)

        adapter.fit = fit
        return adapter

    def execute(data, rules, directory):
        return BenchmarkRunner().run(
            data,
            rules,
            factory,
            directory,
            selector_factory=SpySelector,
            transformer_factory=SpyTransformer,
        )

    first = execute(dataset, protocol, tmp_path / "first")
    changed = dataset.frame.copy()
    heldout = changed.index > protocol.split.train_end
    changed.loc[heldout, ["open", "high", "low", "close"]] *= 10
    changed.loc[heldout, "volume"] *= 100
    altered = MarketDataset(changed, dataset.metadata)
    altered_protocol = replace(protocol, dataset_fingerprint=altered.fingerprint)
    execute(altered, altered_protocol, tmp_path / "altered")
    for name in ("selector", "transformer", "estimator"):
        assert len(calls[name]) == 2
        assert calls[name][0].index.max() <= pd.Timestamp(protocol.split.train_end)
        pd.testing.assert_frame_equal(calls[name][0], calls[name][1])
    np.testing.assert_array_equal(calls["y"][0], calls["y"][1])
    assert first.metadata["splits"]["train"]["label_end"][:10] <= protocol.split.train_end


def test_rejects_changed_snapshot_before_fitting(dataset, protocol, tmp_path):
    changed = dataset.frame.copy()
    changed.iloc[1, 4] += 1
    changed = MarketDataset(changed, dataset.metadata)
    with pytest.raises(ValueError, match="authorized protocol"):
        run_candidate(changed, protocol, tmp_path / "bad")
    with pytest.raises(ValueError, match="authorized protocol"):
        run_candidate(dataset, replace(protocol, dataset_version="other"), tmp_path / "bad-version")


def test_explicit_test_phase_cannot_be_compared_with_validation(
    dataset, protocol, tmp_path, fast_reports
):
    validation = run_candidate(dataset, protocol, tmp_path / "validation")
    final = run_candidate(dataset, protocol, tmp_path / "test", phase="test")
    table = pd.read_parquet(final.directory / "predictions.parquet")
    assert table.origin.min() > pd.Timestamp(protocol.split.validation_end)
    assert table.target_timestamp.max() <= pd.Timestamp(protocol.split.test_end)
    assert validation.metadata["splits"]["train"] == final.metadata["splits"]["train"]
    with pytest.raises(ValueError, match="incompatible"):
        compare_runs([validation.directory, final.directory], tmp_path / "bad-comparison")


def test_compare_rejects_different_target_protocols(
    dataset, protocol, multi_protocol, tmp_path, fast_reports
):
    one = run_candidate(dataset, protocol, tmp_path / "one")
    multiple = run_candidate(
        dataset, multi_protocol, tmp_path / "multiple", strategy_factory=DirectMultiHorizonStrategy
    )
    with pytest.raises(ValueError, match="benchmark_fingerprint"):
        compare_runs([one.directory, multiple.directory], tmp_path / "bad")
    assert {"rmse/h1/realized_volatility", "rmse/h5/realized_volatility"} <= multiple.metrics.keys()


def test_bad_fitted_component_cannot_reorder_rows(dataset, protocol, tmp_path, fast_reports):
    class ReorderingSelector(IdentityFeatureSelector):
        def transform(self, X):
            return super().transform(X).iloc[::-1]

    with pytest.raises(ValueError, match="exact feature row index"):
        run_candidate(dataset, protocol, tmp_path / "bad", selector_factory=ReorderingSelector)


def test_seeded_candidate_is_reproducible_and_artifacts_never_overwritten(
    dataset, protocol, tmp_path, fast_reports
):
    first = run_candidate(dataset, protocol, tmp_path / "tree1", name="tree")
    second = run_candidate(dataset, protocol, tmp_path / "tree2", name="tree")
    assert first.metrics == second.metrics
    pd.testing.assert_frame_equal(
        pd.read_parquet(first.directory / "predictions.parquet"),
        pd.read_parquet(second.directory / "predictions.parquet"),
    )
    with pytest.raises(FileExistsError):
        run_candidate(dataset, protocol, first.directory)


def test_tracker_gets_artifacts_without_being_part_of_estimators(
    dataset, protocol, tmp_path, fast_reports
):
    class SpyTracker:
        def record(self, metadata, metrics, directory):
            self.metadata, self.metrics = metadata, metrics
            assert (tmp_path / "tracked" / "predictions.parquet").exists()

    tracker = SpyTracker()
    result = BenchmarkRunner(tracker).run(
        dataset, protocol, lambda: make_estimator("naive"), tmp_path / "tracked"
    )
    assert tracker.metadata == result.metadata
    assert tracker.metrics == result.metrics
