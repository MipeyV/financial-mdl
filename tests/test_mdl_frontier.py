import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from financial_mdl.cli import main
from financial_mdl.experiments import controlled_regression
from financial_mdl.features import StandardizeFeatures
from financial_mdl.mdl import MDLCodingSpec
from financial_mdl.mdl_frontier import FrontierConfig, evaluate_subsets, run_mdl_frontier
from financial_mdl.metrics import point_metrics
from financial_mdl.models import SklearnAdapter


def test_frontier_plan_is_the_original_synthetic_control_configuration():
    assert FrontierConfig.load(Path("experiments/mdl-frontier-v1.json")) == FrontierConfig()


def test_every_replayed_subset_matches_independently_fitted_ridge():
    X, y = controlled_regression("signal", 2, 130)
    table, diagnostic = evaluate_subsets(X.iloc[:90], y[:90], X.iloc[90:], y[90:], MDLCodingSpec())
    assert len(table) == table.subset_key.nunique() == 64
    assert table.selected_by_mdl.sum() == table.is_full.sum() == table.is_constant.sum() == 1
    assert table.loc[table.selected_by_mdl, "total_bits"].iloc[0] == table.total_bits.min()
    for row in table.itertuples():
        names = json.loads(row.subset_key)
        train, heldout = X.iloc[:90][names], X.iloc[90:][names]
        transformer = StandardizeFeatures().fit(train)
        estimator = SklearnAdapter(Ridge(alpha=1)).fit(transformer.transform(train), y[:90])
        expected = point_metrics(y[90:], estimator.predict(transformer.transform(heldout)))
        assert row.heldout_rmse == pytest.approx(expected["rmse"], abs=1e-12)
        assert row.heldout_mae == pytest.approx(expected["mae"], abs=1e-12)
    chosen = table.loc[table.selected_by_mdl].iloc[0]
    assert json.loads(chosen.subset_key) == diagnostic["selected_features"]


def test_heldout_errors_do_not_change_codes_or_selection():
    X, y = controlled_regression("weak_signal", 0, 140)
    original, before = evaluate_subsets(
        X.iloc[:100], y[:100], X.iloc[100:], y[100:], MDLCodingSpec()
    )
    changed, after = evaluate_subsets(
        X.iloc[:100], y[:100], X.iloc[100:] * 100, y[100:] + 10, MDLCodingSpec()
    )
    assert before == after
    columns = original.columns.difference(["heldout_mae", "heldout_rmse"])
    pd.testing.assert_frame_equal(original[columns], changed[columns])
    assert not original.heldout_rmse.equals(changed.heldout_rmse)


def test_complete_subset_experiment_is_reproducible_and_keeps_seeds_paired(tmp_path, monkeypatch):
    monkeypatch.setattr("financial_mdl.mdl_frontier._plot_frontier", lambda *args: None)
    config = FrontierConfig(n_train=40, n_heldout=20, seeds=(0, 1))
    first = run_mdl_frontier(tmp_path / "one", config)
    second = run_mdl_frontier(tmp_path / "two", config)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3 * 2 * 64
    for _, group in first.groupby(["scenario", "seed"]):
        assert group.training_fingerprint.nunique() == group.heldout_fingerprint.nunique() == 1
        assert len(group) == 64
        assert group.selected_by_mdl.sum() == 1
    summary = pd.read_csv(tmp_path / "one/summary.csv")
    assert len(summary) == 3 * 64
    assert summary.n_seeds.eq(2).all()
    assert summary.groupby("scenario").selection_count.sum().eq(2).all()
    for row in summary.itertuples():
        group = first[(first.scenario == row.scenario) & (first.subset_key == row.subset_key)]
        assert row.mean_heldout_rmse == pytest.approx(group.heldout_rmse.mean())
        assert row.std_heldout_rmse == pytest.approx(group.heldout_rmse.std())
    with pytest.raises(FileExistsError):
        run_mdl_frontier(tmp_path / "one", config)


def test_frontier_cli_exports_headless_static_and_interactive_figures(tmp_path, capsys):
    source = tmp_path / "config.json"
    source.write_text(json.dumps(asdict(FrontierConfig(n_train=30, n_heldout=20, seeds=(0,)))))
    output = tmp_path / "run"
    main(["mdl-frontier", "--config", str(source), "--output", str(output)])
    assert "noise_only" in capsys.readouterr().out
    plan = json.loads((output / "experiment.json").read_text())
    assert len(plan["experiment_fingerprint"]) == 64
    assert plan["coding_spec"]["target_precision"] == 0.0001
    assert len(pd.read_csv(output / "subsets.csv")) == 192
    for scenario in ("signal", "noise_only", "weak_signal"):
        for view in ("all", "zoom"):
            for extension in ("html", "json", "png", "svg"):
                assert (output / f"{scenario}-{view}.{extension}").stat().st_size > 100
        chart = json.loads((output / f"{scenario}-all.json").read_text())
        assert len(chart["data"][0]["x"]) == 64
        assert len(chart["data"]) == 4
        assert chart["layout"]["xaxis"].get("range") is None
        zoom = json.loads((output / f"{scenario}-zoom.json").read_text())
        summary = pd.read_csv(output / "summary.csv")
        full = summary[(summary.scenario == scenario) & summary.is_full].iloc[0]
        visible = summary[
            (summary.scenario == scenario)
            & (summary.mean_heldout_rmse <= 1.10 * full.mean_heldout_rmse)
        ]
        lower, upper = zoom["layout"]["yaxis"]["range"]
        assert lower <= visible.mean_heldout_rmse.min() <= visible.mean_heldout_rmse.max() <= upper
    overview = (output / "index.html").read_text(encoding="utf-8")
    for relative in re.findall(r'(?:href|src)="([^"]+)"', overview):
        assert (output / relative).is_file(), f"Broken report link: {relative}"


@pytest.mark.parametrize("version", ["other", 1])
def test_unsupported_frontier_versions_are_rejected(version):
    with pytest.raises(ValueError):
        FrontierConfig(version=version)


def test_frontier_rejects_mismatched_or_nonfinite_heldout_data():
    X, y = controlled_regression("signal", 0, 40)
    with pytest.raises(ValueError, match="columns/order"):
        evaluate_subsets(X.iloc[:20], y[:20], X.iloc[20:, ::-1], y[20:], MDLCodingSpec())
    with pytest.raises(ValueError, match="aligned finite"):
        evaluate_subsets(X.iloc[:20], y[:20], X.iloc[20:], y[20:, None], MDLCodingSpec())
    with pytest.raises(ValueError, match="aligned finite"):
        evaluate_subsets(X.iloc[:20], y[:20], X.iloc[20:] * np.nan, y[20:], MDLCodingSpec())


def test_invalid_frontier_cli_config_does_not_create_output(tmp_path):
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps({**asdict(FrontierConfig()), "seeds": "0"}))
    with pytest.raises(SystemExit) as error:
        main(["mdl-frontier", "--config", str(source), "--output", str(tmp_path / "unused")])
    assert error.value.code == 2
    assert not (tmp_path / "unused").exists()
