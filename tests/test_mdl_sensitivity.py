import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financial_mdl.cli import main
from financial_mdl.experiments import controlled_regression
from financial_mdl.mdl import ExhaustiveMDLSelector
from financial_mdl.mdl_sensitivity import SensitivityConfig, run_mdl_sensitivity


def test_committed_sensitivity_plan_matches_declared_defaults():
    assert (
        SensitivityConfig.load(Path("experiments/mdl-sensitivity-v1.json")) == SensitivityConfig()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_sizes": ()},
        {"train_sizes": (10, 10)},
        {"train_sizes": (20, 10)},
        {"train_sizes": (True,)},
        {"precisions": (0.0,)},
        {"precisions": (float("nan"),)},
        {"precisions": (0.1, 0.1)},
        {"seeds": (0, 0)},
        {"seeds": (-1,)},
        {"n_heldout": 1},
        {"version": "unknown"},
    ],
)
def test_invalid_sensitivity_plans_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SensitivityConfig(**kwargs)


def test_grid_has_nested_training_fixed_holdout_and_reproducible_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr("financial_mdl.mdl_sensitivity.save_figure", lambda *args: None)
    config = SensitivityConfig(
        train_sizes=(30, 70), precisions=(0.0001, 0.1), seeds=(0,), n_heldout=20
    )
    seen = []
    original_fit = ExhaustiveMDLSelector.fit

    def record_training(self, X, y):
        # The complete plan must exist before the first selector sees data.
        assert (tmp_path / "one/experiment.json").exists()
        seen.append((X.copy(), y.copy()))
        return original_fit(self, X, y)

    monkeypatch.setattr(ExhaustiveMDLSelector, "fit", record_training)
    first = run_mdl_sensitivity(tmp_path / "one", config)
    for block in range(0, len(seen), 4):
        small, repeated, large, _ = seen[block : block + 4]
        pd.testing.assert_frame_equal(small[0], repeated[0])
        pd.testing.assert_frame_equal(small[0], large[0].iloc[:30])
        np.testing.assert_array_equal(small[1], large[1][:30])
        assert len(large[0]) == 70  # held-out rows never enter selection
    second = run_mdl_sensitivity(tmp_path / "two", config)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 24
    for _, scenario in first.groupby(["scenario", "seed"]):
        assert scenario.heldout_fingerprint.nunique() == 1
        assert scenario.training_fingerprint.nunique() == 2
        # Full Ridge does not depend on the coding precision.
        assert (scenario[scenario.method == "full"].groupby("n_train").rmse.nunique() == 1).all()
    for _, pair in first.groupby(["scenario", "seed", "n_train", "target_precision"]):
        assert pair.training_fingerprint.nunique() == 1
        assert pair.heldout_fingerprint.nunique() == 1
        codes = pair.set_index("method").total_bits
        assert codes["mdl"] <= codes["full"]
    summary = pd.read_csv(tmp_path / "one/summary.csv")
    assert len(summary) == 24
    assert summary.n_seeds.eq(1).all()
    with pytest.raises(FileExistsError):
        run_mdl_sensitivity(tmp_path / "one", config)


def test_heldout_perturbation_cannot_change_any_selection_or_training_code(tmp_path, monkeypatch):
    monkeypatch.setattr("financial_mdl.mdl_sensitivity.save_figure", lambda *args: None)
    config = SensitivityConfig(train_sizes=(25, 50), precisions=(0.01,), seeds=(0,), n_heldout=20)
    first = run_mdl_sensitivity(tmp_path / "original", config)

    def perturb(scenario, seed, n):
        X, y = controlled_regression(scenario, seed, n)
        X.iloc[50:] *= 100
        y[50:] += 10
        return X, y

    monkeypatch.setattr("financial_mdl.mdl_sensitivity.controlled_regression", perturb)
    second = run_mdl_sensitivity(tmp_path / "changed", config)
    for source in (tmp_path / "original").glob("*-selection.json"):
        assert json.loads(source.read_text()) == json.loads(
            (tmp_path / "changed" / source.name).read_text()
        )
    assert first.training_fingerprint.equals(second.training_fingerprint)
    assert not first.heldout_fingerprint.equals(second.heldout_fingerprint)
    assert not first.rmse.equals(second.rmse)


def test_sensitivity_cli_exports_plan_all_cells_and_figures(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            asdict(
                SensitivityConfig(train_sizes=(20,), precisions=(0.01,), seeds=(0,), n_heldout=10)
            )
        )
    )
    output = tmp_path / "run"
    main(["mdl-sensitivity", "--config", str(config), "--output", str(output)])
    plan = json.loads((output / "experiment.json").read_text())
    assert plan["train_sizes"] == [20]
    assert plan["reference_code"]["ridge_alpha"] == 1
    assert len(plan["experiment_fingerprint"]) == 64
    assert len(pd.read_csv(output / "results.csv")) == 6
    assert len(pd.read_csv(output / "selection_frequency.csv")) == 18
    assert (output / "selected_features.html").exists()
    assert (output / "heldout_rmse_ratio.json").exists()
    assert "noise_only" in capsys.readouterr().out


@pytest.mark.parametrize("values", [{}, {"seeds": "0"}, []])
def test_cli_rejects_incomplete_plan_before_creating_output(tmp_path, values):
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(values))
    with pytest.raises(SystemExit) as error:
        main(["mdl-sensitivity", "--config", str(source), "--output", str(tmp_path / "unused")])
    assert error.value.code == 2
    assert not (tmp_path / "unused").exists()
