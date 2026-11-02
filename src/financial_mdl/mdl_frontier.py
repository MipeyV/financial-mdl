"""Inspect every synthetic subset's training code and held-out error, without retuning."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from financial_mdl.benchmark import _provenance
from financial_mdl.experiments import controlled_regression
from financial_mdl.mdl import ExhaustiveMDLSelector, MDLCodingSpec
from financial_mdl.mdl_experiment import SYNTHETIC_SCENARIOS
from financial_mdl.mdl_sensitivity import SensitivityConfig, _fingerprint
from financial_mdl.metrics import point_metrics
from financial_mdl.reporting import save_figure

SCENARIO_TITLES = {
    "signal": "Signaux forts et redondance",
    "noise_only": "Bruit seul",
    "weak_signal": "Signal faible",
}


@dataclass(frozen=True)
class FrontierConfig:
    n_train: int = 600
    n_heldout: int = 300
    target_precision: float = 0.0001
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    version: str = "synthetic_mdl_frontier_v1"

    def __post_init__(self) -> None:
        SensitivityConfig(
            train_sizes=(self.n_train,),
            precisions=(self.target_precision,),
            seeds=self.seeds,
            n_heldout=self.n_heldout,
        )
        if self.version != "synthetic_mdl_frontier_v1":
            raise ValueError("Unsupported frontier experiment version")

    @classmethod
    def load(cls, path: Path) -> FrontierConfig:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {"n_train", "n_heldout", "target_precision", "seeds", "version"}
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f"Frontier configuration requires exactly {sorted(expected)}")
        if not isinstance(values["seeds"], list):
            raise ValueError("seeds must be a JSON array")
        values["seeds"] = tuple(values["seeds"])
        return cls(**values)


def evaluate_subsets(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_heldout: pd.DataFrame,
    y_heldout: np.ndarray,
    spec: MDLCodingSpec,
) -> tuple[pd.DataFrame, dict]:
    """Replay the fitted binary64 parameters of all candidates on common held-out rows.

    Selection and parameter fitting happen before held-out prediction. No model is
    selected using the diagnostic held-out errors. The scaler derives only from X_train.
    """
    if tuple(X_heldout.columns) != tuple(X_train.columns):
        raise ValueError("Held-out columns/order must match the training feature universe")
    if (
        len(X_heldout) < 2
        or np.asarray(y_heldout).shape != (len(X_heldout),)
        or not np.isfinite(X_heldout.to_numpy(dtype=float)).all()
        or not np.isfinite(y_heldout).all()
    ):
        raise ValueError("Require at least two aligned finite held-out observations")
    diagnostic = ExhaustiveMDLSelector(spec).fit(X_train, y_train).get_diagnostics()
    standardized = StandardScaler().fit(X_train).transform(X_heldout)
    positions = {name: i for i, name in enumerate(X_train.columns)}
    rows = []
    for candidate in diagnostic["candidates"]:
        names = candidate["features"]
        parameters = struct.unpack(f">{len(names) + 2}d", bytes.fromhex(candidate["parameter_hex"]))
        prediction = parameters[0] + standardized[:, [positions[n] for n in names]] @ np.array(
            parameters[1:-1]
        )
        scores = point_metrics(y_heldout, prediction)
        rows.append(
            {
                "subset_key": json.dumps(names, separators=(",", ":")),
                "features": ", ".join(names) if names else "(constant)",
                **{
                    key: candidate[key]
                    for key in (
                        "n_features",
                        "subset_bits",
                        "model_bits",
                        "target_bits",
                        "total_bits",
                        "training_rmse",
                    )
                },
                "heldout_mae": scores["mae"],
                "heldout_rmse": scores["rmse"],
                "selected_by_mdl": names == diagnostic["selected_features"],
                "is_full": len(names) == len(X_train.columns),
                "is_constant": not names,
            }
        )
    return pd.DataFrame(rows), diagnostic


def _plot_limits(summary: pd.DataFrame, zoom: bool) -> tuple[list | None, list | None]:
    if not zoom:
        return None, None
    full_rmse = summary.loc[summary.is_full, "mean_heldout_rmse"].iloc[0]
    upper = 1.10 * full_rmse
    visible = summary.loc[summary.mean_heldout_rmse <= upper]
    minimum, maximum = visible.mean_heldout_rmse.min(), visible.mean_heldout_rmse.max()
    error_padding = max(float(maximum - minimum) * 0.20, float(full_rmse) * 0.0005)
    left, right = visible.mean_total_bits.min(), visible.mean_total_bits.max()
    padding = max(1.0, float(right - left) * 0.08)
    return [float(left - padding), float(right + padding)], [
        max(0.0, float(minimum - error_padding)),
        float(maximum + error_padding),
    ]


def _plot_frontier(summary: pd.DataFrame, output: Path, scenario: str, config: FrontierConfig):
    import plotly.graph_objects as go
    from matplotlib import colormaps, rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    title = SCENARIO_TITLES[scenario]
    subtitle = (
        f"n = {config.n_train} ; précision = {config.target_precision:g} ; "
        f"{len(config.seeds)} graines ; 64 sous-ensembles"
    )
    highlights = (
        ("is_constant", "Constante", "diamond-open", "D", 240, "#202020"),
        ("is_full", "Ridge complet", "square-open", "s", 180, "#202020"),
        ("selection_count", "Choisi par MDL", "star", "*", 180, "#d62757"),
    )
    for zoom in (False, True):
        suffix = "zoom" if zoom else "all"
        name = f"{scenario}-{suffix}"
        zoom_title = " — zoom prédictif" if zoom else ""
        x_range, y_range = _plot_limits(summary, zoom)
        figure = go.Figure(
            go.Scatter(
                x=summary.mean_total_bits.tolist(),
                y=summary.mean_heldout_rmse.tolist(),
                mode="markers",
                name="Sous-ensembles",
                marker={
                    "size": 9,
                    "color": summary.n_features.tolist(),
                    "colorscale": "Viridis",
                    "cmin": 0,
                    "cmax": 6,
                    "colorbar": {"title": "Variables", "dtick": 1},
                    "line": {"color": "white", "width": 0.6},
                },
                customdata=summary[["features", "std_heldout_rmse", "selection_count"]].values,
                hovertemplate=(
                    "%{customdata[0]}<br>MDL moyen : %{x:.1f} bits<br>RMSE moyenne : %{y:.6f}"
                    "<br>Écart-type : %{customdata[1]:.6f}"
                    "<br>Sélections MDL : %{customdata[2]}<extra></extra>"
                ),
            )
        )
        for column, label, symbol, _, _, color in highlights:
            subset = summary.loc[summary[column].astype(bool)]
            if zoom and (subset.mean_heldout_rmse > y_range[1]).all():
                label += " (hors zoom)"
            figure.add_trace(
                go.Scatter(
                    x=subset.mean_total_bits.tolist(),
                    y=subset.mean_heldout_rmse.tolist(),
                    mode="markers",
                    name=label,
                    marker={"symbol": symbol, "size": 19, "color": color, "line": {"width": 2}},
                    text=subset.features.tolist(),
                    hovertemplate="%{text}<extra>" + label + "</extra>",
                )
            )
        figure.update_layout(
            title={"text": title + zoom_title + "<br><sup>" + subtitle + "</sup>"},
            template="plotly_white",
            width=1100,
            height=680,
            xaxis={"title": "Longueur MDL moyenne sur l’entraînement (bits)", "range": x_range},
            yaxis={"title": "RMSE moyenne hors entraînement", "range": y_range},
            legend={"orientation": "h", "y": -0.18},
            margin={"b": 125},
        )
        save_figure(figure, output, name)
        with rc_context({"font.size": 11, "svg.fonttype": "none"}):
            static = Figure(figsize=(11, 6.8), layout="constrained")
            FigureCanvasAgg(static)
            axes = static.subplots()
            points = axes.scatter(
                summary.mean_total_bits,
                summary.mean_heldout_rmse,
                c=summary.n_features,
                cmap=colormaps["viridis"],
                norm=Normalize(0, 6),
                s=50,
                edgecolors="white",
                linewidths=0.5,
            )
            for column, label, _, marker, size, color in highlights:
                subset = summary.loc[summary[column].astype(bool)]
                if zoom and (subset.mean_heldout_rmse > y_range[1]).all():
                    label += " (hors zoom)"
                axes.scatter(
                    subset.mean_total_bits,
                    subset.mean_heldout_rmse,
                    marker=marker,
                    s=size,
                    facecolors=color if marker == "*" else "none",
                    edgecolors=color,
                    linewidths=1.8,
                    label=label,
                    zorder=5,
                )
            if zoom:
                axes.set_xlim(x_range)
                axes.set_ylim(y_range)
            axes.set_title(title + zoom_title + "\n" + subtitle, pad=15)
            axes.set_xlabel("Longueur MDL moyenne sur l’entraînement (bits)")
            axes.set_ylabel("RMSE moyenne hors entraînement")
            axes.grid(alpha=0.18)
            axes.set_axisbelow(True)
            static.legend(loc="outside lower center", ncols=3, frameon=False, fontsize=10)
            static.colorbar(points, ax=axes, ticks=range(7), label="Variables retenues")
            static.savefig(output / f"{name}.png", dpi=200)
            static.savefig(output / f"{name}.svg")


def run_mdl_frontier(output: Path, config: FrontierConfig) -> pd.DataFrame:
    """Report the entire declared synthetic universe; keep market test data untouched."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    specification = {
        **asdict(config),
        "scenarios": SYNTHETIC_SCENARIOS,
        "generator": "controlled_regression_v1",
        "coding_spec": asdict(MDLCodingSpec(target_precision=config.target_precision)),
        "split": "first n_train independent rows; last n_heldout rows",
        "selection": "per-seed training MDL minimum; held-out errors are diagnostic only",
        "plot": "one point per named subset: mean training bits and mean held-out RMSE",
        "dispersion": "standard deviation across seeds, not a confidence interval",
        "zoom": (
            "points with RMSE <= 1.10 * full Ridge mean; axes enclose these points with "
            "20% error-span padding (at least 0.0005 * full RMSE) and 8% bit-span padding "
            "(at least 1 bit); always export the complete view too"
        ),
    }
    fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    (output / "experiment.json").write_text(
        json.dumps(
            {**specification, "experiment_fingerprint": fingerprint, **_provenance()}, indent=2
        ),
        encoding="utf-8",
    )
    results = []
    spec = MDLCodingSpec(target_precision=config.target_precision)
    for scenario in SYNTHETIC_SCENARIOS:
        for seed in config.seeds:
            X, y = controlled_regression(scenario, seed, config.n_train + config.n_heldout)
            train_X, heldout_X = X.iloc[: config.n_train], X.iloc[config.n_train :]
            train_y, heldout_y = y[: config.n_train], y[config.n_train :]
            rows, diagnostic = evaluate_subsets(train_X, train_y, heldout_X, heldout_y, spec)
            rows = rows.assign(
                scenario=scenario,
                seed=seed,
                training_fingerprint=_fingerprint(train_X, train_y),
                heldout_fingerprint=_fingerprint(heldout_X, heldout_y),
            )
            results.append(rows)
            (output / f"{scenario}-s{seed}-selection.json").write_text(
                json.dumps(diagnostic, indent=2, allow_nan=False), encoding="utf-8"
            )
    results = pd.concat(results, ignore_index=True)
    results.to_csv(output / "subsets.csv", index=False)
    summary = results.groupby(["scenario", "subset_key"], as_index=False).agg(
        features=("features", "first"),
        n_features=("n_features", "first"),
        n_seeds=("seed", "nunique"),
        mean_total_bits=("total_bits", "mean"),
        mean_training_rmse=("training_rmse", "mean"),
        mean_heldout_rmse=("heldout_rmse", "mean"),
        std_heldout_rmse=("heldout_rmse", "std"),
        selection_count=("selected_by_mdl", "sum"),
        is_full=("is_full", "first"),
        is_constant=("is_constant", "first"),
    )
    summary.to_csv(output / "summary.csv", index=False)
    for scenario in SYNTHETIC_SCENARIOS:
        _plot_frontier(summary.loc[summary.scenario == scenario], output, scenario, config)
    sections = []
    for scenario, title in SCENARIO_TITLES.items():
        figures = "".join(
            f'<a href="{scenario}-{view}.html"><img src="{scenario}-{view}.svg" '
            f'alt="{escape(title)} — {label}"></a><p>{label} : '
            f'<a href="{scenario}-{view}.html">Interactif</a> · '
            f'<a href="{scenario}-{view}.png">PNG</a> · '
            f'<a href="{scenario}-{view}.svg">SVG</a></p>'
            for view, label in (("all", "Vue complète"), ("zoom", "Zoom prédictif"))
        )
        sections.append(f"<section><h2>{escape(title)}</h2>{figures}</section>")
    (output / "index.html").write_text(
        '<!doctype html><html lang="fr"><meta charset="utf-8">'
        "<title>MDL et erreur prédictive</title><style>"
        "body{font:18px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;"
        "color:#172033;background:#fff}img{width:100%;height:auto}section{margin:60px 0}"
        "a{color:#2455a4}p{line-height:1.6}</style><h1>MDL et erreur prédictive</h1>"
        f"<p>{config.n_train} observations d’entraînement, {config.n_heldout} d’évaluation, "
        f"précision {config.target_precision:g}, {len(config.seeds)} graines. "
        "Chaque point représente un sous-ensemble, moyenné sur les mêmes graines.</p>"
        "<p>La sélection MDL utilise uniquement l’entraînement. Les erreurs hors entraînement "
        "décrivent ses compromis ; elles ne servent pas à sélectionner un modèle. "
        "Les marqueurs MDL identifient les sous-ensembles choisis sur au moins une graine. "
        "Le zoom cadre les sous-ensembles dont la RMSE moyenne ne dépasse pas "
        "110 % de celle de Ridge complet. Les petits écarts sont agrandis ; "
        "la vue complète conserve leur contexte.</p>"
        + "".join(sections)
        + '<p><a href="subsets.csv">Résultats par graine</a> · '
        '<a href="summary.csv">Moyennes et écarts-types</a></p></html>',
        encoding="utf-8",
    )
    return results
