"""Deterministic Plotly artifacts, with no application or dashboard server."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def save_figure(figure: go.Figure, directory: Path, name: str) -> None:
    figure.write_json(directory / f"{name}.json", pretty=True)
    figure.write_html(directory / f"{name}.html", include_plotlyjs=True, div_id=name)


def forecast_figures(predictions: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    plot = predictions.copy()
    plot["series"] = "h" + plot.horizon.astype(str) + "/" + plot.target
    plot["residual"] = plot.predicted - plot.actual
    melted = plot.melt(
        id_vars=["origin", "series"],
        value_vars=["actual", "predicted"],
        var_name="kind",
        value_name="value",
    )
    save_figure(
        px.line(
            melted,
            x="origin",
            y="value",
            color="kind",
            facet_row="series",
            title="Actual versus predicted",
        ),
        directory,
        "forecasts",
    )
    save_figure(
        px.histogram(
            plot,
            x="residual",
            color="series",
            barmode="overlay",
            title="Residuals: predicted minus actual",
        ),
        directory,
        "residuals",
    )
    errors = (
        plot.assign(squared_error=plot.residual**2).groupby("series").squared_error.mean() ** 0.5
    )
    save_figure(
        px.bar(
            x=errors.index,
            y=errors.values,
            labels={"x": "Horizon/target", "y": "RMSE"},
            title="Per-horizon performance",
        ),
        directory,
        "per_horizon",
    )


def training_figures(market: pd.DataFrame, X_train: pd.DataFrame, directory: Path) -> None:
    """Exploration artifacts deliberately exclude validation/test feature distributions."""
    save_figure(
        px.line(market, x=market.index, y="close", title="Training-period close"),
        directory,
        "prices",
    )
    returns = market.close.pipe(lambda c: c.pct_change(fill_method=None))
    save_figure(
        px.line(x=returns.index, y=returns, title="Training-period simple returns"),
        directory,
        "returns",
    )
    vol = X_train.filter(like="realized_volatility")
    if not vol.empty:
        save_figure(
            px.line(vol, title="Trailing realized volatility (training)"),
            directory,
            "rolling_volatility",
        )
    save_figure(
        px.imshow(
            X_train.corr(),
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title="Training feature correlations",
        ),
        directory,
        "correlations",
    )


def mdl_figures(selection: dict, directory: Path) -> None:
    """Training reference-code diagnostics, never per-subset validation scores."""
    directory.mkdir(parents=True, exist_ok=True)
    decomposition = []
    for reference in ("selected", "full", "constant"):
        for component in ("subset_bits", "model_bits", "target_bits"):
            decomposition.append(
                {
                    "reference": reference,
                    "component": component,
                    "bits": selection[reference][component],
                }
            )
    save_figure(
        px.bar(
            pd.DataFrame(decomposition),
            x="reference",
            y="bits",
            color="component",
            title="MDL training reference code: decomposition",
        ),
        directory,
        "mdl_decomposition",
    )
    candidates = pd.DataFrame(selection["candidates"])
    candidates["representation_bits"] = candidates.subset_bits + candidates.model_bits
    candidates["features"] = candidates.features.map(lambda names: ", ".join(names) or "constant")
    save_figure(
        px.scatter(
            candidates,
            x="representation_bits",
            y="training_rmse",
            color="n_features",
            hover_data=["features", "total_bits"],
            title="MDL training candidates: representation cost versus error",
        ),
        directory,
        "mdl_complexity_error",
    )
