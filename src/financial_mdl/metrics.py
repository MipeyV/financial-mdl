"""Version 1 metrics: equal-weight point errors, plus each horizon/variable."""

import numpy as np

from financial_mdl.schemas import Forecast


def point_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics: tuple[str, ...] = ("mae", "rmse"),
) -> dict[str, float]:
    """Shared arithmetic for aligned forecasts and standalone synthetic controls."""
    if not metrics or set(metrics) - {"mae", "rmse"}:
        raise ValueError("Unsupported metric")
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Metric inputs must have identical nonempty shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric inputs must be finite")
    errors = predicted - actual
    functions = {"mae": lambda e: np.mean(np.abs(e)), "rmse": lambda e: np.sqrt(np.mean(e**2))}
    result = {name: float(functions[name](errors)) for name in metrics}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Metric calculation overflowed")
    return result


def evaluate(actual: Forecast, predicted: Forecast, metrics: tuple[str, ...]) -> dict[str, float]:
    actual.coordinates.assert_equal(predicted.coordinates)
    result = point_metrics(actual.values, predicted.values, metrics)
    for h, horizon in enumerate(actual.coordinates.horizons):
        for k, target in enumerate(actual.coordinates.target_names):
            scores = point_metrics(actual.values[:, h, k], predicted.values[:, h, k], metrics)
            result.update({f"{name}/h{horizon}/{target}": value for name, value in scores.items()})
    return result
