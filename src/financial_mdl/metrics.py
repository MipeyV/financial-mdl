"""Version 1 metrics: equal-weight point errors, plus each horizon/variable."""

import numpy as np

from financial_mdl.schemas import Forecast


def evaluate(actual: Forecast, predicted: Forecast, metrics: tuple[str, ...]) -> dict[str, float]:
    actual.coordinates.assert_equal(predicted.coordinates)
    if not metrics or set(metrics) - {"mae", "rmse"}:
        raise ValueError("Unsupported metric")
    errors = predicted.values - actual.values
    functions = {"mae": lambda e: np.mean(np.abs(e)), "rmse": lambda e: np.sqrt(np.mean(e**2))}
    result = {}
    for name in metrics:
        result[name] = float(functions[name](errors))
        for h, horizon in enumerate(actual.coordinates.horizons):
            for k, target in enumerate(actual.coordinates.target_names):
                result[f"{name}/h{horizon}/{target}"] = float(functions[name](errors[:, h, k]))
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Metric calculation overflowed")
    return result
