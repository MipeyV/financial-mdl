"""Compose built-in estimators with preprocessing appropriate to their input units."""

from collections.abc import Callable
from functools import partial

from financial_mdl.contracts import FeatureTransformer, ModelFactory
from financial_mdl.features import IdentityFeatures, StandardizeFeatures
from financial_mdl.models import ESTIMATOR_NAMES, PERSISTENCE_WINDOWS, make_estimator


def make_candidate(
    name: str,
    seed: int = 42,
    annualization: int = 252,
) -> tuple[ModelFactory, Callable[[], FeatureTransformer]]:
    """Candidate choices stay separate from the immutable benchmark protocol.

    Pass the protocol target's annualization when using a non-default target scale.
    Each factory returns a new instance; persistence preserves original feature units.
    """
    if name not in ESTIMATOR_NAMES:
        raise ValueError(f"Unknown estimator {name!r}; choose from {ESTIMATOR_NAMES}")
    transformer = IdentityFeatures if name in PERSISTENCE_WINDOWS else StandardizeFeatures
    return partial(make_estimator, name, seed=seed, annualization=annualization), transformer
