"""Deterministic partitions with target-window purging and optional session gaps."""

import numpy as np
import pandas as pd

from financial_mdl.protocol import TemporalSplit
from financial_mdl.schemas import Targets


def chronological_partitions(
    targets: Targets,
    sessions: pd.DatetimeIndex,
    split: TemporalSplit,
) -> dict[str, np.ndarray]:
    origins = targets.observations.coordinates.origins
    label_ends = targets.observations.coordinates.target_timestamps.max(axis=1)
    boundaries = (
        ("train", pd.Timestamp(split.train_start), pd.Timestamp(split.train_end)),
        (
            "validation",
            pd.Timestamp(split.train_end),
            pd.Timestamp(split.validation_end),
        ),
        (
            "test",
            pd.Timestamp(split.validation_end),
            pd.Timestamp(split.test_end),
        ),
    )
    partitions = {}
    for name, start, end in boundaries:
        cutoff_sessions = sessions[sessions <= end]
        gap = split.gap_sessions if name != "test" else 0
        if len(cutoff_sessions) <= gap:
            raise ValueError(f"Insufficient sessions for {name} gap")
        cutoff = cutoff_sessions[-gap - 1]
        after_start = origins >= start if name == "train" else origins > start
        positions = np.flatnonzero(
            after_start & (origins <= end) & (label_ends <= cutoff.to_datetime64())
        )
        if len(positions) == 0:
            raise ValueError(f"Empty {name} partition after eligibility, purging and gap")
        partitions[name] = positions
    return partitions
