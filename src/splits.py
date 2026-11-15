"""Generate reproducible temporal, spatial, and random validation folds."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from src.config import N_SPLITS, SEED

Fold = dict[str, Any]


def temporal_fold_indices(
    dates: pd.Series,
    n_splits: int = N_SPLITS,
    gap: int = 1,
) -> list[Fold]:
    """Split a complete daily calendar and map each fold back to data rows."""
    observed_dates = pd.to_datetime(pd.Series(dates), errors="raise").reset_index(drop=True)
    if observed_dates.empty:
        raise ValueError("Cannot create temporal folds from an empty date series.")

    calendar = pd.date_range(
        start=observed_dates.min(),
        end=observed_dates.max(),
        freq="D",
    )
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    folds = []
    for fold_number, (train_days, validation_days) in enumerate(
        splitter.split(calendar),
        start=1,
    ):
        train_indices = np.flatnonzero(
            observed_dates.isin(calendar[train_days]).to_numpy()
        )
        validation_indices = np.flatnonzero(
            observed_dates.isin(calendar[validation_days]).to_numpy()
        )
        folds.append({
            "fold": fold_number,
            "train_indices": train_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
        })
    return folds


def spatial_fold_indices(
    groups: pd.Series,
    n_splits: int = N_SPLITS,
) -> list[Fold]:
    """Create GroupKFold indices that keep each station in one fold."""
    group_values = pd.Series(groups).reset_index(drop=True)
    if group_values.nunique() < n_splits:
        raise ValueError("There must be at least n_splits distinct station groups.")

    splitter = GroupKFold(n_splits=n_splits)
    placeholder_features = np.zeros((len(group_values), 1))
    folds = []
    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(placeholder_features, groups=group_values),
        start=1,
    ):
        folds.append({
            "fold": fold_number,
            "train_indices": train_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
        })
    return folds


def random_fold_indices(
    n_rows: int,
    n_splits: int = N_SPLITS,
    random_state: int = SEED,
) -> list[Fold]:
    """Create shuffled K-Fold indices with a fixed seed."""
    if n_rows < n_splits:
        raise ValueError("There must be at least n_splits rows.")

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    row_indices = np.arange(n_rows)
    folds = []
    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(row_indices),
        start=1,
    ):
        folds.append({
            "fold": fold_number,
            "train_indices": train_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
        })
    return folds


def save_folds_json(folds: list[Fold], path: Path) -> None:
    """Write fold row indices to a readable JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(folds, output_file, indent=2)
        output_file.write("\n")
