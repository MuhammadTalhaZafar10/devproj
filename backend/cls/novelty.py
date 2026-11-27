"""Small, pure helpers for the novelty experiment (labels, building-level scoring, bootstrap)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def tercile_labels(train_eui: pd.Series, eui: pd.Series) -> np.ndarray:
    """Class 0/1/2 (Low/Medium/High) for ``eui`` using cutoffs fitted on ``train_eui`` only.

    The cutoffs are the 1/3 and 2/3 quantiles of the training values. A value equal to a
    cutoff goes to the lower class, which matches ``pd.cut`` with right-closed bins.
    """
    cuts = np.asarray(pd.Series(train_eui).quantile([1 / 3, 2 / 3]), dtype=float)
    return np.searchsorted(cuts, np.asarray(eui, dtype=float), side="left").astype(int)


def building_level(df_test: pd.DataFrame, proba: np.ndarray, y_building: pd.Series):
    """Average class probabilities over each building's test weeks, then argmax per building.

    ``df_test`` needs a ``building_id`` column and is row-aligned with ``proba``;
    ``y_building`` maps building_id to its true class. Returns ``(y_true, y_pred)``
    ordered by sorted building_id.
    """
    proba = np.asarray(proba, dtype=float)
    if len(df_test) != len(proba):
        raise ValueError("df_test and proba must have the same number of rows")
    table = pd.DataFrame(proba)
    table["_building"] = df_test["building_id"].to_numpy()
    mean = table.groupby("_building").mean()
    y_pred = mean.to_numpy().argmax(axis=1)
    y_true = y_building.loc[mean.index].to_numpy()
    return y_true, y_pred


def bootstrap_ci(y_true, y_pred, metric: Callable, n: int = 1000, seed: int = 0):
    """95% percentile bootstrap interval, resampling buildings (array rows) with replacement."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    size = len(y_true)
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, size, size)
        values[i] = metric(y_true[pick], y_pred[pick])
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)
