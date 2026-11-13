"""Validation protocols S1-S4.

S1 random k-fold        -> what the field currently reports
S2 blocked chronological-> skill on future hours of seen buildings
S3 leave-one-building-out -> skill on unseen buildings, same sites
S4 leave-one-site-out   -> skill on unseen sites/weather regimes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


def random_folds(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    positions = np.arange(len(df))
    return [(train, test) for train, test in kf.split(positions)]


def blocked_temporal(
    df: pd.DataFrame, cutoff: str = "2017-07-01"
) -> list[tuple[np.ndarray, np.ndarray]]:
    timestamps = pd.to_datetime(df["timestamp"])
    cutoff_ts = pd.Timestamp(cutoff)
    train = np.flatnonzero((timestamps < cutoff_ts).to_numpy())
    test = np.flatnonzero((timestamps >= cutoff_ts).to_numpy())
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"cutoff {cutoff} leaves an empty split")
    return [(train, test)]


def leave_one_group_out(
    df: pd.DataFrame, group_col: str, n_folds: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = df[group_col].to_numpy()
    unique = np.unique(groups)
    n_splits = int(min(n_folds, len(unique)))
    if n_splits < 2:
        raise ValueError(f"{group_col} needs at least two distinct groups")
    gkf = GroupKFold(n_splits=n_splits)
    positions = np.arange(len(df))
    return [(train, test) for train, test in gkf.split(positions, groups=groups)]


def protocol_splits(
    df: pd.DataFrame, protocol: str, n_folds: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    if protocol == "S1":
        return random_folds(df, n_splits=n_folds, seed=seed)
    if protocol == "S2":
        return blocked_temporal(df)
    if protocol == "S3":
        return leave_one_group_out(df, "building_id", n_folds=n_folds, seed=seed)
    if protocol == "S4":
        return leave_one_group_out(df, "site_id", n_folds=n_folds, seed=seed)
    raise ValueError(f"unknown protocol: {protocol}")
