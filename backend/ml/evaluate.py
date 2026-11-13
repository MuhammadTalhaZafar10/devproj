"""Metric contract shared by every runner, CSV, JSON, and the frontend."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def per_building_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y_true, "pred": y_pred, "group": groups})
    rows = []
    for group, part in frame.groupby("group", sort=True):
        y = part["y"].to_numpy()
        pred = part["pred"].to_numpy()
        rows.append(
            {
                "group": group,
                "n": len(part),
                "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
                "mae": float(np.mean(np.abs(y - pred))),
                "r2": _r2(y, pred),
            }
        )
    return pd.DataFrame(rows)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred, groups = y_true[mask], y_pred[mask], groups[mask]

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    positive = y_true > 0
    mape = (
        float(np.mean(np.abs((y_true[positive] - y_pred[positive]) / y_true[positive])) * 100)
        if positive.any()
        else float("nan")
    )

    per_group = per_building_metrics(y_true, y_pred, groups)["r2"].dropna()
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": _r2(y_true, y_pred),
        "mape": mape,
        "n": int(len(y_true)),
        "r2_median": float(per_group.median()) if len(per_group) else float("nan"),
        "r2_iqr_low": float(per_group.quantile(0.25)) if len(per_group) else float("nan"),
        "r2_iqr_high": float(per_group.quantile(0.75)) if len(per_group) else float("nan"),
    }
