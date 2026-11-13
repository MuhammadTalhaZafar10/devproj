"""Pooled and per-building metrics must be arithmetically correct."""
from __future__ import annotations

import numpy as np
import pytest

from backend.ml.evaluate import per_building_metrics, regression_metrics


def test_regression_metrics_matches_hand_computed_rmse():
    m = regression_metrics(
        np.array([1.0, 2.0]), np.array([1.0, 4.0]), np.array(["a", "a"])
    )
    assert m["rmse"] == pytest.approx(np.sqrt(2.0))
    assert m["mae"] == pytest.approx(1.0)
    assert m["mape"] == pytest.approx(50.0)
    assert m["n"] == 2


def test_pooled_r2_is_not_mean_of_group_r2():
    y = np.array([1.0, 1.0, 100.0, 100.0])
    pred = np.array([1.0, 1.0, 100.0, 200.0])
    groups = np.array(["small", "small", "big", "big"])
    pooled = regression_metrics(y, pred, groups)["r2"]
    mean_of_groups = per_building_metrics(y, pred, groups)["r2"].mean()
    assert pooled != pytest.approx(mean_of_groups)


def test_mape_ignores_zero_targets():
    m = regression_metrics(np.array([0.0, 2.0]), np.array([0.0, 3.0]), np.array(["a", "a"]))
    assert m["mape"] == pytest.approx(50.0)
