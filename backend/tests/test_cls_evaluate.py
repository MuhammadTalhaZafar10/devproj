import numpy as np
import pandas as pd

from backend.cls.evaluate import (
    building_group_split, classification_metrics, paper_protocol, random_row_split,
)


def test_metrics_perfect_and_worst():
    y = np.array([0, 1, 2, 0, 1, 2])
    perfect = classification_metrics(y, y)
    assert perfect["accuracy"] == 1.0 and perfect["f1_weighted"] == 1.0
    wrong = classification_metrics(y, (y + 1) % 3)
    assert wrong["accuracy"] == 0.0


def test_group_split_never_shares_a_building_but_row_split_does():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 20)
    y = np.repeat(rng.integers(0, 3, 60), 20)
    tr, te = building_group_split(y, groups)
    assert not set(groups[tr]) & set(groups[te])
    tr, te = random_row_split(y)
    assert set(groups[tr]) & set(groups[te])


def test_paper_protocol_runs_on_small_data():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(150, 6)), columns=list("abcdef"))
    y = (X["a"] + 0.1 * rng.normal(size=150) > 0).astype(int) + (X["b"] > 1).astype(int)
    out = paper_protocol(X, y.to_numpy(), seeds=(0,), models=("random_forest", "lightgbm"))
    assert {r["model"] for r in out} == {"random_forest", "lightgbm"}
    assert all(0 <= r["accuracy"] <= 1 for r in out)
