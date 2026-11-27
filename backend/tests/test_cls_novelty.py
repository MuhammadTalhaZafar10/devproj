import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from backend.cls.novelty import bootstrap_ci, building_level, tercile_labels


def test_tercile_cutoffs_use_only_train():
    train = pd.Series(np.arange(1, 10, dtype=float))        # cutoffs 3.67, 6.33
    test = pd.Series([1000.0, 0.0, 5.0, 3.0, 7.0])           # extreme values must not move cutoffs
    labels = tercile_labels(train, test)
    assert labels.tolist() == [2, 0, 1, 0, 2]
    assert labels.dtype.kind == "i"
    # changing the test values changes nothing about the cutoffs
    assert tercile_labels(train, pd.Series([5.0])).tolist() == [1]


def test_building_level_averages_probabilities():
    df = pd.DataFrame({"building_id": ["a", "a", "a", "b", "b"]})
    # building a: row argmaxes are 0, 0, 1 but the mean probability favours class 1
    proba = np.array([
        [0.40, 0.35, 0.25],
        [0.40, 0.35, 0.25],
        [0.00, 0.95, 0.05],
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
    ])
    y_b = pd.Series({"a": 1, "b": 2})
    y_true, y_pred = building_level(df, proba, y_b)
    assert y_true.tolist() == [1, 2]
    assert y_pred.tolist() == [1, 2]
    assert proba.argmax(1).tolist()[:3] == [0, 0, 1]   # a plain row vote would pick 0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, 200)
    y_pred = np.where(rng.random(200) < 0.8, y_true, (y_true + 1) % 3)
    point = accuracy_score(y_true, y_pred)
    low, high = bootstrap_ci(y_true, y_pred, accuracy_score, n=500, seed=0)
    assert low <= high
    assert low <= point <= high
    assert bootstrap_ci(y_true, y_pred, accuracy_score, n=500, seed=0) == (low, high)
