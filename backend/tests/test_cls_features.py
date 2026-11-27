import numpy as np
import pytest

from backend.cls.features import feature_names, shape_features

HOURS = np.tile(np.arange(24), 7)
DOWS = np.repeat(np.arange(7), 24)


def _series(n=5, seed=0):
    rng = np.random.default_rng(seed)
    base = 50 + 20 * np.sin(2 * np.pi * HOURS / 24)
    return np.abs(base + rng.normal(0, 4, (n, 168))) + 1.0


def test_features_are_scale_free_so_absolute_consumption_cannot_leak():
    x = _series()
    temp = np.random.default_rng(1).normal(15, 5, x.shape)
    a = shape_features(x, HOURS, DOWS, temp)
    b = shape_features(x * 1234.5, HOURS, DOWS, temp)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-9)


def test_feature_count_exceeds_course_minimum():
    assert len(feature_names()) >= 35


def test_nan_hours_do_not_break_features():
    x = _series()
    x[0, :10] = np.nan
    out = shape_features(x, HOURS, DOWS)
    assert not out.isna().any().any()


def test_non_positive_mean_is_rejected():
    x = _series()
    x[2] = 0.0
    with pytest.raises(ValueError):
        shape_features(x, HOURS, DOWS)


def test_hour_profile_peaks_where_signal_peaks():
    x = np.ones((1, 168)) * 10
    x[0, HOURS == 14] = 100
    out = shape_features(x, HOURS, DOWS)
    assert out["peak_hour"].iloc[0] == 14
