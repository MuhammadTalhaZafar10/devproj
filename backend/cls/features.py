"""Scale-free consumption-shape features.

Raymand et al. (2023) remove every feature "directly resultant of consumption"
for the performance-class target, because the label is per-area consumption.
We enforce that structurally: every feature is computed on the series divided
by its own mean, so multiplying a series by a constant cannot change it. The
test-suite asserts this invariance, so absolute consumption cannot leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PERIODS = {"night": (0, 6), "morning": (6, 12), "afternoon": (12, 18), "evening": (18, 24)}
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    out = np.zeros(a.shape[0])
    ok = denom > 1e-12
    out[ok] = (a[ok] * b[ok]).sum(axis=1) / denom[ok]
    return out


def shape_features(
    x: np.ndarray,
    hours: np.ndarray,
    dows: np.ndarray,
    temp: np.ndarray | None = None,
    months: np.ndarray | None = None,
) -> pd.DataFrame:
    """Features for N series of length T (T must be a multiple of 24).

    x      (N, T) hourly kWh, NaN allowed (callers drop rows with many NaN).
    hours  (T,) hour of day 0-23.
    dows   (T,) day of week 0=Mon..6=Sun.
    temp   (N, T) site air temperature, NaN-free, or None.
    months (T,) month 1-12 -> adds 12 seasonal ratios (building-level table only).
    Rows whose mean is not strictly positive must be removed by the caller.
    """
    x = np.asarray(x, dtype=np.float64)
    n, t = x.shape
    if t % 24:
        raise ValueError("series length must be a multiple of 24")
    nan = np.isnan(x)
    valid = (~nan).sum(axis=1)
    mean = np.nanmean(x, axis=1)
    if np.any(mean <= 0):
        raise ValueError("rows with non-positive mean must be dropped first")
    zero_frac = ((x == 0) & ~nan).sum(axis=1) / valid
    r = np.where(nan, mean[:, None], x) / mean[:, None]

    f: dict[str, np.ndarray] = {}
    for h in range(24):
        f[f"hour_{h:02d}"] = r[:, hours == h].mean(axis=1)
    for d in range(7):
        f[f"dow_{d}"] = r[:, dows == d].mean(axis=1)
    weekend = np.isin(dows, (5, 6))
    wk_mean = r[:, ~weekend].mean(axis=1)
    f["weekend_ratio"] = np.divide(
        r[:, weekend].mean(axis=1), wk_mean, out=np.ones(n), where=wk_mean > 0
    )
    for name, (lo, hi) in PERIODS.items():
        f[f"period_{name}"] = r[:, (hours >= lo) & (hours < hi)].mean(axis=1)
    f["cv"] = r.std(axis=1)
    f["skew"] = np.nan_to_num(stats.skew(r, axis=1))
    f["kurtosis"] = np.nan_to_num(stats.kurtosis(r, axis=1))
    rmax = r.max(axis=1)
    f["load_factor"] = 1.0 / rmax
    f["min_max_ratio"] = r.min(axis=1) / rmax
    for q in QUANTILES:
        f[f"q{int(q * 100):02d}"] = np.quantile(r, q, axis=1)
    profile = np.stack([f[f"hour_{h:02d}"] for h in range(24)], axis=1)
    f["peak_hour"] = profile.argmax(axis=1).astype(float)
    f["trough_hour"] = profile.argmin(axis=1).astype(float)
    days = r.reshape(n, t // 24, 24)
    dmax, dmin = days.max(axis=2), days.min(axis=2)
    dlr = np.divide(dmin, dmax, out=np.zeros_like(dmin), where=dmax > 0)
    f["daily_load_ratio_mean"] = dlr.mean(axis=1)
    f["daily_load_ratio_std"] = dlr.std(axis=1)
    f["acf_lag1"] = _corr_rows(r[:, :-1], r[:, 1:])
    f["acf_lag24"] = _corr_rows(r[:, :-24], r[:, 24:])
    f["zero_fraction"] = zero_frac
    if temp is not None:
        f["temp_corr"] = _corr_rows(r, np.asarray(temp, dtype=np.float64))
    if months is not None:
        for m in range(1, 13):
            sel = months == m
            f[f"month_{m:02d}"] = r[:, sel].mean(axis=1) if sel.any() else np.ones(n)
    return pd.DataFrame(f)


def feature_names(with_temp: bool = True, with_months: bool = False) -> list[str]:
    hours = np.tile(np.arange(24), 7)
    dows = np.repeat(np.arange(7), 24)
    x = np.ones((1, 168)) + np.arange(168) % 5 * 0.01
    temp = np.arange(168, dtype=float)[None, :] if with_temp else None
    months = np.repeat(np.arange(1, 8), 24) if with_months else None
    return list(shape_features(x, hours, dows, temp, months).columns)
