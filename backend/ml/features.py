"""Leakage-aware feature construction.

Tier A: context only (calendar, weather, building metadata).
Tier B: + causal energy lags (all strictly before t).
Tier C: + rolling statistics over windows ending at t-1.
Tier D: + the same rolling statistics including t (deliberate positive control).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIERS = ("A", "B", "C", "D")

USAGE_CODES = {
    "Education": 1,
    "Entertainment/public assembly": 2,
    "Food sales": 3,
    "Food service": 4,
    "Healthcare": 5,
    "Laboratory": 6,
    "Lodging/residential": 7,
    "Mixed use": 8,
    "Office": 9,
    "Other": 10,
    "Parking": 11,
    "Public services": 12,
    "Retail": 13,
    "Services": 14,
    "Technology/science": 15,
    "Utility": 16,
    "Warehouse/storage": 17,
}

WINDOWS = (6, 12, 24)

CALENDAR = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_business_hour",
]
CONTEXT = [
    "airTemperature",
    "airTemperature_lag1",
    "dewTemperature",
    "windSpeed",
    "cloudCoverage",
    "precipDepth1HR",
    "sqm",
    "usage_code",
]
LAGS = ["lag_1", "lag_24", "lag_168", "lag_diff_1", "lag_diff_24"]
ROLLING_CAUSAL = [f"roll_{stat}_{w}" for w in WINDOWS for stat in ("mean", "std", "min", "max")]
ROLLING_LEAKY = [f"leaky_{stat}_{w}" for w in WINDOWS for stat in ("mean", "std", "min", "max")]

FEATURES: dict[str, list[str]] = {
    "A": CALENDAR + CONTEXT,
    "B": CALENDAR + CONTEXT + LAGS,
    "C": CALENDAR + CONTEXT + LAGS + ROLLING_CAUSAL,
    "D": CALENDAR + CONTEXT + LAGS + ROLLING_CAUSAL + ROLLING_LEAKY,
}

AVAILABILITY: dict[str, str] = {name: "t" for name in CALENDAR + CONTEXT}
AVAILABILITY.update(
    {
        "lag_1": "t-1",
        "lag_24": "t-24",
        "lag_168": "t-168",
        "lag_diff_1": "t-2..t-1",
        "lag_diff_24": "t-24..t-1",
    }
)
AVAILABILITY.update({name: "<=t-1" for name in ROLLING_CAUSAL})
AVAILABILITY.update({name: "t" for name in ROLLING_LEAKY})


class LeakageGuardError(RuntimeError):
    """Raised when a causal tier is asked to expose current-timestep energy."""


def availability_report(tier: str) -> pd.DataFrame:
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier}")
    return pd.DataFrame(
        {"feature": FEATURES[tier], "available_at": [AVAILABILITY[f] for f in FEATURES[tier]]}
    )


def _add_calendar(df: pd.DataFrame) -> None:
    ts = df["timestamp"]
    hour = ts.dt.hour
    dow = ts.dt.day_of_week
    month = ts.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (dow >= 5).astype("int8")
    df["is_business_hour"] = ((hour >= 8) & (hour <= 18) & (dow < 5)).astype("int8")


def _add_context(df: pd.DataFrame) -> None:
    df["airTemperature_lag1"] = df.groupby("building_id")["airTemperature"].shift(1)
    df["usage_code"] = (
        df["primaryspaceusage"].map(USAGE_CODES).fillna(0).astype("int16")
    )
    weather = ["airTemperature", "dewTemperature", "windSpeed", "cloudCoverage", "precipDepth1HR"]
    for col in weather:
        df[col] = df.groupby("building_id")[col].transform(
            lambda s: s.ffill().bfill()
        )


def _add_lags(df: pd.DataFrame) -> None:
    grouped = df.groupby("building_id")["y"]
    df["lag_1"] = grouped.shift(1)
    df["lag_24"] = grouped.shift(24)
    df["lag_168"] = grouped.shift(168)
    df["lag_diff_1"] = df["lag_1"] - grouped.shift(2)
    df["lag_diff_24"] = df["lag_1"] - grouped.shift(24)


def _rolling_stats(df: pd.DataFrame, leaky: bool) -> None:
    prefix = "leaky" if leaky else "roll"
    grouped = df.groupby("building_id")["y"]
    for w in WINDOWS:
        base = grouped.transform(lambda s, w=w: s if leaky else s.shift(1))
        for stat in ("mean", "std", "min", "max"):
            df[f"{prefix}_{stat}_{w}"] = base.groupby(df["building_id"]).transform(
                lambda s, w=w, stat=stat: getattr(
                    s.rolling(w, min_periods=w), stat
                )()
            )


def build_features(
    panel: pd.DataFrame,
    tier: str,
    enforce_causal: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier}")
    if enforce_causal and tier == "D":
        raise LeakageGuardError("tier D includes current-timestep energy and cannot be causal")

    df = panel.sort_values(["building_id", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    _add_calendar(df)
    _add_context(df)
    if tier in ("B", "C", "D"):
        _add_lags(df)
    if tier in ("C", "D"):
        _rolling_stats(df, leaky=False)
    if tier == "D":
        _rolling_stats(df, leaky=True)

    cols = FEATURES[tier]
    values = df[cols].replace([np.inf, -np.inf], np.nan)
    valid = values.notna().all(axis=1)
    X = values.loc[valid, cols].astype("float32")
    y = df.loc[valid, "y"].astype("float64")
    X.index = df.index[valid]
    y.index = df.index[valid]
    return X, y


def check_causality(panel: pd.DataFrame, tier: str) -> None:
    """Mutate y at the final timestamp; causal tiers must not react."""
    if tier == "D":
        raise LeakageGuardError("tier D is deliberately leaky")
    base, _ = build_features(panel, tier)
    mutated = panel.copy()
    last = mutated.index[-1]
    mutated.loc[last, "y"] = float(mutated.loc[last, "y"]) + 1000.0
    changed, _ = build_features(mutated, tier)
    if last in base.index:
        row_base = base.loc[[last]]
        row_changed = changed.loc[[last]]
        if not row_base.equals(row_changed):
            raise LeakageGuardError(f"tier {tier} reacted to the current target")
