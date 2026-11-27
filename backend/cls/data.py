"""Cohort, labels and the two BDG2 classification tables.

Table A (paper protocol): one row per building, features from 2016-2017.
Table B (course scale):   one row per building-week, >=100k rows, 50+ features.

Both share one label: the building's performance class, i.e. a tercile of
mean electricity per floor area (kWh/m2/yr). Raymand et al. (2023) say only
"low, medium and high" and their repository is an empty placeholder, so
terciles are our documented assumption.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.cls.features import shape_features

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CLASSES = ["Low", "Medium", "High"]
WEEK = 168
MIN_COVERAGE = 0.80          # share of non-null hours a meter needs
MAX_WINDOW_MISSING = 0.10    # share of NaN hours tolerated inside one week
ID_COLUMNS = ["building_id", "site_id", "primaryspaceusage", "week_start"]


def load_raw(raw_dir: Path = RAW_DIR):
    elec = pd.read_csv(Path(raw_dir) / "electricity.csv", parse_dates=["timestamp"])
    elec = elec.sort_values("timestamp").reset_index(drop=True)
    meta = pd.read_csv(Path(raw_dir) / "metadata.csv")
    weather = pd.read_csv(
        Path(raw_dir) / "weather.csv", parse_dates=["timestamp"],
        usecols=["timestamp", "site_id", "airTemperature"],
    )
    return elec, meta, weather


def select_cohort(elec: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Buildings with usable electricity, positive area and non-zero use."""
    cols = [c for c in elec.columns if c != "timestamp"]
    values = elec[cols].to_numpy(dtype=np.float64)
    values[values < 0] = np.nan
    coverage = np.mean(~np.isnan(values), axis=0)
    mean_kwh = np.nanmean(np.where(np.isnan(values), np.nan, values), axis=0)
    table = pd.DataFrame({"building_id": cols, "coverage": coverage, "mean_kwh": mean_kwh})
    table = table.merge(
        meta[["building_id", "site_id", "primaryspaceusage", "sqm"]], on="building_id", how="inner"
    )
    keep = (
        (table["coverage"] >= MIN_COVERAGE)
        & (table["mean_kwh"] > 0)
        & (table["sqm"] > 0)
    )
    table = table[keep].reset_index(drop=True)
    table["eui_kwh_m2_yr"] = table["mean_kwh"] * 8760.0 / table["sqm"]
    return table


def assign_classes(cohort: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """Tercile labels on kWh/m2/yr. Returns the cohort and the two cutoffs."""
    cuts = cohort["eui_kwh_m2_yr"].quantile([1 / 3, 2 / 3]).tolist()
    out = cohort.copy()
    out["performance_class"] = pd.cut(
        out["eui_kwh_m2_yr"], [-np.inf, cuts[0], cuts[1], np.inf], labels=CLASSES
    ).astype(str)
    return out, cuts


def _site_temperature(elec: pd.DataFrame, weather: pd.DataFrame) -> dict[str, np.ndarray]:
    stamps = pd.DatetimeIndex(elec["timestamp"])
    out: dict[str, np.ndarray] = {}
    for site, part in weather.groupby("site_id"):
        series = part.drop_duplicates("timestamp").set_index("timestamp")["airTemperature"]
        series = series.reindex(stamps).interpolate(limit_direction="both")
        out[site] = series.fillna(series.mean()).to_numpy(dtype=np.float64)
    return out


def build_building_table(elec, meta, weather) -> pd.DataFrame:
    """Table A: one feature row per building (paper protocol)."""
    cohort, _ = assign_classes(select_cohort(elec, meta))
    stamps = pd.DatetimeIndex(elec["timestamp"])
    usable = (len(stamps) // 24) * 24
    hours = stamps.hour.to_numpy()[:usable]
    dows = stamps.dayofweek.to_numpy()[:usable]
    months = stamps.month.to_numpy()[:usable]
    temps = _site_temperature(elec, weather)
    x = elec[cohort["building_id"]].to_numpy(dtype=np.float64)[:usable].T
    x[x < 0] = np.nan
    temp = np.stack([temps[s][:usable] for s in cohort["site_id"]])
    feats = shape_features(x, hours, dows, temp, months)
    ids = cohort[["building_id", "site_id", "primaryspaceusage", "eui_kwh_m2_yr", "performance_class"]]
    return pd.concat([ids.reset_index(drop=True), feats], axis=1)


def build_window_table(elec, meta, weather, n_rows: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Table B: one row per building-week, sampled to ``n_rows``."""
    cohort, _ = assign_classes(select_cohort(elec, meta))
    stamps = pd.DatetimeIndex(elec["timestamp"])
    n_weeks = len(stamps) // WEEK
    usable = n_weeks * WEEK
    hours = stamps.hour.to_numpy()[:WEEK]
    dows = stamps.dayofweek.to_numpy()[:WEEK]
    week_starts = stamps[:usable:WEEK]
    temps = _site_temperature(elec, weather)
    frames = []
    for row in cohort.itertuples(index=False):
        series = elec[row.building_id].to_numpy(dtype=np.float64)[:usable]
        series = np.where(series < 0, np.nan, series).reshape(n_weeks, WEEK)
        ok = np.isnan(series).mean(axis=1) <= MAX_WINDOW_MISSING
        with np.errstate(all="ignore"):
            ok &= np.nan_to_num(np.nanmean(series, axis=1)) > 0
        if not ok.any():
            continue
        temp = temps[row.site_id][:usable].reshape(n_weeks, WEEK)[ok]
        feats = shape_features(series[ok], hours, dows, temp)
        meta_cols = pd.DataFrame(
            {
                "building_id": row.building_id,
                "site_id": row.site_id,
                "primaryspaceusage": row.primaryspaceusage,
                "week_start": week_starts[ok],
                "performance_class": row.performance_class,
            }
        )
        frames.append(pd.concat([meta_cols.reset_index(drop=True), feats], axis=1))
    table = pd.concat(frames, ignore_index=True)
    if len(table) > n_rows:
        table = table.sample(n=n_rows, random_state=seed).sort_values(
            ["building_id", "week_start"]
        ).reset_index(drop=True)
    return table


def feature_columns(table: pd.DataFrame) -> list[str]:
    drop = set(ID_COLUMNS) | {"performance_class", "eui_kwh_m2_yr"}
    return [c for c in table.columns if c not in drop]
