"""BDG2 loading, cohort selection, and canonical panel construction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

WEATHER_COLUMNS = [
    "airTemperature",
    "dewTemperature",
    "windSpeed",
    "cloudCoverage",
    "precipDepth1HR",
]
CONTEXT_COLUMNS = ["primaryspaceusage", "sqm", "site_eui"]


def load_metadata(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    meta = pd.read_csv(Path(raw_dir) / "metadata.csv")
    return meta[["building_id", "site_id", "primaryspaceusage", "sqm", "site_eui"]]


def load_weather(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    weather = pd.read_csv(Path(raw_dir) / "weather.csv", parse_dates=["timestamp"])
    return weather[["timestamp", "site_id", *WEATHER_COLUMNS]]


def load_meter_wide(raw_dir: Path = RAW_DIR, meter: str = "electricity") -> pd.DataFrame:
    path = Path(raw_dir) / f"{meter}.csv"
    wide = pd.read_csv(path, parse_dates=["timestamp"])
    wide = wide.sort_values("timestamp").reset_index(drop=True)
    return wide


def coverage_table(wide: pd.DataFrame) -> pd.Series:
    return wide.drop(columns=["timestamp"]).notna().mean()


REPRODUCTION_USES = (
    "Lodging/residential",
    "Office",
    "Education",
    "Retail",
    "Other",
)


def select_reproduction_cohort(metadata: pd.DataFrame) -> list[str]:
    """Two buildings per primary use from the Panther site, deterministically.

    The base paper reports five use types, so the five named uses are fixed.
    """
    panther = metadata[
        (metadata["site_id"] == "Panther")
        & (metadata["primaryspaceusage"].isin(REPRODUCTION_USES))
    ]
    chosen: list[str] = []
    for use in REPRODUCTION_USES:
        candidates = sorted(panther.loc[panther["primaryspaceusage"] == use, "building_id"])
        chosen.extend(candidates[:2])
    return chosen


def select_novelty_cohort(
    metadata: pd.DataFrame,
    coverage: pd.Series,
    n: int = 120,
    min_coverage: float = 0.9,
) -> list[str]:
    """Deterministic site-stratified sample of well-covered meters."""
    eligible = metadata[metadata["building_id"].map(coverage) >= min_coverage].copy()
    eligible = eligible.sort_values(["site_id", "primaryspaceusage", "building_id"])
    by_site = {
        site: group["building_id"].tolist()
        for site, group in eligible.groupby("site_id", sort=True)
    }
    chosen: list[str] = []
    while len(chosen) < n and any(by_site.values()):
        for site in sorted(by_site):
            if by_site[site] and len(chosen) < n:
                chosen.append(by_site[site].pop(0))
    return chosen


def build_panel(
    raw_dir: Path = RAW_DIR,
    cohort: str = "novelty",
    n: int = 120,
    meter: str = "electricity",
) -> pd.DataFrame:
    metadata = load_metadata(raw_dir)
    weather = load_weather(raw_dir)
    wide = load_meter_wide(raw_dir, meter=meter)
    coverage = coverage_table(wide)

    if cohort == "reproduction":
        ids = select_reproduction_cohort(metadata)
    elif cohort == "novelty":
        ids = select_novelty_cohort(metadata, coverage, n=n)
    else:
        raise ValueError(f"unknown cohort: {cohort}")

    present = [b for b in ids if b in wide.columns]
    long = wide[["timestamp", *present]].melt(
        id_vars=["timestamp"], var_name="building_id", value_name="y"
    )
    long["y"] = pd.to_numeric(long["y"], errors="coerce")
    long = long.dropna(subset=["y"])
    long = long[np.isfinite(long["y"]) & (long["y"] >= 0)]

    panel = long.merge(metadata, on="building_id", how="left")
    panel = panel.merge(weather, on=["site_id", "timestamp"], how="left")
    panel = panel.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    missing = [c for c in [*CONTEXT_COLUMNS, *WEATHER_COLUMNS] if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing expected columns: {missing}")
    return panel


def write_panel(panel: pd.DataFrame, processed_dir: Path = PROCESSED_DIR) -> Path:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / "bdg2_panel.parquet"
    panel.to_parquet(path, index=False)
    return path


def read_panel(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = Path(processed_dir) / "bdg2_panel.parquet"
    return pd.read_parquet(path)
