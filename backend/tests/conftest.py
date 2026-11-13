"""Shared pytest fixtures and synthetic panel factory."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_panel(buildings: int = 2, hours: int = 400, seed: int = 7) -> pd.DataFrame:
    """Small but realistic hourly panel used by every test.

    Starts before the default chronological cutoff so that year-splitting
    protocols have data on both sides.
    """
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    uses = ["Lodging/residential", "Office", "Education", "Retail", "Other"]
    timestamps = pd.date_range("2017-05-01", periods=hours, freq="h")
    for b in range(buildings):
        site = f"site{b % 3}"
        base = 40.0 + 10.0 * b
        daily = 8.0 * np.sin(2 * np.pi * timestamps.hour.to_numpy() / 24)
        y = np.maximum(0.0, base + daily + rng.normal(0, 1.5, hours))
        frames.append(
            pd.DataFrame(
                {
                    "building_id": f"b{b}",
                    "site_id": site,
                    "timestamp": timestamps,
                    "y": y,
                    "primaryspaceusage": uses[b % len(uses)],
                    "sqm": 500.0 + 100.0 * b,
                    "site_eui": 80.0 + b,
                    "airTemperature": 12.0 + 6.0 * np.sin(2 * np.pi * np.arange(hours) / 24),
                    "dewTemperature": 6.0,
                    "windSpeed": 2.0,
                    "cloudCoverage": 0.4,
                    "precipDepth1HR": 0.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def make_panel():
    return _make_panel
