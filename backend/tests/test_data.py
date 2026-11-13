"""Cohort selection for the BDG2 study."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.ml.data import select_novelty_cohort, select_reproduction_cohort


def _metadata(uses: list[str], per_use: int = 3, prefix: str = "Panther") -> pd.DataFrame:
    rows = []
    for use in uses:
        for i in range(per_use):
            rows.append(
                {
                    "building_id": f"{prefix}_{use.split('/')[0].lower()}_{i}",
                    "site_id": prefix,
                    "primaryspaceusage": use,
                }
            )
    return pd.DataFrame(rows)


def test_reproduction_cohort_is_ten_panther_buildings_covering_five_uses():
    meta = _metadata(
        ["Lodging/residential", "Office", "Education", "Retail", "Other"]
    )
    ids = select_reproduction_cohort(meta)
    assert len(ids) == 10
    assert all(b.startswith("Panther_") for b in ids)
    uses = {meta.set_index("building_id").loc[b, "primaryspaceusage"] for b in ids}
    assert len(uses) == 5


def test_reproduction_cohort_is_deterministic():
    meta = _metadata(["Lodging/residential", "Office", "Education", "Retail", "Other"])
    assert select_reproduction_cohort(meta) == select_reproduction_cohort(meta)


def test_novelty_cohort_excludes_low_coverage_buildings():
    meta = _metadata(["Office", "Education"], per_use=10)
    coverage = pd.Series(1.0, index=meta["building_id"])
    coverage.iloc[:10] = 0.5
    ids = select_novelty_cohort(meta, coverage, n=8, min_coverage=0.9)
    assert len(ids) == 8
    assert not set(ids) & set(meta["building_id"].iloc[:10])
