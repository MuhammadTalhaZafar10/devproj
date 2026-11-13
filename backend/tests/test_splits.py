"""Split protocols must never leak time or groups across the boundary."""
from __future__ import annotations

import numpy as np
import pytest

from backend.ml.splits import (
    blocked_temporal,
    leave_one_group_out,
    protocol_splits,
    random_folds,
)


def test_blocked_temporal_has_no_time_overlap(make_panel):
    df = make_panel(hours=1500)
    train, test = blocked_temporal(df)[0]
    assert df.iloc[train]["timestamp"].max() < df.iloc[test]["timestamp"].min()
    assert len(np.intersect1d(train, test)) == 0


def test_random_folds_cover_every_row_once(make_panel):
    df = make_panel(hours=200)
    seen = np.concatenate([test for _, test in random_folds(df, n_splits=5)])
    assert len(seen) == len(df)
    assert len(np.unique(seen)) == len(df)


def test_leave_one_site_out_never_shares_a_site(make_panel):
    df = make_panel(buildings=6)
    for train, test in leave_one_group_out(df, "site_id"):
        assert set(df.iloc[train]["site_id"]).isdisjoint(set(df.iloc[test]["site_id"]))


def test_leave_one_building_out_never_shares_a_building(make_panel):
    df = make_panel(buildings=6)
    for train, test in leave_one_group_out(df, "building_id"):
        assert set(df.iloc[train]["building_id"]).isdisjoint(set(df.iloc[test]["building_id"]))


def test_unknown_protocol_raises(make_panel):
    with pytest.raises(ValueError):
        protocol_splits(make_panel(), "S9")
