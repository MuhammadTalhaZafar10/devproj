"""Feature tiers must be causal, except tier D which is the positive control."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.ml.features import FEATURES, LeakageGuardError, availability_report, build_features


def test_tier_c_ignores_current_target_but_tier_d_uses_it(make_panel):
    panel = make_panel(hours=220)
    x_c, _ = build_features(panel, "C")
    x_d, _ = build_features(panel, "D")
    last = panel.index[-1]
    mutated = panel.copy()
    mutated.loc[last, "y"] = mutated.loc[last, "y"] + 1000.0

    x_c2, _ = build_features(mutated, "C")
    x_d2, _ = build_features(mutated, "D")

    pd.testing.assert_frame_equal(x_c.loc[[last]], x_c2.loc[[last]])
    assert not x_d.loc[[last]].equals(x_d2.loc[[last]])


def test_tier_a_contains_no_energy_history(make_panel):
    panel = make_panel()
    x, _ = build_features(panel, "A")
    assert not [c for c in x.columns if c.startswith(("lag_", "roll_", "leaky_"))]


def test_availability_report_labels_every_feature():
    report = availability_report("C").set_index("feature")
    assert set(report.index) == set(FEATURES["C"])
    energy = [c for c in FEATURES["C"] if c.startswith(("lag_", "roll_"))]
    assert (report.loc[energy, "available_at"] != "t").all()
    leaky = availability_report("D").set_index("feature")
    assert (leaky.loc[[c for c in FEATURES["D"] if c.startswith("leaky_")], "available_at"] == "t").all()


def test_warmup_rows_are_dropped(make_panel):
    panel = make_panel(hours=200)
    x, y = build_features(panel, "C")
    assert len(x) == len(y)
    assert len(x) < len(panel)


def test_causality_guard_rejects_leaky_tier(make_panel):
    panel = make_panel(hours=72)
    with pytest.raises(LeakageGuardError):
        build_features(panel, "D", enforce_causal=True)
