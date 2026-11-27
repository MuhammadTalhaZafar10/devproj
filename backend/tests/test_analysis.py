"""The decomposition must be additive and subtract matching cells."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.ml.analysis import channel_decomposition, feature_construction_gap


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "lightgbm", "protocol": "S1", "tier": "C", "r2": 0.95},
            {"model": "lightgbm", "protocol": "S2", "tier": "C", "r2": 0.60},
            {"model": "lightgbm", "protocol": "S3", "tier": "C", "r2": 0.40},
            {"model": "lightgbm", "protocol": "S4", "tier": "C", "r2": 0.25},
            {"model": "lightgbm", "protocol": "S1", "tier": "D", "r2": 0.99},
            {"model": "lightgbm", "protocol": "S2", "tier": "D", "r2": 0.80},
            {"model": "lightgbm", "protocol": "S3", "tier": "D", "r2": 0.55},
            {"model": "lightgbm", "protocol": "S4", "tier": "D", "r2": 0.30},
        ]
    )


def test_channel_decomposition_subtracts_matching_cells():
    out = channel_decomposition(_results())
    tier_c = out[(out["model"] == "lightgbm") & (out["tier"] == "C")].set_index("channel")["delta_r2"]
    assert tier_c["temporal"] == pytest.approx(0.35)
    assert tier_c["building"] == pytest.approx(0.20)
    assert tier_c["site"] == pytest.approx(0.15)
    assert tier_c["total"] == pytest.approx(0.70)


def test_channels_are_additive():
    out = channel_decomposition(_results())
    for (_, tier), part in out.groupby(["model", "tier"]):
        values = part.set_index("channel")["delta_r2"]
        assert values["temporal"] + values["building"] + values["site"] == pytest.approx(
            values["total"]
        )


def test_feature_construction_gap_is_d_minus_c():
    out = feature_construction_gap(_results()).set_index("protocol")["delta_r2"]
    assert out["S1"] == pytest.approx(0.04)
    assert out["S2"] == pytest.approx(0.20)
