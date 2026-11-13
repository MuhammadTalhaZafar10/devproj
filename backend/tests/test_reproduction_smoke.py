"""Smoke test for the six-model reproduction runner."""
from __future__ import annotations

from backend.ml.models import MODEL_NAMES
from backend.scripts.run_reproduction import run_reproduction


def test_smoke_reproduction_writes_one_row_per_model(make_panel, tmp_path):
    out = run_reproduction(panel=make_panel(buildings=2, hours=2200), out_dir=tmp_path)
    assert set(out["model"]) == set(MODEL_NAMES)
    assert out["rmse"].notna().all()
    assert (tmp_path / "reproduction_metrics.csv").exists()
