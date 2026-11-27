"""Reproduce Raymand et al. (2023) performance-class classification on BDG2.

Usage:
    python -m backend.scripts.run_cls_reproduction [--rebuild] [--quick]

Writes reports/classification_results.json, backend/models/cls_model.joblib,
backend/models/cls_schema.json and reports/cls_examples.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.cls import data as D
from backend.cls import paper
from backend.cls.evaluate import (
    building_group_split, paper_protocol, random_row_split, run_split,
)
from backend.cls.models import MODEL_INFO, MODEL_NAMES

REPORTS = D.REPO_ROOT / "reports"
MODELS_DIR = D.REPO_ROOT / "backend" / "models"
CLASS_TO_INT = {c: i for i, c in enumerate(D.CLASSES)}


def load_tables(rebuild: bool):
    pa = D.PROCESSED_DIR / "bdg2_cls_buildings.parquet"
    pb = D.PROCESSED_DIR / "bdg2_cls_100k.parquet"
    if rebuild or not (pa.exists() and pb.exists()):
        elec, meta, weather = D.load_raw()
        A = D.build_building_table(elec, meta, weather)
        B = D.build_window_table(elec, meta, weather)
        A.to_parquet(pa)
        B.to_parquet(pb)
    return pd.read_parquet(pa), pd.read_parquet(pb)


def _label(info_models):
    for row in info_models:
        row["display"], row["in_paper"] = MODEL_INFO[row["model"]]
    return info_models


def _examples(B: pd.DataFrame, test_idx, n_per_class: int = 10) -> list[dict]:
    elec, _, weather = D.load_raw()
    temps = D._site_temperature(elec, weather)
    stamps = pd.DatetimeIndex(elec["timestamp"])
    pos = {t: i for i, t in enumerate(stamps[:: D.WEEK])}
    rng = np.random.default_rng(7)
    test = B.iloc[test_idx]
    out = []
    for cls in D.CLASSES:
        pool = test[test["performance_class"] == cls]
        for i in rng.choice(pool.index.to_numpy(), size=min(n_per_class, len(pool)), replace=False):
            row = B.loc[i]
            lo = pos[row["week_start"]] * D.WEEK
            hourly = elec[row["building_id"]].to_numpy(dtype=float)[lo: lo + D.WEEK]
            out.append({
                "building_id": row["building_id"],
                "site_id": row["site_id"],
                "primaryspaceusage": row["primaryspaceusage"],
                "week_start": str(row["week_start"]),
                "true_class": cls,
                "hourly_kwh": [None if np.isnan(v) else round(float(v), 3) for v in hourly],
                "air_temperature": [round(float(v), 2) for v in temps[row["site_id"]][lo: lo + D.WEEK]],
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quick", action="store_true", help="smoke run on small samples")
    args = parser.parse_args()

    A, B = load_tables(args.rebuild)
    feats_a = D.feature_columns(A)
    feats_b = D.feature_columns(B)
    shared = [c for c in feats_b if c in feats_a]
    if args.quick:
        A = A.sample(300, random_state=0)
        B = B[B["building_id"].isin(B["building_id"].unique()[:60])].reset_index(drop=True)
    ya = A["performance_class"].map(CLASS_TO_INT).to_numpy()
    yb = B["performance_class"].map(CLASS_TO_INT).to_numpy()

    print(f"Table A (paper protocol): {A.shape}, features={len(feats_a)}")
    print(f"Table B (course scale):   {B.shape}, features={len(feats_b)}")

    print("\n== Paper protocol on building table (random 80/20, 5-fold CV, 5 seeds) ==")
    paper_rows = _label(paper_protocol(A[feats_a], ya, seeds=(0,) if args.quick else (0, 1, 2, 3, 4)))
    for r in paper_rows:
        print(f"{r['display']:18s} f1_test={r['f1_weighted']:.3f}±{r['f1_weighted_std']:.3f} "
              f"acc_test={r['accuracy']:.3f} f1_cv={r['f1_val']:.3f}")

    print("\n== Scale table (100k rows): random-row split vs whole-building split ==")
    X = B[feats_b]
    rows_random, _ = run_split(X, yb, random_row_split(yb))
    group_split = building_group_split(yb, B["building_id"].to_numpy())
    rows_group, fitted = run_split(X, yb, group_split)
    for name, rows in (("random rows", rows_random), ("whole buildings", rows_group)):
        for r in _label(rows):
            print(f"{name:16s} {r['display']:18s} f1={r['f1_weighted']:.3f} acc={r['accuracy']:.3f} "
                  f"({r['fit_seconds']:.0f}s)")

    best = max(rows_group, key=lambda r: r["f1_weighted"])["model"]
    served = "lightgbm"  # small artifact, near-best; RF would be >100 MB
    model = fitted[served]
    importances = sorted(
        zip(feats_b, model.feature_importances_.tolist()), key=lambda kv: -kv[1]
    )[:15]

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "citation": paper.CITATION,
            "split": paper.PAPER_SPLIT,
            "buildings": paper.PAPER_BUILDINGS,
            "table3_performance_class": paper.TABLE3_PERFORMANCE_CLASS,
            "fig6a_f1_performance_class": paper.FIG6A_F1_PERFORMANCE_CLASS,
        },
        "cohort": {
            "buildings": int(len(A)),
            "sites": int(A["site_id"].nunique()),
            "class_counts": A["performance_class"].value_counts().to_dict(),
            "class_cutoffs_kwh_m2_yr": [
                float(A.loc[A["performance_class"] == "Low", "eui_kwh_m2_yr"].max()),
                float(A.loc[A["performance_class"] == "Medium", "eui_kwh_m2_yr"].max()),
            ],
            "label_note": "Terciles of mean kWh/m2/yr; paper cutoffs unpublished (repo is a placeholder).",
        },
        "dataset_100k": {
            "rows": int(len(B)), "columns": int(B.shape[1]), "feature_columns": len(feats_b),
            "buildings": int(B["building_id"].nunique()), "sites": int(B["site_id"].nunique()),
            "unit": "one building-week (168 hourly readings)",
        },
        "models": {k: {"display": v[0], "in_paper": v[1]} for k, v in MODEL_INFO.items()},
        "paper_protocol": paper_rows,
        "scale_protocol": {"random_rows": rows_random, "whole_buildings": rows_group},
        "best_model_whole_buildings": best,
        "served_model": served,
        "feature_importance_top15": importances,
        "classes": D.CLASSES,
    }
    if not args.quick:
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "classification_results.json").write_text(json.dumps(result, indent=2))
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODELS_DIR / "cls_model.joblib", compress=3)
        (MODELS_DIR / "cls_schema.json").write_text(json.dumps({
            "features": feats_b, "classes": D.CLASSES, "window_hours": D.WEEK,
            "model": MODEL_INFO[served][0],
            "trained_on": "whole-building 80 % split of the 100k-row table",
        }, indent=2))
        (REPORTS / "cls_examples.json").write_text(json.dumps(_examples(B, group_split[1]), indent=1))
        print("\nwrote reports/classification_results.json, backend/models/cls_model.joblib")


if __name__ == "__main__":
    main()
