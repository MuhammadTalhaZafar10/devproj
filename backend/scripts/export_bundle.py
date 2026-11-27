"""Package the served model the way the lab manual expects, plus a sample payload.

Writes
    backend/models/model_bundle.joblib   {"pipeline", "features", "task", "model_version"}
    backend/test_payload.json            one real held-out record (feature name -> value)
    data/sample_inputs.csv               the same record as a one-row CSV (no target column)
    backend/models/cls_schema.json       adds per-feature medians and ranges for the Angular form

Run after run_cls_reproduction:  python -m backend.scripts.export_bundle
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from backend.app import features_from_hourly
from backend.cls import data as D

MODELS = D.REPO_ROOT / "backend" / "models"
REPORTS = D.REPO_ROOT / "reports"


def main() -> None:
    schema = json.loads((MODELS / "cls_schema.json").read_text())
    features = schema["features"]
    estimator = joblib.load(MODELS / "cls_model.joblib")
    # Feature engineering is done by backend.cls.features, so the sklearn pipeline is the model alone.
    pipeline = Pipeline([("model", estimator)])
    bundle = {"pipeline": pipeline, "features": features, "task": "classification", "model_version": "1.0"}
    joblib.dump(bundle, MODELS / "model_bundle.joblib", compress=3)

    example = json.loads((REPORTS / "cls_examples.json").read_text())[0]
    frame, error = features_from_hourly(example["hourly_kwh"], example["air_temperature"], example["week_start"][:10])
    assert error is None, error
    row = frame[features].iloc[0]
    payload = {k: float(v) for k, v in row.items()}
    (D.REPO_ROOT / "backend" / "test_payload.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame([payload]).to_csv(D.REPO_ROOT / "data" / "sample_inputs.csv", index=False)

    table = pd.read_parquet(D.PROCESSED_DIR / "bdg2_cls_100k.parquet")[features]
    schema["defaults"] = {c: float(table[c].median()) for c in features}
    schema["ranges"] = {c: [float(table[c].min()), float(table[c].max())] for c in features}
    schema["model_version"] = "1.0"
    (MODELS / "cls_schema.json").write_text(json.dumps(schema, indent=2))
    print(f"bundle with {len(features)} features; sample record from {example['building_id']} "
          f"(true class {example['true_class']})")


if __name__ == "__main__":
    main()
