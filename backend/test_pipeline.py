"""Independent inference check (lab manual stage 5): load the bundle and predict one record.

Run from the project root:  python backend/test_pipeline.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
bundle = joblib.load(ROOT / "backend/models/model_bundle.joblib")
payload = json.loads((ROOT / "backend/test_payload.json").read_text(encoding="utf-8"))
features = bundle["features"]
assert set(features) == set(payload), "Payload keys must match features"
frame = pd.DataFrame([[payload[f] for f in features]], columns=features)
model = bundle["pipeline"]
classes = json.loads((ROOT / "backend/models/cls_schema.json").read_text())["classes"]
print("Prediction:", classes[int(model.predict(frame)[0])])
print("Probabilities:", dict(zip(classes, model.predict_proba(frame)[0].round(4).tolist())))
print("PASS: local inference")
