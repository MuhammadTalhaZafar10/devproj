"""Flask API for the BDG2 performance-class classifier.

Endpoints
    GET  /health      status + whether a trained artifact is loaded
    GET  /model-info  served model, feature list, classes
    GET  /results     reproduction results (paper vs ours, 4 models)
    GET  /novelty     leakage-protocol results, if computed
    GET  /examples    real held-out building-weeks with ground truth
    POST /predict     {"hourly_kwh": [168 values], "air_temperature": [168]?, "week_start": "YYYY-MM-DD"?}
                      or {"features": {name: value, ...}}

There is no fake "demo" model: without a trained artifact /predict answers 503.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

from backend.cls.features import shape_features

BACKEND_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BACKEND_DIR.parent / "reports"
MODEL_PATH = BACKEND_DIR / "models" / "cls_model.joblib"
SCHEMA_PATH = BACKEND_DIR / "models" / "cls_schema.json"
WINDOW = 168
MAX_MISSING = 0.10
DEFAULT_ORIGINS = "http://localhost:4200,https://muhammadtalhazafar10.github.io"


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def features_from_hourly(
    hourly: Any, temperature: Any = None, week_start: str | None = None
) -> tuple[pd.DataFrame | None, str | None]:
    """Turn 168 hourly kWh readings into the model's feature row."""
    if not isinstance(hourly, list) or len(hourly) != WINDOW:
        return None, f"'hourly_kwh' must be a list of exactly {WINDOW} hourly readings."
    try:
        x = np.array([np.nan if v is None else float(v) for v in hourly], dtype=float)
    except (TypeError, ValueError):
        return None, "'hourly_kwh' must contain numbers (or null for missing)."
    x[~np.isfinite(x)] = np.nan
    x[x < 0] = np.nan
    if np.isnan(x).mean() > MAX_MISSING:
        return None, "Too many missing/negative hours (limit 10 %)."
    if not np.nanmean(x) > 0:
        return None, "Mean consumption must be positive."
    # Training windows start on a Friday (2016-01-01); allow an explicit start date.
    start_dow = 4
    if week_start:
        try:
            start_dow = date.fromisoformat(week_start).weekday()
        except ValueError:
            return None, "'week_start' must be an ISO date (YYYY-MM-DD)."
    hours = np.tile(np.arange(24), 7)
    dows = (start_dow + np.arange(WINDOW) // 24) % 7
    temp = None
    if temperature is not None:
        try:
            temp = np.array([float(v) for v in temperature], dtype=float)
        except (TypeError, ValueError):
            return None, "'air_temperature' must be a list of numbers."
        if temp.shape != (WINDOW,) or not np.isfinite(temp).all():
            return None, f"'air_temperature' must be {WINDOW} finite numbers."
    feats = shape_features(
        x[None, :], hours, dows, temp[None, :] if temp is not None else np.zeros((1, WINDOW))
    )
    return feats, None


def create_app(model: Any = None, schema: dict | None = None) -> Flask:
    app = Flask(__name__)
    origins = {o.strip() for o in os.getenv("CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()}

    if model is None and joblib is not None and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    schema = schema or _read_json(SCHEMA_PATH)
    ready = model is not None and schema is not None
    features: list[str] = schema["features"] if ready else []
    classes: list[str] = schema["classes"] if ready else []

    @app.after_request
    def add_cors(resp):
        origin = request.headers.get("Origin")
        if origin and (origin in origins or "*" in origins):
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.get("/health")
    def health() -> Any:
        return jsonify({
            "status": "ok",
            "inference_mode": "model" if ready else "unavailable",
            "model": schema.get("model") if ready else None,
            "feature_count": len(features),
            "classes": classes,
        })

    @app.get("/model-info")
    def model_info() -> Any:
        if not ready:
            return jsonify({"error": "No trained model artifact loaded."}), 503
        return jsonify({**schema, "feature_count": len(features)})

    def _report(name: str):
        data = _read_json(REPORTS_DIR / name)
        if data is None:
            return jsonify({"error": f"{name} has not been generated yet."}), 404
        return jsonify(data)

    @app.get("/results")
    def results() -> Any:
        return _report("classification_results.json")

    @app.get("/novelty")
    def novelty() -> Any:
        return _report("cls_novelty_results.json")

    @app.get("/examples")
    def examples() -> Any:
        data = _read_json(REPORTS_DIR / "cls_examples.json")
        if data is None:
            return jsonify({"error": "cls_examples.json has not been generated yet."}), 404
        try:
            limit = max(1, min(int(request.args.get("n", len(data))), len(data)))
        except ValueError:
            return jsonify({"error": "'n' must be an integer."}), 400
        return jsonify(data[:limit])

    @app.post("/predict")
    def predict() -> Any:
        if not ready:
            return jsonify({"error": "No trained model artifact loaded."}), 503
        try:
            payload = request.get_json(force=False, silent=False)
        except Exception:
            return jsonify({"error": "Invalid JSON payload."}), 400
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON body must be an object."}), 400

        if "hourly_kwh" in payload:
            frame, error = features_from_hourly(
                payload.get("hourly_kwh"), payload.get("air_temperature"), payload.get("week_start")
            )
            if error:
                return jsonify({"error": error}), 400
            frame = frame.reindex(columns=features)
        elif "features" in payload and isinstance(payload["features"], dict):
            given = payload["features"]
            missing = sorted(set(features) - set(given))
            if missing:
                return jsonify({"error": f"Missing required fields: {missing[:10]}"}), 400
            try:
                row = {k: float(given[k]) for k in features}
            except (TypeError, ValueError):
                return jsonify({"error": "All feature values must be numeric."}), 400
            if not all(np.isfinite(v) for v in row.values()):
                return jsonify({"error": "All feature values must be finite."}), 400
            frame = pd.DataFrame([row], columns=features)
        elif set(features) & set(payload):
            # Flat form used by the lab manual: {feature_name: value, ...}
            missing = [f for f in features if f not in payload]
            extra = [k for k in payload if k not in features]
            if missing or extra:
                return jsonify({"error": "Input schema mismatch", "missing": missing, "extra": extra}), 400
            try:
                row = {k: float(payload[k]) for k in features}
            except (TypeError, ValueError):
                return jsonify({"error": "All feature values must be numeric."}), 400
            if not all(np.isfinite(v) for v in row.values()):
                return jsonify({"error": "All feature values must be finite."}), 400
            frame = pd.DataFrame([row], columns=features)
        else:
            return jsonify({"error": "Provide 'hourly_kwh' (168 values) or the model features."}), 400

        try:
            proba = model.predict_proba(frame)[0]
            idx = int(np.argmax(proba))
        except Exception:
            return jsonify({"error": "Prediction failed due to internal error."}), 500
        return jsonify({
            "prediction": classes[idx],
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
            "task": "classification",
            "inference_mode": "model",
            "model": schema.get("model"),
            "disclaimer": "Research/teaching prototype only. Classes are terciles of kWh per m2 per year.",
        })

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
