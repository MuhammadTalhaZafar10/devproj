from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


ROOT_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = Path(__file__).resolve().parent / "models" / "feature_schema.json"
MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / "models" / "model.joblib",
    Path(__file__).resolve().parent / "models" / "model.pkl",
]


class DemoModel:
    """CI-safe deterministic fallback. Not a trained model."""

    labels = ["Low", "Medium", "High"]

    def predict(self, frame: pd.DataFrame) -> list[str]:
        score = frame.mean(axis=1).iloc[0]
        if score < -0.25:
            return [self.labels[0]]
        if score > 0.25:
            return [self.labels[2]]
        return [self.labels[1]]


def load_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_runtime_model() -> tuple[Any, str]:
    if joblib is None:
        return DemoModel(), "demo"
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return joblib.load(candidate), "model"
    return DemoModel(), "demo"


def create_app() -> Flask:
    app = Flask(__name__)
    schema = load_schema()
    feature_names: list[str] = schema["features"]
    model, mode = load_runtime_model()

    def validate_payload(payload: Any) -> tuple[dict[str, float] | None, str | None]:
        if not isinstance(payload, dict):
            return None, "JSON body must be an object."

        payload_keys = set(payload.keys())
        expected_keys = set(feature_names)
        missing = sorted(expected_keys - payload_keys)
        extra = sorted(payload_keys - expected_keys)

        if missing:
            return None, f"Missing required fields: {missing[:10]}"
        if extra:
            return None, f"Unexpected fields: {extra[:10]}"

        numeric_payload: dict[str, float] = {}
        for key in feature_names:
            try:
                value = float(payload[key])
            except (TypeError, ValueError):
                return None, f"Field '{key}' must be numeric."
            if not np.isfinite(value):
                return None, f"Field '{key}' must be finite."
            numeric_payload[key] = value
        return numeric_payload, None

    @app.get("/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "inference_mode": mode,
                "feature_count": len(feature_names),
            }
        )

    @app.post("/predict")
    def predict() -> Any:
        try:
            payload = request.get_json(force=False, silent=False)
        except Exception:
            return jsonify({"error": "Invalid JSON payload."}), 400

        validated, error = validate_payload(payload)
        if error:
            return jsonify({"error": error}), 400

        try:
            frame = pd.DataFrame([validated], columns=feature_names)
            prediction = model.predict(frame)[0]
            response = {
                "prediction": str(prediction),
                "inference_mode": mode,
            }
            if mode != "model":
                response["note"] = (
                    "Demo fallback prediction is active. No deployable trained "
                    "model artifact was found in backend/models/."
                )
            return jsonify(response)
        except Exception:
            return jsonify({"error": "Prediction failed due to internal error."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
