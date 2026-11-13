from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

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

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        score = float(frame.mean(axis=1).iloc[0])
        distances = np.array([-(score + 0.5) ** 2, -score**2, -(score - 0.5) ** 2])
        probabilities = np.exp(distances - distances.max())
        return np.array([probabilities / probabilities.sum()])


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


def unpack_model(model: Any) -> tuple[Any, str, float | None, list[str]]:
    if isinstance(model, dict) and "estimator" in model:
        return (
            model["estimator"],
            str(model.get("model_version", "trained-unknown")),
            float(model["conformal_threshold"])
            if model.get("conformal_threshold") is not None
            else None,
            [str(value) for value in model.get("classes", DemoModel.labels)],
        )
    return model, "legacy-artifact", None, DemoModel.labels


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": os.getenv("CORS_ORIGINS", "*")}})
    schema = load_schema()
    feature_names: list[str] = schema["features"]
    loaded_model, mode = load_runtime_model()
    model, model_version, conformal_threshold, class_labels = unpack_model(loaded_model)

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
                "model_version": model_version,
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
            raw_prediction = model.predict(frame)[0]
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(frame)[0]
                classes = [
                    class_labels[int(value)]
                    for value in getattr(model, "classes_", range(len(probabilities)))
                ]
                probability_map = {
                    label: round(float(probability), 6)
                    for label, probability in zip(classes, probabilities)
                }
                if conformal_threshold is not None:
                    prediction_set = [
                        label
                        for label, probability in probability_map.items()
                        if probability >= 1.0 - conformal_threshold
                    ]
                    if not prediction_set:
                        prediction_set = [classes[int(np.argmax(probabilities))]]
                else:
                    prediction_set = [classes[int(np.argmax(probabilities))]]
            else:
                probability_map = {}
                prediction_set = [str(raw_prediction)]
            prediction = (
                class_labels[int(raw_prediction)]
                if isinstance(raw_prediction, (int, np.integer))
                and 0 <= int(raw_prediction) < len(class_labels)
                else str(raw_prediction)
            )
            response = {
                "prediction": prediction,
                "inference_mode": mode,
                "model_version": model_version,
                "probabilities": probability_map,
                "conformal_prediction_set": prediction_set,
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
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
