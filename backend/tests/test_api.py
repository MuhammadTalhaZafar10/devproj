from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import create_app

SAMPLE_PATH = REPO_ROOT / "data" / "sample_request.json"


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def load_sample_payload() -> dict:
    with SAMPLE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_required_structure_exists():
    assert (REPO_ROOT / "backend").is_dir()
    assert (REPO_ROOT / "backend" / "tests").is_dir()
    assert (REPO_ROOT / "backend" / "models" / "feature_schema.json").is_file()
    assert (REPO_ROOT / "data").is_dir()
    assert (REPO_ROOT / "docs").is_dir()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["feature_count"] > 0


def test_predict_valid_payload(client):
    payload = load_sample_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"] in {"Low", "Medium", "High"}
    assert body["inference_mode"] in {"model", "demo"}


def test_predict_missing_fields(client):
    payload = load_sample_payload()
    payload.pop(next(iter(payload)))
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Missing required fields" in response.get_json()["error"]


def test_predict_invalid_json(client):
    response = client.post(
        "/predict",
        data='{"broken": true',
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid JSON payload."


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("extra_field", 1, "Unexpected fields"),
        ("hour", "not-a-number", "must be numeric"),
        ("hour", float("nan"), "must be finite"),
        ("hour", float("inf"), "must be finite"),
    ],
)
def test_predict_rejects_invalid_fields(client, field, value, message):
    payload = load_sample_payload()
    if field == "extra_field":
        payload[field] = value
    else:
        payload[field] = value
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_predict_response_contains_uncertainty_fields(client):
    response = client.post("/predict", json=load_sample_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_version"]
    assert set(body["probabilities"]) <= {"Low", "Medium", "High"}
    assert body["conformal_prediction_set"]
    assert body["prediction"] in {"Low", "Medium", "High"}
