from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import create_app
from backend.cls.features import feature_names

CLASSES = ["Low", "Medium", "High"]


@pytest.fixture(scope="module")
def bundle():
    names = feature_names()
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, len(names))), columns=names)
    y = rng.integers(0, 3, 300)
    model = LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y)
    return model, {"features": names, "classes": CLASSES, "model": "LightGBM (test)"}


@pytest.fixture()
def client(bundle):
    app = create_app(model=bundle[0], schema=bundle[1])
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _week(seed=0):
    rng = np.random.default_rng(seed)
    hours = np.tile(np.arange(24), 7)
    return [round(float(v), 3) for v in 40 + 20 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 2, 168)]


def test_required_structure_exists():
    assert (REPO_ROOT / "backend").is_dir()
    assert (REPO_ROOT / "backend" / "tests").is_dir()
    assert (REPO_ROOT / "docs").is_dir()


def test_health_reports_model_mode(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["inference_mode"] == "model"
    assert body["feature_count"] >= 35


def test_predict_from_hourly_readings(client):
    r = client.post("/predict", json={"hourly_kwh": _week(), "air_temperature": [15.0] * 168})
    assert r.status_code == 200
    body = r.get_json()
    assert body["prediction"] in CLASSES
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-2


def test_predict_from_feature_dict(client, bundle):
    row = {n: 0.5 for n in bundle[1]["features"]}
    assert client.post("/predict", json={"features": row}).status_code == 200
    row.pop(next(iter(row)))
    r = client.post("/predict", json={"features": row})
    assert r.status_code == 400 and "Missing required fields" in r.get_json()["error"]


def test_predict_from_flat_feature_payload_like_the_lab_manual(client, bundle):
    row = {n: 0.5 for n in bundle[1]["features"]}
    body = client.post("/predict", json=row).get_json()
    assert body["prediction"] in CLASSES and body["task"] == "classification" and body["disclaimer"]
    row["not_a_feature"] = 1
    r = client.post("/predict", json=row)
    assert r.status_code == 400 and r.get_json()["extra"] == ["not_a_feature"]


@pytest.mark.parametrize("payload", [
    {"hourly_kwh": [1.0] * 10},
    {"hourly_kwh": [0.0] * 168},
    {"hourly_kwh": [None] * 100 + [5.0] * 68},
    {"hourly_kwh": _week(), "week_start": "not-a-date"},
    {"nothing": 1},
    [1, 2, 3],
])
def test_predict_rejects_bad_payloads(client, payload):
    assert client.post("/predict", json=payload).status_code == 400


def test_predict_invalid_json(client):
    r = client.post("/predict", data='{"broken": true', content_type="application/json")
    assert r.status_code == 400 and r.get_json()["error"] == "Invalid JSON payload."


def test_without_artifact_predict_is_503_not_a_fake_answer(tmp_path, monkeypatch):
    import backend.app as mod

    monkeypatch.setattr(mod, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(mod, "SCHEMA_PATH", tmp_path / "missing.json")
    app = mod.create_app()
    c = app.test_client()
    assert c.get("/health").get_json()["inference_mode"] == "unavailable"
    assert c.post("/predict", json={"hourly_kwh": _week()}).status_code == 503


def test_cors_header_only_for_allowed_origin(client):
    ok = client.get("/health", headers={"Origin": "http://localhost:4200"})
    bad = client.get("/health", headers={"Origin": "https://evil.example"})
    assert ok.headers.get("Access-Control-Allow-Origin") == "http://localhost:4200"
    assert "Access-Control-Allow-Origin" not in bad.headers
