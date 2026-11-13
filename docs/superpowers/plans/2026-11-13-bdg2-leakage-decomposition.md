# BDG2 Leakage-Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the base paper's six-model BDG2 energy benchmark, measure how much of the random-vs-temporal validation gap comes from temporal autocorrelation, building identity, site context, and feature construction, and ship the results through a real Flask API and an Angular app deployed to GitHub Pages.

**Architecture:** A small `backend/ml` library builds a canonical BDG2 panel, constructs four feature-availability tiers and four validation protocols, and fits frozen tree-ensemble models. Two CLI runners write CSV/JSON results and figures. A training script exports a model bundle consumed by Flask; the Angular app renders bundled JSON statically on Pages and upgrades to live inference when the API is reachable.

**Tech Stack:** Python 3.11, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, Flask, pytest; Angular 19, TypeScript, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-13-bdg2-leakage-decomposition-design.md`

## Global Constraints

- Python 3.11; pin dependencies in `backend/requirements.txt`; use a repo-local `.venv` (gitignored).
- Never fit a scaler, imputer, or encoder outside the training fold.
- Tier C rolling windows must end at `t-1`; tier D is the only deliberately leaky tier and must be labelled as such everywhere.
- Primary metrics: pooled RMSE (kWh) and pooled R²; also MAE, MAPE on `y > 0`, per-building R² median/IQR.
- Model configurations are frozen across every cell of the matrix; no per-cell tuning.
- Fallback panel cap: 120 meters. If the full matrix exceeds 90 minutes, reduce group folds before reducing meters.
- Reproduction target bands (base paper Table 14): RMSE 5–9 kWh, R² 0.59–0.76, MAPE 8–13%.
- Committed model artifacts must be < 20 MB; otherwise the release/generation path is documented instead.
- No invented citations, metrics, or dataset sizes. BDG2 is 3,053 meters / 1,636 buildings / 19 sites.
- GitHub Pages is static-only: the site must render from bundled JSON and must not claim live inference when the API is absent.

## Review Focus

- A current-timestep energy value reaching a causal tier (the single failure that invalidates the study).
- Grouped splits leaking a building or site into both train and test.
- Metrics computed on log1p predictions instead of the original kWh scale.
- The API accepting a feature dict with wrong length/order and silently predicting.
- The Pages build fetching `/api` on a static host and rendering an error or a blank panel.

---

### Task 1: Canonical BDG2 panel and cohorts

**Files:**
- Create: `backend/ml/__init__.py`
- Create: `backend/ml/data.py`
- Create: `backend/scripts/fetch_data.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_data.py`
- Modify: `backend/requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `make_panel(buildings=2, hours=400) -> pd.DataFrame` test factory in `conftest.py` (columns `building_id, site_id, timestamp, y` plus tier-A context); `RAW_DIR`, `PROCESSED_DIR`, `load_metadata(raw_dir) -> pd.DataFrame`, `load_weather(raw_dir) -> pd.DataFrame`, `load_meter_panel(raw_dir, meter="electricity") -> pd.DataFrame` (columns `building_id, timestamp, y`), `build_panel(raw_dir=RAW_DIR, cohort="novelty") -> pd.DataFrame` (adds `site_id, primaryspaceusage, sqm, site_eui, airTemperature, dewTemperature, windSpeed, cloudCoverage, precipDepth1HR`), `select_reproduction_cohort(metadata) -> list[str]`, `select_novelty_cohort(panel, metadata, n=120, min_coverage=0.9) -> list[str]`.

- [ ] **Step 1: Add dependencies and ignore rules**

`backend/requirements.txt`: add `kagglehub`, `lightgbm`, `xgboost`, `catboost`, `scikit-learn`, `pyarrow`, `matplotlib`, `seaborn`, `joblib` at working versions. `.gitignore`: add `data/raw/`, `data/processed/`, `.venv/`, `reports/figures/`.

- [ ] **Step 2: Write the failing cohort test**

```python
def test_reproduction_cohort_is_ten_panther_buildings_covering_five_uses():
    meta = pd.DataFrame({
        "building_id": [f"Panther_{u}_{i}" for u in ["lodging","office","education","retail","other"] for i in range(3)],
        "site_id": "Panther",
        "primaryspaceusage": [u for u in ["Lodging/residential","Office","Education","Retail","Other"] for _ in range(3)],
    })
    ids = select_reproduction_cohort(meta)
    assert len(ids) == 10
    assert all(b.startswith("Panther_") for b in ids)
    assert len({meta.set_index("building_id").loc[b, "primaryspaceusage"] for b in ids}) == 5
```

- [ ] **Step 3: Run it and verify it fails**

Run: `pytest backend/tests/test_data.py::test_reproduction_cohort_is_ten_panther_buildings_covering_five_uses -v`
Expected: FAIL with `ImportError`/`NameError`.

- [ ] **Step 4: Implement `backend/ml/data.py`**

`load_meter_panel` reads the wide CSV in chunks, melting only requested meter columns; `build_panel` inner-joins metadata and left-joins site weather on `site_id, timestamp`. `fetch_data.py` calls `kagglehub.dataset_download("claytonmiller/buildingdatagenomeproject2")`, writes the canonical panel to `data/processed/bdg2_panel.parquet`, and prints row/building/site counts.

- [ ] **Step 5: Run the tests**

Run: `pytest backend/tests/test_data.py -v`
Expected: PASS.

- [ ] **Step 6: Fetch and build the real panel**

Run: `python -m backend.scripts.fetch_data`
Expected: prints `rows > 40,000,000`, `buildings = 1636`, `sites = 19`, and writes the parquet.

- [ ] **Step 7: Commit**

```bash
git add backend/ml backend/scripts/fetch_data.py backend/tests/test_data.py backend/requirements.txt .gitignore
git commit -m "feat(data): canonical BDG2 panel and cohort selection"
```

---

### Task 2: Feature tiers with a leakage guard

**Files:**
- Create: `backend/ml/features.py`
- Create: `backend/tests/test_features.py`

**Interfaces:**
- Produces: `TIERS = ("A","B","C","D")`, `FEATURES: dict[str, list[str]]`, `build_features(panel, tier) -> tuple[pd.DataFrame, pd.Series]`, `availability_report(tier) -> pd.DataFrame`, `LeakageGuardError`.
- Consumes: `backend.ml.data.build_panel`.

- [ ] **Step 1: Write the failing causality test**

```python
def test_tier_c_ignores_current_target_but_tier_d_uses_it():
    panel = make_panel(hours=48)
    x_c, _ = build_features(panel, "C")
    x_d, _ = build_features(panel, "D")
    mutated = panel.copy()
    mutated.loc[mutated.index[-1], "y"] = panel["y"].iloc[-1] + 1000
    x_c2, _ = build_features(mutated, "C")
    x_d2, _ = build_features(mutated, "D")
    pd.testing.assert_frame_equal(x_c.tail(1), x_c2.tail(1))
    assert not x_d.tail(1).equals(x_d2.tail(1))
```

- [ ] **Step 2: Run it and verify it fails**

Run: `pytest backend/tests/test_features.py::test_tier_c_ignores_current_target_but_tier_d_uses_it -v`
Expected: FAIL.

- [ ] **Step 3: Implement the tiers**

Group by `building_id` before every shift/rolling. Tier C uses `.shift(1).rolling(w).mean()`; tier D uses `.rolling(w).mean()` including the current row. `build_features` returns `float32` X with `FEATURES[tier]` column order and drops warm-up rows still containing NaN lags.

- [ ] **Step 4: Add the availability report test**

```python
def test_availability_report_labels_every_feature():
    report = availability_report("C")
    assert set(report.columns) == {"feature", "available_at"}
    assert (report["available_at"] <= "t-1").all()
```

- [ ] **Step 5: Run the tests**

Run: `pytest backend/tests/test_features.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/ml/features.py backend/tests/test_features.py
git commit -m "feat(features): causal tiers and leakage guard"
```

---

### Task 3: Validation protocols

**Files:**
- Create: `backend/ml/splits.py`
- Create: `backend/tests/test_splits.py`

**Interfaces:**
- Produces: `random_folds(df, n_splits=5, seed=42)`, `blocked_temporal(df, cutoff="2017-07-01")`, `leave_one_group_out(df, group_col, n_folds=5, seed=42)`, each returning `list[tuple[np.ndarray, np.ndarray]]` of positional indices; `protocol_splits(df, protocol, n_folds=5, seed=42)` with `protocol in {"S1","S2","S3","S4"}`.

- [ ] **Step 1: Write the failing leak tests**

```python
def test_blocked_temporal_has_no_time_overlap():
    df = make_panel(hours=200)
    train, test = blocked_temporal(df)[0]
    assert df.iloc[train]["timestamp"].max() < df.iloc[test]["timestamp"].min()

def test_leave_one_site_out_never_shares_a_site():
    df = make_panel(buildings=6)
    for train, test in leave_one_group_out(df, "site_id"):
        assert set(df.iloc[train]["site_id"]).isdisjoint(set(df.iloc[test]["site_id"]))
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest backend/tests/test_splits.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the four protocols**

S1 = shuffled `KFold`; S2 = single cutoff pair; S3/S4 = `GroupKFold` over `building_id`/`site_id` (cap at `n_folds` only if runtime forces it). Raise `ValueError` for unknown protocol.

- [ ] **Step 4: Run the tests**

Run: `pytest backend/tests/test_splits.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/ml/splits.py backend/tests/test_splits.py
git commit -m "feat(splits): random, blocked, building-holdout, site-holdout"
```

---

### Task 4: Models and metric contract

**Files:**
- Create: `backend/ml/models.py`
- Create: `backend/ml/evaluate.py`
- Create: `backend/tests/test_evaluate.py`

**Interfaces:**
- Produces: `MODEL_NAMES = ("random_forest","adaboost","gbm","xgboost","lightgbm","catboost")`, `make_model(name, seed=42)`; `regression_metrics(y_true, y_pred, groups) -> dict` with keys `rmse, mae, r2, mape, n, r2_median, r2_iqr_low, r2_iqr_high`; `per_building_metrics(y_true, y_pred, groups) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing metric test**

```python
def test_regression_metrics_matches_hand_computed_rmse():
    m = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 4.0]), np.array(["a","a"]))
    assert m["rmse"] == pytest.approx(np.sqrt(2.0))
    assert m["mae"] == pytest.approx(1.0)
    assert m["mape"] == pytest.approx(50.0)
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest backend/tests/test_evaluate.py::test_regression_metrics_matches_hand_computed_rmse -v`
Expected: FAIL.

- [ ] **Step 3: Implement models and metrics**

RF: 100 trees, `max_depth=20`, `n_jobs=-1`. LightGBM: 300 estimators, lr 0.05, 31 leaves. XGBoost: 300, lr 0.05, depth 6. CatBoost: 500, depth 6, verbose off. GBM: 200, lr 0.05. AdaBoost: 100. Compute R² on pooled predictions; per-building R² from grouped sums of squares, never mean-of-means.

- [ ] **Step 4: Run the tests**

Run: `pytest backend/tests/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/ml/models.py backend/ml/evaluate.py backend/tests/test_evaluate.py
git commit -m "feat(eval): frozen model registry and pooled/per-building metrics"
```

---

### Task 5: Reproduction runner (base paper Table 14)

**Files:**
- Create: `backend/scripts/run_reproduction.py`
- Create: `backend/tests/test_reproduction_smoke.py`

**Interfaces:**
- Produces: `run_reproduction(panel, cohort: list[str] | None, out_dir) -> pd.DataFrame`; writes `reports/reproduction_metrics.csv` and `.json` with one row per model and columns `model, rmse, mae, r2, mape, train_seconds, n_train, n_test`.
- CLI: `python -m backend.scripts.run_reproduction [--smoke]`.

- [ ] **Step 1: Write the smoke test**

```python
def test_smoke_reproduction_writes_one_row_per_model(tmp_path):
    out = run_reproduction(make_panel(buildings=2, hours=400), cohort="repro", out_dir=tmp_path)
    assert set(out["model"]) == set(MODEL_NAMES)
    assert out["rmse"].notna().all()
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest backend/tests/test_reproduction_smoke.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the runner**

Use tier C and S2 on the 10-building reproduction cohort; log1p-fit then `expm1` before metrics; seed 42; write a `reproduction_vs_paper.md` table comparing our values to the paper's bands.

- [ ] **Step 4: Run tests, then the real run**

Run: `pytest backend/tests/test_reproduction_smoke.py -v` → PASS
Run: `python -m backend.scripts.run_reproduction` → writes the CSV; every model's RMSE must fall in 5–9 kWh and R² in 0.59–0.76. If outside, investigate leakage or cohort size before proceeding.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/run_reproduction.py backend/tests/test_reproduction_smoke.py reports/reproduction_metrics.csv reports/reproduction_metrics.json
git commit -m "feat(repro): six-model BDG2 chronological benchmark"
```

---

### Task 6: Novelty pilot (the lab-tomorrow fallback)

**Files:**
- Create: `backend/scripts/run_novelty.py`
- Create: `backend/ml/analysis.py`
- Create: `backend/tests/test_analysis.py`

**Interfaces:**
- Produces: `channel_decomposition(results: pd.DataFrame) -> pd.DataFrame` (columns `channel, delta_r2, model, tier`); `run_matrix(panel, cohort, models, protocols, tiers, folds) -> pd.DataFrame` with one row per fit (`model, protocol, tier, fold, rmse, mae, r2, mape, seconds`); CLI `--pilot` and `--out`.
- Consumes: Tasks 1–4.

- [ ] **Step 1: Write the failing decomposition test**

```python
def test_channel_decomposition_subtracts_matching_cells():
    r = pd.DataFrame([
        {"model":"lightgbm","protocol":"S1","tier":"C","r2":0.95},
        {"model":"lightgbm","protocol":"S2","tier":"C","r2":0.60},
        {"model":"lightgbm","protocol":"S3","tier":"C","r2":0.40},
        {"model":"lightgbm","protocol":"S4","tier":"C","r2":0.25},
    ])
    out = channel_decomposition(r).set_index("channel")["delta_r2"]
    assert out["temporal"] == pytest.approx(0.35)
    assert out["building"] == pytest.approx(0.20)
    assert out["site"] == pytest.approx(0.15)
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest backend/tests/test_analysis.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the matrix runner and analysis**

Pilot: 30 meters, `lightgbm` + `random_forest`, 3 folds for S3/S4. Full: 120 meters, 3 models, 5 folds. Write `reports/novelty_results.csv`, `reports/novelty_results.json`, `reports/figures/delta_r2_heatmap.png`, `reports/figures/per_building_r2.png`.

- [ ] **Step 4: Add the positive-control assertion**

```python
def test_leaky_tier_beats_causal_tier_in_random_split(results):
    c = results.query("protocol=='S1' and tier=='C' and model=='lightgbm'")["r2"].mean()
    d = results.query("protocol=='S1' and tier=='D' and model=='lightgbm'")["r2"].mean()
    assert d - c > 0.10, "positive control failed: harness is not detecting the deliberate leak"
```

- [ ] **Step 5: Run the pilot and inspect**

Run: `python -m backend.scripts.run_novelty --pilot`
Expected: 48–96 fits, a heatmap, and a printed summary naming the largest channel. Save the summary for tomorrow's lab even if the full run is not done.

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/run_novelty.py backend/ml/analysis.py backend/tests/test_analysis.py reports/novelty_results.csv reports/novelty_results.json reports/figures
git commit -m "feat(novelty): leakage-channel decomposition pilot"
```

---

### Task 7: Full novelty matrix

**Files:**
- Modify: `backend/scripts/run_novelty.py`
- Modify: `reports/novelty_results.*`

**Interfaces:**
- Produces: the full 3 models × 4 protocols × 4 tiers result set plus `reports/novelty_summary.md` (one paragraph per channel, largest first).

- [ ] **Step 1: Run the full matrix**

Run: `python -m backend.scripts.run_novelty`
Expected: completes under 90 minutes; positive control passes.

- [ ] **Step 2: Verify the headline claim is computable**

Run: `python -c "import json;d=json.load(open('reports/novelty_results.json'));print(d['channels'])"`
Expected: four channel values plus `total`, with `temporal + building + site` reconciling to the S1→S4 difference within 1e-6.

- [ ] **Step 3: Commit**

```bash
git add reports/novelty_results.csv reports/novelty_results.json reports/novelty_summary.md reports/figures
git commit -m "feat(novelty): full protocol x feature-tier decomposition"
```

---

### Task 8: Model artifact and schema export

**Files:**
- Create: `backend/ml/artifacts.py`
- Create: `backend/scripts/train_artifacts.py`
- Create: `backend/tests/test_artifacts.py`
- Modify: `backend/models/feature_schema.json`
- Create: `backend/models/metrics.json`
- Create: `backend/models/model_card.json`

**Interfaces:**
- Produces: `save_bundle(path, model, feature_names, tier, metrics, meta)`, `load_bundle(path) -> dict` with keys `model, feature_names, tier, metrics, meta`; CLI writes `backend/models/model.joblib` and the JSON files.

- [ ] **Step 1: Write the failing round-trip test**

```python
def test_bundle_round_trip(tmp_path):
    p = tmp_path/"m.joblib"
    save_bundle(p, "dummy", ["a","b"], "C", {"rmse": 1.0}, {"version": "test"})
    b = load_bundle(p)
    assert b["feature_names"] == ["a","b"] and b["tier"] == "C"
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest backend/tests/test_artifacts.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement export**

Train LightGBM on tier C / S2 over the reproduction cohort, then save bundle + schema (`features`, `tier`, `target`, `units`, `trained_at`) + metrics + model card (dataset, cohort, protocol, limitations, git SHA).

- [ ] **Step 4: Run tests and export**

Run: `pytest backend/tests/test_artifacts.py -v` → PASS
Run: `python -m backend.scripts.train_artifacts` → bundle < 20 MB, schema feature count matches `len(FEATURES["C"])`.

- [ ] **Step 5: Commit**

```bash
git add backend/ml/artifacts.py backend/scripts/train_artifacts.py backend/tests/test_artifacts.py backend/models
git commit -m "feat(model): export real LightGBM artifact and schema"
```

---

### Task 9: Flask API with real inference

**Files:**
- Modify: `backend/app.py`
- Modify: `backend/tests/test_api.py`
- Modify: `backend/tests/conftest.py`

**Interfaces:**
- Produces endpoints: `GET /health`, `GET /model-info`, `GET /examples?n=`, `POST /predict`, `POST /predict/batch`, `GET /novelty`.
- `/predict` request `{"features": {name: number}}`; response `{"prediction_kwh": float, "inference_mode": "model"|"demo", "model": str, "artifact_version": str}`.

- [ ] **Step 1: Write the failing API tests**

```python
def test_predict_rejects_missing_features(client, payload):
    payload["features"].pop(next(iter(payload["features"])))
    assert client.post("/predict", json=payload).status_code == 400

def test_predict_returns_model_mode_with_real_artifact(client, payload):
    r = client.post("/predict", json=payload).get_json()
    assert r["inference_mode"] == "model" and r["prediction_kwh"] > 0
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest backend/tests/test_api.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement with a fixture bundle**

`create_app(bundle_path=None)` loads the committed bundle; validation rejects missing/extra fields, non-numeric, NaN/inf. `/examples` returns 5 held-out rows from a committed `backend/models/examples.json`. Keep `DemoModel` only for tests without a bundle.

- [ ] **Step 4: Run all backend tests**

Run: `pytest backend/tests -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app.py backend/tests backend/models/examples.json
git commit -m "feat(api): real model inference, examples, model info"
```

---

### Task 10: Frontend data layer

**Files:**
- Modify: `frontend/src/app/prediction-api.service.ts`
- Create: `frontend/src/app/app.models.ts`
- Create: `frontend/public/data/metrics.json`
- Create: `frontend/public/data/novelty_results.json`
- Create: `frontend/public/data/model_card.json`

**Interfaces:**
- Produces: `health()`, `modelInfo()`, `examples(n)`, `predict(features)`, `novelty()`; `MetricsRow`, `NoveltyResults`, `ExampleRow` interfaces; `loadStatic<T>(path)` using `assets`-relative `data/*.json`.

- [ ] **Step 1: Write the failing service test**

```ts
it('falls back to static assets when the API is unreachable', async () => {
  http.get.and.returnValue(throwError(() => new Error('offline')));
  const data = await service.noveltyWithFallback();
  expect(data.channels.temporal).toBeGreaterThanOrEqual(0);
});
```

- [ ] **Step 2: Run and verify failure**

Run: `cd frontend && npm test -- --watch=false --browsers=ChromeHeadless`
Expected: FAIL (method missing).

- [ ] **Step 3: Implement the service and copy artifacts**

Copy `reports/novelty_results.json`, `backend/models/metrics.json`, `backend/models/model_card.json` into `frontend/public/data/` (a script `frontend/scripts/sync-data.mjs` or documented manual copy). Fetch static data first; attempt `/api` and replace on success; expose `apiAvailable` signal.

- [ ] **Step 4: Run tests and build**

Run: `cd frontend && npm test -- --watch=false --browsers=ChromeHeadless && npm run build`
Expected: PASS and build output under `dist/frontend/browser`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app frontend/public/data frontend/scripts
git commit -m "feat(ui): typed API/static data layer"
```

---

### Task 11: Frontend report and live-inference UI

**Files:**
- Modify: `frontend/src/app/app.component.ts`
- Modify: `frontend/src/app/app.component.html`
- Modify: `frontend/src/app/app.component.css`
- Modify: `frontend/src/app/app.component.spec.ts`

**Interfaces:**
- Consumes: Task 10 service.
- Produces sections: reproduction benchmark table, novelty heatmap + channel cards, live prediction panel, provenance/limitations. No hard-coded metric arrays.

- [ ] **Step 1: Write the failing component test**

```ts
it('renders the channel decomposition from data, not constants', () => {
  const el = fixture.nativeElement as HTMLElement;
  expect(el.textContent).toContain('Temporal autocorrelation');
  expect(el.textContent).toContain(component.channels[0].value.toFixed(3));
});
```

- [ ] **Step 2: Run and verify failure**

Run: `cd frontend && npm test -- --watch=false --browsers=ChromeHeadless`
Expected: FAIL.

- [ ] **Step 3: Implement the UI**

On init, `Promise.all([metrics, novelty, modelCard])`; render table/heatmap from data. Live panel: call `examples(5)`, let the user select a row, POST its features, show predicted vs actual and `inference_mode`. When `apiAvailable === false`, show "Static results — start the API for live inference" and disable the button with an explanation.

- [ ] **Step 4: Run tests and build**

Run: `cd frontend && npm test -- --watch=false --browsers=ChromeHeadless && npm run build`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app
git commit -m "feat(ui): benchmark, novelty decomposition, live inference panel"
```

---

### Task 12: Local end-to-end run and Pages deployment

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/pages.yml` (only if asset paths break)
- Modify: `README.md`

- [ ] **Step 1: Run both services locally**

Run: `python backend/app.py` and `cd frontend && npm start`
Expected: `/api/health` returns `inference_mode: "model"`; the UI shows real numbers and a successful live prediction.

- [ ] **Step 2: Verify the Pages build locally**

Run: `cd frontend && npm run build -- --base-href "/devproj/" && python -m http.server 8080 -d dist/frontend/browser`
Expected: page loads at `/devproj/` assets, renders JSON data, and shows the offline notice.

- [ ] **Step 3: Update CI**

`ci.yml`: install backend deps, `pytest backend/tests -q`, `python -m backend.scripts.run_reproduction --smoke` (or a fast fixture), `npm ci`, `npm run build`.

- [ ] **Step 4: Commit and push**

```bash
git add .github README.md
git commit -m "ci: test API and build Pages bundle with real artifacts"
git push origin main
```

- [ ] **Step 5: Watch the workflow and verify the live URL**

Run: `gh run list --limit 3` (or the Actions tab)
Expected: both workflows green; `https://muhammadtalhazafar10.github.io/devproj/` renders the new report.

---

### Task 13: Research write-up and citations

**Files:**
- Create: `docs/novelty.md`
- Create: `docs/limitations.md`
- Modify: `docs/model_provenance.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/novelty.md`**

Structure: problem (composite ΔR²), gap, method (four channels, tiers A–D, splits S1–S4), results (filled from `novelty_summary.md`), threats to validity, references. Cite only verified entries from `reports/BDG2 boosting literature review.md`; mark unverified ones as "not used."

- [ ] **Step 2: Write `docs/limitations.md`**

Cover: 10/120-meter sampling, no deep baselines, single meter type, log1p target, quantile-free evaluation, Pages static-only, artifact reproducibility.

- [ ] **Step 3: Update provenance**

Record dataset version, cohort rule, seeds, library versions, artifact hash, and the exact command to regenerate every number.

- [ ] **Step 4: Commit**

```bash
git add docs README.md
git commit -m "docs: novelty, limitations, reproducible provenance"
```

---

### Task 14: Lab presentation packet

**Files:**
- Create: `docs/lab-presentation.md`
- Create: `reports/figures/` (final figures)

- [ ] **Step 1: Write the slide outline**

Ten slides: title; base paper in one slide; what we reproduced + table; the leakage problem in the old notebook (before/after numbers); novelty claim; experimental design diagram; pilot/full results heatmap; per-building distribution; live demo; limitations + next steps + references.

- [ ] **Step 2: Add speaker notes with the three defensible sentences**

Use the exact wording in `reports/BDG2 boosting literature review.md` §"What you can say tomorrow without overclaiming", and the verification caveats.

- [ ] **Step 3: Dry-run the demo**

Run: start backend + frontend, load the Pages URL, click one example, confirm predicted vs actual renders.
Expected: no console errors; offline notice correct when the API is stopped.

- [ ] **Step 4: Commit**

```bash
git add docs/lab-presentation.md reports/figures
git commit -m "docs: lab presentation packet and final figures"
```

---

## Self-Review

**Spec coverage:** §3 goals map to Tasks 5 (reproduction), 6–7 (novelty), 8–9 (artifact/API), 10–12 (UI/deploy), 13–14 (evidence). §6 data → Task 1. §7 tiers → Task 2. §8 splits → Task 3. §9 metrics/analysis → Tasks 4, 6, 7. §10 architecture → Tasks 8–12. §11 deliverables → Tasks 12–14. §12 risks: disk handled by Task 1 parquet + `.gitignore`; runtime handled by pilot-first ordering and the fold-reduction rule; citation integrity handled in Task 13.

**Placeholder scan:** no TBD/unwritten steps; every task names its files, signatures, test assertions, and the command that proves it.

**Type consistency:** `build_features` returns `(X, y)` and is consumed identically in Tasks 5, 6, 8. `regression_metrics` keys are reused verbatim in the CSV/JSON writers and the frontend interfaces. Protocol IDs `S1–S4` and tier IDs `A–D` are used identically across Tasks 3, 6, 7, 10.

**Review Focus tests:** current-timestep leak → Task 2 Step 1; grouped leakage → Task 3 Step 1; log1p scale → Task 4 Step 1 and Task 5 Step 3; API field mismatch → Task 9 Step 1; Pages offline render → Task 10 Step 1 and Task 12 Step 2.
