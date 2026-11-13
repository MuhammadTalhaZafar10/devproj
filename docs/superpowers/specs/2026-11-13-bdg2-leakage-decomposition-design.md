# BDG2 Leakage-Decomposition Study — Design

**Date:** 2026-11-13
**Status:** Approved direction (novelty confirmed by project owner)
**Base paper:** Xie, Y., & Tummala, S. P. (2025). *Machine Learning for Sensor Analytics: A Comprehensive Review and Benchmark of Boosting Algorithms in Healthcare, Environmental, and Energy Applications.* Sensors, 25(23), 7294. https://doi.org/10.3390/s25237294
**Dataset:** Building Data Genome Project 2 (BDG2), Miller et al. 2020, *Scientific Data* 7:438. https://doi.org/10.1038/s41597-020-00712-x

---

## 1. Intent and why

The course requires (a) a successful implementation of the base paper with running code and benchmark-level models, (b) a locally deployed and GitHub-pushed frontend + backend, (c) a GitHub Pages frontend deployment, (d) a defensible research novelty with visible progress for tomorrow's lab, and (e) literature support with citations.

The base paper is a six-model benchmark of tree ensembles on sensor data, including a 10-building BDG2 energy-forecasting task. Its central methodological finding is that random cross-validation inflates R² to 0.90–0.99 on BDG2 while forward-chaining validation yields 0.24–0.34 — a ΔR² of 0.63–0.74 — and that sensor drift reverses model rankings.

The existing repository implements neither of those claims. Its notebook trains a Low/Medium/High classifier on a 100,000-row sample using rolling statistics that include the current timestep, a shuffled row-level split, and quantile bins computed over the full dataset. The reported 92.12% LightGBM accuracy is therefore a leakage artifact, not a benchmark. The web app displays those static numbers and the Flask API falls back to a `DemoModel`.

This project replaces that with an honest reproduction of the base paper and a novel, literature-grounded extension.

## 2. Novelty

**Claim:** the BDG2 literature reports a single composite random-versus-temporal validation gap. We decompose that gap into its separable channels — temporal autocorrelation, building identity, site/weather context, and feature construction — and measure which dominates, holding the model family fixed.

Supporting evidence (see `reports/BDG2 boosting literature review.md`):

- Only one post-2021 peer-reviewed paper benchmarks the six tree ensembles on BDG2: the base paper, on 10 buildings from a single site.
- No BDG2 study reports leave-one-site-out generalisation.
- No building-energy study prices the rolling-window leakage channel, despite the leakage taxonomy being well established (Kapoor & Narayanan 2023; Kaufman et al. 2012).
- Modern GBDTs are statistically indistinguishable on BDG2 (pairwise p > 0.05), so "a better booster" is not a viable novelty claim.

**Positive control:** an intentionally leaky feature tier must show sharply higher R² than its causal counterpart. If it does not, the harness is broken and no result is reported.

## 3. Goals

1. Reproduce the base paper's BDG2 regression benchmark with running code: six models (RF, AdaBoost, GBM, XGBoost, LightGBM, CatBoost), 70/30 chronological split, per-building metrics in the paper's bands (RMSE 5–9 kWh, R² 0.59–0.76, MAPE 8–13%).
2. Produce the novelty experiment: protocol × feature-availability matrix with ΔR² per channel and per-building distributions.
3. Ship a trained model artifact and a real inference backend (no demo fallback on the happy path).
4. Ship an Angular frontend that renders real metrics/novelty results and can call the backend live; deploy it to GitHub Pages.
5. Write the evidence: model provenance, limitations, novelty note with citations, and a lab presentation.

## 4. Non-goals

- Beating the base paper's accuracy.
- Full 53-million-row BDG2 census (scoped to a stratified panel).
- Hosting Flask publicly (GitHub Pages is static-only; backend runs locally and is documented).
- Deep-learning baselines.
- Production-grade security, authentication, or multi-tenant serving.

## 5. Anti-goals (would make this a failure even if it runs)

- Any headline metric produced with current-timestep energy in the features.
- Reporting a single pooled R² without per-building distributions.
- A Pages site that only shows hard-coded numbers with no reproducible artifact behind them.
- A novelty claim that is "LightGBM/CatBoost wins."
- Citations that were not verified (no invented references or metrics).

## 6. Data

**Source:** Kaggle `claytonmiller/buildingdatagenomeproject2`, fetched with `kagglehub` (anonymous download verified, 263 MB archive, 713 MB extracted).

Files used: `metadata.csv` (1,636 buildings, 19 sites), `weather.csv` (site-level hourly), `electricity_cleaned.csv` (wide: 1,578 meter columns × 17,544 hourly rows).

**Reproduction cohort:** 10 Panther-site buildings, one per each of five use types × 2, chosen by a deterministic rule (complete 2016-01-01→2017-12-31 coverage, then alphabetical building_id within use type). Document the exact list; the paper does not publish its 10 IDs, so matching is by construction and bands, not by identity.

**Novelty cohort:** 120 meters stratified by site (all 19 sites that have ≥2 eligible meters) and primary space usage, requiring ≥90% non-null hourly coverage over 2016–2017. Target column `y_t` = electricity consumption at hour t (kWh), rows with `y_t <= 0` excluded from MAPE.

**Target:** regression, one-hour-ahead, log1p-transformed target for fitting; metrics reported on the original kWh scale.

## 7. Feature-availability tiers (the treatment variable)

Every feature carries an availability timestamp relative to prediction time `t`. The tier is the only thing that changes between rows of the matrix.

| Tier | Name | Contents | Legitimate for `y_t`? |
|---|---|---|---|
| A | Context | calendar (hour/day-of-week/month sin+cos, is_weekend), site weather at `t` and `t-1`, building metadata (sqm, primary use, site EUI) | Yes |
| B | Causal lags | `y_{t-1}`, `y_{t-24}`, `y_{t-168}`, `y_{t-1}-y_{t-2}`, `y_{t-24}-y_{t-168}` | Yes |
| C | Causal rolling | mean/std/min/max over 6/12/24 h windows **ending at `t-1`** | Yes |
| D | Leaky rolling (positive control) | the same windows **including `t`** | **No — deliberate leak** |

Scalers and imputers are fit inside each training fold only.

## 8. Validation protocols (the other treatment variable)

| ID | Protocol | Question it answers |
|---|---|---|
| S1 | Random 5-fold over rows | What the field currently reports |
| S2 | Blocked chronological 70/30 (train ≤ 2017-06-30, test after) | Skill on future hours of seen buildings |
| S3 | Leave-one-building-out, 5 grouped folds | Skill on unseen buildings, same sites |
| S4 | Leave-one-site-out, grouped folds | Skill on unseen sites/weather regimes |

Models: LightGBM, Random Forest, CatBoost (CatBoost is the base paper's clean winner; RF is the bagging baseline). Model configurations frozen across cells.

## 9. Measurements and analysis

Primary metric: **pooled RMSE (kWh)** and **pooled R²** on held-out predictions. Secondary: MAE, MAPE (y > 0), per-building R² (median and IQR, including negatives), training time.

Channel decomposition (computed on identical cohorts and models):

- Temporal autocorrelation = R²(S1) − R²(S2)
- Building identity = R²(S2) − R²(S3)
- Site/weather = R²(S3) − R²(S4)
- Feature construction = R²(tier D) − R²(tier C), reported per split
- Total = R²(S1, tier C) − R²(S4, tier C)

Deliverables: 4×4 ΔR² heatmap, per-building violin plots, one CSV of every (model, split, tier, fold) result, and a `novelty_results.json` consumed by the frontend.

## 10. Engineering architecture

```
BDG2 raw (kagglehub)
   └─ backend/ml/data.py        cohort selection, panel building
      └─ features.py           tier A/B/C/D construction + availability guard
         └─ splits.py          S1..S4
            ├─ scripts/run_reproduction.py -> reports/reproduction_metrics.{csv,json}
            └─ scripts/run_novelty.py      -> reports/novelty_results.{csv,json}, figures
                                              └─ backend/models/artifacts/
                                                 lightgbm_a.joblib, feature_schema.json,
                                                 metrics.json, model_card.json
```

- **Backend:** Flask. `GET /health` (mode=model, artifact version), `GET /model-info`, `GET /examples` (real held-out rows with ground truth), `POST /predict` (single feature dict → kWh + interval-free point estimate), `POST /predict/batch`, `GET /novelty` (server-side results). Real artifact is required; `DemoModel` remains only as a CI-safe fallback and is labelled in the response.
- **Frontend:** Angular 19 standalone. `PredictionApiService` gains health/model-info/examples/predict/novelty calls. App sections: (1) reproduction benchmark table, (2) novelty decomposition heatmap + per-building distributions, (3) live inference panel using `/examples`, (4) provenance/limitations. Static build reads `assets/metrics.json` and `assets/novelty_results.json` so Pages always renders; API calls are attempted at runtime and degrade gracefully.
- **Deployment:** GitHub Actions builds with `--base-href /devproj/` and deploys `frontend/dist/frontend/browser`; backend runs locally via `python backend/app.py` on :5000 with the Angular proxy.
- **Artifacts policy:** model `.joblib` is committed only if < 20 MB; otherwise generated by `scripts/train_artifacts.py` and attached to a release. Decision recorded in `docs/model_provenance.md`.

## 11. Deliverables checklist (mapped to the five course requirements)

| # | Requirement | Deliverable |
|---|---|---|
| 1 | Base paper implemented, benchmark models | `scripts/run_reproduction.py`, metrics table vs paper Table 14, 6 models |
| 2 | Frontend + backend local, pushed | Flask app with real artifacts, Angular dev server via proxy, commits on `main` |
| 3 | Frontend on GitHub Pages | updated `pages.yml`, static assets bundled, live URL verified |
| 4 | Novelty progress for lab | pilot results (evening) + full matrix (overnight) + slides |
| 5 | Literature support | `docs/novelty.md`, reference list, `reports/BDG2 boosting literature review.md` |

## 12. Constraints and risks

- **Disk:** 6.5 GB free; raw data 713 MB, artifacts and caches must stay small. Delete the extracted raw cache after the canonical panel is written, or keep only the panel parquet (~100–300 MB).
- **Time:** one night. The pilot (30 meters) must run before the full matrix so there is always a result to present.
- **Compute:** no GPU. Cap the panel at 120 meters and RF at 100 trees; measure fit time and shrink if a full cell exceeds ~10 minutes.
- **Reproduction fidelity:** the paper does not publish its 10 building IDs or random seeds; report bands and exact protocol, not identity.
- **Pages cannot host Flask:** the deployed site must be honest about static results and offer live inference only when the API is reachable.
- **Citation integrity:** six entries in the literature report are marked unverified; they must not appear on slides as BDG2 studies.

## 13. Decisions left to the builder

- Exact 10-building reproduction list (rule above is binding; the sort key is not).
- Whether S3/S4 use 5 folds or 3 when runtime exceeds budget (prefer 5; document any reduction).
- Model hyperparameters: start from the base paper's defaults, tune nothing; freeze.
- Whether the live-inference panel ships before or after the novelty panel if time is short (novelty first).
