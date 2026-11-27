# BDG2 building classification: reproducing Raymand et al. (2023)

Classifies buildings from the **Building Data Genome Project 2** into Low / Medium / High
electricity performance classes from the shape of their hourly consumption, then measures how much
the usual validation shortcuts inflate the score.

**Base paper:** Raymand, Najafi, Haghighat Mamaghani, Moazami, Rinaldi (2023). *Machine learning-based
estimation of buildings' characteristics employing electrical and chilled water consumption data:
Pipeline optimization.* Energy and Buildings 295, 113327. https://doi.org/10.1016/j.enbuild.2023.113327

Deployed page: https://muhammadtalhazafar10.github.io/devproj/

## What is in the repo

| Path | Purpose |
|---|---|
| `backend/cls/` | Features, cohort and labels, four classifiers, evaluation protocols, paper reference numbers |
| `backend/scripts/run_cls_reproduction.py` | Reproduction: paper protocol on 1,470 buildings, plus the 100,000-row table |
| `backend/scripts/run_cls_novelty.py` | Novelty: random rows vs whole buildings vs whole sites |
| `backend/app.py` | Flask API (`/health`, `/model-info`, `/results`, `/novelty`, `/examples`, `/predict`) |
| `frontend/` | Angular 19 page: paper-vs-ours table, split comparison, live classifier |
| `reports/` | `classification_results.json`, `cls_novelty_results.json`, `cls_examples.json` |
| `backend/ml/`, `reports/reproduction_metrics.*` | Earlier hourly-forecasting (regression) reproduction of Xie and Tummala (2025) |

## Data

`data/raw/` holds symlinks to the Kaggle BDG2 files (`electricity.csv`, `metadata.csv`, `weather.csv`),
fetched with `python -m backend.scripts.fetch_data`. Two tables are derived from them:

- **Building table**, 1,470 buildings x 66 features: the paper protocol (one row per building).
- **100k table**, 100,000 building-weeks x 59 columns (54 features): the course-scale dataset.

Class labels are terciles of mean kWh/m2/year. The paper does not publish its cutoffs and its code
repository is an empty placeholder, so this is an assumption. Every feature is computed on a series
divided by its own mean, so absolute consumption cannot leak into the label. A test enforces that.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python -m backend.scripts.fetch_data                 # BDG2 from Kaggle (about 260 MB)
python -m backend.scripts.run_cls_reproduction --rebuild
python -m backend.scripts.run_cls_novelty
python -m backend.scripts.sync_frontend_data         # copies result files into frontend/public
pytest backend/tests -q

python -m backend.app                                # API on http://127.0.0.1:5000
cd frontend && npm ci && npm start                   # page on http://localhost:4200 (proxies /api)
```

Without a trained artifact `/predict` answers 503. There is no fake fallback model.

## Honest limits

- Not the paper's 374 features or its exact cohort (1,494 buildings), so compare sizes, not decimals.
- Gradient Boosting uses scikit-learn defaults: the paper does not list its tuned values.
- GitHub Pages is static, so the live classifier needs the Flask API running locally.
