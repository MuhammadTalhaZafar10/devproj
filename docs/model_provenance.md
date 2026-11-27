# Model and data provenance

## Task
Classify a building into Low, Medium or High electricity performance class from the shape of its hourly
electricity use. The class is a tercile of mean kWh per m2 per year over the 1,470-building cohort. The paper
(Raymand et al. 2023) does not publish its cutoffs, so terciles are our assumption.

## Data
- Source: Building Data Genome Project 2 (Miller et al. 2020, Scientific Data 7:368), Kaggle
  `claytonmiller/buildingdatagenomeproject2`: `electricity.csv`, `metadata.csv`, `weather.csv`.
- Cohort: electricity meters with at least 80 % non-missing hours, positive floor area and non-zero use:
  1,470 buildings, 18 sites.
- Building table: one row per building, 66 features from 2016-2017 (paper protocol).
- 100k table: 100,000 rows (a random sample of building-weeks, seed 42), 59 columns, 54 features. Each row is one
  week of one building and carries the building's class, so rows of one building are not independent.

## Features
Computed on each series divided by its own mean (hour-of-day and weekday profiles, day-period ratios, spread,
percentiles, peak and trough hour, daily load ratio, autocorrelation, zero fraction, correlation with air
temperature). Multiplying a series by a constant cannot change them, which a unit test enforces, so absolute
consumption and floor area never reach the model.

## Models
Random Forest (paper's tuned settings from its Appendix Table B.2) and Gradient Boosting (scikit-learn defaults) are
from the paper. LightGBM and XGBoost are not. The served model is LightGBM trained on a whole-building 80 % split of
the 100k table. `backend/models/model_bundle.joblib` follows the lab manual layout
(`pipeline`, `features`, `task`, `model_version`).

## Files that are not committed
`data/raw/` and `data/processed/` (regenerate with the scripts in the README) and `backend/models/*.joblib`
(generate with `run_cls_reproduction` then `export_bundle`). CI therefore tests the API with a small in-test model.

## Limits and ethics
- Research and teaching prototype. It is not an energy audit and must not drive investment or compliance decisions.
- Numbers are in the same band as the paper's, not an exact reproduction (different cohort and features).
- The class cutoffs are relative to this cohort, not absolute standards.
