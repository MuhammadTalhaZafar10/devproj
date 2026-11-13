# Research progress

## Methodological framing

This project adapts the ensemble-learning and explainable-AI comparison described
by the 2024 PLOS ONE study, rather than claiming an exact reproduction: the
dataset, feature engineering, and target task differ. The BDG2 paper is cited
as the source and provenance of the energy-meter data.

## Proposed contribution

The target classes have an order (`Low < Medium < High`). The implementation
adds an uncertainty-aware evaluation layer to a gradient-boosted baseline:

- a fixed stratified train/test split;
- a separate calibration split that is never used to fit the estimator;
- split-conformal prediction sets calibrated at 90% nominal coverage;
- accuracy, balanced accuracy, macro-F1, ordinal MAE, log loss, coverage, and
  prediction-set size;
- a reproducible model version, random seed, dataset SHA-256, and metrics
  manifest.

## Current run

The reproducible run on `BDG2_final_100k.zip` uses seed 42 and 300 schema
features. It produced 91.68% accuracy, 90.51% balanced accuracy, 0.8985
macro-F1, 0.1176 ordinal MAE, 0.1768 log loss, and 91.68% empirical conformal
coverage on the frozen 20,000-row test set. The artifact is intentionally
ignored by Git because it is a generated binary; rerun
`python backend/train_model.py` to regenerate it.

## Public backend deployment

The Flask service is containerized for Render with `Dockerfile` and
`render.yaml`. The validated `backend/models/model.joblib` is included in the deployment
branch so the container can load it. GitHub Pages cannot host this service.
Set `CORS_ORIGINS` to the Pages origin and set
`globalThis.__BDG2_API_URL__` before the Angular build when connecting the
deployed frontend to a hosted API. By default, the Pages build connects to
`127.0.0.1:5000`, allowing a presenter to run Flask locally and use the live
controls from the deployed page. Local development continues to use the
Angular `/api` proxy.

## References

- Miller et al. (2020), *Scientific Data*, DOI:
  [10.1038/s41597-020-00712-x](https://doi.org/10.1038/s41597-020-00712-x).
- *Advancing ensemble learning techniques for residential building electricity
  consumption forecasting: Insight from explainable artificial intelligence*
  (2024), *PLOS ONE*, DOI:
  [10.1371/journal.pone.0307654](https://doi.org/10.1371/journal.pone.0307654).
- Borrotti (2024), *Energies*, DOI:
  [10.3390/en17174348](https://doi.org/10.3390/en17174348).
- Yu, Chang, and Dong (2022), *Buildings*, DOI:
  [10.3390/buildings12081284](https://doi.org/10.3390/buildings12081284).
- Tian (2024), *Journal of Building Performance Simulation*, DOI:
  [10.1080/19401493.2024.2387071](https://doi.org/10.1080/19401493.2024.2387071).
