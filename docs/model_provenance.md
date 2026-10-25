# Model and Schema Provenance

## Repository evidence used
- Notebook: `/home/runner/work/devproj/devproj/devfinal;.ipynb`
- Dataset archive: `/home/runner/work/devproj/devproj/BDG2_final_100k.zip`
- Derived schema artifact: `/home/runner/work/devproj/devproj/backend/models/feature_schema.json`

## What was discovered
- The notebook constructs a 3-class target (`Low`, `Medium`, `High`) from energy-consumption data.
- The ZIP contains `BDG2_final_100k.csv` with 301 columns total: 300 model input features + `target`.
- No deployable trained model artifact was found in the repository (`backend/models/model.joblib` or `.pkl`).

## API behavior
- `POST /predict` validates requests against the exact 300-feature schema from `feature_schema.json`.
- If no trained model artifact is present, the backend returns a **documented demo fallback prediction** (`inference_mode: demo`).
- Demo fallback is CI-safe and deterministic; it is not presented as real production inference.

## Ethics and usage disclaimer
- Predictions from demo mode are for integration/testing only.
- Do not use demo-mode outputs for high-stakes decisions.
- Any production inference claim requires a validated, versioned, reproducible trained artifact and an updated model card.
