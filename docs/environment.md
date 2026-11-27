# Environment and reproducibility notes

## Backend
- Python 3.11 or newer (developed on 3.11.2, CI uses 3.12).
- Key packages (see `backend/requirements.txt`): Flask, pandas, numpy, scikit-learn, LightGBM, XGBoost, joblib, pyarrow, pytest.
- Training scripts use 4 CPU threads. Do not run two training jobs at once: LightGBM and XGBoost slow down by orders of magnitude when threads are oversubscribed.

## Frontend
- Angular 19 standalone app in `frontend/`. Node 20 or newer (developed on Node 24, CI uses Node 20).
- The page reads static JSON from `frontend/public/` (copied with `python -m backend.scripts.sync_frontend_data`) and works on GitHub Pages without a backend.
- The backend address is set in `frontend/public/config.json` (`apiBase`). The dev server on port 4200 uses its `/api` proxy instead.

## CI assumptions
- Backend tests run without private files: the API tests build a small model inside the test.
- Frontend build is static and deployable to GitHub Pages. Flask is not deployed to Pages.

## Time and size
- `run_cls_reproduction` took about 85 minutes on 4 cores while other jobs ran (Gradient Boosting on 80,000 rows alone took 13 to 16 minutes per split).
- `data/processed/` is about 65 MB and `data/raw` links to a 700 MB download.
