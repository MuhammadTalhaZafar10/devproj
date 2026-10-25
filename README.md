# devproj - Week 6 DevOps Lab

This repository now includes a complete Week 6-style backend + frontend + CI/CD setup while preserving the existing notebook/dataset context.

## Deployed application

[Open the latest Angular GitHub Pages viewer for the Jupyter Notebook contents](https://muhammadtalhazafar10.github.io/devproj/)

## Repository structure

- `backend/` Flask API (`/health`, `/predict`)
- `backend/tests/` pytest coverage for API + required structure
- `backend/models/` safe schema artifact (`feature_schema.json`) and model notes
- `data/` non-sensitive sample request payload
- `docs/` provenance, environment, and limitations
- `frontend/` Angular 19 standalone app
- `.github/workflows/ci.yml` backend tests + frontend build on push/PR to `main`
- `.github/workflows/pages.yml` static Angular deployment to GitHub Pages

## Provenance and model context

- Existing notebook analyzed: `devfinal;.ipynb`
- Existing dataset archive analyzed: `BDG2_final_100k.zip` (`BDG2_final_100k.csv`)
- Derived schema: 300 feature columns + `target` label (`Low`/`Medium`/`High`)
- Generated artifact: `backend/models/feature_schema.json`

No deployable trained model artifact was present in this repository.  
Therefore, backend inference defaults to a **documented demo mode** unless `backend/models/model.joblib` (or `model.pkl`) is added.

## Backend API

### Endpoints

- `GET /health`  
  Returns API status, feature count, and inference mode (`model` or `demo`).

- `POST /predict`  
  Requires an exact JSON object matching all 300 schema fields.
  - `400` for invalid JSON, missing fields, extra fields, or non-numeric values
  - `500` for unexpected prediction-time failures

### Local run

```bash
cd /home/runner/work/devproj/devproj
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/app.py
```

### Backend tests

```bash
cd /home/runner/work/devproj/devproj
pip install -r backend/requirements.txt
pytest backend/tests -q
```

## Frontend (Angular)

The Angular app uses schema-driven JSON validation and calls `/api/health` + `/api/predict`.

Local development uses proxy forwarding:
- `/api/*` -> `http://127.0.0.1:5000/*`

### Local run

```bash
cd /home/runner/work/devproj/devproj/frontend
npm ci
npm start
```

### Production build

```bash
cd /home/runner/work/devproj/devproj/frontend
npm run build
```

For GitHub Pages, workflow builds with:

```bash
npm run build -- --base-href "/<repo-name>/"
```

## CI/CD behavior

### CI (`.github/workflows/ci.yml`)

On push/PR to `main`:
1. Install backend dependencies and run `pytest backend/tests -q`
2. Run `npm ci` and `npm run build` in `frontend/`

### Pages deployment (`.github/workflows/pages.yml`)

On push to `main`:
1. Build Angular frontend with repository base href
2. Upload static artifact `frontend/dist/frontend/browser`
3. Deploy to GitHub Pages using official Pages actions

## Important limitation: Pages is static-only

GitHub Pages cannot host Flask/Python inference.

- The deployed site is frontend-only.
- Live inference requires running Flask locally or hosting backend separately.
- The frontend handles backend-unavailable scenarios with clear messaging.

## Security and data handling

- No secrets/tokens are committed.
- `.gitignore` excludes virtual envs, node modules, build output, cache, env files, and unapproved model/data artifacts.
- Only commit-safe schema/sample artifacts are included for CI reproducibility.

## Research / ethics disclaimer

The default demo mode output is integration-oriented and must not be interpreted as validated real-world predictive performance.  
Any production use requires validated training reproducibility, versioned model artifact governance, and domain-appropriate risk review.
