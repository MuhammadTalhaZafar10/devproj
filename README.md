# CyberGuard Malware Detection Platform

CyberGuard is a full-stack malware-risk analysis project built around a schema-validated Flask API and a responsive Angular security dashboard. The application analyzes a fixed set of 300 malware features and returns a risk classification from the available inference model.

## Live application

The Angular frontend is deployed to GitHub Pages:

**https://muhammadtalhazafar10.github.io/devproj/**

GitHub Pages hosts the static frontend only. The Flask API must be running locally or deployed separately for live health checks and predictions.

## Features

- CyberGuard-style security dashboard with responsive dark UI
- Hash-based Angular routing compatible with GitHub Pages
- Schema-driven validation for all 300 prediction features
- Sample payload loading for quick demonstrations
- Backend health and inference-mode indicators
- Threat analysis result panel with clear error states
- Flask `/health` and `/predict` endpoints
- Automated backend tests and Angular production builds with GitHub Actions
- Automatic GitHub Pages deployment from the `main` branch

## Project structure

```text
backend/                 Flask API and backend tests
backend/models/          Feature schema and model documentation
data/                    Sample request data
docs/                    Environment and model documentation
frontend/                Angular 19 standalone application
.github/workflows/       CI and GitHub Pages deployment workflows
```

## Technology stack

- **Frontend:** Angular 19, TypeScript, standalone components, reactive forms
- **Backend:** Python, Flask, pandas, NumPy
- **Testing:** pytest and Angular/Karma test tooling
- **Deployment:** GitHub Actions and GitHub Pages

## Backend API

### `GET /health`

Returns the API status, configured feature count, and inference mode.

### `POST /predict`

Accepts a JSON object containing exactly the 300 fields defined in `backend/models/feature_schema.json`.

The API returns `400 Bad Request` for:

- Invalid JSON
- Missing required fields
- Unexpected fields
- Non-numeric or non-finite values

If no trained model artifact is available, the API uses the documented deterministic demo model and identifies that mode in the response.

## Local development

### Run the backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/app.py
```

The backend runs at `http://127.0.0.1:5000`.

### Run the Angular frontend

```bash
cd frontend
npm ci
npm start
```

The Angular development server runs at `http://localhost:4200` and proxies `/api` requests to the local Flask service.

### Build for production

```bash
cd frontend
npm run build -- --configuration production
```

The Pages artifact is generated at `frontend/dist/frontend/browser`.

## Continuous integration and deployment

The repository includes two workflows:

- **CI:** runs backend tests and verifies the Angular production build on pushes and pull requests targeting `main`.
- **GitHub Pages:** builds the Angular application with the repository base path, verifies the browser artifact, and deploys it to GitHub Pages after changes reach `main`.

## Model and data notes

The project includes the derived 300-feature schema and safe sample payloads from the existing malware dataset and notebook context. No deployable trained model artifact is committed. For that reason, the backend uses demo inference until `backend/models/model.joblib` or `backend/models/model.pkl` is supplied.

Demo predictions are intended for integration and demonstration purposes. They should not be interpreted as validated production malware detections without reproducible training, model evaluation, versioned artifacts, and an appropriate security review.

## Security

- No secrets or tokens are committed.
- Local environments, dependency directories, build output, and unapproved model artifacts are excluded through `.gitignore`.
- Prediction inputs are strictly validated against the feature schema before inference.

## Maintainer

- Muhammad Talha Zafar
