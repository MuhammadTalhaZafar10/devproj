# Environment and Reproducibility Notes

## Backend
- Python 3.12+
- Install from `/home/runner/work/devproj/devproj/backend/requirements.txt`

## Frontend
- Angular 19 standalone app under `/home/runner/work/devproj/devproj/frontend`
- Node 20+ recommended in CI

## CI assumptions
- Backend tests run without private files by using commit-safe schema/sample artifacts.
- Frontend build is static and deployable to GitHub Pages.
- Live Flask inference is intentionally not deployed to Pages.
