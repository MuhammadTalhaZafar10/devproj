This folder stores backend inference artifacts that are safe to commit.

- `feature_schema.json` is derived from the repository dataset archive (`BDG2_final_100k.zip`).
- No trained deployable model artifact is currently committed.
- If you add a validated trained model (for example `model.joblib`), the API will load it automatically.
