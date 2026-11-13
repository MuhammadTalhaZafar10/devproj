This folder stores backend inference artifacts that are safe to commit.

- `feature_schema.json` is derived from the repository dataset archive (`BDG2_final_100k.zip`).
- `train_model.py` creates a validated `model.joblib` and `metrics.json`.
- The current validated model artifact is committed so the container deployment
  can load it without depending on an unavailable local filesystem.
