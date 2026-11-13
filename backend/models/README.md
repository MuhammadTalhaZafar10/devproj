This folder stores backend inference artifacts that are safe to commit.

- `feature_schema.json` is derived from the repository dataset archive (`BDG2_final_100k.zip`).
- `train_model.py` creates a validated local `model.joblib` and `metrics.json`.
- Generated binary artifacts remain ignored by Git; rerun the training command
  from the repository root to regenerate them.
