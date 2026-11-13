"""Train the reproducible BDG2 ordinal uncertainty-aware classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "backend" / "models" / "feature_schema.json"


def conformal_threshold(probabilities: np.ndarray, labels: np.ndarray, quantile: float) -> float:
    nonconformity = 1.0 - probabilities[np.arange(len(labels)), labels]
    return float(np.quantile(nonconformity, quantile, method="higher"))


def prediction_sets(probabilities: np.ndarray, threshold: float) -> list[list[int]]:
    sets = []
    for row in probabilities:
        prediction_set = [index for index, probability in enumerate(row) if probability >= 1.0 - threshold]
        sets.append(prediction_set or [int(np.argmax(row))])
    return sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "BDG2_final_100k.zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=float, default=0.2)
    args = parser.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    features = schema["features"]
    classes = schema["target_classes"]
    with zipfile.ZipFile(args.data) as archive:
        member = schema["source_member"]
        with archive.open(member) as csv_file:
            data = pd.read_csv(csv_file, usecols=features + [schema["target_column"]])

    data[features] = data[features].apply(pd.to_numeric, errors="coerce")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target = data[schema["target_column"]]
    if pd.api.types.is_numeric_dtype(target):
        labels = pd.to_numeric(target, errors="coerce").to_numpy()
        valid = np.isfinite(labels) & np.isin(labels, np.arange(len(classes)))
        data = data.loc[valid].reset_index(drop=True)
        labels = labels[valid].astype(int)
    else:
        mapping = {name: index for index, name in enumerate(classes)}
        data = data[data[schema["target_column"]].isin(classes)].reset_index(drop=True)
        labels = data[schema["target_column"]].map(mapping).to_numpy()
    values = data[features].to_numpy(dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        values, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x_train,
        y_train,
        test_size=args.calibration_size,
        random_state=args.seed,
        stratify=y_train,
    )
    estimator = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=args.seed,
    )
    estimator.fit(x_fit, y_fit)
    calibration_probabilities = estimator.predict_proba(x_cal)
    threshold = conformal_threshold(calibration_probabilities, y_cal, 0.9)
    test_probabilities = estimator.predict_proba(x_test)
    test_predictions = estimator.predict(x_test)
    sets = prediction_sets(test_probabilities, threshold)
    metrics = {
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_predictions)),
        "macro_f1": float(f1_score(y_test, test_predictions, average="macro")),
        "ordinal_mae": float(mean_absolute_error(y_test, test_predictions)),
        "log_loss": float(log_loss(y_test, test_probabilities, labels=list(range(len(classes))))),
        "conformal_coverage": float(np.mean([y in prediction_set for y, prediction_set in zip(y_test, sets)])),
        "mean_prediction_set_size": float(np.mean([len(prediction_set) for prediction_set in sets])),
    }

    models_dir = ROOT / "backend" / "models"
    models_dir.mkdir(exist_ok=True)
    artifact = {
        "estimator": estimator,
        "feature_names": features,
        "classes": classes,
        "conformal_threshold": threshold,
        "model_version": f"histgb-seed{args.seed}",
    }
    joblib.dump(artifact, models_dir / "model.joblib", compress=3)
    dataset_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    manifest = {
        "model_version": artifact["model_version"],
        "seed": args.seed,
        "feature_count": len(features),
        "rows": int(len(data)),
        "train_rows": int(len(x_fit)),
        "calibration_rows": int(len(x_cal)),
        "test_rows": int(len(x_test)),
        "dataset_sha256": dataset_hash,
        "conformal_threshold": threshold,
        "metrics": metrics,
    }
    (models_dir / "metrics.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
