"""Metrics and protocol runners for the classification study."""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit, cross_validate,
    train_test_split,
)

from backend.cls.models import MODEL_NAMES, make_model

LABELS = [0, 1, 2]


def classification_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": [float(v) for v in f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)],
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def fit_and_score(name: str, X_train, y_train, X_test, y_test, seed: int = 42):
    model = make_model(name, seed=seed)
    started = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - started
    out = classification_metrics(y_test, model.predict(X_test))
    out["fit_seconds"] = seconds
    return model, out


def cv_scores(name: str, X, y, seed: int = 42, folds: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    res = cross_validate(
        make_model(name, seed=seed), X, y, cv=cv,
        scoring={"f1": "f1_weighted", "acc": "accuracy"},
    )
    return {"f1_val": float(res["test_f1"].mean()), "acc_val": float(res["test_acc"].mean())}


def paper_protocol(X, y, seeds=(0, 1, 2, 3, 4), models=MODEL_NAMES) -> list[dict]:
    """20 % stratified random test split + 5-fold CV on the rest, repeated over seeds."""
    rows = []
    for name in models:
        runs = []
        for seed in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed
            )
            cv = cv_scores(name, X_tr, y_tr, seed=seed)
            _, test = fit_and_score(name, X_tr, y_tr, X_te, y_te, seed=seed)
            runs.append({**test, **cv})
        rows.append({"model": name, "runs": len(runs), **_aggregate(runs)})
    return rows


def _aggregate(runs: list[dict]) -> dict:
    out = {}
    for key in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
                "f1_macro", "f1_val", "acc_val", "fit_seconds"):
        vals = np.array([r[key] for r in runs], dtype=float)
        out[key] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
    out["confusion"] = np.sum([np.array(r["confusion"]) for r in runs], axis=0).tolist()
    out["f1_per_class"] = np.mean([r["f1_per_class"] for r in runs], axis=0).tolist()
    return out


def random_row_split(y, test_size: float = 0.2, seed: int = 42):
    """Row-level split: rows of one building land on both sides (the leaky way)."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(sss.split(np.zeros(len(y)), y))


def building_group_split(y, groups, seed: int = 42):
    """Whole buildings are held out: no building appears on both sides."""
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return next(sgk.split(np.zeros(len(y)), y, groups))


def run_split(X, y, split: tuple, models=MODEL_NAMES, seed: int = 42):
    train, test = split
    rows, fitted = [], {}
    for name in models:
        model, scores = fit_and_score(name, X.iloc[train], y[train], X.iloc[test], y[test], seed)
        rows.append({"model": name, **scores})
        fitted[name] = model
    return rows, fitted
