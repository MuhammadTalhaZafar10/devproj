"""Bounded, leakage-free attempt to beat the paper's performance-class F1 on the building table.

Protocol = the paper's: random stratified 80/20 split, tuning only inside 5-fold CV on the
80 %. Five pre-specified seeds (0-4); we report the mean and sd over all five, never the best.
Variants (each adds one thing, nothing uses kWh level or floor area):
  V0  paper-style Random Forest on the 66 scale-free features (our reproduction)
  V1  V0 + primaryspaceusage one-hot (disclosed extra input)
  V2  V1, Random Forest tuned by 8-draw randomized search inside the training split (budget cut after a first
      run showed 20 draws took 8 minutes per seed; seed 0 gave no gain at either budget)
  V3  V1, ordinal: regress log(kWh/m2/yr), cut at the TRAINING terciles
Usage: python -m backend.scripts.run_cls_improve
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from backend.cls import data as D
from backend.cls.models import make_model

SEEDS = (0, 1, 2, 3, 4)
PAPER = {"rf_all_features_f1": 0.604, "rf_selected_f1": 0.616, "best_pipeline_f1": 0.622}


def f1w(y, p) -> float:
    return float(f1_score(y, p, average="weighted", zero_division=0))


def main() -> None:
    A = pd.read_parquet(D.PROCESSED_DIR / "bdg2_cls_buildings.parquet")
    feats = D.feature_columns(A)
    y = A["performance_class"].map({c: i for i, c in enumerate(D.CLASSES)}).to_numpy()
    X0 = A[feats]
    use = pd.get_dummies(A["primaryspaceusage"], prefix="use", dummy_na=True).astype(float)
    X1 = pd.concat([X0, use], axis=1)
    log_eui = np.log(A["eui_kwh_m2_yr"].to_numpy())

    rows: dict[str, list[tuple[float, float]]] = {k: [] for k in ("V0", "V1", "V2", "V3")}
    for seed in SEEDS:
        idx = np.arange(len(A))
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)

        for key, X in (("V0", X0), ("V1", X1)):
            m = make_model("random_forest", seed=seed).fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[te])
            rows[key].append((f1w(y[te], p), accuracy_score(y[te], p)))

        search = RandomizedSearchCV(
            make_model("random_forest", seed=seed),
            {
                "n_estimators": randint(150, 350),
                "max_features": uniform(0.1, 0.5),
                "min_samples_leaf": randint(1, 8),
                "min_samples_split": randint(2, 16),
                "criterion": ["gini", "entropy"],
            },
            n_iter=8, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            scoring="f1_weighted", random_state=seed, n_jobs=1,
        ).fit(X1.iloc[tr], y[tr])
        p = search.predict(X1.iloc[te])
        rows["V2"].append((f1w(y[te], p), accuracy_score(y[te], p)))

        reg = RandomForestRegressor(
            n_estimators=400, max_features=0.3, min_samples_leaf=3, n_jobs=4, random_state=seed
        ).fit(X1.iloc[tr], log_eui[tr])
        cuts = np.quantile(log_eui[tr], [1 / 3, 2 / 3])
        p = np.digitize(reg.predict(X1.iloc[te]), cuts)
        rows["V3"].append((f1w(y[te], p), accuracy_score(y[te], p)))
        print(f"seed {seed}: " + "  ".join(f"{k}={v[-1][0]:.3f}" for k, v in rows.items()), flush=True)

    labels = {
        "V0": "Random Forest, scale-free features (reproduction)",
        "V1": "V0 plus use type as an input",
        "V2": "V1 with Random Forest tuned inside cross-validation",
        "V3": "V1, ordinal: regress log kWh/m2, cut at training terciles",
    }
    out = []
    for k, vals in rows.items():
        f1s = np.array([v[0] for v in vals]); accs = np.array([v[1] for v in vals])
        out.append({"variant": k, "label": labels[k], "f1_mean": float(f1s.mean()), "f1_sd": float(f1s.std()),
                    "acc_mean": float(accs.mean()), "per_seed_f1": [float(x) for x in f1s]})
        print(f"{k}  F1 {f1s.mean():.3f} ± {f1s.std():.3f}  acc {accs.mean():.3f}  {labels[k]}")
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_reference": PAPER,
        "seeds": list(SEEDS),
        "protocol": "random stratified 80/20 split of 1,470 buildings; tuning only inside 5-fold CV on the training 80 %",
        "variants": out,
        "rule": "Report the mean of all five seeds. 'Beats the paper' only if mean exceeds 0.622 by more than one sd.",
    }
    (D.REPO_ROOT / "reports" / "cls_improvement.json").write_text(json.dumps(result, indent=2))
    print("wrote reports/cls_improvement.json")


if __name__ == "__main__":
    main()
