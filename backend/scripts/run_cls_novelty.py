"""Novelty: how much does the validation protocol change a building-classification score?

Three models (RF, LightGBM, XGBoost) plus two baselines, four validation protocols:
  P1  random rows                    weeks of one building land in train and test
  P1b later weeks, same buildings    train on 2016 weeks, test on 2017 weeks
  P2  whole buildings                every test building is unseen (5 seeds)
  P3  whole sites                    leave one site out; sites with <5 buildings never tested
For P2 and P3 the tercile cutoffs are recomputed from the training buildings of each fold.
Scores are reported per row and per building (class probabilities averaged over a
building's test weeks), with a bootstrap interval over buildings.
Writes reports/cls_novelty_results.json (not in --quick mode).

Usage:  python -m backend.scripts.run_cls_novelty [--rows 30000] [--quick]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

from backend.cls import data as D
from backend.cls.evaluate import classification_metrics
from backend.cls.models import MODEL_INFO, MODEL_NAMES, make_model
from backend.cls.novelty import bootstrap_ci, building_level, tercile_labels

REPORTS = D.REPO_ROOT / "reports"
MIN_SITE_BUILDINGS = 5
BASELINES = {"baseline_prior": "Majority class", "baseline_use_type": "Use type only"}
# Gradient Boosting (scikit-learn) is single-threaded and ~10x slower than the other three on tens of
# thousands of rows; across about 35 folds it does not fit the time budget. It is still in the
# reproduction table (run_cls_reproduction.py), so the ladder uses the other three models.
MODELS_RUN = tuple(m for m in MODEL_NAMES if m != "gradient_boosting")
ALL_MODELS = MODELS_RUN + tuple(BASELINES)
DISPLAY = {**{k: v[0] for k, v in MODEL_INFO.items()}, **BASELINES}
PROTOCOLS = [
    ("P1", "Random rows", "Weeks from the same building land in train and test."),
    ("P1b", "Later weeks, same buildings", "Train on 2016 weeks, test on 2017 weeks of the same buildings."),
    ("P2", "Whole buildings", "Test buildings are unseen; their site is not. Mean and sd over seeds."),
    ("P3", "Whole sites", "Leave one site out; tercile cutoffs refitted on the training buildings."),
]


def f1w(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def fit_predict_proba(name: str, X: pd.DataFrame, X_use: pd.DataFrame, tr, te, y_tr) -> np.ndarray:
    """Fit one model (or baseline) on rows ``tr`` and return 3-column probabilities for ``te``."""
    if name == "baseline_prior":
        model, Xa = DummyClassifier(strategy="prior"), X
    elif name == "baseline_use_type":
        model, Xa = make_model("lightgbm"), X_use
    else:
        model, Xa = make_model(name), X
    model.fit(Xa.iloc[tr], y_tr)
    raw = model.predict_proba(Xa.iloc[te])
    out = np.zeros((len(te), len(D.CLASSES)))
    out[:, np.asarray(model.classes_, dtype=int)] = raw
    return out


def run_folds(folds, X, X_use, label_fn=None) -> dict:
    """folds: list of (tr, te, y_tr, y_te). Returns name -> {idx, y, proba, fold}."""
    out = {n: {"idx": [], "y": [], "proba": [], "fold": []} for n in ALL_MODELS}
    for k, (tr, te, y_tr, y_te) in enumerate(folds):
        for name in ALL_MODELS:
            proba = fit_predict_proba(name, X, X_use, tr, te, y_tr)
            r = out[name]
            r["idx"].append(te); r["y"].append(y_te); r["proba"].append(proba)
            r["fold"].append(np.full(len(te), k))
    return {n: {k: np.concatenate(v) for k, v in r.items()} for n, r in out.items()}


def score_run(B: pd.DataFrame, res: dict, boot_n: int) -> dict:
    """Row-level and building-level metrics (plus CI) for one run of one model."""
    idx, y, proba = res["idx"], res["y"], res["proba"]
    pred = proba.argmax(axis=1)
    m = classification_metrics(y, pred)
    df_test = B.iloc[idx]
    y_building = pd.Series(y).groupby(df_test["building_id"].to_numpy()).first()
    yb, pb = building_level(df_test, proba, y_building)
    lo, hi = bootstrap_ci(yb, pb, f1w, n=boot_n, seed=0)
    return {
        "accuracy": m["accuracy"], "f1_weighted": m["f1_weighted"], "f1_macro": m["f1_macro"],
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "building_accuracy": float(accuracy_score(yb, pb)), "building_f1_weighted": f1w(yb, pb),
        "ci_low": lo, "ci_high": hi, "test_buildings": int(len(yb)),
    }


def aggregate(runs: list[dict]) -> dict:
    """Mean over runs; sd of the row and building weighted F1 when there is more than one run."""
    keys = [k for k in runs[0] if k != "test_buildings"]
    out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    out["test_buildings"] = int(round(np.mean([r["test_buildings"] for r in runs])))
    if len(runs) > 1:
        out["f1_weighted_std"] = float(np.std([r["f1_weighted"] for r in runs]))
        out["building_f1_weighted_std"] = float(np.std([r["building_f1_weighted"] for r in runs]))
    return out


def relabel_fold(eui_rows, buildings, tr, te):
    """Labels for train and test rows from cutoffs fitted on the training buildings only."""
    train_eui = pd.Series(eui_rows[tr], index=buildings[tr])
    train_eui = train_eui[~train_eui.index.duplicated()]
    return (tercile_labels(train_eui, pd.Series(eui_rows[tr])),
            tercile_labels(train_eui, pd.Series(eui_rows[te])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=30_000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    n_folds, seeds, boot_n = (2, [0], 200) if args.quick else (3, [0, 1, 2, 3, 4], 1000)
    if args.quick:
        args.rows = 3000

    B = pd.read_parquet(D.PROCESSED_DIR / "bdg2_cls_100k.parquet")
    B = B.sample(n=min(args.rows, len(B)), random_state=42).reset_index(drop=True)
    A = pd.read_parquet(D.PROCESSED_DIR / "bdg2_cls_buildings.parquet")
    feats = D.feature_columns(B)
    X = B[feats]
    use = pd.get_dummies(B["primaryspaceusage"], dummy_na=True).astype("float32")
    use.columns = [f"use_{i}" for i in range(use.shape[1])]
    y_global = B["performance_class"].map({c: i for i, c in enumerate(D.CLASSES)}).to_numpy()
    buildings = B["building_id"].to_numpy()
    sites = B["site_id"].to_numpy()
    eui_rows = B["building_id"].map(A.set_index("building_id")["eui_kwh_m2_yr"]).to_numpy(dtype=float)
    pool = np.where(~np.isnan(eui_rows))[0]            # rows whose building is in Table A
    zeros = np.zeros(len(B))
    print(f"sample {X.shape}, buildings={len(set(buildings))}, sites={len(set(sites))}, "
          f"rows with Table A eui={len(pool)}")

    results: dict[str, dict[str, dict]] = {}   # protocol -> model -> aggregated metrics
    p3_runs: dict[str, dict] = {}

    def record(pid: str, runs_by_model: dict[str, list[dict]]):
        results[pid] = {n: aggregate(r) for n, r in runs_by_model.items()}
        for n in ALL_MODELS:
            a = results[pid][n]
            print(f"{pid:3s} {DISPLAY[n]:18s} row_f1={a['f1_weighted']:.3f} "
                  f"bldg_f1={a['building_f1_weighted']:.3f} [{a['ci_low']:.3f},{a['ci_high']:.3f}]")

    # P1: random rows, global label
    folds = [(tr, te, y_global[tr], y_global[te])
             for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=0).split(zeros, y_global)]
    res = run_folds(folds, X, X_use=use)
    record("P1", {n: [score_run(B, res[n], boot_n)] for n in ALL_MODELS})

    # P1b: train 2016, test 2017, global label
    year = B["week_start"].dt.year.to_numpy()
    tr, te = np.where(year == 2016)[0], np.where(year == 2017)[0]
    res = run_folds([(tr, te, y_global[tr], y_global[te])], X, X_use=use)
    record("P1b", {n: [score_run(B, res[n], boot_n)] for n in ALL_MODELS})

    # P2: whole buildings, cutoffs refit per fold, repeated over seeds
    runs_by_model = {n: [] for n in ALL_MODELS}
    for seed in seeds:
        folds = []
        splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        for a, b in splitter.split(zeros[pool], y_global[pool], buildings[pool]):
            tr, te = pool[a], pool[b]
            y_tr, y_te = relabel_fold(eui_rows, buildings, tr, te)
            folds.append((tr, te, y_tr, y_te))
        res = run_folds(folds, X, X_use=use)
        for n in ALL_MODELS:
            runs_by_model[n].append(score_run(B, res[n], boot_n))
        print(f"P2 seed {seed} done")
    record("P2", runs_by_model)

    # P3: leave one site out. Small sites stay in training and are never tested.
    per_site_buildings = pd.Series(buildings[pool]).groupby(sites[pool]).nunique()
    eligible = set(per_site_buildings[per_site_buildings >= MIN_SITE_BUILDINGS].index)
    excluded = sorted(set(per_site_buildings.index) - eligible)
    folds, fold_sites = [], []
    for a, b in LeaveOneGroupOut().split(zeros[pool], y_global[pool], sites[pool]):
        tr, te = pool[a], pool[b]
        site = sites[te[0]]
        if site not in eligible:
            continue
        y_tr, y_te = relabel_fold(eui_rows, buildings, tr, te)
        folds.append((tr, te, y_tr, y_te))
        fold_sites.append(str(site))
    res = run_folds(folds, X, X_use=use)
    record("P3", {n: [score_run(B, res[n], boot_n)] for n in ALL_MODELS})

    # Per-site table: LightGBM and the use-type baseline on each P3 fold
    per_site = []
    for k, site in enumerate(fold_sites):
        row = {"site": site}
        for tag, name in (("", "lightgbm"), ("use_type_", "baseline_use_type")):
            r = res[name]
            sel = r["fold"] == k
            df_test = B.iloc[r["idx"][sel]]
            y_b = pd.Series(r["y"][sel]).groupby(df_test["building_id"].to_numpy()).first()
            yb, pb = building_level(df_test, r["proba"][sel], y_b)
            row[f"{tag}accuracy"] = float(accuracy_score(yb, pb))
            row[f"{tag}f1_weighted"] = f1w(yb, pb)
            if not tag:
                row["buildings"] = int(len(yb))
        per_site.append(row)
    per_site.sort(key=lambda r: r["f1_weighted"])

    # Headline from the numbers: best real model by mean building-level F1 across protocols
    best = max(MODELS_RUN, key=lambda n: np.mean([results[p][n]["building_f1_weighted"] for p in results]))
    g = lambda p, n: results[p][n]["building_f1_weighted"]
    headline = (
        f"Building-level weighted F1 of the best model: P1 {g('P1', best):.2f}, P1b {g('P1b', best):.2f}, "
        f"P2 {g('P2', best):.2f} (sd {results['P2'][best].get('building_f1_weighted_std', 0.0):.2f} "
        f"over {len(seeds)} seeds), P3 {g('P3', best):.2f}. "
        f"A use-type-only baseline scores {g('P3', 'baseline_use_type'):.2f} under P3 "
        f"and the majority class {g('P3', 'baseline_prior'):.2f}."
    )

    protocols = []
    for pid, label, description in PROTOCOLS:
        models = [{"model": n, "display": DISPLAY[n], **results[pid][n]} for n in ALL_MODELS]
        protocols.append({"protocol": pid, "label": label, "description": description, "models": models})

    folds_detail = (
        f"P1: StratifiedKFold({n_folds}) on rows, seed 0. P1b: train on 2016 weeks, test on 2017 weeks. "
        f"P2: StratifiedGroupKFold({n_folds}) by building_id, seeds {seeds}; mean and sd over seeds, "
        f"confidence interval is the mean of the per-seed bounds. "
        f"P3: leave-one-site-out over {len(fold_sites)} test sites ({', '.join(sorted(fold_sites))}); "
        f"sites with fewer than {MIN_SITE_BUILDINGS} sampled buildings ({', '.join(map(str, excluded)) or 'none'}) "
        f"are always in training and never tested. "
        f"P2 and P3 refit the tercile cutoffs on each fold's training buildings; P1 and P1b use the table label. "
        f"Building-level scores average class probabilities over a building's test weeks; "
        f"interval is a 95% percentile bootstrap over buildings ({boot_n} resamples)."
    )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "headline": headline,
        "sample_rows": int(len(B)),
        "folds": n_folds,
        "seeds": seeds,
        "protocols": protocols,
        "per_site": per_site,
        "folds_detail": folds_detail,
        "notes": [
            f"{len(B):,}-row random sample (seed 42) of the 100,000-row table; rows of one building share a label.",
            "Row-level columns (accuracy, f1_weighted, f1_macro, balanced_accuracy) score every test week; "
            "building_* columns score one averaged prediction per building.",
            "Baselines use the same folds: 'Majority class' predicts the training prior, "
            "'Use type only' is LightGBM on one-hot primaryspaceusage.",
            "Differences between protocols mix several changes at once (unseen buildings, unseen sites, "
            "different class balance in test folds); this experiment does not separate them.",
            f"Headline model: {DISPLAY[best]} (highest mean building-level weighted F1 over the four protocols).",
            "Models: Random Forest, LightGBM, XGBoost. Gradient Boosting is in the reproduction table but not "
            "in this ladder (too slow for about 35 folds).",
            "Per-site rows use LightGBM and the use-type baseline on the P3 fold for that site.",
        ],
    }
    print("\n" + headline)
    if not args.quick:
        (REPORTS / "cls_novelty_results.json").write_text(json.dumps(result, indent=2))
        print("wrote reports/cls_novelty_results.json")


if __name__ == "__main__":
    main()
