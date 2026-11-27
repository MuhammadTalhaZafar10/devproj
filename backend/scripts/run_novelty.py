"""Pilot/full leakage-decomposition matrix.

Usage:
    python -m backend.scripts.run_novelty --pilot
    python -m backend.scripts.run_novelty --n 120 --folds 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.ml.analysis import channel_decomposition, feature_construction_gap, summarize  # noqa: E402
from backend.ml.data import REPO_ROOT, read_panel  # noqa: E402
from backend.ml.evaluate import per_building_metrics, regression_metrics  # noqa: E402
from backend.ml.features import TIERS, build_features, check_causality  # noqa: E402
from backend.ml.models import make_model  # noqa: E402
from backend.ml.splits import protocol_splits  # noqa: E402

REPORTS = REPO_ROOT / "reports"
FIGURES = REPORTS / "figures"
PROTOCOLS = ("S1", "S2", "S3", "S4")


def run_matrix(
    panel: pd.DataFrame,
    models: tuple[str, ...] = ("lightgbm",),
    protocols: tuple[str, ...] = PROTOCOLS,
    tiers: tuple[str, ...] = TIERS,
    folds: int = 3,
    predict_sample: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    per_building: list[pd.DataFrame] = []
    for tier in tiers:
        started = time.time()
        if tier != "D":
            check_causality(panel.groupby("building_id", group_keys=False).head(400), tier)
        X, y = build_features(panel, tier)
        frame = panel.loc[X.index, ["timestamp", "building_id", "site_id"]]
        print(f"[tier {tier}] rows={len(X):,} features={X.shape[1]} in {time.time() - started:.1f}s")

        for protocol in protocols:
            splits = protocol_splits(frame, protocol, n_folds=folds)
            for fold, (train, test) in enumerate(splits):
                X_train, X_test = X.iloc[train], X.iloc[test]
                y_train, y_test = np.log1p(y.iloc[train]), y.iloc[test]
                groups = frame.iloc[test]["building_id"].to_numpy()
                for name in models:
                    model = make_model(name)
                    started = time.time()
                    model.fit(X_train, y_train)
                    seconds = time.time() - started
                    pred = np.expm1(model.predict(X_test))
                    metrics = regression_metrics(y_test.to_numpy(), pred, groups)
                    rows.append(
                        {
                            "model": name,
                            "protocol": protocol,
                            "tier": tier,
                            "fold": fold,
                            "rmse": metrics["rmse"],
                            "mae": metrics["mae"],
                            "r2": metrics["r2"],
                            "mape": metrics["mape"],
                            "n_test": metrics["n"],
                            "seconds": seconds,
                        }
                    )
                    pb = per_building_metrics(y_test.to_numpy(), pred, groups)
                    pb["model"], pb["protocol"], pb["tier"], pb["fold"] = name, protocol, tier, fold
                    per_building.append(pb)
                    print(
                        f"  {name:10s} {protocol} {tier} fold{fold} "
                        f"r2={metrics['r2']:.3f} rmse={metrics['rmse']:.2f} ({seconds:.1f}s)"
                    )
    return pd.DataFrame(rows), pd.concat(per_building, ignore_index=True)


def _plot_heatmap(channels: pd.DataFrame, path: Path) -> None:
    models = sorted(channels["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.6), squeeze=False)
    for ax, model in zip(axes[0], models):
        part = channels[channels["model"] == model].pivot(
            index="tier", columns="channel", values="delta_r2"
        )
        order = [c for c in ["temporal", "building", "site", "total"] if c in part.columns]
        matrix = part[order]
        im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.1, vmax=1.0)
        ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_title(model)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="ΔR²")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_distributions(per_building: pd.DataFrame, path: Path) -> None:
    if per_building.empty:
        return
    part = per_building[per_building["protocol"].isin(["S2", "S3", "S4"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, data = [], []
    for (tier, protocol), group in part.groupby(["tier", "protocol"]):
        labels.append(f"{tier}/{protocol}")
        data.append(group["r2"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("per-building R²")
    ax.set_title("Building-level distributions by tier and protocol")
    ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--models", default=None)
    args = parser.parse_args()

    panel = read_panel()
    if args.pilot:
        keep = sorted(panel["building_id"].unique())[:30]
        panel = panel[panel["building_id"].isin(keep)].copy()
        models = ("lightgbm",)
        folds = 3
        suffix = "pilot"
    else:
        if args.n < panel["building_id"].nunique():
            keep = sorted(panel["building_id"].unique())[: args.n]
            panel = panel[panel["building_id"].isin(keep)].copy()
        models = tuple(args.models.split(",")) if args.models else ("lightgbm", "catboost")
        folds = args.folds
        suffix = "full"

    result, per_building = run_matrix(panel, models=models, folds=folds)
    REPORTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS / f"novelty_results_{suffix}.csv", index=False)
    per_building.to_csv(REPORTS / f"novelty_per_building_{suffix}.csv", index=False)

    channels = channel_decomposition(result)
    feature_gap = feature_construction_gap(result)
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "suffix": suffix,
        "n_buildings": int(panel["building_id"].nunique()),
        "n_rows": int(len(panel)),
        "models": list(models),
        "folds": folds,
        "channels": summarize(result),
        "feature_construction": feature_gap.to_dict(orient="records"),
        "cells": (
            result.groupby(["model", "protocol", "tier"], as_index=False)
            .agg(r2=("r2", "mean"), rmse=("rmse", "mean"), mape=("mape", "mean"))
            .to_dict(orient="records")
        ),
    }
    (REPORTS / f"novelty_results_{suffix}.json").write_text(json.dumps(payload, indent=2))

    _plot_heatmap(channels, FIGURES / f"delta_r2_heatmap_{suffix}.png")
    _plot_distributions(per_building, FIGURES / f"per_building_r2_{suffix}.png")

    c = channels[(channels["model"] == models[0]) & (channels["tier"] == "C")].set_index("channel")
    print("\nChannel decomposition (tier C, first model):")
    print(c["delta_r2"].to_string())
    print(f"\nwrote reports/novelty_results_{suffix}.json")


if __name__ == "__main__":
    main()
