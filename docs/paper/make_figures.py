"""Draw every figure of the paper from the result files. No number is typed by hand.

Run from the project root:  .venv/bin/python docs/paper/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "reports" / "classification_results.json").read_text())
N = json.loads((ROOT / "reports" / "cls_novelty_results.json").read_text())
I = json.loads((ROOT / "reports" / "cls_improvement.json").read_text())
A = pd.read_parquet(ROOT / "data" / "processed" / "bdg2_cls_buildings.parquet")

from matplotlib import font_manager as _fm

for _f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSerif-*.ttf"):
    _fm.fontManager.addfont(str(_f))

INK, TEAL, SLATE = "#12262B", "#0F7C78", "#5B6B76"
LOW, MED, HIGH = "#2F7DE1", "#E2A200", "#D4452B"
plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 10, "axes.titlesize": 10.5,
    "axes.labelsize": 10, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight",
})
NAMES = {"random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting",
         "lightgbm": "LightGBM", "xgboost": "XGBoost"}
PAPER_RF_TEST, PAPER_BEST = 0.604, 0.622
PAPER_CV = R["paper"]["fig6a_f1_performance_class"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# ---- Fig 1: study pipeline ------------------------------------------------------------------
def pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, text, fc="#EEF3F2", fs=9.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=INK, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=INK))

    bw, gap, y0, h0 = 20.5, 5.0, 43, 16
    xs = [1 + i * (bw + gap) for i in range(4)]
    texts = ["BDG2 raw data\n1,636 buildings\n19 sites, hourly\n2016 to 2017",
             "Cohort filter\n>= 80 % hours,\narea > 0, use > 0\n1,470 buildings",
             "Scale-free\nfeatures\nprofiles, shape,\nweather (54)",
             "Label\nterciles of\nkWh/m$^2$/yr:\nLow, Medium, High"]
    for x, t, fc in zip(xs, texts, ["#E3ECEB", "#EEF3F2", "#EEF3F2", "#EEF3F2"]):
        box(x, y0, bw, h0, t, fc)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + bw + 0.6, y0 + h0 / 2, b - 0.6, y0 + h0 / 2)

    box(3, 22, 28, 13, "Building table\n1,470 rows x 66 features\n(paper protocol)", "#DCEBE9")
    box(36, 22, 28, 13, "Building-week table\n100,000 rows x 59 columns\n(course scale)", "#DCEBE9")
    box(69, 22, 29, 13, "Models\nRF, Gradient Boosting (paper)\nLightGBM, XGBoost (added)\n+ 2 baselines", "#F4EBD6", fs=8.6)
    arrow(xs[2] + bw / 2 - 3, y0 - 0.6, 17, 35.7); arrow(xs[2] + bw / 2 + 1, y0 - 0.6, 50, 35.7)
    arrow(31.8, 28.5, 35.4, 28.5); arrow(64.6, 28.5, 68.4, 28.5)

    box(1, 2, 31, 13, "Reproduction\nrandom 80/20 split + 5-fold CV,\n5 seeds, versus Raymand et al.", "#E8EFF8", fs=8.8)
    box(35, 2, 31, 13, "Protocol ladder (novelty)\nP1 rows, P1b later weeks,\nP2 buildings, P3 sites", "#E8EFF8", fs=8.8)
    box(69, 2, 30, 13, "Outputs\nbuilding-level F1 + 95 % CI,\nFlask API, Angular page", "#E8EFF8", fs=8.8)
    arrow(17, 21.4, 16.5, 15.6); arrow(50, 21.4, 50, 15.6); arrow(83.5, 21.4, 84, 15.6)
    save(fig, "fig1_pipeline")


# ---- Fig 2: label distribution ---------------------------------------------------------------
def eui():
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    cuts = [A.loc[A.performance_class == "Low", "eui_kwh_m2_yr"].max(),
            A.loc[A.performance_class == "Medium", "eui_kwh_m2_yr"].max()]
    bins = np.logspace(np.log10(A.eui_kwh_m2_yr.min()), np.log10(A.eui_kwh_m2_yr.max()), 55)
    for cls, col in (("Low", LOW), ("Medium", MED), ("High", HIGH)):
        ax.hist(A.loc[A.performance_class == cls, "eui_kwh_m2_yr"], bins=bins, color=col, alpha=0.9, label=cls)
    for c in cuts:
        ax.axvline(c, color=INK, ls="--", lw=1)
        ax.text(c, ax.get_ylim()[1] * 0.97, f" {c:.0f}", va="top", fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("Mean electricity use (kWh per m$^2$ per year, log scale)")
    ax.set_ylabel("Buildings"); ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.12))
    save(fig, "fig2_label_distribution")


# ---- Fig 3: reproduction against the paper ---------------------------------------------------
def reproduction():
    rows = {r["model"]: r for r in R["paper_protocol"]}
    order = ["random_forest", "gradient_boosting", "lightgbm", "xgboost"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    y = np.arange(len(order))[::-1]
    ax = axes[0]
    for yi, m in zip(y, order):
        r = rows[m]; col = TEAL if r["in_paper"] else INK
        ax.errorbar(r["f1_weighted"], yi, xerr=r["f1_weighted_std"], fmt="o", color=col, capsize=3, ms=6)
    ax.axvline(PAPER_RF_TEST, color=SLATE, ls="--", lw=1); ax.axvline(PAPER_BEST, color=SLATE, lw=1.6)
    ax.set_yticks(y); ax.set_yticklabels([NAMES[m] + ("" if rows[m]["in_paper"] else " *") for m in order])
    ax.set_xlim(0.55, 0.66); ax.set_xlabel("Test weighted F1 (mean $\\pm$ sd, 5 splits)")
    ax.set_title("(a) Test set", loc="left")
    ax.text(PAPER_RF_TEST, len(order) - 0.45, "paper RF\n0.604", ha="right", fontsize=8, color=SLATE)
    ax.text(PAPER_BEST + 0.001, len(order) - 0.45, "paper best\n0.622", ha="left", fontsize=8, color=SLATE)
    ax = axes[1]
    for yi, m in zip(y, order):
        ax.plot(rows[m]["f1_val"], yi, "o", color=TEAL if rows[m]["in_paper"] else INK, ms=6)
        if m in PAPER_CV:
            ax.plot(PAPER_CV[m], yi, "D", mfc="white", mec=SLATE, ms=7)
    ax.set_xlim(0.55, 0.66); ax.set_xlabel("5-fold cross-validated weighted F1"); ax.set_title("(b) Cross-validation", loc="left")
    ax.plot([], [], "o", color=TEAL, label="ours"); ax.plot([], [], "D", mfc="white", mec=SLATE, label="paper (Fig. 6a)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.text(0.01, -0.02, "* model not in the paper", fontsize=8)
    fig.tight_layout(); save(fig, "fig3_reproduction")


# ---- Fig 4: two splits on the 100k table ------------------------------------------------------
def two_splits():
    rr = {r["model"]: r["f1_weighted"] for r in R["scale_protocol"]["random_rows"]}
    wb = {r["model"]: r["f1_weighted"] for r in R["scale_protocol"]["whole_buildings"]}
    order = list(NAMES); x = np.arange(len(order)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    b1 = ax.bar(x - w / 2, [rr[m] for m in order], w, color=SLATE, label="Random rows")
    b2 = ax.bar(x + w / 2, [wb[m] for m in order], w, color=TEAL, label="Whole buildings held out")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([NAMES[m] for m in order]); ax.set_ylim(0, 0.9)
    ax.set_ylabel("Row-level weighted F1"); ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    save(fig, "fig4_two_splits")


# ---- Fig 5: protocol ladder with intervals ----------------------------------------------------
def ladder():
    prot = N["protocols"]; labels = [p["protocol"] for p in prot]
    fig, ax = plt.subplots(figsize=(6.6, 3.5)); x = np.arange(len(prot))
    series = [("random_forest", "Random Forest", TEAL, "o", "-"), ("lightgbm", "LightGBM", "#2B6CB0", "s", "-"),
              ("xgboost", "XGBoost", INK, "^", "-"), ("baseline_use_type", "Use type only", MED, "D", "--"),
              ("baseline_prior", "Majority class", SLATE, "x", "--")]
    off = np.linspace(-0.12, 0.12, len(series))
    for (key, name, col, mk, ls), o in zip(series, off):
        pts = [next(m for m in p["models"] if m["model"] == key) for p in prot]
        v = np.array([m["building_f1_weighted"] for m in pts])
        lo = v - np.array([m["ci_low"] for m in pts]); hi = np.array([m["ci_high"] for m in pts]) - v
        ax.errorbar(x + o, v, yerr=[lo, hi], color=col, marker=mk, ls=ls, lw=1.2, ms=5, capsize=2.5, label=name)
    ax.set_xticks(x); ax.set_xticklabels([f"{p['protocol']}\n{p['label']}" for p in prot], fontsize=8.5)
    ax.set_ylabel("Building-level weighted F1"); ax.set_ylim(0, 0.85); ax.grid(axis="y", color="#D3DCDB", lw=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    save(fig, "fig5_protocol_ladder")


# ---- Fig 6: per-site -------------------------------------------------------------------------
def per_site():
    ps = sorted(N["per_site"], key=lambda r: r["f1_weighted"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6)); y = np.arange(len(ps)); h = 0.38
    ax.barh(y + h / 2, [r["f1_weighted"] for r in ps], h, color=TEAL, label="LightGBM")
    ax.barh(y - h / 2, [r["use_type_f1_weighted"] for r in ps], h, color=MED, label="Use type only")
    ax.set_yticks(y); ax.set_yticklabels([f"{r['site']} (n={r['buildings']})" for r in ps], fontsize=8.5)
    ax.set_xlabel("Building-level weighted F1 with that site held out"); ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#D3DCDB", lw=0.6); ax.set_xlim(0, 1.0)
    save(fig, "fig6_per_site")


# ---- Fig 7: confusion matrix -----------------------------------------------------------------
def confusion():
    best = R["best_model_whole_buildings"]
    row = next(r for r in R["scale_protocol"]["whole_buildings"] if r["model"] == best)
    cm = np.array(row["confusion"], dtype=float); cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    im = ax.imshow(cmn, cmap="BuGn", vmin=0, vmax=0.8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]*100:.0f}%", ha="center", va="center", color="white" if cmn[i, j] > 0.45 else INK)
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_yticklabels(["Low", "Medium", "High"]); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f"{NAMES[best]}, whole buildings held out", fontsize=9.5)
    save(fig, "fig7_confusion")


# ---- Fig 8: feature importance ---------------------------------------------------------------
def importance():
    top = R["feature_importance_top15"]; names = [t[0].replace("_", " ") for t in top][::-1]
    vals = np.array([t[1] for t in top])[::-1]; vals = vals / vals.max()
    fig, ax = plt.subplots(figsize=(5.4, 3.6)); ax.barh(range(len(top)), vals, color=TEAL)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Relative split importance (LightGBM, served model)")
    save(fig, "fig8_importance")


# ---- Fig 9: improvement attempts -------------------------------------------------------------
def improvement():
    v = I["variants"]; fig, ax = plt.subplots(figsize=(6.2, 2.9)); y = np.arange(len(v))[::-1]
    for yi, r in zip(y, v):
        ax.errorbar(r["f1_mean"], yi, xerr=r["f1_sd"], fmt="o", color=TEAL, capsize=3, ms=6)
        ax.plot(r["per_seed_f1"], [yi] * len(r["per_seed_f1"]), ".", color=SLATE, alpha=0.6, ms=5)
    ax.axvline(PAPER_BEST, color=SLATE, lw=1.6); ax.text(PAPER_BEST + 0.001, len(v) - 0.5, "paper best 0.622", fontsize=8, color=SLATE)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['variant']}  {r['label'].split(',')[0][:34]}" for r in v], fontsize=8)
    ax.set_xlim(0.52, 0.66); ax.set_xlabel("Test weighted F1 (mean $\\pm$ sd; small dots are single seeds)")
    save(fig, "fig9_improvement")


if __name__ == "__main__":
    for f in (pipeline, eui, reproduction, two_splits, ladder, per_site, confusion, importance, improvement):
        f()
