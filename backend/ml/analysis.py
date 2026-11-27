"""Channel decomposition of the random-vs-temporal validation gap."""
from __future__ import annotations

import pandas as pd


def _cell(results: pd.DataFrame, model: str, protocol: str, tier: str) -> float:
    rows = results.query(
        "model == @model and protocol == @protocol and tier == @tier"
    )["r2"]
    if rows.empty:
        raise KeyError(f"missing cell {model}/{protocol}/{tier}")
    return float(rows.mean())


def channel_decomposition(results: pd.DataFrame) -> pd.DataFrame:
    """Decompose the total inflation into four additive channels."""
    rows = []
    for model in sorted(results["model"].unique()):
        for tier in sorted(results["tier"].unique()):
            s1 = _cell(results, model, "S1", tier)
            s2 = _cell(results, model, "S2", tier)
            s3 = _cell(results, model, "S3", tier)
            s4 = _cell(results, model, "S4", tier)
            rows.extend(
                [
                    {"model": model, "tier": tier, "channel": "temporal", "delta_r2": s1 - s2},
                    {"model": model, "tier": tier, "channel": "building", "delta_r2": s2 - s3},
                    {"model": model, "tier": tier, "channel": "site", "delta_r2": s3 - s4},
                    {"model": model, "tier": tier, "channel": "total", "delta_r2": s1 - s4},
                ]
            )
    return pd.DataFrame(rows)


def feature_construction_gap(results: pd.DataFrame) -> pd.DataFrame:
    """Leaky tier D minus causal tier C, per model and protocol."""
    rows = []
    for model in sorted(results["model"].unique()):
        for protocol in sorted(results["protocol"].unique()):
            c = _cell(results, model, protocol, "C")
            d = _cell(results, model, protocol, "D")
            rows.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "channel": "feature_construction",
                    "delta_r2": d - c,
                }
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    channels = channel_decomposition(results)
    feature = feature_construction_gap(results)
    out: dict = {"channels": {}, "feature_construction": {}}
    for (model, tier), part in channels.groupby(["model", "tier"]):
        key = f"{model}|{tier}"
        out["channels"][key] = {
            row["channel"]: round(float(row["delta_r2"]), 6)
            for _, row in part.iterrows()
        }
    for (model, protocol), part in feature.groupby(["model", "protocol"]):
        out["feature_construction"][f"{model}|{protocol}"] = round(
            float(part["delta_r2"].iloc[0]), 6
        )
    return out
