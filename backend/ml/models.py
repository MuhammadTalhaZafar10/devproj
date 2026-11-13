"""Frozen model registry for the BDG2 study.

Configurations follow the base paper's defaults and are never tuned per cell.
"""
from __future__ import annotations

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

MODEL_NAMES = ("random_forest", "adaboost", "gbm", "xgboost", "lightgbm", "catboost")


def make_model(name: str, seed: int = 42):
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=seed
        )
    if name == "adaboost":
        return AdaBoostRegressor(n_estimators=100, random_state=seed)
    if name == "gbm":
        return GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=seed
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            random_state=seed,
            verbosity=-1,
        )
    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=500,
            depth=6,
            loss_function="RMSE",
            random_seed=seed,
            thread_count=-1,
            verbose=0,
            allow_writing_files=False,
        )
    raise ValueError(f"unknown model: {name}")
