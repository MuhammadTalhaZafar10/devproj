"""Four classifiers: two from Raymand et al. (2023), two that are not in the paper."""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

# name -> (display name, in_paper)
MODEL_INFO = {
    "random_forest": ("Random Forest", True),
    "gradient_boosting": ("Gradient Boosting", True),
    "lightgbm": ("LightGBM", False),
    "xgboost": ("XGBoost", False),
}
MODEL_NAMES = tuple(MODEL_INFO)


def make_model(name: str, seed: int = 42, n_jobs: int = 4):
    if name == "random_forest":
        # Appendix Table B.2 of the paper (Subset A, performance class).
        return RandomForestClassifier(
            n_estimators=100, criterion="entropy", max_features=0.2,
            min_samples_leaf=3, min_samples_split=9, n_jobs=n_jobs, random_state=seed,
        )
    if name == "gradient_boosting":
        # The paper does not publish its tuned GB values; scikit-learn defaults.
        return GradientBoostingClassifier(random_state=seed)
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            n_jobs=n_jobs, random_state=seed, verbose=-1,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
            n_jobs=n_jobs, random_state=seed, eval_metric="mlogloss",
        )
    raise ValueError(f"unknown model {name!r}")
