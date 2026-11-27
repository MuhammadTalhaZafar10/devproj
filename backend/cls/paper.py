"""Published reference numbers, transcribed from the paper's PDF.

Raymand, F., Najafi, B., Haghighat Mamaghani, A., Moazami, A., Rinaldi, F. (2023).
Machine learning-based estimation of buildings' characteristics employing electrical
and chilled water consumption data: Pipeline optimization.
Energy and Buildings 295, 113327. https://doi.org/10.1016/j.enbuild.2023.113327

Table 3 (Subset A, electricity only, 1,494 buildings), target "Performance Class",
weighted F1 and accuracy on a 20 % test set. Fig. 6a gives 5-fold-CV weighted F1
of all five of their classifiers on the same target.
"""

CITATION = (
    "Raymand et al. (2023), Energy and Buildings 295:113327, "
    "doi:10.1016/j.enbuild.2023.113327"
)

TABLE3_PERFORMANCE_CLASS = {
    "random_forest_all_features": {"f1_val": 0.602, "f1_test": 0.604, "acc_val": 0.602, "acc_test": 0.605},
    "random_forest_selected_features": {"f1_val": 0.611, "f1_test": 0.616, "acc_val": 0.612, "acc_test": 0.615},
    "optimal_pipeline": {"f1_val": 0.620, "f1_test": 0.622, "acc_val": 0.621, "acc_test": 0.622},
}

# Fig. 6a, weighted F1 (5-fold CV, all features), performance class.
FIG6A_F1_PERFORMANCE_CLASS = {
    "random_forest": 0.62,
    "gradient_boosting": 0.60,
    "logistic_regression": 0.61,
    "svc": 0.59,
    "ann": 0.60,
}

PAPER_BUILDINGS = 1494
PAPER_SPLIT = "20 % random test split; 5-fold CV on the remaining 80 %"
