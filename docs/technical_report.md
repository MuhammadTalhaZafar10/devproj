# Technical report: BDG2 performance-class classification

Base paper: Raymand, Najafi, Haghighat Mamaghani, Moazami, Rinaldi (2023), *Machine learning-based estimation of
buildings' characteristics employing electrical and chilled water consumption data: Pipeline optimization*,
Energy and Buildings 295:113327, https://doi.org/10.1016/j.enbuild.2023.113327

All numbers below come from `reports/classification_results.json`, `reports/cls_novelty_results.json` and
`reports/cls_improvement.json`, produced by the scripts in `backend/scripts/`.

## 1. Task and data
Classify a building as Low, Medium or High electricity performance from its hourly electricity use. The class is a
tercile of mean kWh per m2 per year. The paper does not publish its cutoffs and its repository is an empty placeholder,
so the cutoffs (78 and 158 kWh/m2/yr) are our assumption.

Data: Building Data Genome Project 2, electricity meters with at least 80 % non-missing hours and a positive floor
area: 1,470 buildings at 18 sites (the paper used 1,494). Two tables:

| Table | Rows | Columns | Use |
|---|---|---|---|
| Building table | 1,470 | 71 (66 features) | The paper's protocol: one row per building |
| 100k table | 100,000 | 59 (54 features) | Course-scale table: one row per building-week |

Rows of the 100k table are not independent: every week of a building carries that building's class. The effective
sample size is the number of buildings.

Every feature is computed on a series divided by its own mean, so absolute consumption and floor area cannot enter
the model. A unit test (`test_features_are_scale_free_so_absolute_consumption_cannot_leak`) enforces this.

## 2. Reproduction (paper protocol, building table)
Random stratified 80/20 split, 5-fold cross-validation on the training part, five splits (seeds 0 to 4), weighted F1.

| Model | In paper | Test F1 (mean ± sd) | Test accuracy | CV F1 | Paper |
|---|---|---|---|---|---|
| Random Forest | yes | 0.604 ± 0.015 | 0.604 | 0.579 | test F1 0.604, accuracy 0.605 (Table 3); CV F1 0.62 (Fig. 6a) |
| Gradient Boosting | yes | 0.602 ± 0.015 | 0.601 | 0.577 | CV F1 0.60 (Fig. 6a) |
| LightGBM | no | 0.595 ± 0.013 | 0.596 | 0.575 | not in paper |
| XGBoost | no | 0.607 ± 0.019 | 0.607 | 0.583 | not in paper |

The paper's best tuned pipeline scores 0.622 F1 / 0.622 accuracy. Our Random Forest equals the paper's Random Forest
baseline (0.604), which may partly be chance (sd 0.015). The right wording is "in the same band as the paper", not
"reproduced exactly": different cohort, 54 to 66 features instead of 374, and assumed class cutoffs.

## 3. Did we beat the paper?
No. A bounded, leakage-free attempt (`run_cls_improve.py`, five fixed seeds, tuning only inside training folds):

| Variant | Test F1 (mean ± sd) |
|---|---|
| V0 Random Forest, scale-free features | 0.604 ± 0.015 |
| V1 plus use type as an input | 0.602 ± 0.021 |
| V2 V1 with tuned Random Forest (8-draw search inside CV) | 0.601 ± 0.011 |
| V3 V1, ordinal: regress log kWh/m2, cut at training terciles | 0.585 ± 0.019 |

None exceeds the paper's 0.622 by more than one sd. Reaching 0.80 would require inputs built from kWh level or floor
area, which the label is made of, so the score would be arithmetic and not a prediction. We did not do that.
An Opus 5.5 review estimated a legitimate ceiling of roughly 0.62 to 0.68 (an assumption, not a measurement).

## 4. Course-scale table: two splits
Row-level weighted F1 on the 100k table (80/20):

| Model | Random rows | Whole buildings held out |
|---|---|---|
| Random Forest | 0.746 | 0.556 |
| Gradient Boosting | 0.607 | 0.550 |
| LightGBM | 0.689 | 0.553 |
| XGBoost | 0.715 | 0.548 |

## 5. Novelty: classification with whole sites held out
Claim (wording approved after prior-art checks, see `docs/novelty.md`): to our knowledge, no prior study evaluates
BDG2 performance-class classification with whole sites held out. Site holdout has been used on BDG2 for forecasting
and imputation, so we do not claim to be first to hold out sites.

Setup: 30,000-row sample, Random Forest, LightGBM, XGBoost plus two baselines. Gradient Boosting is excluded here
because it is too slow for about 35 folds. Scores are per building (a building's weeks averaged); intervals are 95 %
bootstrap over buildings. P2 uses five seeds. For P2 and P3 the tercile cutoffs are refit on each fold's training
buildings.

| Protocol | Random Forest | LightGBM | XGBoost | Majority class | Use type only |
|---|---|---|---|---|---|
| P1 random rows | 0.75 (0.72 to 0.77) | 0.73 (0.70 to 0.75) | 0.77 (0.75 to 0.79) | 0.17 | 0.42 |
| P1b train 2016, test 2017 | 0.71 (0.69 to 0.73) | 0.70 (0.68 to 0.72) | 0.72 (0.69 to 0.74) | 0.17 | 0.42 |
| P2 whole buildings | 0.59 (0.56 to 0.61) | 0.59 (0.56 to 0.61) | 0.59 (0.56 to 0.61) | 0.17 | 0.40 |
| P3 whole sites | 0.55 (0.52 to 0.58) | 0.53 (0.50 to 0.55) | 0.53 (0.50 to 0.55) | 0.16 | 0.34 |

What the numbers support:
- The model keeps real skill on unseen sites: 0.55 against 0.34 for a model that knows only the building's use type
  and 0.16 for the majority class. The intervals do not overlap, so the claim is not falsified.
- Most of the drop comes from not having seen the building (P1 0.75 to P2 0.59). Part of that is built into the data,
  because all weeks of a building share a label.
- Holding out the site costs a further 0.04 (P2 0.59 to P3 0.55). The intervals overlap partly, so at this sample size
  we call this "a small drop that we cannot separate from noise", not a proven effect.
- The result varies by site: with the site held out, LightGBM scores 0.86 on Lamb and 0.66 on Bear, but only 0.05 on
  Gator, where the use-type baseline scores 0.56. Per-site numbers rest on few buildings.

Limits: protocol differences mix several changes at once (unseen buildings, unseen sites, different class balance
per fold), and the experiment does not separate them. Single seed for P1, P1b and P3.

## 6. System
Flask API (`backend/app.py`): `/health`, `/model-info`, `/results`, `/novelty`, `/examples`, `/predict`. `/predict`
takes 168 hourly readings, or the 54 features as a flat object. Without a trained artifact it answers 503; there is
no fake fallback. On 30 held-out single weeks the live API was right 14 times (47 %), consistent with the 0.55
row-level whole-building score: one week is a noisy sample.

Angular 19 page on GitHub Pages: static JSON for all results, live classification when the backend is reachable
(address set in `frontend/public/config.json`). GitHub Pages cannot run Flask.

Tests: 46 backend tests and 6 frontend tests pass locally; CI runs the backend tests and the Angular build.

## 7. Limitations
- Cohort, features and class cutoffs differ from the paper's; comparison is indicative.
- Gradient Boosting uses scikit-learn defaults because the paper lists no tuned values.
- Prior-art search was abstract-level plus open-access full text; three paywalled papers (Yan 2023, Li 2024,
  Piscitelli 2023) were not read. "Not found" is not "does not exist".
- Research and teaching prototype; not an energy audit.
