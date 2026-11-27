# Novelty: classification of BDG2 buildings with whole sites held out

## Claim (safe wording)

> To our knowledge, based on OpenAlex full-text, arXiv and Europe PMC searches, no prior study evaluates
> BDG2 building performance-class classification with whole sites held out. We report how the
> Raymand et al. (2023) classifier performs under four split protocols, compared with a majority-class
> baseline and a use-type-only baseline, with building-level bootstrap confidence intervals.

"Not found" is not "does not exist". Confidence in the not-found result is medium. See the open checks below.

## Do not say

- "We are the first to use leave-one-site-out evaluation on BDG2." Site holdout is already used on BDG2 for forecasting.
- "Random splits inflate scores." In our 100,000-row table every week of a building carries the building's label, so part of that gap is built into the data.

## What the protocols separate

| Protocol | Train and test share |
|---|---|
| P1 random weeks | the same buildings, sites and time |
| P1b train 2016, test 2017 | the same buildings and sites |
| P2 whole buildings | the same sites |
| P3 whole sites | nothing |

Rows of one building are not independent, so all headline scores are building-level (a building's weeks are averaged)
and confidence intervals are bootstrapped over buildings. Baselines: majority class, and a model that sees only the
building's use type.

## Precedents to cite and position against

| Work | What it did | How we differ |
|---|---|---|
| Raymand et al. (2023), Energy and Buildings 295:113327 | Classifies use type, performance class and operation group on BDG2. Random 20 % test split and 5-fold CV, no site holdout (read from the PDF). | We keep their task and add site holdout. |
| Vivekanandan and Rajeswari (2026), Zenodo 10.5281/zenodo.22229229 | Leave-one-site-out on BDG2 for probabilistic forecasting. Record description only, the article was not found. | Forecasting, not classification. |
| Li et al. (2026), Zenodo 10.5281/zenodo.22643217 | Forecasting on 300 GEPIII buildings; mentions leave-one-site-out. Record description only. | Forecasting, not classification. |
| Fu, Quintana, Nagy, Miller (2023), doi 10.1016/j.applthermaleng.2023.121545 | Imputation of meter data; split by site ID, test sites unseen. | Imputation, not classification. |
| Mishra et al. (2026), doi 10.1922/comdem.v3i6s.19074 | Forecasting with a building-group holdout; says its scores are not verified efficiency classifications. | Forecasting. |
| Lien and Rajasekharan (2024), doi 10.1016/j.enbuild.2024.114954 | Classifies building category from smart-meter data of unseen Norwegian buildings (84 %). | Not BDG2, no site analysis. |

## Open checks (not yet verified)

1. Yan et al. (2023), Li et al. (2024) and Piscitelli et al. (2023) could not be read (paywall or no abstract).
   Cheapest check: get the PDFs through a library and search for "site", "leave-one" and "classif".
2. The main articles behind the two 2026 data records were not found. Confirm they are forecasting only.
3. Semantic Scholar and Google Scholar full text were not covered. One query to run by hand:
   `"Building Data Genome" "leave-one-site-out" classification`.

## Outcomes we can report honestly

- Skill falls under site holdout by more than the confidence intervals: report the size of the gap at each step.
- Intervals overlap: report "no detectable degradation at this sample size", not "the split does not matter".
- The model does not beat the use-type-only baseline under site holdout: report it as a negative result. It means
  performance class is mostly explained by use type, or that load-shape features do not transfer across sites.

BDG2 has about 19 sites, so intervals for the site-holdout protocol may be wide. Report the number of sites and buildings
per fold, and macro F1 next to accuracy.
