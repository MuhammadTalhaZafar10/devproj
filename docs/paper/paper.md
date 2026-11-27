---
title: "Reproducing a smart-meter building performance-class classifier on BDG2 and evaluating it on unseen buildings and unseen sites"
author: "Author Name (Institution, Department, City, Country)"
abstract: |
  Raymand et al. (2023) classified buildings of the Building Data Genome Project 2 (BDG2) into Low, Medium and High electricity-consumption classes from smart-meter features. With a random 20% test split of buildings, they reported a test weighted F1 of 0.604 for a Random Forest and 0.622 for their best pipeline. We reproduced this task on 1,470 BDG2 buildings at 18 sites with 54 to 66 scale-free load-shape features, which by construction cannot encode absolute consumption or floor area; the unpublished class cutoffs were set at cohort terciles. Under the paper's protocol, four tree ensembles scored 0.595 to 0.607 (standard deviation 0.013 to 0.019 over five splits), in the same band as the paper; Random Forest scored 0.604 ± 0.015. Three leakage-free modifications did not raise the score above 0.604, so the paper's 0.622 was not exceeded. We then evaluated the classifiers under a ladder of stricter protocols on a 30,000-row sample of the 100,000-row building-week table, with majority-class and use-type-only baselines and building-level 95% bootstrap intervals. Building-level weighted F1 was 0.73 to 0.77 with random rows, 0.59 with whole buildings held out and 0.53 to 0.55 with whole sites held out, against 0.34 for the use-type baseline and 0.16 for the majority class. Part of the first drop is built into the data, because all weeks of a building share its label. Per-site scores ranged from 0.05 to 0.86.
keywords:
  - Building Data Genome Project 2
  - building performance class
  - smart meter
  - reproducibility
  - data leakage
  - leave-one-site-out validation
  - random forest
  - gradient boosting
bibliography: references.bib
link-citations: true
---

# 1. Introduction

Smart-meter deployment has made hourly electricity data widely available for non-residential buildings [@raymand2023]. These data make it possible to screen many buildings for poor energy performance without a site visit, a task that @raymand2023 call remote auditing. Take a model that sorts buildings into low, medium and high consumption classes from the shape of their load. An owner of a large portfolio, or a utility, could use it to decide where to send auditors first. The Building Data Genome Project 2 (BDG2) [@miller2020] provides hourly meter data from 1,636 non-residential buildings at 19 sites for 2016 and 2017. It is the natural public test bed for such models.

@raymand2023 built and optimised machine-learning (ML) pipelines on BDG2 that classify buildings by principal use type, performance class and operation group. The pipelines use features computed from the buildings' meter data. For the performance class, their Random Forest (RF) reached a test weighted F1 score of 0.604 and their best pipeline reached 0.622. These scores come from a random 20% test split of buildings, so the training and test sets share sites. Buildings at one BDG2 site share a location and climate and, as the base paper notes, are probably operated with similar strategies. The same paper also classified buildings by operation group, a target that separates buildings by site, and reached a test weighted F1 of 0.842 on it. So meter features carry information about which site a building belongs to. When sites are shared between training and test sets, a performance-class model can draw on that site-level information. Its score may then not describe how it would perform at a site it has never seen. The general hazard of evaluation designs that let information cross from training to test data is well documented [@kaufman2012; @kapoor2023; @roberts2017]. On BDG2 itself, a regression benchmark found that random cross-validation (CV) gave far higher scores than time-ordered validation [@xie2025].

This study had two aims: to reproduce the performance-class result of @raymand2023 with transparent features that cannot leak the label and test whether legitimate changes improve on it, and to measure how the score changes as evaluation moves from random rows to unseen buildings and unseen sites. The contributions are:

1. **A reproduction.** We used 1,470 BDG2 buildings at 18 sites, four tree ensembles and 54 to 66 scale-free features, which cannot encode absolute consumption or floor area. Test weighted F1 fell in the same band as the paper's: 0.595 to 0.607 across models, and 0.604 ± 0.015 for Random Forest.
2. **A negative result on improvement.** We made a bounded, pre-specified attempt to improve on the paper without using the label's ingredients. It did not succeed: the best variant scored 0.604, against the paper's 0.622.
3. **A protocol ladder.** We evaluated four protocols on a 30,000-row sample of the 100,000-row building-week table: random rows, later weeks of the same buildings, whole buildings and whole sites. Each came with majority-class and use-type-only baselines and building-level 95% bootstrap intervals. We searched OpenAlex, Europe PMC and arXiv at abstract level and in open-access full text. To our knowledge, from those searches, no prior study evaluates BDG2 building performance-class classification with whole sites held out.
4. **Open code.** The code regenerates every table and figure in this paper. It also includes a small Flask application programming interface (API) and an Angular web page that serve the model.

Figure 1 (Section 4) summarises the study pipeline. Section 2 reviews related work, and Section 3 describes the data, cohort and labels. Section 4 sets out the methods and Section 5 the results. Section 6 discusses what the results do and do not support, Section 7 lists limitations and Section 8 concludes.

# 2. Background and related work

## 2.1 BDG2 and the Great Energy Predictor III competition

BDG2 [@miller2020] is an open collection of hourly meter readings from 1,636 non-residential buildings at 19 sites, recorded in 2016 and 2017. It covers electricity, chilled water, steam, hot water, gas, water, irrigation and solar meters. Each building has metadata such as floor area and primary space usage, and each site has a weather file. Site names are anonymised as animal names (for example Panther, Lamb or Gator). The data set was the basis of the American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE) Great Energy Predictor III (GEPIII) forecasting competition on Kaggle. @miller2022 analysed the errors of that competition's solutions and discussed the limits of ML for building energy prediction. Much of the later BDG2 work that we found concerns forecasting, imputation or transfer learning (Section 2.4). Classification of building attributes is less common.

## 2.2 Building characterisation from meter data and the base paper

@raymand2023 is the base paper of this study, so we describe its performance-class pipeline in some detail. They first cleaned the meter data. A building was kept only if at most 5% of its readings were missing and at most 10% were zero. Outlying peaks and drops were trimmed. This left 1,494 buildings with electricity data (their Subset A) and a smaller group that also had chilled-water data (Subset B). For Subset A they generated 374 features per building, re-implementing a feature set proposed earlier for BDG2's predecessor. The features fall into three groups:

- **Statistical features:** temporal statistics over days, months and seasons, ratios between them, and rank correlations with outdoor temperature.
- **Regression-model features:** a load-shape model, weather change-point models and a seasonal-trend decomposition.
- **Pattern-based features:** motifs and discords found with a symbolic representation of daily profiles, a measure of how typical a building is of its use type, and a test for long-term changes in steady-state consumption.

Three targets were defined. The first is principal use type, regrouped into a few broad categories. The second is performance class: three classes (low, medium and high consumption) from consumption normalised by floor area. The third is operation group, which separates buildings by site. For the performance class, every feature directly derived from consumption was removed to prevent leakage, which left 221 features.

Five classifiers were compared by 5-fold CV: RF, Gradient Boosting (GB), logistic regression, a support vector classifier and an artificial neural network. Their Fig. 6a reports weighted F1 values for the performance class of 0.62 (RF), 0.60 (GB), 0.61 (logistic regression), 0.59 (support vector classifier) and 0.60 (neural network). RF was chosen as the benchmark. A forward feature-selection procedure then reduced the performance-class features from 221 to 26. The Tree-based Pipeline Optimization Tool, a genetic-programming search over scikit-learn pipelines, then picked the final pipeline.

On the 20% test set the RF with all features reached weighted F1 0.604 and accuracy 0.605. With the 26 selected features it reached 0.616 and 0.615, and the optimised pipeline reached 0.622 and 0.622. The paper's Appendix Table B.2 gives the optimised performance-class pipeline as a single RF: entropy criterion, 20% of features per split, at least 3 samples per leaf, at least 9 samples to split, and 100 trees. The text of its Section 4.2.1 instead says that the performance-class and operation-group pipelines use Gradient Boosting. We followed the table. The paper also notes that buildings in the intermediate class were the ones most often misclassified.

Three details matter for a reproduction. The numerical class cutoffs are not published. The test split is random over buildings, so test buildings come from sites that also appear in training. And the linked code repository (DataOptimaLab/BuildingClassifcation_DeterminationOfOptimalFeatureSet on GitHub) contained only a README and a licence when we checked it.

Two other studies are close to this task. As described in its abstract, @lien2024 classified the standard building category of Norwegian buildings from smart-meter data. The test buildings were unseen in training, and the model was correct in 84% of test cases. That study does not use BDG2 and, as far as its abstract shows, does not analyse site transfer. We know @piscitelli2023 only from its abstract. It proposes a time-series-based energy benchmarking framework for large building stocks, which is related to performance classification but is not the same supervised task.

## 2.3 Data leakage and validation design

@kaufman2012 defined leakage in data mining as the use, during model building or evaluation, of information that would not be legitimately available when the model is applied, and they described ways to detect and avoid it. @kapoor2023 documented leakage in many published ML-based studies across scientific fields and proposed a taxonomy of its forms. They argued that leakage is a major cause of irreproducible results. @roberts2017 showed that random cross-validation underestimates prediction error when observations are dependent in time, space or hierarchy. They recommended blocking the folds along that structure. Building-week data are hierarchical: weeks are nested in buildings, and buildings are nested in sites. Blocking by building and by site is therefore the natural way to apply that advice here.

For BDG2 the size of the effect has been measured in one regression benchmark. @xie2025 compared boosting algorithms on 10 BDG2 buildings. Random CV gave coefficients of determination of 0.90 to 0.99, and time-series CV gave 0.24 to 0.34. That gap concerns a regression target rather than a building-level class. It nevertheless suggests that, on these data, the choice of protocol can matter more than the choice of algorithm.

## 2.4 Unseen-building and site-level evaluation on BDG2

Holding out whole sites is not new on BDG2. @fu2023 imputed gaps in building energy data with a multidimensional context autoencoder and split the data by site, so their test sites were unseen. Two data records on Zenodo describe site-level evaluation for forecasting. @vivekanandan2026, as described in its repository record, uses BDG2 with a "strict leave-one-site-out evaluation" for probabilistic forecasting of unseen buildings. @li2026, also as described in its repository record, benchmarks load forecasters on 300 GEPIII buildings and mentions leave-one-site-out evaluation. We did not find the articles behind these two records. @mishra2026 forecast BDG2 consumption with a building-group holdout. @kim2025 evaluated automated feature extraction and selection for forecasting on 100 BDG2 buildings with a random 80/20 split. As described in its abstract, @spencer2025 compared transfer-learning strategies for transformer models in building energy forecasting.

The studies we found that hold out sites on BDG2 therefore address forecasting or imputation. None of them addresses classifying buildings into performance classes. Our searches covered abstracts plus open-access full text in OpenAlex, Europe PMC and arXiv. Two further candidate papers, Yan et al. (2023) and Li et al. (2024), were paywalled and could not be read, so we do not cite them. We read Piscitelli et al. [-@piscitelli2023] only as far as its abstract. Table 1 summarises the closest work and how much of each we read.

**Table 1.** Closest prior work. "Holdout" is the evaluation design as far as we could establish it; "Evidence read" states whether we read the full text, only the abstract, or only a data-repository record.

| Work | Task | BDG2 | Holdout | Evidence read |
|---|---|---|---|---|
| @raymand2023 | Classification: use type, performance class, operation group | Yes (1,494 buildings, electricity) | Random 20% test split of buildings; 5-fold CV on the rest | Full text |
| @xie2025 | Regression benchmark of boosting algorithms | Yes (10 buildings) | Random CV compared with time-series CV | Full text |
| @kim2025 | Forecasting with automated feature extraction and selection | Yes (100 buildings) | Random 80/20 split | Full text (manuscript) |
| @mishra2026 | Forecasting and multi-criteria performance assessment | Yes | Building-group holdout | Full text |
| @fu2023 | Imputation of meter-data gaps | Yes | Split by site; test sites unseen | Full text |
| @vivekanandan2026 | Probabilistic forecasting for unseen buildings | Yes | Leave-one-site-out | Repository record |
| @li2026 | Benchmark of load forecasters (300 buildings) | GEPIII data | Mentions leave-one-site-out | Repository record |
| @lien2024 | Classification of standard building category | No (Norwegian data) | Unseen buildings | Abstract |
| @spencer2025 | Transfer learning for transformer forecasting | Not verified | Not established | Abstract |
| @piscitelli2023 | Time-series energy benchmarking of building stocks | Not verified | Not established | Abstract |
| This study | Classification: performance class | Yes (1,470 buildings, 18 sites) | Random rows, later weeks, whole buildings, whole sites | Not applicable |

# 3. Data, cohort and labels

## 3.1 Source files

We used three files of the public Kaggle distribution of BDG2: hourly electricity readings per building (`electricity.csv`), building metadata (`metadata.csv`), and hourly site weather (`weather.csv`). From the metadata we took each building's site, primary space usage and floor area in square metres. From the weather file we took air temperature. Gaps in each site's temperature series were filled by linear interpolation. Negative electricity readings were treated as missing.

## 3.2 Cohort

A building entered the cohort if three conditions held. At least 80% of its hours had a non-missing electricity reading. Its floor area was positive. Its mean electricity use over the non-missing hours was positive. These rules gave 1,470 buildings at 18 of the 19 BDG2 sites. @raymand2023 used 1,494 buildings selected by different rules: at most 5% missing readings, at most 10% zero readings, and trimming of outliers. We did not trim outliers or remove periods of zero readings. The two cohorts are not identical, so the comparison in Section 5.1 is between similar, not identical, data sets.

## 3.3 Performance-class label

For each building $b$ we computed the annual electricity use intensity

$$e_b = \frac{8760\,\bar{x}_b}{A_b},$$

where $\bar{x}_b$ is the mean hourly electricity use in kWh over the non-missing hours and $A_b$ is the floor area in m$^2$. The performance class was set by the terciles of $e_b$ over a reference set of buildings:

$$y_b = \begin{cases} \text{Low} & \text{if } e_b \le q_{1/3},\\ \text{Medium} & \text{if } q_{1/3} < e_b \le q_{2/3},\\ \text{High} & \text{if } e_b > q_{2/3}, \end{cases}$$

where $q_{1/3}$ and $q_{2/3}$ are the empirical one-third and two-thirds quantiles of $e$ over the reference set. For the reproduction and the course-scale table, the reference set was the whole cohort. This gave 490 buildings in each class, with boundaries at 77.7 and 158.3 kWh m$^{-2}$ yr$^{-1}$ (the largest values in the Low and Medium classes). For the stricter protocols the cutoffs were refitted on each fold's training buildings (Section 4.5).

The tercile rule is our assumption. @raymand2023 describe the classes only as low, medium and high consumption, and they publish neither cutoffs nor class sizes. Terciles give balanced classes, which keeps the majority-class baseline at a known level (Section 4.6). Figure 2 shows the distribution and the cutoffs. The Medium band is narrow: its upper boundary is only 2.04 times its lower boundary. Many buildings therefore lie close to a boundary, where a small error in an estimate of intensity changes the class.

![Figure 2. Distribution of annual electricity use intensity (kWh per m² per year, logarithmic axis) for the 1,470 cohort buildings, coloured by performance class. Dashed lines mark the tercile boundaries at 78 and 158 kWh/m²/yr (77.7 and 158.3 to one decimal); each class holds 490 buildings. Most buildings lie within roughly two orders of magnitude around the boundaries, and a thin tail of very low values falls in the Low class.](figures/fig2_label_distribution.png){width=85%}

The low tail visible in Figure 2 probably reflects metering or floor-area records rather than real use. We kept these buildings because the paper's cleaning rule for them is not known in enough detail to copy, and because removing them would change the terciles.

## 3.4 Two analysis tables

Two tables were derived from the cohort, and both carry the same building-level label.

The **building table** has one row per building: 1,470 rows and 71 columns. The columns are building identifier, site, primary space usage, intensity $e_b$, class, and 66 features computed over the full two-year record. This table matches the paper's unit of analysis and was used for the reproduction (Section 4.3) and the improvement attempt (Section 4.7).

The **building-week table** has one row per building-week: 100,000 rows and 59 columns. The columns are building identifier, site, primary space usage, week start, class, and 54 features computed on one 168-hour window. The table was built to meet a course requirement of at least 100,000 rows and 35 columns. Weeks were aligned to the start of the record. A week was kept if at most 10% of its hours were missing and its mean was positive. From all valid building-weeks, 100,000 were drawn at random (seed 42). The table covers 1,468 buildings at 18 sites, about 68 weeks per building on average. Every row inherits its building's class, which has consequences for evaluation (Section 4.4).

# 4. Methods

Figure 1 shows the study pipeline. The two tables of Section 3.4 feed two strands of work. The first is a reproduction of the paper's protocol on the building table, together with a bounded attempt to improve on it. The second is a ladder of evaluation protocols on the building-week table. Both strands use the same scale-free features and the same model settings.

![Figure 1. Study pipeline. BDG2 raw data (1,636 buildings, 19 sites, hourly, 2016 to 2017) pass a cohort filter (at least 80% of hours present, positive floor area and use), leaving 1,470 buildings. Scale-free features (54 per week; 66 per building with monthly means) and tercile labels form a 1,470-row building table and a 100,000-row by 59-column building-week table. Four models (two from the paper, two added) and two baselines feed the reproduction (random 80/20 split with 5-fold CV over five seeds), the protocol ladder (P1 rows, P1b later weeks, P2 buildings, P3 sites) and the outputs (building-level F1 with 95% intervals, a Flask API and an Angular page).](figures/fig1_pipeline.png){width=95%}

## 4.1 Scale-free features and the leakage guard

The label is a ratio of mean consumption to floor area, so any input that carries the consumption level or the floor area carries part of the label. @raymand2023 handled this by removing features "directly resultant of consumption". We enforced the same rule by construction. Every feature was computed on the series divided by its own mean,

$$r_{b,t} = \frac{x_{b,t}}{\bar{x}_b},$$

so that a feature function $f$ satisfies $f(c\,\mathbf{x}) = f(\mathbf{x})$ for every constant $c>0$. Missing hours were set to the mean ($r=1$) after normalisation. The fraction of zero readings was counted on the raw series, and it is also unchanged by scaling. Floor area does not enter any feature. A unit test, `test_features_are_scale_free_so_absolute_consumption_cannot_leak`, multiplies synthetic series by 1,234.5 and checks that every feature is unchanged. The features therefore describe the shape of the load (when it peaks, how flat it is, how it varies by day and with temperature) and not its size.

Table 2 groups the features by family. The building-week table uses the 54 features listed. The building table adds 12 monthly means of the normalised series, which describe the seasonal profile over the two-year record, for 66 features in total. Appendix B lists every name.

**Table 2.** Feature families. All features are computed on the mean-normalised series $r$; counts are for one building-week (54) and one building (66).

| Family | Features | Count |
|---|---|---|
| Hour-of-day profile | Mean of $r$ at each hour 00 to 23 | 24 |
| Day-of-week profile | Mean of $r$ on each weekday, Monday to Sunday | 7 |
| Weekend ratio | Weekend mean divided by weekday mean | 1 |
| Day periods | Mean of $r$ at night (00 to 06 h), morning, afternoon, evening | 4 |
| Spread and shape | Standard deviation (coefficient of variation), skewness, kurtosis, load factor (mean over maximum), minimum-to-maximum ratio | 5 |
| Quantiles | 5th, 25th, 50th, 75th and 95th percentiles of $r$ | 5 |
| Timing | Hour of the peak and of the trough of the daily profile | 2 |
| Daily load ratio | Mean and standard deviation of the daily minimum-to-maximum ratio | 2 |
| Autocorrelation | Lag-1 h and lag-24 h autocorrelation | 2 |
| Zero fraction | Share of hours with a zero reading | 1 |
| Weather | Correlation of $r$ with site air temperature | 1 |
| *Subtotal, building-week table* | | *54* |
| Monthly profile (building table only) | Mean of $r$ in each calendar month | 12 |
| *Total, building table* | | *66* |

The 54 to 66 features are far fewer than the paper's 221 non-consumption features for this target. They also omit its pattern-mining and regression-model families. The reproduction is therefore a reproduction of the task and protocol, not of the feature set.

## 4.2 Models and hyperparameters

Four classifiers were used, all with scikit-learn 1.9.1 interfaces [@pedregosa2011]:

- **Random Forest** [@breiman2001]: the settings of the paper's Appendix Table B.2. These are 100 trees, entropy criterion, maximum features 0.2, minimum 3 samples per leaf and minimum 9 samples per split.
- **Gradient Boosting** [@friedman2001]: scikit-learn defaults, because the paper publishes no tuned values for this target.
- **LightGBM** [@ke2017]: 300 trees, learning rate 0.05 and 31 leaves. This model is not in the paper.
- **XGBoost** [@chen2016]: 300 rounds, learning rate 0.1, maximum depth 6 and histogram tree construction. This model is not in the paper.

The two added models were included because they are widely used gradient-boosting libraries that scale to the 100,000-row table. We did not scale features, select features or weight classes. The random seed of each model was set to the split seed in the reproduction and to 42 elsewhere. Appendix A lists all settings.

## 4.3 Reproduction protocol

The reproduction followed the paper's protocol on the building table with all 66 features. A stratified random 20% of buildings (294) formed the test set. On the remaining 80%, a stratified 5-fold CV gave the CV score. The model was then refitted on the full 80% and scored on the test set. This was repeated for five split seeds (0 to 4). We report the mean and the standard deviation (sd; population form, over the five splits) of the test weighted F1, test accuracy and CV weighted F1. Confusion matrices were summed over the five test sets. The class cutoffs were the whole-cohort terciles of Section 3.3.

## 4.4 The building-week table and why its rows are dependent

On the building-week table, two 80/20 splits were compared. In the *random-rows* split, a stratified random 20% of rows was used for testing (seed 42). In the *whole-buildings* split, the first fold of a 5-fold stratified split grouped by building (seed 42) was used. Each test set had 20,000 rows. Both were scored per row.

The rows of this table are not independent. All weeks of a building carry the same label and have similar load shapes. In the random-rows split, roughly four-fifths of each test building's weeks are in the training set. A model that recognises a building from its weekly signature can then return the label it saw in training, without having learned anything that would transfer to a new building. Part of the gap between the two splits is therefore built into how the table was made, and the effective sample size is closer to the number of buildings than to the number of rows. We report the comparison because the course required this table. It shows how much recognition contributes to a row-level score on these data. It is not evidence for a new finding about random splits.

## 4.5 Protocol ladder

To separate what a model learns about buildings, sites and time, we compared four protocols that share progressively less between training and test data. To fit the available hardware, the ladder used a random sample of 30,000 rows (seed 42) of the building-week table, covering 1,467 buildings at 18 sites. Three models were run: Random Forest, LightGBM and XGBoost. Gradient Boosting was excluded for compute reasons: the ladder needs 37 fits per model (3 for P1, 1 for P1b, 15 for P2, 18 for P3), and the single-threaded scikit-learn implementation took 805 s and 954 s per fit on the 80,000-row training part of the full table, against 136 to 441 s for the other models (Appendix C).

- **P1, random rows.** Stratified 3-fold CV on rows (seed 0). Training and test data share buildings, sites and time.
- **P1b, later weeks of the same buildings.** Training on weeks that start in 2016 and testing on weeks that start in 2017. The two sets share buildings and sites but not time. There were 1,462 test buildings.
- **P2, whole buildings.** Stratified 3-fold CV grouped by building, repeated for seeds 0 to 4. Test buildings are unseen, but their sites appear in training.
- **P3, whole sites.** Leave-one-site-out over the 18 sites. Neither buildings nor sites are shared, although both sets cover the same calendar period. A site with fewer than 5 sampled buildings would have been kept in training and never tested; no site met that condition.

In P1 and P1b the label is the table's label, from whole-cohort terciles. In P2 and P3 the tercile cutoffs $q_{1/3}$ and $q_{2/3}$ were refitted on the training buildings of each fold, using one intensity value per building. Training and test buildings were then labelled with those cutoffs. This keeps the test buildings' intensities out of the label definition, as would be the case when a model is applied to a new site. Within each protocol, the out-of-fold predictions were pooled over folds and scored once. For P2 this was done once per seed.

## 4.6 Baselines, metrics and bootstrap intervals

Two baselines were run on exactly the same folds. The *majority-class* baseline predicts the training-set class prior, so it always outputs the most frequent training class. The *use-type-only* baseline is LightGBM with the settings above, trained on a one-hot encoding of primary space usage (with an indicator for missing values) and nothing else. It measures how much of the performance class can be predicted from use type alone. A load-shape model should beat it before it can be credited with learning anything beyond use type.

The main metric is the weighted F1 score,

$$F_1^{w} = \sum_{c=1}^{3} \frac{n_c}{N}\,\frac{2P_cR_c}{P_c+R_c},$$

where $P_c$ and $R_c$ are the precision and recall of class $c$, $n_c$ is the number of test items of class $c$ and $N$ is their total. With three balanced classes, a classifier that always predicts one class scores $\tfrac{1}{3}\cdot\tfrac{2\cdot(1/3)\cdot 1}{1/3+1} = 1/6 \approx 0.167$, which is the level the majority-class baseline should reach.

Because rows of one building are dependent, the ladder's headline scores are building-level. For a test building $b$ with test weeks $W_b$, the predicted class probabilities $\hat{\mathbf{p}}_w$ were averaged and the class with the largest mean was taken:

$$\bar{\mathbf{p}}_b = \frac{1}{|W_b|}\sum_{w \in W_b}\hat{\mathbf{p}}_w, \qquad \hat{y}_b = \arg\max_{c}\,\bar{p}_{b,c}.$$

Building-level weighted F1 was computed over the pairs $(y_b, \hat{y}_b)$. Its 95% interval is a percentile bootstrap over buildings [@efron1993]. We drew $K = 1000$ resamples $S^{(k)}$ of the test buildings with replacement (seed 0) and computed $F^{*(k)} = F_1^{w}\big(\{(y_b,\hat{y}_b) : b \in S^{(k)}\}\big)$. The 2.5th and 97.5th percentiles of $F^{*(1)},\dots,F^{*(K)}$ form the interval. For P2 we report the mean and sd over the five seeds, and the interval is the mean of the per-seed bounds.

The intervals describe sampling variation of the test buildings for fixed fitted models, not variation from refitting. Every protocol scores largely the same buildings, so intervals from different protocols are not independent, and their overlap is only a rough guide to whether two scores differ. Row-level scores, which treat every test week as one item, are given in Appendix C.

## 4.7 Improvement attempt

To test whether the paper's 0.622 could be exceeded legitimately, we ran four pre-specified variants on the building table. All used the paper's protocol (stratified 80/20 split) with seeds 0 to 4, and all tuning was done inside the training split. We report the mean over all five seeds, never the best seed. The decision rule was fixed in advance: a variant would count as better than the paper only if its mean exceeded 0.622 by more than one sd. No variant used consumption level or floor area as an input.

- **V0.** The reproduction Random Forest on the 66 scale-free features.
- **V1.** V0 plus a one-hot encoding of primary space usage as an extra input.
- **V2.** V1 with the Random Forest tuned by an 8-draw randomised search inside 5-fold CV on the training split. The search covered 150 to 349 trees, maximum features 0.1 to 0.6, minimum 1 to 7 samples per leaf, minimum 2 to 15 samples per split, and Gini or entropy criteria. The budget was cut from 20 to 8 draws after a first run took 8 minutes per seed and gave no gain on seed 0 at either budget.
- **V3.** V1 recast as an ordinal problem. A Random Forest regressor (400 trees, maximum features 0.3, minimum 3 samples per leaf) predicted $\log e_b$. The prediction was cut at the training split's terciles of $\log e_b$. Intensity is the regression target here, never an input.

# 5. Results

## 5.1 Reproduction

Table 3 and Figure 3 compare the four models with the paper. Mean test weighted F1 ranged from 0.595 (LightGBM) to 0.607 (XGBoost). The split-to-split sd was 0.013 to 0.019, so the differences between models are smaller than the variation between splits. Random Forest scored 0.604 ± 0.015, the same value to three decimals as the paper's Random Forest with all features (0.604). Given an sd of 0.015, this agreement is partly a matter of chance and is not evidence of an exact match. The defensible statement is that our scores are in the same band as the paper's. The paper's best pipeline (0.622) lies about 1.2 sd above our Random Forest mean, and below the best of our five Random Forest splits (0.630, Table 4).

**Table 3.** Reproduction on the building table (1,470 buildings, 66 features): random stratified 80/20 split with 5-fold CV on the training part, five seeds. Values are means ± sd over the five splits. Paper values are from Table 3 and Fig. 6a of @raymand2023 (1,494 buildings; Fig. 6a values are given to two decimals).

| Model | In paper | Test weighted F1 | Test accuracy | CV weighted F1 | Paper |
|---|---|---|---|---|---|
| Random Forest | Yes | 0.604 ± 0.015 | 0.604 | 0.579 ± 0.006 | Test F1 0.604, accuracy 0.605, validation F1 0.602 (all features); CV F1 0.62 (Fig. 6a) |
| Gradient Boosting | Yes | 0.602 ± 0.015 | 0.601 | 0.577 ± 0.009 | CV F1 0.60 (Fig. 6a) |
| LightGBM | No | 0.595 ± 0.013 | 0.596 | 0.575 ± 0.005 | Not in paper |
| XGBoost | No | 0.607 ± 0.019 | 0.607 | 0.583 ± 0.013 | Not in paper |
| Paper: Random Forest, 26 selected features | | 0.616 | 0.615 | | Validation F1 0.611 |
| Paper: best tuned pipeline | | 0.622 | 0.622 | | Validation F1 0.620 |

![Figure 3. Reproduction compared with the base paper. (a) Mean test weighted F1 ± sd over five random 80/20 splits for our four models (asterisk: not in the paper); the dashed line marks the paper's Random Forest (0.604) and the solid line its best pipeline (0.622). Our scores range from 0.595 (LightGBM) to 0.607 (XGBoost), with Random Forest at 0.604. (b) Our 5-fold cross-validated weighted F1 (circles) against the paper's Fig. 6a values (diamonds): 0.579 against 0.62 for Random Forest and 0.577 against 0.60 for Gradient Boosting.](figures/fig3_reproduction.png){width=95%}

Our CV scores were 0.020 to 0.025 lower than our test scores for every model. Each CV fold trains on four-fifths of the training split (about 940 buildings), against 1,176 for the final model. The gap is therefore consistent with a learning curve that has not yet flattened, but we did not test this. The paper's validation and test scores for the same Random Forest were closer together (0.602 and 0.604). Its Fig. 6a CV values (0.62 and 0.60) are 0.02 to 0.04 above ours.

For all four models the Medium class was the hardest. Random Forest's per-class F1 was 0.651 for Low, 0.529 for Medium and 0.632 for High. The Medium F1 of the other three models was 0.512 to 0.525. Random Forest recalled 65.9% of Low, 53.9% of Medium and 61.4% of High test buildings. This matches the paper's remark that intermediate buildings were the ones most often misclassified.

## 5.2 Attempt to exceed the paper

Table 4 and Figure 9 report the improvement attempt. No variant reached the decision threshold, and none of V1 to V3 had a mean above V0's 0.604. Adding use type as an input (V1) gave 0.602 with a larger spread. Tuning the forest inside CV (V2) gave 0.601 with the smallest spread. The ordinal regression (V3) gave 0.585. The paper's 0.622 was therefore not exceeded, and we do not claim an improvement over it.

**Table 4.** Improvement attempt on the building table: same protocol as Table 3, five seeds, all tuning inside the training split. A variant would count as better than the paper only if its mean exceeded 0.622 by more than one sd.

| Variant | Change | Test weighted F1 (mean ± sd) | Test accuracy | Per-seed F1 range |
|---|---|---|---|---|
| V0 | Random Forest, scale-free features (reproduction) | 0.604 ± 0.015 | 0.604 | 0.582 to 0.630 |
| V1 | V0 plus use type as an input | 0.602 ± 0.021 | 0.601 | 0.575 to 0.630 |
| V2 | V1 with Random Forest tuned inside CV | 0.601 ± 0.011 | 0.600 | 0.586 to 0.613 |
| V3 | V1 as ordinal regression of $\log e_b$, cut at training terciles | 0.585 ± 0.019 | 0.582 | 0.558 to 0.606 |
| Paper | Best tuned pipeline | 0.622 | 0.622 | |

![Figure 9. Improvement variants against the paper's best pipeline. Large markers show mean test weighted F1 ± sd over five seeds; small dots show the single seeds; the vertical line marks the paper's 0.622. V0 (Random Forest, scale-free features) scored 0.604, V1 (plus use type as input) 0.602, V2 (V1 with the forest tuned inside cross-validation) 0.601 and V3 (V1 recast as ordinal regression of log intensity, cut at training terciles; truncated to "V3 V1" on the axis) 0.585. No mean reaches 0.622.](figures/fig9_improvement.png){width=90%}

Use type adds nothing on top of the load-shape features, and V2 shows no gain within the small tuning budget. Both results suggest that the limit lies in the information the features carry, not in model capacity. Section 6.3 returns to this.

## 5.3 Course-scale table: random rows against whole buildings

Table 5 and Figure 4 give the row-level results on the 100,000-row table. With random rows, weighted F1 ranged from 0.607 (Gradient Boosting) to 0.746 (Random Forest). With whole buildings held out, all four models fell within 0.008 of each other, from 0.548 to 0.556.

**Table 5.** Building-week table (100,000 rows, 54 features): row-level scores on a 20,000-row test set for a random-row split and a split that holds out whole buildings. Seed 42 for splits and models.

| Model | Random rows: weighted F1 | Whole buildings: weighted F1 | Difference | Random rows: accuracy | Whole buildings: accuracy |
|---|---|---|---|---|---|
| Random Forest | 0.746 | 0.556 | 0.190 | 0.746 | 0.554 |
| Gradient Boosting | 0.607 | 0.550 | 0.058 | 0.606 | 0.547 |
| LightGBM | 0.689 | 0.553 | 0.136 | 0.688 | 0.553 |
| XGBoost | 0.715 | 0.548 | 0.167 | 0.715 | 0.548 |

![Figure 4. Row-level weighted F1 on the 100,000-row building-week table when test rows are drawn at random (grey) and when whole buildings are held out (teal). Random rows: Random Forest 0.75, Gradient Boosting 0.61, LightGBM 0.69, XGBoost 0.72. Whole buildings: 0.55 to 0.56 for all four models.](figures/fig4_two_splits.png){width=85%}

The size of the random-row advantage followed how closely each model can fit individual rows. It was largest for the Random Forest, whose leaves may hold as few as 3 samples (0.190). It was smallest for the default Gradient Boosting, whose trees are shallow (0.058). This pattern fits the reading in Section 4.4. Under random rows, a flexible model can recognise weeks of buildings it has already seen. Once buildings are held out, that advantage disappears and the four models score within 0.008 of each other. Because all weeks of a building share a label, this gap is expected from the design of the table and is not a new finding.

## 5.4 Protocol ladder

Table 6 and Figure 5 give building-level weighted F1 with 95% intervals for every protocol.

**Table 6.** Protocol ladder on a 30,000-row sample of the building-week table (1,467 buildings, 18 sites): building-level weighted F1 with 95% bootstrap interval over buildings. P2 values are means over five seeds, with the sd over seeds after the mean. Gradient Boosting is not in the ladder (compute cost, Section 4.5).

| Protocol | Random Forest | LightGBM | XGBoost | Majority class | Use type only |
|---|---|---|---|---|---|
| P1 random rows | 0.747 (0.724 to 0.769) | 0.727 (0.704 to 0.749) | 0.771 (0.749 to 0.792) | 0.167 (0.146 to 0.187) | 0.416 (0.391 to 0.443) |
| P1b later weeks, same buildings | 0.710 (0.686 to 0.734) | 0.699 (0.676 to 0.723) | 0.717 (0.694 to 0.741) | 0.168 (0.148 to 0.190) | 0.416 (0.393 to 0.441) |
| P2 whole buildings | 0.589 ± 0.004 (0.564 to 0.613) | 0.586 ± 0.006 (0.561 to 0.611) | 0.589 ± 0.009 (0.565 to 0.613) | 0.165 ± 0.001 (0.145 to 0.186) | 0.398 ± 0.008 (0.374 to 0.424) |
| P3 whole sites | 0.552 (0.524 to 0.577) | 0.527 (0.500 to 0.551) | 0.525 (0.499 to 0.550) | 0.158 (0.138 to 0.180) | 0.341 (0.317 to 0.367) |

![Figure 5. Building-level weighted F1 with 95% bootstrap intervals over buildings for each protocol: P1 random rows, P1b later weeks of the same buildings, P2 whole buildings (mean over five seeds) and P3 whole sites. Lines join the three models (Random Forest, LightGBM, XGBoost) and the two baselines (use type only, majority class). The models fall from 0.73 to 0.77 (P1) to 0.70 to 0.72 (P1b), 0.59 (P2) and 0.53 to 0.55 (P3); the use-type baseline falls from 0.42 to 0.34 and the majority class stays near 0.16 to 0.17.](figures/fig5_protocol_ladder.png){width=90%}

Moving from P1 to P1b (new weeks of known buildings) cost the models 0.028 to 0.053. The largest step was from P1b to P2, when the buildings themselves were unseen: a fall of 0.113 to 0.128, to 0.586 to 0.589. P2 varied little across seeds (sd 0.004 to 0.009). All differences were computed before rounding.

Holding out whole sites (P3) lowered the scores further, to 0.552 for Random Forest, 0.527 for LightGBM and 0.525 for XGBoost. The drops from P2 were 0.037, 0.060 and 0.064. For Random Forest the P2 and P3 intervals overlap. For LightGBM and XGBoost they are separated by narrow margins of 0.010 and 0.015. These are marginal intervals on largely the same buildings, not a paired test. P3 also ran with a single seed, and the refitted cutoffs change the class balance of each test site. We therefore describe the P2-to-P3 change as a small drop of 0.04 to 0.06 that this design cannot firmly separate from noise.

Under P3 all three models stayed well above both baselines. Their lower interval bounds (0.499 to 0.524) lie above the upper bound of the use-type-only baseline (0.367), and the margin over that baseline was 0.184 to 0.211. The use-type baseline itself fell from 0.398 under P2 to 0.341 under P3. This suggests that the link between use type and performance class also differs between sites. The majority-class baseline stayed at 0.158 to 0.168, close to the value of 1/6 expected for balanced classes.

Averaged over the four protocols XGBoost scored highest, but under P3 Random Forest did. Row-level scores were lower throughout (for example 0.473 against 0.525 for XGBoost under P3; Appendix C), because averaging over a building's weeks removes week-to-week noise.

## 5.5 Per-site results

Figure 6 breaks the P3 result for LightGBM down by held-out site, next to the use-type-only baseline on the same fold. Table 7 lists the four best and four worst sites. The building-level weighted F1 of LightGBM ranged from 0.054 (Gator) to 0.857 (Lamb), with a median over sites of 0.517. LightGBM scored above the use-type baseline at 16 of the 18 sites. The two exceptions were Gator, where the baseline scored 0.555 against 0.054, and Eagle, where it scored 0.427 against 0.417.

**Table 7.** Best and worst held-out sites under P3 (LightGBM against the use-type-only baseline on the same leave-one-site-out fold; building-level scores). $n$ is the number of test buildings at the site.

| Site | $n$ | LightGBM weighted F1 | LightGBM accuracy | Use type only weighted F1 | Use type only accuracy |
|---|---|---|---|---|---|
| Lamb | 145 | 0.857 | 0.855 | 0.046 | 0.048 |
| Mouse | 7 | 0.714 | 0.714 | 0.600 | 0.714 |
| Bear | 86 | 0.664 | 0.651 | 0.374 | 0.453 |
| Crow | 5 | 0.633 | 0.600 | 0.000 | 0.000 |
| Bobcat | 22 | 0.419 | 0.455 | 0.369 | 0.364 |
| Eagle | 102 | 0.417 | 0.422 | 0.427 | 0.422 |
| Shrew | 9 | 0.390 | 0.444 | 0.167 | 0.333 |
| Gator | 67 | 0.054 | 0.164 | 0.555 | 0.642 |

![Figure 6. Leave-one-site-out results by held-out site: building-level weighted F1 of LightGBM (teal) and of the use-type-only baseline (gold) when that site is excluded from training; n is the number of test buildings. Sites are sorted by the LightGBM score, from Lamb (0.86, n = 145) to Gator (0.05, n = 67, where the baseline scores 0.56). LightGBM exceeds the baseline at 16 of 18 sites.](figures/fig6_per_site.png){width=88%}

Four sites have fewer than 15 test buildings: Crow (5), Mouse (7), Shrew (9) and Moose (12). At these sizes one building changes accuracy by between 0.08 and 0.20, so their scores carry little information. Among the larger sites, the spread is still wide. Lamb (145 buildings) scored 0.857 and Bear (86) 0.664, while the largest site, Rat (267 buildings), scored 0.442 against 0.352 for the baseline. At Lamb the use-type baseline almost always predicted the wrong class (0.046), while the load-shape model was mostly right. Gator shows the opposite pattern (Section 6.4).

## 5.6 Error analysis and features

Figure 7 shows the row-normalised confusion matrix of the Random Forest, the best model on the 100,000-row table with whole buildings held out. The correct-class rates were 56% for Low, 50% for Medium and 61% for High. Most errors fell into the neighbouring class: 30% of Low rows and 29% of High rows were predicted as Medium. Confusions between the extreme classes were less common (15% of Low rows predicted High, 10% of High rows predicted Low). Medium had the lowest per-class F1 (0.480), as in the reproduction.

![Figure 7. Row-normalised confusion matrix of Random Forest on the 100,000-row table with whole buildings held out (row-level, 20,000 test rows). Correct-class rates are 56% (Low), 50% (Medium) and 61% (High); most errors go to the neighbouring class, and only 15% of Low and 10% of High rows are assigned to the opposite extreme.](figures/fig7_confusion.png){width=55%}

Figure 8 shows the 15 most-used features of the served LightGBM model, trained on the whole-building 80% split of the 100,000-row table. Importance is measured as the number of splits that use a feature. The lag-1 h autocorrelation was used most (1,183 splits), followed by the normalised load at 06 h (864), at 22 h (835) and the 5th percentile of the normalised load (829). Eleven of the 15 are hour-of-day profile values, concentrated in the morning ramp (06 to 09 h) and the evening (17 to 23 h). Two describe base load relative to the mean: the 5th percentile and the minimum-to-maximum ratio (692). One plausible reading is that the timing of start-up and shut-down, and the size of the base load relative to the mean, carry most of the shape information about intensity. Split counts measure use, not causal effect, so this reading is a hypothesis.

![Figure 8. Top-15 features of the served LightGBM model by number of splits, scaled to the most-used feature (lag-1 h autocorrelation, 1,183 splits). The next are the normalised load at 06 h (864) and 22 h (835) and the 5th percentile of the normalised load (829); 11 of the 15 are hour-of-day profile values, and the minimum-to-maximum ratio (692) and lag-24 h autocorrelation (703) also appear.](figures/fig8_importance.png){width=80%}

# 6. Discussion

## 6.1 What the results support

Three statements are supported by the numbers. First, a performance-class classifier restricted to scale-free load-shape features scores in the same band as the published one when evaluated the same way: 0.595 to 0.607 against 0.604 to 0.622. This holds despite a different cohort, a feature set a quarter to a third the size, and assumed class cutoffs. Second, the classifier keeps real skill on buildings from sites it has never seen. Under leave-one-site-out evaluation it reached 0.525 to 0.552 building-level weighted F1, against 0.341 for a model that knows only the building's use type and 0.158 for the majority class, with non-overlapping intervals. Third, most of the fall from the most optimistic protocol to the strictest occurs when the building itself becomes unseen (P1b to P2). A smaller part occurs when the site becomes unseen (P2 to P3).

## 6.2 What the results do not support

The results do not show that random splits inflate scores as a general discovery. In the building-week table every week of a building carries the same label, so a random-row split is expected to reward recognition of buildings. The P1-to-P2 gap and the random-row advantage in Table 5 are partly a property of the table, and they say little about studies whose unit of analysis is the building. The results also do not establish a site effect of a particular size. The P2-to-P3 drop of 0.04 to 0.06 has overlapping intervals for Random Forest and narrowly separated marginal intervals for the other two models, from a single P3 run. We read it as a small drop that we cannot separate from noise. Finally, the results do not show an improvement on the paper: the best variant scored 0.604, and the paper's best pipeline scored 0.622.

## 6.3 Why an F1 near 0.8 is out of reach without leakage

At several points in the project a target of 80% F1 was discussed. We think it cannot be reached on this task without leakage, for four reasons.

First, the label is a ratio of two quantities that the features are designed not to see: the consumption level and the floor area. Load shape is related to intensity only indirectly, for example through operating hours or a large base load. Second, the classes are cuts through a continuous distribution, and many buildings sit near a boundary (Figure 2). The Medium band spans only a factor of 2.04, so even a good estimate of intensity from shape misclassifies many buildings near the cutoffs. The confusion matrices agree: most errors are between neighbouring classes (Figure 7), and Medium is the weakest class in every evaluation reported here. Third, the direct attempts to extract more from the same information failed. Use type added nothing (V1), tuning added nothing (V2), and predicting intensity directly and cutting it was worse (V3). The paper itself, with 221 non-consumption features, feature selection and automated pipeline search, reached 0.622.

Fourth, the obvious ways to raise the score are leakage. If mean consumption and floor area were inputs, the class would be a deterministic function of them, $e_b = 8760\,\bar{x}_b/A_b$ followed by two thresholds. A tree model would learn the thresholds and score close to 1. That score would be arithmetic on the label's ingredients, not a prediction, which is the kind of leakage defined by @kaufman2012 and @kapoor2023. Absolute consumption alone would leak part of the label too, because larger buildings use more energy. The base paper removed consumption-derived features for exactly this reason. Site identity as an input would be legitimate only if the model were always applied at sites already in its training data, and it cannot help under P3.

## 6.4 The Gator site

Gator is the clearest failure. With Gator held out, LightGBM classified 16.4% of its 67 buildings correctly (weighted F1 0.054). That is well below the one-third expected from guessing, so the errors are systematic, not random. On the same fold the use-type-only baseline classified 64.2% correctly (weighted F1 0.555). So at Gator, use type predicted the class far better than its average across sites (0.341 under P3), while load shape pointed the wrong way.

The result files do not record Gator's class distribution or the predicted classes, so we cannot diagnose the cause. Several explanations are possible, and none has been tested. Gator's buildings may combine load shapes with intensities in a way that differs from the other 17 sites, for example through climate or through a different split of building services between the electricity meter and other meters. Floor-area or metering conventions at the site may differ. Or Gator's buildings may be concentrated in one class under the refitted cutoffs, which would make any systematic shift costly. Whatever the cause, Gator shows that a model with good average skill on unseen sites can fail badly on a particular site. The per-site spread in Figure 6 is as important as the mean.

## 6.5 Practical meaning of whole-site generalisation

P3 is the protocol closest to a common use case: a model trained on labelled buildings from existing portfolios is applied to a new campus or city where no building has been labelled. In that setting our models placed buildings in the correct tercile about half the time, and the building-level accuracy of Random Forest under P3 was 0.548. That is clearly better than knowing only use type, and it could support screening: ranking buildings for a closer look, or flagging likely high-intensity buildings for metering checks. It is not accurate enough to certify a building's class.

Two cautions apply. Per-site results range from 0.054 to 0.857, so the score at a new site is hard to predict; a check against the use-type baseline or a few labelled buildings at that site would be needed before relying on the model, and we did not test such a step. And the classes are relative to this cohort: they say how a building compares with BDG2 buildings, not whether it meets an absolute standard.

## 6.6 Relation to the base paper's protocol

The base paper's unit of analysis is the building and its test set is a random 20% of buildings, so its protocol corresponds to our P2 (unseen buildings at seen sites), not to the row-level P1, and its scores do not contain the recognition effect that raises P1. Our P2 scores (0.586 to 0.589) and our building-table reproduction (0.595 to 0.607) sit near the paper's 0.604 to 0.622, consistent with this correspondence, although features and aggregation differ. What the paper's protocol does share is the site. Our P3 results suggest that moving to unseen sites costs a further small amount on average, 0.04 to 0.06 in our runs, and a large amount at some sites. Reporting a site-held-out score next to the random-split score, together with a use-type baseline, would make such classifiers easier to judge.

# 7. Limitations and threats to validity

- **Cohort and labels.** Our cohort (1,470 buildings, 18 sites) and cleaning rules differ from the paper's (1,494 buildings). The class cutoffs are our assumption (terciles), because the paper's are unpublished and its repository held no code when checked. A different rule, for example unequal classes or fixed thresholds, would change every score. In the reproduction, the improvement attempt, P1 and P1b, the cutoffs were computed on the whole cohort including test buildings. This uses no feature information, and its effect on two quantiles of 1,470 values is small, but it is not a strict separation.
- **Features.** The 54 to 66 scale-free features are a subset of the kind of information in the paper's 221 non-consumption features. They omit pattern-mining and regression-model features entirely, so the comparison with the paper is indicative.
- **Gradient Boosting settings.** Gradient Boosting ran with library defaults, because the paper gives no tuned values for this target. It was excluded from the protocol ladder for compute reasons.
- **Seeds.** P1, P1b and P3 were run with a single seed, and only P2 was repeated over five seeds. The 100,000-row two-split comparison used one split per protocol.
- **Intervals.** The bootstrap intervals cover test-set sampling for fixed fitted models, not refitting variation. They are marginal, not paired, across protocols, and for P2 they are means of per-seed bounds. The P3 intervals in particular should not be read as a test of the P2-to-P3 difference.
- **Sampling and row dependence.** The ladder used a 30,000-row sample of the 100,000-row table, about 20 weeks per building, to fit the available hardware (4 CPU cores, 6 GB memory, no GPU). Rows of one building are dependent, so row-level scores overstate the effective sample size. We addressed this with building-level scoring, but the random-row results remain partly determined by the table's design.
- **Confounded protocol steps.** Each step of the ladder changes more than one thing: whether buildings or sites are unseen, and the class balance of the test fold after cutoffs are refitted. The experiment does not separate these.
- **Per-site sample sizes.** Four sites have fewer than 15 test buildings, so their per-site scores are very uncertain. The Gator failure could not be diagnosed from the stored results.
- **Prior-art search.** Our search covered abstracts and open-access full text in OpenAlex, Europe PMC and arXiv. Two candidate papers (Yan et al., 2023; Li et al., 2024) were paywalled and not read. @piscitelli2023, @lien2024 and @spencer2025 were read only as far as their abstracts. @vivekanandan2026 and @li2026 were read only as far as their repository records. Not finding a prior study is not proof that none exists, so the novelty statement is limited to what these searches covered.
- **Scope.** The study used electricity only, and the 2016 to 2017 data of one public collection. The models are research and teaching prototypes, not energy audits.

# 8. Conclusion

We reproduced the performance-class classification of @raymand2023 on BDG2 with transparent, scale-free load-shape features that cannot encode consumption level or floor area. With the paper's random building split, four tree ensembles scored 0.595 to 0.607 weighted F1, in the same band as the paper's 0.604 to 0.622. A pre-specified, leakage-free attempt to improve on the paper did not succeed. We argue that substantially higher scores on this label require inputs built from its own ingredients. A ladder of stricter protocols showed that building-level weighted F1 falls from 0.73 to 0.77 with random rows to 0.59 with unseen buildings, and to 0.53 to 0.55 with unseen sites. The scores stay well above a use-type-only baseline (0.34) and a majority-class baseline (0.16). The step to unseen sites was small on average and not firmly separable from noise, but per-site results ranged from 0.05 to 0.86. To our knowledge, within the searches described above, no prior study has evaluated BDG2 performance-class classification with whole sites held out. Future work could repeat P3 over several seeds and with paired intervals, add the paper's pattern-based features, and test whether a few labelled buildings at a new site can correct failures such as Gator's.

# Reproducibility and software availability

All code, tests and result files are in the repository <https://github.com/MuhammadTalhaZafar10/devproj>. The data are the public BDG2 files (`electricity.csv`, `metadata.csv`, `weather.csv`) from the Kaggle distribution of @miller2020. They are fetched with `python -m backend.scripts.fetch_data` and are not redistributed. From the repository root, the results are regenerated with these commands, in this order:

1. `python -m backend.scripts.run_cls_reproduction --rebuild`. This builds both tables, runs the reproduction and the two-split comparison, and writes `reports/classification_results.json` and the served model.
2. `python -m backend.scripts.export_bundle`. This packages the served model for the API.
3. `python -m backend.scripts.run_cls_novelty`. This runs the protocol ladder and writes `reports/cls_novelty_results.json`.
4. `python -m backend.scripts.run_cls_improve`. This runs the improvement attempt and writes `reports/cls_improvement.json`.
5. `python -m backend.scripts.sync_frontend_data`. This copies the result files to the web page.
6. `python docs/paper/make_figures.py`. This draws Figures 1 to 9 from the result files.

Seeds are as follows:

- Week sampling for the 100,000-row table: 42.
- Reproduction and improvement splits: 0 to 4. In the reproduction the model seed equals the split seed.
- Two-split comparison and served model: 42.
- Ladder sample: 42. P1 folds: 0. P2 folds: 0 to 4.
- Model seed in the ladder: 42.
- Bootstrap: 0.

Software: Python 3.11.2, scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.2.0, numpy 2.4.6 and pandas 3.1.1. Hardware: 4 CPU cores, 6 GB memory, no GPU.

The test suite includes the scale-invariance test of Section 4.1; 46 backend tests and 6 frontend tests passed locally. A Flask API (`/health`, `/model-info`, `/results`, `/novelty`, `/examples`, `/predict`) serves the model; `/predict` accepts 168 hourly readings or the 54 features and returns HTTP 503 when no trained model is present. An Angular page shows the stored results and, when the API is reachable, classifies a submitted week. On 30 held-out single weeks the API returned the correct class 14 times (47%), consistent with the row-level whole-building score, since one week is a noisy sample of a building.

# Declarations

**Declaration of AI assistance.** The code and the manuscript text were prepared with AI assistance (Claude models, Anthropic) under the authors' direction. All content, including every number, citation and statement, must be checked by the authors, who take responsibility for it.

**Funding.** [Funding statement to be completed by the authors.]

**Competing interests.** [Competing-interests statement to be completed by the authors.]

# References

::: {#refs}
:::

# Appendix A. Hyperparameters of all models

**Table A1.** Model settings. Unlisted parameters are library defaults (scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.2.0). "Seed" is the model's random state.

| Model | Used in | Settings | Seed |
|---|---|---|---|
| Random Forest classifier | Reproduction, two splits, ladder, V0, V1 | 100 trees; criterion entropy; maximum features 0.2; minimum samples per leaf 3; minimum samples per split 9; bootstrap on; 4 parallel jobs (paper's Appendix Table B.2) | Split seed (reproduction, V0 to V3); 42 (two splits, ladder) |
| Gradient Boosting classifier | Reproduction, two splits | scikit-learn defaults | Split seed (reproduction); 42 |
| LightGBM classifier | Reproduction, two splits, ladder, use-type baseline, served model | 300 trees; learning rate 0.05; 31 leaves; 4 parallel jobs; logging off | Split seed (reproduction); 42 |
| XGBoost classifier | Reproduction, two splits, ladder | 300 rounds; learning rate 0.1; maximum depth 6; histogram tree method; evaluation metric multi-class log loss; 4 parallel jobs | Split seed (reproduction); 42 |
| Majority class | Ladder | scikit-learn dummy classifier, strategy "prior" | Not applicable |
| Use type only | Ladder | LightGBM as above on one-hot primary space usage with a missing-value indicator | 42 |
| V2 tuned Random Forest | Improvement | Randomised search, 8 draws, stratified 5-fold CV, weighted-F1 scoring; trees 150 to 349; maximum features uniform on 0.1 to 0.6; minimum samples per leaf 1 to 7; minimum samples per split 2 to 15; criterion Gini or entropy | Split seed |
| V3 Random Forest regressor | Improvement | 400 trees; maximum features 0.3; minimum samples per leaf 3; target $\log e_b$; output cut at the training terciles of $\log e_b$ | Split seed |

# Appendix B. Feature names

All features are computed on the mean-normalised series $r$ (Section 4.1) by `backend/cls/features.py`. Names are as stored in the tables.

- **Hour-of-day profile (24):** `hour_00`, `hour_01`, `hour_02`, `hour_03`, `hour_04`, `hour_05`, `hour_06`, `hour_07`, `hour_08`, `hour_09`, `hour_10`, `hour_11`, `hour_12`, `hour_13`, `hour_14`, `hour_15`, `hour_16`, `hour_17`, `hour_18`, `hour_19`, `hour_20`, `hour_21`, `hour_22`, `hour_23`.
- **Day-of-week profile (7):** `dow_0` (Monday), `dow_1`, `dow_2`, `dow_3`, `dow_4`, `dow_5`, `dow_6` (Sunday).
- **Weekend ratio (1):** `weekend_ratio`.
- **Day periods (4):** `period_night` (00 to 06 h), `period_morning` (06 to 12 h), `period_afternoon` (12 to 18 h), `period_evening` (18 to 24 h).
- **Spread and shape (5):** `cv`, `skew`, `kurtosis`, `load_factor`, `min_max_ratio`.
- **Quantiles (5):** `q05`, `q25`, `q50`, `q75`, `q95`.
- **Timing (2):** `peak_hour`, `trough_hour`.
- **Daily load ratio (2):** `daily_load_ratio_mean`, `daily_load_ratio_std`.
- **Autocorrelation (2):** `acf_lag1`, `acf_lag24`.
- **Zero fraction (1):** `zero_fraction` (computed on the raw series).
- **Weather (1):** `temp_corr`.

These 54 features make up the building-week table. The building table adds 12 monthly means, `month_01` to `month_12`, for 66 features.

# Appendix C. Protocol details and run times

**Table C1.** Fold construction of the protocol ladder (30,000-row sample, seed 42; 1,000 bootstrap resamples over buildings, seed 0).

| Protocol | Splitter | Repeats | Label cutoffs | Test buildings |
|---|---|---|---|---|
| P1 | Stratified 3-fold on rows | Seed 0 | Whole-cohort terciles (table label) | 1,467 |
| P1b | Train on weeks starting in 2016, test on weeks starting in 2017 | One split | Whole-cohort terciles (table label) | 1,462 |
| P2 | Stratified group 3-fold, grouped by building | Seeds 0 to 4; mean and sd over seeds; interval is the mean of per-seed bounds | Refitted on each fold's training buildings | 1,467 |
| P3 | Leave one site out over 18 sites (Bear, Bobcat, Bull, Cockatoo, Crow, Eagle, Fox, Gator, Hog, Lamb, Moose, Mouse, Panther, Peacock, Rat, Robin, Shrew, Wolf); sites with fewer than 5 sampled buildings would stay in training (none) | One run | Refitted on each fold's training buildings | 1,467 |

**Table C2.** Row-level weighted F1 of the protocol ladder (every test week scored as one item). P2 values are means ± sd over five seeds.

| Protocol | Random Forest | LightGBM | XGBoost | Majority class | Use type only |
|---|---|---|---|---|---|
| P1 | 0.675 | 0.661 | 0.675 | 0.174 | 0.418 |
| P1b | 0.605 | 0.597 | 0.601 | 0.177 | 0.412 |
| P2 | 0.545 ± 0.003 | 0.541 ± 0.005 | 0.537 ± 0.005 | 0.172 ± 0.001 | 0.401 ± 0.008 |
| P3 | 0.492 | 0.480 | 0.473 | 0.164 | 0.344 |

**Table C3.** Model fitting times in seconds (4 CPU cores; Gradient Boosting is single-threaded). Building table: mean (sd) over the five reproduction splits, one fit on the 80% training part. Building-week table: one fit on 80,000 rows. The ladder's result file does not record run times.

| Model | Building table (1,176 rows) | Building-week table, random rows | Building-week table, whole buildings |
|---|---|---|---|
| Random Forest | 1.3 (0.2) | 136 | 179 |
| Gradient Boosting | 7.2 (0.5) | 805 | 954 |
| LightGBM | 9.8 (5.8) | 441 | 311 |
| XGBoost | 11.2 (0.6) | 237 | 259 |
