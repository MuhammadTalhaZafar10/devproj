# Gradient-Boosting and Tree-Ensemble Benchmarks for Building Energy Prediction (2021–2026), with emphasis on BDG2

Scope note: this note covers peer-reviewed work published 2021 or later that benchmarks LightGBM, XGBoost, CatBoost, Random Forest (RF), generic gradient boosting machines (GBM), and/or AdaBoost on the Building Data Genome Project 2 (BDG2) or on closely comparable whole-building hourly/sub-hourly energy data. The BDG2 data descriptor itself (Miller et al., 2020) and the ASHRAE Great Energy Predictor III (GEPIII) competition overview (2020) predate the window and are cited only as dataset/protocol background, clearly marked.

---

## Which post-2021 papers benchmark LightGBM, XGBoost, CatBoost, or Random Forest on BDG2 or closely comparable whole-building hourly energy data?

### Takeaway
A small but usable set exists. The single most directly reusable benchmark is Xie & Tummala (2025, *Sensors*), which runs exactly RF, AdaBoost, GBM, XGBoost, LightGBM and CatBoost on a 10-building BDG2 subset (175,440 hourly observations) under one pipeline and reports RMSE/MAE/R²/MAPE plus a random-vs-chronological validation comparison. Supporting evidence comes from BDG2-specific applications (LightGBM for training-length sensitivity; BDG2 building-characteristic classification), a single-building 15-min LightGBM/XGBoost/CatBoost/RF comparison, and GEPIII-based lessons-learned work that is qualitative rather than numeric.

### Cited Findings

**Primary BDG2 benchmark (all six model families, one protocol)**

- Xie, Y., & Tummala, S. P. (2025). *Machine Learning for Sensor Analytics: A Comprehensive Review and Benchmark of Boosting Algorithms in Healthcare, Environmental, and Energy Applications.* **Sensors**, 25(23), 7294. DOI: 10.3390/s25237294. Open access (PMC12694449). — The energy case study uses **BDG2**: 10 buildings stratified over five primary-use categories (education ×3, lodging ×2, office ×1, parking ×2, retail ×2), 17,544 hourly samples per building, **175,440 observations total**, 2016–2017, **1-hour-ahead horizon**. Building IDs given: Panther_education_Aine/Bella/Ceil, Panther_lodging_Adora/Cora, Panther_office_Darnell, Panther_parking_Erna/Lorriane, Panther_retail_Felix/Gale. — [Source](https://doi.org/10.3390/s25237294)
- The same paper states the six-model design explicitly: "Bagging baseline: Random Forest … Classical boosting: AdaBoost … modern gradient boosting frameworks (XGBoost, LightGBM, and CatBoost)" under "an identical leakage-safe pipeline". — [Source](https://doi.org/10.3390/s25237294)
- BDG2 scope as described there: "full Building Data Genome 2 dataset containing 1578 buildings … hourly electricity meter readings for 1578 non-residential buildings across six countries … 2016–2017" (note: this is the electricity subset; the BDG2 descriptor itself reports 1,636 buildings / 3,053 meters / 19 sites). — [Source](https://doi.org/10.3390/s25237294); dataset descriptor: Miller et al. (2020), *Scientific Data* 7, 368, DOI 10.1038/s41597-020-00712-x — [Source](https://doi.org/10.1038/s41597-020-00712-x)

**BDG2 applications using boosting**

- Chen, Y., & Chen, Z. (2022). *Short-term load forecasting for multiple buildings: A length sensitivity-based approach.* **Energy Reports**, 8, 14274–14288. DOI: 10.1016/j.egyr.2022.10.425. — Uses **LightGBM** to derive per-building CV(RMSE)-vs-training-length curves on the Building Data Genome, then k-means to flag length-sensitive buildings: "approximately 20% of buildings in the Building Data Genome are labeled as length-sensitive buildings, and adopting appropriate training data lengths can reduce the prediction error of these buildings by up to 15%." Day-ahead forecasting task. — [Source](https://doi.org/10.1016/j.egyr.2022.10.425)
- Raymand, F., Najafi, B., Haghighat Mamaghani, A., Moazami, A., & Rinaldi, F. (2023). *Machine learning-based estimation of buildings' characteristics employing electrical and chilled water consumption data: Pipeline optimization.* **Energy and Buildings**, 295, 113327. DOI: 10.1016/j.enbuild.2023.113327. — Uses **BDG2 (1,636 buildings)** for three classification pipelines: (A) use type, (B) **performance class**, (C) operation group, from hourly electrical and chilled-water consumption. Feature generation + selection + pipeline optimization improved accuracy by **5.3%, 2.9%, and 3.9%** for A/B/C versus a benchmark model while cutting feature counts by **94.7%, 88.3%, 89.4%**. — [Source](https://doi.org/10.1016/j.enbuild.2023.113327)

**Comparable whole-building data (not BDG2)**

- Koukaras, P., Mustapha, A., Mystakidis, A., & Tjortjis, C. (2024). *Optimizing Building Short-Term Load Forecasting: A Comparative Analysis of Machine Learning Models.* **Energies**, 17(6), 1450. DOI: 10.3390/en17061450. Open access. — One university building in Thessaloniki, Greece; 15-min resolution; chronological split ("first … to train the models and the last six months to test"); compares HGBR, Extra Trees (ETR), **LightGBM (LGBMR)**, CatBoost (CBR), GBR, Elastic Net, Bayesian Ridge, etc. Results: 1-step/15-min — HGBR R²=0.939, ETR 0.939, **LGBMR 0.938** (RMSE 0.116, MAE 0.071, 11.2 s training), CBR 0.929; 4-step/15-min — HGBR 0.884, **LGBMR 0.884** (RMSE 0.159, MAE 0.092), ETR 0.879, GBR 0.878, **RFR 0.868**, CBR 0.855. LightGBM is the best-or-tied model in both tables at a fraction of ETR/RFR training time. — [Source](https://doi.org/10.3390/en17061450)
- Muqtadir, A., Li, B., Zhou, Y., Chen, S., & Kazmi, S. N. (2025). *Nowcasting the next hour of residential load using boosting ensemble machines.* **Scientific Reports**, 15, 7157. DOI: 10.1038/s41598-025-91767-6. — Integrates **LightGBM + XGBoost + CatBoost** and evaluates on "13 residential locations in North America and Europe"; reports **RMSLE = 0.1898, R² = 0.9745** for the fused model, with RMSE/CVRMSE/MBE and an ablation over incremental model addition. Cites BDG2 but the dataset description does not match BDG2's site structure; treat as comparable-load evidence, not a BDG2 result. — [Source](https://doi.org/10.1038/s41598-025-91767-6)

**GEPIII / competition-scale evidence (qualitative)**

- Miller, C., Hao, L., & Fu, C. (2022). *Gradient boosting machines and careful pre-processing work best: ASHRAE Great Energy Predictor III lessons learned.* arXiv:2202.02898 (published in ASHRAE Transactions). — Survey of top-5% GEPIII teams plus analysis of shared notebooks: "The top-performing solutions mostly used ensembles of Gradient Boosting Machine (GBM) tree-based models, with the LightGBM package being the most popular"; LightGBM, XGBoost and CatBoost "show up as the top three libraries used by the survey respondents and the winning solutions"; pre-processing and feature engineering were rated extremely/very important by **>70%** of respondents; winners emphasised removing outliers/anomalous behaviour. — [Source](https://arxiv.org/abs/2202.02898)
- Miller, C., Picchetti, B., Fu, C., & Pantelic, J. (2022). *Limitations of machine learning for building energy prediction: ASHRAE GEPIII Kaggle competition error analysis.* **Science and Technology for the Built Environment**, 28(5), 610–627. DOI: 10.1080/23744731.2022.2067466. — Test set: "two years of hourly whole building readings from **2380 meters in 1448 buildings at 16 locations**"; aggregating the top-50 solutions, errors were within acceptable range (RMSLE_scaled < 0.1) on **79.1%** of test data, 0.1–0.3 on **16.1%**, and >0.3 on **4.8%**. — [Source](https://doi.org/10.1080/23744731.2022.2067466)
- GEPIII competition overview (background, 2020): Miller et al., **Science and Technology for the Built Environment** 26(9), DOI 10.1080/23744731.2020.1795514 — 39,402 submissions; winners were GBM/LightGBM ensembles. — [Source](https://doi.org/10.1080/23744731.2020.1795514)

**Method citations for the model families (all pre-2021 but standard)**

- LightGBM: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 30, 3146–3154. — [Source](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- XGBoost: Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16, 785–794. DOI: 10.1145/2939672.2939785. — [Source](https://doi.org/10.1145/2939672.2939785)
- CatBoost: Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: unbiased boosting with categorical features.* NeurIPS 31 (arXiv:1706.09516). — [Source](https://arxiv.org/abs/1706.09516)
- Random Forest: Breiman, L. (2001). *Random Forests.* Machine Learning 45, 5–32. DOI: 10.1023/A:1010933404324. — [Source](https://doi.org/10.1023/A:1010933404324)
- GBM: Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine.* Annals of Statistics 29(5), 1189–1232. DOI: 10.1214/aos/1013203451. — [Source](https://doi.org/10.1214/aos/1013203451)
- AdaBoost: Freund, Y., & Schapire, R. E. (1997). *A decision-theoretic generalization of on-line learning and an application to boosting.* Journal of Computer and System Sciences 55(1), 119–139. DOI: 10.1006/jcss.1997.1504. — [Source](https://doi.org/10.1006/jcss.1997.1504)

### Inferences
- For a LightGBM + RF study on BDG2, Xie & Tummala (2025) is the natural head-to-head citation: it gives directly comparable numbers for exactly those two families on the same 10-building BDG2 subset, plus CatBoost/XGBoost/GBM/AdaBoost as context.
- The published BDG2 boosting literature is thin on independent replication: most 2021–2026 BDG2 papers use boosting as a tool (forecasting, classification, benchmarking) rather than as the object of a model comparison. Only one paper (Sensors 2025) provides a full six-model table on BDG2.
- The GEPIII literature supports the model-choice justification qualitatively (GBM/LightGBM dominated a 4,000-participant competition), not quantitatively on BDG2.

### Gaps
- No 2021–2026 paper was found that benchmarks **CatBoost specifically on BDG2 against LightGBM and RF with a building-holdout protocol**; CatBoost appears only in the mixed 10-building BDG2 subset of Sensors 2025.
- The Electrical Engineering 2025 paper "Improving energy management practices … LightGBM, RF, and XGBoost … with advanced optimization strategies" (Dai & Huang, DOI 10.1007/s00202-025-03167-8) is topically ideal but its abstract and full text could not be retrieved (publisher paywall/empty OpenAlex abstract), so its dataset and metrics are unverified and not reported here.
- Several BDG2-citing 2025–2026 papers (e.g., *Buildings* 16(16) 3343 robustness study, *Energy and Buildings* 2026 "Beyond RMSE") were not accessible in full text and are not characterised here.

---

## What performance (accuracy, F1, RMSE, MAE, R², MAPE) is reported, on what validation protocol?

### Takeaway
On BDG2 1-hour-ahead load, the best reported chronological-split numbers cluster at **R² ≈ 0.59–0.76, RMSE ≈ 5.7–6.6 kWh, MAPE ≈ 8–10%**, with CatBoost best and RF/RMSE weaker; under random k-fold CV the same models appear at **R² ≈ 0.90–0.99**. The gap between those two regimes (ΔR² 0.63–0.74) is the single most important number for benchmarking claims.

### Cited Findings

**BDG2, 1-h ahead, six models (Xie & Tummala 2025, Table 14; per-building simple average, 10 buildings)**

| Model | RMSE (kWh) | MAE (kWh) | R² | MAPE (%) | Train time (s) |
|---|---|---|---|---|---|
| Random Forest | 6.62 | 4.23 | 0.591 | 9.6 | 57.73 |
| AdaBoost | 9.00 | 6.65 | 0.661 | 13.3 | 216.87 |
| GBM | 6.33 | 4.04 | 0.709 | 8.9 | 416.49 |
| XGBoost | 6.12 | 3.95 | 0.692 | 8.8 | 2.28 |
| LightGBM | 6.06 | 3.95 | 0.723 | 8.6 | 3.35 |
| CatBoost | 5.67 | 3.82 | 0.764 | 8.3 | 3.41 |

- Reported protocol: "split 70% training / 30% testing via **chronological splits**"; metrics are unweighted averages across the ten buildings ("each building weighted equally"); building-specific RMSE ranges **3.2 kWh/h (small retail) to 12.8 kWh/h (large education)**; CatBoost lowest error in **8 of 10** buildings. — [Source](https://doi.org/10.3390/s25237294)
- Statistical testing: Friedman χ²(5) = 19.3, **p = 0.002**; Nemenyi: CatBoost vs AdaBoost p < 0.001, CatBoost vs RF **p = 0.027**; all pairwise comparisons among CatBoost/XGBoost/LightGBM/GBM **p > 0.05** (i.e., modern GBDT variants statistically indistinguishable on 10 buildings). Reported headline improvements: CatBoost 14% lower RMSE than RF, "33.1% improvement over AdaBoost; 31.3% over Random Forest". — [Source](https://doi.org/10.3390/s25237294)
- Caution flag from the same source: AdaBoost has a *higher* R² (0.661) than RF (0.591) despite much worse RMSE (9.00 vs 6.62 kWh); the authors attribute RF's low R² to occasional **negative R² for volatile sites** (parking structures with weekday/weekend regime reversals). This makes the R² column internally non-monotonic with the error columns and should be reported as-is, not "cleaned up". — [Source](https://doi.org/10.3390/s25237294)

**Random k-fold vs chronological/forward-chaining (same BDG2 subset, Xie & Tummala 2025, Table 17)**

| Model | Random CV R² | Time-series CV R² | Gap ΔR² |
|---|---|---|---|
| Random Forest | 0.986 ± 0.008 | 0.287 ± 0.762 | 0.699 |
| AdaBoost | 0.903 ± 0.076 | 0.236 ± 0.794 | 0.667 |
| GBM | 0.967 ± 0.022 | 0.340 ± 0.734 | 0.628 |
| XGBoost | 0.985 ± 0.009 | 0.247 ± 0.832 | 0.738 |
| LightGBM | 0.984 ± 0.010 | 0.303 ± 0.689 | 0.681 |
| CatBoost | 0.986 ± 0.009 | 0.287 ± 0.749 | 0.699 |

- Text: "Random CV yields inflated R² (0.90–0.99) since temporally adjacent samples share nearly identical load patterns. When evaluated chronologically, true predictive power drops to 0.24–0.34, leading to gaps ΔR² = 0.63–0.74 across models"; the time-series protocol is **5 forward-chaining folds that strictly preserve chronology**. — [Source](https://doi.org/10.3390/s25237294)
- The large standard deviations (±0.69–0.83) on the time-series R² indicate that per-building results range from strongly positive to strongly negative; the mean is not a safe point estimate for a single building. — [Source](https://doi.org/10.3390/s25237294) (inference from reported ± values)

**Comparable hourly/sub-hourly building data**

- Energies 2024 (single university building, 15-min, chronological 70/30): **LightGBM R² = 0.938** (RMSE 0.116, MAE 0.071) for 1-step; **R² = 0.884** (RMSE 0.159, MAE 0.092) for 4-step; RF (RFR) 0.868 at 4-step; CatBoost 0.929 / 0.855; HGBR 0.939 / 0.884. Random Forest training time 1,944 s vs LightGBM 11.2 s at 1-step. — [Source](https://doi.org/10.3390/en17061450)
- Sci Rep 2025 (13 residential locations, 1-h nowcast): LightGBM+XGBoost+CatBoost fusion achieves **RMSLE 0.1898 and R² 0.9745**; individual-model figures are in the paper's tables but were not extractable in this pass. — [Source](https://doi.org/10.1038/s41598-025-91767-6)
- GEPIII error bounds (top-50 solutions): acceptable error (RMSLE_scaled < 0.1) on 79.1% of test rows; 16.1% in 0.1–0.3; 4.8% > 0.3. — [Source](https://doi.org/10.1080/23744731.2022.2067466)

**Interpretability-adjacent result (same BDG2 benchmark)**

- SHAP rank stability across folds (Spearman ρ): RF **0.822**, CatBoost 0.812, XGBoost 0.775, LightGBM 0.756, GBM 0.708, AdaBoost 0.650 — i.e., RF is the most stable explainer even though it is not the most accurate. — [Source](https://doi.org/10.3390/s25237294)

### Inferences
- A defensible benchmark table for a new LightGBM + RF BDG2 study should report at minimum: RMSE (kWh), MAE, R², MAPE, per-building spread, and the validation protocol, because the same model moves by ΔR² ≈ 0.68 depending on protocol.
- Published "BDG2 LightGBM R² ≈ 0.72" and "RF R² ≈ 0.59" (chronological) are the fairest external comparators; quoting R² ≈ 0.98 from a random split would be non-comparable and is exactly the failure mode documented in the same paper.
- The Sensors 2025 Table 14 (chronological, R² 0.59–0.76) and Table 17 random-CV column (R² 0.90–0.99) are not directly reconcilable from the text; the safest reading is that Table 14 is the intended deployment estimate and Table 17 quantifies protocol sensitivity.

### Gaps
- No paper was found reporting **F1 or accuracy for a regression task** on BDG2 (expected — regression dominates); classification metrics on BDG2 are only available for building-attribute classification, not consumption-level prediction.
- No BDG2 paper reported results under a strict **building-holdout** (leave-buildings-out) protocol with boosting; all located BDG2 benchmarks use temporal or random splits within buildings. This is a genuine methodological gap in the literature, not just a search limitation.
- Absolute kWh RMSE values are not comparable across papers without normalisation (building size differs by ~4× within the Sensors subset alone).

---

## Do any papers report classification of consumption into Low/Medium/High or similar ordinal bins, and what accuracy is typical?

### Takeaway
No 2021–2026 peer-reviewed paper was found that trains LightGBM/XGBoost/CatBoost/RF to classify **hourly consumption into Low/Medium/High bins on BDG2 and reports accuracy/F1**. The closest published practice is (a) BDG2 **building performance-class** classification (ordinal building-level label, not hourly bins), (b) quantile-derived low/medium/high benchmarking lines, and (c) building use-type classification. Typical reported classification gains are in the 3–5 percentage-point range from pipeline optimisation, but headline accuracy figures could not be verified.

### Cited Findings
- BDG2 performance-class classification: Raymand et al. (2023) define pipeline (B) "performance class" alongside (A) use type and (C) operation group, using hourly electrical and chilled-water data from **1,636 BDG2 buildings**; feature selection/optimisation improved accuracy by **2.9% for the performance-class pipeline** (5.3% use type, 3.9% operation group) versus a benchmark model, while removing 88.3% of features. — [Source](https://doi.org/10.1016/j.enbuild.2023.113327)
- Quantile-defined low/medium/high consumption lines: Wu, T., Wang, B., Zhang, D., Zhao, Z., & Zhu, H. (2023). *Benchmarking Evaluation of Building Energy Consumption Based on Data Mining.* **Sustainability**, 15(6), 5211. DOI: 10.3390/su15065211. — Random Forest screens building features contributing to energy intensity and building class; K-means reclassifies buildings; "the quartiles of buildings in the category of new buildings are calculated to obtain the **low energy consumption line, medium energy consumption line and high energy consumption line**"; cluster quality improved (DBI −60.8%, CH ×3.73). This is a *binning/benchmarking* method, not a supervised Low/Med/High classifier with accuracy. — [Source](https://doi.org/10.3390/su15065211)
- Consumption-pattern classification as a preprocessing step (not ordinal bins): Dong, Z., Liu, J., Liu, B., Li, K., & Li, X. (2021). *Hourly energy consumption prediction of an office building based on ensemble learning and energy consumption pattern classification.* **Energy and Buildings**, 241, 110929. DOI: 10.1016/j.enbuild.2021.110929. — Classifies daily consumption patterns first, then applies ensemble learning per pattern; abstract/full text not retrievable in this pass, so no accuracy figure is reported here. — [Source](https://doi.org/10.1016/j.enbuild.2021.110929)
- GEPIII building-level heterogeneity is a warning for any binning scheme: errors are concentrated in specific buildings/timeframes, with 4.8% of test rows effectively unpredictable from meter+weather+metadata inputs. — [Source](https://doi.org/10.1080/23744731.2022.2067466)

### Inferences
- If a study introduces Low/Medium/High consumption classes on BDG2, it is likely **novel relative to the published literature** and should say so, citing Raymand et al. (performance class) and Wu et al. (quantile lines) as the nearest precedents rather than implying an established benchmark exists.
- Class boundaries derived from global quantiles will interact strongly with building type (education vs lodging vs parking load shapes differ by an order of magnitude), so per-building or per-use-type quantiles are the defensible choice; otherwise the classifier mostly learns building identity.
- Because boosting models are near-ceiling on this kind of tabular task, a Low/Med/High classifier will likely report high accuracy that is inflated by class imbalance and temporal autocorrelation; balanced accuracy / macro-F1 and a chronological split are the honest metrics.

### Gaps
- **No verifiable accuracy/F1 figure for a Low/Medium/High consumption classifier on BDG2 was found.** Multiple searches (OpenAlex full-text, Crossref bibliographic, arXiv) returned only forecasting studies or building-attribute classification.
- Raymand et al.'s actual classifier family (RF vs XGBoost vs LightGBM) and its absolute accuracy/F1 are behind a paywall in this session; only the relative improvement percentages are verified. The paper should be obtained via NTNU/PoliMi repository or institutional access before quoting absolute numbers.
- Whether the "performance class" label in Raymand et al. is ordinal (e.g., low/medium/high EUI) or categorical could not be confirmed from the abstract.

---

## What feature engineering is standard (calendar/cyclical time, lags, rolling statistics, weather, building metadata)?

### Takeaway
The de-facto BDG2 boosting feature set is: lagged load (1–24 h), rolling statistics (6/12/24 h mean, std, min, max), weather (temperature, humidity, wind, cloud), calendar encodings (hour-of-day, day-of-week, month, holiday), degree days (HDD/CDD base 65 °F), and static building metadata (floor area, primary use, vintage). Site-specific (not global) weather normalisation is used to avoid forcing one temperature response across climate zones.

### Cited Findings
- Xie & Tummala (2025) BDG2 feature list, verbatim: "lagged consumption (1–24 h), rolling statistics (6/12/24-h mean/std/min/max capturing load patterns), weather variables (temperature, humidity, wind speed, and cloud cover from co-located NOAA stations), and calendar encodings (hour-of-day, day-of-week, month, binary holiday indicator, and heating/cooling degree-day base 65 °F). Building metadata (square footage, primary-use category, and vintage) provide static context." — [Source](https://doi.org/10.3390/s25237294)
- Normalisation practice: "Weather variables were normalized using z-score standardization within each building's local climate context rather than global normalization across all sites … computed mean and standard deviation … from its training period and applied consistent scaling to both training and test sets." — [Source](https://doi.org/10.3390/s25237294)
- Quality filtering that materially changes results: removal of "commissioning periods (first 30 days after meter installation), HVAC shutdown events (consecutive zeros > 48 h …), and extreme outliers (>3× interquartile range …), retaining 95–98% of records per building." — [Source](https://doi.org/10.3390/s25237294)
- The features identified as dominant are consistent across models: "outdoor temperature, previous-hour consumption, hour-of-day, and day-of-week encodings" (SHAP-based, ρ = 0.65–0.82 stability). — [Source](https://doi.org/10.3390/s25237294)
- Energies 2024 uses the same family on a single building: calendar and weather inputs, sliding-window (SW) framing, and autocorrelation/stationarity screening before training; 15-min and 30-min/1-h aggregation are compared, with aggregation shown to change model ranking. — [Source](https://doi.org/10.3390/en17061450)
- GEPIII lessons: "standard model inputs of historical meter, weather, and basic building metadata" define the ceiling of achievable accuracy; feature engineering and preprocessing were rated the most important phases by top competitors (>70% extremely/very important). — [Source](https://arxiv.org/abs/2202.02898); [Source](https://doi.org/10.1080/23744731.2022.2067466)
- LightGBM's method paper justifies the histogram/leaf-wise design that makes these wide lag/rolling feature matrices cheap to train (relevant when 24 lags × 10 buildings × 17,544 rows are used). — [Source](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

### Inferences
- A feature set matching the above is sufficient to claim "standard BDG2 practice" with a direct citation; adding cyclical sin/cos encodings is a defensible refinement but the benchmark papers use raw integer hour/day/month plus holiday and degree days, so sin/cos is not required for comparability.
- Because lag-1 and previous-hour load dominate importance, feature ablations should report performance without lag features — otherwise the model is largely an persistence forecaster and the comparison to published R² is misleading.
- Site-specific z-scoring of weather is worth replicating; global normalisation across 19 BDG2 sites would distort the temperature–load mapping.

### Gaps
- No paper was found that systematically ablates **cyclical (sin/cos) encoding vs integer calendar features** on BDG2, so the marginal value of cyclical encoding is unquantified.
- Holiday calendars for the six BDG2 countries are not standardised across papers; Sensors uses "binary holiday indicator" without specifying country coverage.

---

## Where do these papers caution about leakage from lag/rolling features or random splits?

### Takeaway
The strongest and most citable caution is quantitative: on BDG2 hourly load, random k-fold CV inflates R² by **0.63–0.74** relative to forward-chaining time-series CV, and modern GBDT models (XGBoost, CatBoost, LightGBM) show *larger* leakage gaps than classical boosting because they fit temporal autocorrelation more aggressively. The mechanism is explicit: shuffled folds put hour-adjacent samples (and their lag features) on both sides of the split.

### Cited Findings
- "Random cross-validation (CV) systematically overestimates performance through temporal leakage, with magnitudes varying substantially across domains." — [Source](https://doi.org/10.3390/s25237294)
- Energy-specific quantification: "Random CV yields inflated R² (0.90–0.99) since temporally adjacent samples share nearly identical load patterns. When evaluated chronologically, true predictive power drops to 0.24–0.34, leading to gaps ΔR² = 0.63–0.74 across models"; "Modern GBDT methods show systematically larger gaps than classical boosting, indicating greater susceptibility to temporal leakage through aggressive gradient fitting and deeper trees. This finding mandates time-series CV for all sensor forecasting applications regardless of domain." — [Source](https://doi.org/10.3390/s25237294)
- Mechanism: "For time-series forecasting, this introduces look-ahead bias because temporal correlations let models trained on future data indirectly 'see' past test samples"; "When random shuffling places temporally neighboring samples across train and test folds, these local patterns appear predictive." — [Source](https://doi.org/10.3390/s25237294)
- Recommended protocol: "5 forward-chaining folds that strictly preserve chronology" for the temporal-generalisation table; the main benchmark uses a **70/30 chronological split**. — [Source](https://doi.org/10.3390/s25237294)
- Preprocessing leakage: "Robust preprocessing must therefore be leakage-safe and refitted within each training fold to prevent contamination" (imputers/scalers/normalisers fitted inside folds). — [Source](https://doi.org/10.3390/s25237294)
- Competition-level leakage warning: GEPIII participants "compare the performance of their solution to the training and public data sets that the contestants found online (**public leaderboard leakage data**)", and top teams stressed careful removal of outliers/anomalies to improve generalisation to both public and private test sets. — [Source](https://arxiv.org/abs/2202.02898)
- Error-analysis corroboration: even the top-50 GEPIII solutions leave 4.8% of test rows out of acceptable error range, i.e., high random-split scores do not transfer to unseen buildings/periods. — [Source](https://doi.org/10.1080/23744731.2022.2067466)

### Inferences
- For a BDG2 LightGBM + RF study, the defensible protocol is chronological (or expanding-window forward-chaining) with all lag/rolling features computed only from past values, and with a building-holdout variant as a robustness check; reporting random k-fold alone would be a reviewable flaw given the published ΔR² evidence.
- The expected honest performance band for 1-h-ahead BDG2 load under chronological validation is R² ≈ 0.25–0.35 mean across heterogeneous buildings (with large per-building variance), or R² ≈ 0.6–0.76 if the split is chronological but pooled/per-building-averaged as in the Sensors Table 14 setup — the two are not the same estimator and the difference must be stated explicitly.
- Because lag-1 features are themselves near-copies of the target, any random split effectively leaks the answer; feature-importance and SHAP results under random CV are similarly optimistic and should be recomputed under the chronological protocol.

### Gaps
- No paper was found that isolates the leakage contribution of **rolling statistics specifically** (as opposed to lag features or random splitting) on BDG2; the cautions are stated jointly.
- The literature does not converge on a single recommended number of forward-chaining folds or minimum gap/embargo between train and test for hourly building load; only the 5-fold forward-chaining choice is documented in the BDG2 benchmark reviewed here.
