# Post-2021 peer-reviewed studies using the Building Data Genome Project 2 (BDG2)

*Scope note: 337 works published 2021-01-01 or later cite the BDG2 descriptor (OpenAlex `cites:W3033693292`, retrieved 2026-11-13). That is the sampling frame. Most citing works are reviews, urban-energy papers, or RL/scheduling papers that mention BDG2 only in passing; the table below isolates works where BDG2 is (or is very likely) the modelling dataset. Entries marked **†** cite BDG2 but their dataset use could not be confirmed from an accessible full text/abstract (publisher PDF blocked); treat those as "verify before citing".*

*Important correction to the brief: BDG2 is **3,053 meters from 1,636 non-residential buildings across 19 sites** (2016–2017, hourly, ~53.6 M measurements), not "3,053 buildings" — [Miller et al. 2020, arXiv:2006.02273](https://arxiv.org/abs/2006.02273); [Sci Data 7:368](https://doi.org/10.1038/s41597-020-00712-x). Getting this number right matters because several papers report building counts after cleaning (e.g., 1,828 meters; 327 buildings) that are easy to confuse with the raw total.*

---

## 1. Which papers use BDG2 after 2021, and what exactly do they predict?

### Takeaway
BDG2 is used overwhelmingly for **short-term load forecasting and transfer learning**; a smaller cluster covers **anomaly/discord detection**, **synthetic data generation**, and **benchmarking/characterisation pipelines**. Classification of *energy-consumption levels* is essentially absent as a headline task: a title search across all 337 citing works for "classif*" returns **zero** hits, and the closest work is a multi-task pipeline that predicts building *performance class* as one of three labels.

### Evidence table (core studies)

| # | Study (year, venue) | Task | BDG2 data used | Split / evaluation | Model | Reported performance | Verification |
|---|---|---|---|---|---|---|---|
| 1 | Miller et al. 2020, *Scientific Data* 7:368 — [doi](https://doi.org/10.1038/s41597-020-00712-x) | Dataset descriptor (provenance only) | 3,053 meters / 1,636 buildings / 19 sites; electricity, chilledwater, steam, hotwater, gas, water, irrigation, solar; 2016–2017 hourly | n/a | n/a | n/a | Primary |
| 2 | Miller, Picchetti, Fu, Pantelic 2022, *Sci Technol Built Environ* 28:610 — [doi](https://doi.org/10.1080/23744731.2022.2067466) | **Error analysis** of load prediction (not a new model) | GEPIII test set: 2,380 meters, 1,448 buildings, 16 locations — the same meter pool distributed as BDG2, so a sibling/provenance study, not BDG2 proper | Competition hold-out (2017–2018), aggregated top-50 solutions | Gradient-boosted trees/NN ensembles (top-50 Kaggle solutions) | **79.1%** of test data with RMSLE_scaled ≤ 0.1; **16.1%** in 0.1–0.3; **4.8%** > 0.3 ("unlikely to be accurately predicted") | Verified full text (arXiv:2106.13475) |
| 3 | Kazmi, Fu, Miller 2023, *Building and Environment* 239:110407 — [doi](https://doi.org/10.1016/j.buildenv.2023.110407) | Review / "ten questions" on data-driven demand modelling | BDG2 discussed as one of the few large open benchmarks | n/a | n/a | n/a | Verified full text |
| 4 | Fu, Kazmi, Quintana, Miller 2024, *Energy and Buildings* 312:114216 — [doi](https://doi.org/10.1016/j.enbuild.2024.114216) | **Synthetic meter-data generation** conditioned on metadata | Electricity meters only; 3,053 → **1,828 meters** after cleaning; full 2016–2017 | **75%/25% random split over meters** (unseen meters in test) | CVAE, CGAN, conditional diffusion (image-reshaped weekly profiles) | Outperforms CVAE/CGAN on fidelity metrics; generates full-year hourly profiles | Verified full text (arXiv:2404.00525) |
| 5 | Spencer, Ranathunga, Boulic et al. 2025, *Energy and Buildings* 336:115632 — [doi](https://doi.org/10.1016/j.enbuild.2025.115632) | **Cross-dataset transfer learning** for 24 h / 96 h load forecasting | **16 BDG2 datasets** (site/building subsets, plus "truncated" variants with fewer buildings/weather features/time range); 70/20/10 | 70/20/10 per dataset; transfer = train on source datasets, test on held-out target dataset | 6 data-centric TL strategies × 3 Transformers (vanilla, Informer, PatchTST) | Up to **25.5%** improvement for targets with <10 buildings; 1–8.2% (Fox), 0.6–6.7% (Rat) on individual datasets; PatchTST recommended over FEDformer/vanilla | Verified full text (arXiv:2410.14107) |
| 6 | Raymand, Najafi, Mamaghani et al. 2023, *Energy and Buildings* 295:113327 — [doi](https://doi.org/10.1016/j.enbuild.2023.113327) | **Classification**: (A) use type, (B) **performance class**, (C) operation group | Hourly **electricity + chilled-water**; **1,636 buildings** | Feature generation → selection → pipeline optimisation (per-pipeline); exact split not in abstract | Optimised ML pipelines (feature selection + model selection) | Accuracy gains of **5.3% / 2.9% / 3.9%** (A/B/C) over benchmark; features cut **94.7% / 88.3% / 89.4%** | Abstract verified; split unverified |
| 7 | Quintana, Stoeckmann, Park et al. 2022, *Energy and Buildings* 265:112096 — [doi](https://doi.org/10.1016/j.enbuild.2022.112096) | **Discord/outlier detection** in load profiles | Electricity only; 1,636 buildings; paper states 2016–2018 (BDG2 covers 2016–2017 — minor discrepancy) | Discord labels from GEPIII top-5 preprocessing used as **pseudo-ground-truth**; no train/test split (unsupervised) | ALDI++ (Gaussian-mixture + Wasserstein/DTW daily-profile clustering) | **+20.4%** discord-day detection vs baseline ALDI | Verified full text (arXiv:2203.06618) |
| 8 | Noorchenarboo & Grolinger 2024, *Energy and Buildings* 328:115177 — [doi](https://doi.org/10.1016/j.enbuild.2024.115177) | **Explainable anomaly detection** | **4 BDG2 datasets** (Jan 2016–Dec 2017) + 4 Canadian datasets; weather from Environment Canada | **80/10/10** windowed sequences (temporal ordering not stated) | RNN/CNN/Transformer variants (LSTM, GRU, BLSTM, BGRU) + SHAP variants | Context-weighted SHAP background selection improves explanation stability/consistency vs SHAP/KernelSHAP | Verified full text (arXiv:2501.06099) |
| 9 | Stjelja, Kuzmanovski, Kosonen, Jokisalo 2024, *Energy and Buildings* 313:114249 — [doi](https://doi.org/10.1016/j.enbuild.2024.114249) | Probabilistic load prediction + **collective anomaly detection** | BDG2 (building subset not stated in abstract) **†** | Point + prediction-interval evaluation; "alarm matrix" post-processing | EnbPI (conformal) vs XGBoostLSS | Point **CV-RMSE 9–17%**, better than ASHRAE Guideline 14 thresholds; EnbPI higher recall, XGBoostLSS higher precision | Abstract verified; dataset attribution unverified |
| 10 | Piscitelli, Giudice, Capozzoli 2023, *Applied Energy* 357:122550 — [doi](https://doi.org/10.1016/j.apenergy.2023.122550) | **External energy benchmarking** of building stocks | Large stock (BDG2 use implied by citation graph; not named in abstract) **†** | KPI-based scoring/ranking, peer groups beyond Primary Space Usage | Time-series KPIs: thermal sensitivity, load shape, magnitude, volatility, anomaly rate, schedule | Ranks buildings by composite performance score; no single accuracy metric | Abstract verified; dataset attribution unverified |
| 11 | Canaydin, Fu, Bálint et al. 2024, *Applied Energy* 360:122741 — [doi](https://doi.org/10.1016/j.apenergy.2024.122741) | Interpretable **supervised + unsupervised** learning on demand data | BDG2 (implied) **†** | Not available | Domain-informed vs domain-agnostic feature sets; supervised and clustering tasks | Not available from abstract | Unverified |
| 12 | Fan, Lei, Sun, Piscitelli et al. 2021, *Energy* 240:122775 — [doi](https://doi.org/10.1016/j.energy.2021.122775) | **Transfer learning** in data-scarce contexts (data-centric vs algorithm-centric) | BDG2 (implied) **†** | Source/target building transfer | Various ML/DL with TL | Not available from abstract | Unverified |
| 13 | Yan, Zhao, Rezgui et al. 2023, *J Build Eng* 80:108127 — [doi](https://doi.org/10.1016/j.jobe.2023.108127) | **Cross-building transferability/robustness** | **327 buildings** as source domain (subset consistent with BDG2) **†** | Pre-train on 327 buildings, fine-tune on target with varying data volumes; repeated at **20 time nodes**; negative transfer monitored | Pre-trained + fine-tuned predictor | TL significantly improves accuracy with limited target data; reports negative-transfer cases | Abstract verified; dataset attribution unverified |
| 14 | Yang, Ran, Feng, Hou 2024, *Applied Energy* 377:124584 — [doi](https://doi.org/10.1016/j.apenergy.2024.124584) | **Clustering** of consumption time series to improve predictor accuracy | BDG2 (implied) **†** | Not available | K-PCD clustering + per-cluster predictors | Not available from abstract | Unverified |

### Additional BDG2-citing works worth knowing (breadth, not all core uses)
- **BuildingsBench** (Emami, Sahu, Graf, NeurIPS 2023, [doi](https://doi.org/10.52202/075280-0871)) — 900K-building STLF benchmark; cites BDG2 as prior art, but BDG2 is not its primary dataset.
- **"A Global Feature-Rich Network Dataset of Cities"** (*Sci Data* 2023, [doi](https://doi.org/10.1038/s41597-023-02578-1)) and **"Building Data Genome Directory"** (*J Phys Conf Ser* 2023, [doi](https://doi.org/10.1088/1742-6596/2600/3/032003)) — data-infrastructure papers, not modelling.
- **"Hydronic heating systems in 259 commercial buildings"** (*Energy and Buildings* 2024, [doi](https://doi.org/10.1016/j.enbuild.2024.114543)) — descriptive/characterisation use.
- **"Privacy-preserving knowledge sharing for few-shot building energy prediction"** (*Applied Energy* 2023, [doi](https://doi.org/10.1016/j.apenergy.2023.120860)) and **"Personalized federated learning for cross-building energy knowledge sharing"** (*Applied Energy* 2024, [doi](https://doi.org/10.1016/j.apenergy.2024.123016)) — federated/cross-building learning, BDG2-based per citation graph.
- **"Large-scale comparison and demonstration of continual learning"** (*Applied Energy* 2023, [doi](https://doi.org/10.1016/j.apenergy.2023.121481)) — concept-drift/continual learning.
- **"Zero-Shot Prediction Error from a Pre-Trained Transformer Model as a Building Energy Diagnostic"** (*Buildings* 2026, [doi](https://doi.org/10.3390/buildings16122290)) — newer, explicitly benchmarking-oriented diagnostic use.

### Inferences
- The BDG2 literature is **forecasting-dominated**, with transfer learning as the dominant framing for cross-building work. Prediction targets are almost always the **next 24–96 h of whole-building electricity**, occasionally chilled water or steam.
- "Classification" in this literature almost always means **building-type/metadata classification**, not **consumption-level classification**. Only Raymand et al. (2023) explicitly targets a *performance class* label.
- Benchmarking papers (Piscitelli et al.) effectively produce ordinal performance rankings, but they are KPI/rule-based rather than supervised classification with held-out evaluation.

### Gaps
- Could not confirm dataset attribution for 6 of 14 table entries (publisher PDFs blocked; abstracts omit dataset names). Those are marked **†** and need full-text checks before citation.
- No paper in the 2021–2026 citing set has "classification" in the title (OpenAlex `title.search:classif` ∩ `cites:W3033693292` = **0 results**). Cannot rule out body-text-only classification studies.

### Full citations (for the table above)
- Miller, C.; Kathirgamanathan, A.; Picchetti, B.; Arjunan, P.; Park, J. Y.; Nagy, Z.; Raftery, P.; Hobson, B. W.; Shi, Z.; Meggers, F. (2020). The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. *Scientific Data*, 7, 368. https://doi.org/10.1038/s41597-020-00712-x
- Miller, C.; Picchetti, B.; Fu, C.; Pantelic, J. (2022). Limitations of machine learning for building energy prediction: ASHRAE Great Energy Predictor III Kaggle competition error analysis. *Science and Technology for the Built Environment*, 28, 610. https://doi.org/10.1080/23744731.2022.2067466
- Kazmi, H.; Fu, C.; Miller, C. (2023). Ten questions concerning data-driven modelling and forecasting of operational energy demand at building and urban scale. *Building and Environment*, 239, 110407. https://doi.org/10.1016/j.buildenv.2023.110407
- Fu, C.; Kazmi, H.; Quintana, M.; Miller, C. (2024). Creating synthetic energy meter data using conditional diffusion and building metadata. *Energy and Buildings*, 312, 114216. https://doi.org/10.1016/j.enbuild.2024.114216
- Spencer, R.; Ranathunga, S.; Boulic, M.; van Heerden, A.; Sušnjak, T. (2025). Transfer learning on transformers for building energy consumption forecasting—A comparative study. *Energy and Buildings*, 336, 115632. https://doi.org/10.1016/j.enbuild.2025.115632
- Raymand, F.; Najafi, B.; Haghighat Mamaghani, A.; Moazami, A.; Rinaldi, F. (2023). Machine learning-based estimation of buildings' characteristics employing electrical and chilled water consumption data: Pipeline optimization. *Energy and Buildings*, 295, 113327. https://doi.org/10.1016/j.enbuild.2023.113327
- Quintana, M.; Stoeckmann, T.; Park, J. Y.; Turowski, M.; Hagenmeyer, V.; Miller, C. (2022). ALDI++: Automatic and parameter-less discord and outlier detection for building energy load profiles. *Energy and Buildings*, 265, 112096. https://doi.org/10.1016/j.enbuild.2022.112096
- Noorchenarboo, M.; Grolinger, K. (2024). Explaining deep learning-based anomaly detection in energy consumption data by focusing on contextually relevant data. *Energy and Buildings*, 328, 115177. https://doi.org/10.1016/j.enbuild.2024.115177
- Stjelja, D.; Kuzmanovski, V.; Kosonen, R.; Jokisalo, J. (2024). Building consumption anomaly detection: A comparative study of two probabilistic approaches. *Energy and Buildings*, 313, 114249. https://doi.org/10.1016/j.enbuild.2024.114249
- Piscitelli, M. S.; Giudice, R.; Capozzoli, A. (2023). A holistic time series-based energy benchmarking framework for applications in large stocks of buildings. *Applied Energy*, 357, 122550. https://doi.org/10.1016/j.apenergy.2023.122550
- Canaydin, A.; Fu, C.; Bálint, A.; Khalil, M.; Miller, C.; Kazmi, H. (2024). Interpretable domain-informed and domain-agnostic features for supervised and unsupervised learning on building energy demand data. *Applied Energy*, 360, 122741. https://doi.org/10.1016/j.apenergy.2024.122741
- Fan, C.; Lei, Y.; Sun, Y.; Piscitelli, M. S.; Chiosa, R.; Capozzoli, A. (2021). Data-centric or algorithm-centric: Exploiting the performance of transfer learning for improving building energy predictions in data-scarce context. *Energy*, 240, 122775. https://doi.org/10.1016/j.energy.2021.122775
- Yan, R.; Zhao, T.; Rezgui, Y.; Kubicki, S.; Yu, L. (2023). Transferability and robustness of a data-driven model built on a large number of buildings. *Journal of Building Engineering*, 80, 108127. https://doi.org/10.1016/j.jobe.2023.108127
- Yang, H.; Ran, M.; Feng, H.; Hou, D. (2024). K-PCD: A new clustering algorithm for building energy consumption time series analysis and predicting model accuracy improvement. *Applied Energy*, 377, 124584. https://doi.org/10.1016/j.apenergy.2024.124584

---

## 2. What data preparation did each study use?

### Takeaway
Preparation is **heterogeneous and rarely leakage-safe**. Meter-type restriction to electricity is the norm; missing-data handling ranges from forward/back-fill to linear interpolation with hard column-drop thresholds; splits are mostly random or unspecified, with only a handful of papers doing explicit building- or dataset-holdout.

### Cited findings
- **Electricity-only, 3,053 → 1,828 meters**: IQR outlier removal, forward+backward fill, columns with >5% missing dropped, min-max scaling to [-1,1], 1-D series reshaped into 2-D weekly "images" — [Fu et al. 2024](https://arxiv.org/abs/2404.00525).
- **75%/25% random split over meters** (test = unseen meters, not unseen time) — [Fu et al. 2024](https://arxiv.org/abs/2404.00525).
- **16 BDG2 datasets**; buildings with ≥10% missing removed, linear interpolation, columns with >3,000 zeros dropped, ProEnFo cleaning package; **70/20/10** split (temporal ordering not stated in text) — [Spencer et al. 2025](https://arxiv.org/abs/2410.14107).
- **Electricity + chilled water, all 1,636 buildings**, hourly, used for three separate classification pipelines with feature generation/selection before model fitting — [Raymand et al. 2023](https://doi.org/10.1016/j.enbuild.2023.113327).
- **Electricity only, 1,636 buildings**; no split because the task is unsupervised; labels are inherited from GEPIII top-5 preprocessing choices (a **pseudo-ground-truth**, not human labels) — [Quintana et al. 2022](https://arxiv.org/abs/2203.06618).
- **4 BDG2 datasets + 4 Canadian datasets**, 80/10/10 windowed sequences, weather merged from Environment Canada — [Noorchenarboo & Grolinger 2024](https://arxiv.org/abs/2501.06099).
- **327 source buildings**, fine-tuning with "different volumes of data" in the target, repeated at **20 different time nodes** to test robustness — [Yan et al. 2023](https://doi.org/10.1016/j.jobe.2023.108127).
- **19 BDG2 sites → 16 modelling datasets**, ASHRAE climate-zone metadata, weather features, 17,544 hourly records per dataset, 53.6 M measurements total — [Spencer et al. 2025](https://arxiv.org/abs/2410.14107).
- No paper reviewed here reports **gap-length-aware masking**, **meter-level missingness stratification**, or **site-stratified** train/test reporting.

### Inferences
- Because BDG2 is a *convenience panel* (all sites share the 2016–2017 window), random splits leak information across time: a model can interpolate a held-out meter-hour from neighbouring hours of the same meter. Papers using random splits are therefore reporting **interpolation skill, not forecast skill**.
- Cleaning choices materially change the cohort (3,053 → 1,828 meters in one paper; 327 buildings in another), so cross-paper metric comparisons are not like-for-like.

### Gaps
- Split strategy is **not stated** in several abstracts and could not be recovered from blocked PDFs (Raymand et al., Stjelja et al., Piscitelli et al., Canaydin et al.).
- No study reports the **overlap of buildings retained** after cleaning, making replication of cohorts difficult.

---

## 3. What models and reported accuracy/RMSE/CV performance?

### Takeaway
Boosted trees remain competitive baselines, but the recent literature is dominated by Transformers, GNNs and transfer/federated variants; reported gains are typically **single-digit percent MAE/MAPE improvements**, and the strongest quantified BDG2-specific results come from transfer learning (+25.5%) and pipeline optimisation (+2.9–5.3% accuracy).

### Cited findings
- **Top-50 Kaggle ensembles**: 79.1% of BDG2-provenance test data predicted with RMSLE_scaled ≤ 0.1; 16.1% between 0.1–0.3; **4.8% > 0.3** (effectively unpredictable) — [Miller et al. 2022](https://doi.org/10.1080/23744731.2022.2067466).
- **Transformer TL**: up to **25.5%** improvement for target datasets with <10 buildings; dataset-level gains 0.6–8.2%; PatchTST > FEDformer > vanilla Transformer — [Spencer et al. 2025](https://doi.org/10.1016/j.enbuild.2025.115632).
- **Classification pipelines**: +5.3% (use type), +2.9% (performance class), +3.9% (operation group) accuracy over a benchmark pipeline, while removing 88–95% of features — [Raymand et al. 2023](https://doi.org/10.1016/j.enbuild.2023.113327).
- **Probabilistic anomaly detection**: point-prediction **CV-RMSE 9–17%**, i.e. inside ASHRAE Guideline 14 tolerances; EnbPI trades precision for recall, XGBoostLSS the reverse — [Stjelja et al. 2024](https://doi.org/10.1016/j.enbuild.2024.114249).
- **ALDI++**: +20.4% discord-day detection vs ALDI, evaluated against GEPIII-derived pseudo-labels — [Quintana et al. 2022](https://doi.org/10.1016/j.enbuild.2022.112096).
- **Cross-building TL**: pre-training on 327 buildings significantly improves accuracy when target data are scarce; negative transfer occurs and is explicitly monitored — [Yan et al. 2023](https://doi.org/10.1016/j.jobe.2023.108127).

### Inferences
- The **realistic error floor** on BDG2-like data is around RMSLE_scaled ≈ 0.1 for most meters; papers claiming sub-1% MAPE on random splits are almost certainly measuring interpolation, not forecasting.
- Reported gains of 1–8% MAE from architecture changes are within the noise created by different cleaning/split choices, which supports the case for a **fixed, leakage-free BDG2 benchmark protocol**.

### Gaps
- Few papers report **per-meter or per-site error distributions**; most report a single aggregate. Miller et al. 2022 is the main exception.
- No paper reports **calibration** (interval coverage) on BDG2 for classification-style outputs.

---

## 4. What limitations and future-work directions do they state?

### Takeaway
Authors consistently flag (i) geographic narrowness (North America + Europe only), (ii) data quality/missingness, (iii) coarse source-domain selection in transfer learning, and (iv) benchmark over-fitting. Leakage is discussed mainly in the review literature, not in the modelling papers themselves.

### Cited findings
- **Geographic and architectural scope**: "all these datasets belong to North American and European countries… might not hold for datasets obtained from other parts of the world"; only 3 dataset combinations (2, 3, 16); only 24 h and 96 h horizons; only 3 Transformer architectures; source selection at dataset rather than building-type granularity — [Spencer et al. 2025](https://arxiv.org/abs/2410.14107).
- **Benchmark over-fitting**: "we are now running the risk of overfitting to these particular datasets at the cost of broader generalization"; BDG2 data quality is "often low, with even large intra-dataset variations"; collating datasets forces dropping fields not recorded in most buildings — [Kazmi et al. 2023](https://doi.org/10.1016/j.buildenv.2023.110407).
- **Leakage**: "To avoid spurious results and prevent data leakage, this score should be evaluated using backtesting or time series cross-validation"; competition models built after hold-out release "exhibit better skill, which indicates overfitting on the test set" — [Kazmi et al. 2023](https://doi.org/10.1016/j.buildenv.2023.110407).
- **Distribution shift**: models trained on observed weather but fed forecasts at inference degrade; Covid-19 showed non-stationarity; regular retraining is "imperative" — [Kazmi et al. 2023](https://doi.org/10.1016/j.buildenv.2023.110407).
- **Irreducible error**: 4.8% of BDG2-provenance test data has RMSLE_scaled > 0.3 and is "unlikely to be accurately predicted" with standard inputs (historical meter + weather + basic metadata) — [Miller et al. 2022](https://doi.org/10.1080/23744731.2022.2067466).
- **Simplification**: weather excluded as a direct generative input to keep the conditional model tractable — [Fu et al. 2024](https://arxiv.org/abs/2404.00525).

### Inferences
- The field's own reviews already concede that BDG2 results may not generalise; a new study that demonstrates **site-holdout** generalisation would directly answer a stated limitation rather than adding another architecture comparison.
- None of the modelling papers frame random splitting as a limitation — the critique comes from the review layer, which is an opening for a methods-focused contribution.

### Gaps
- Stated limitations for Raymand et al., Piscitelli et al., Canaydin et al. could not be extracted (paywalled full text).
- No paper states a limitation about **label definition** for performance classes (i.e., that class boundaries are arbitrary quantiles of EUI), which is relevant to classification work.

---

## 5. Which gaps remain around classification of consumption levels, cross-building generalisation, and leakage-free evaluation?

### Takeaway
All three gaps are real and largely unaddressed in the BDG2 literature: consumption-level classification is nearly absent, cross-building work is dominated by *dataset-level* transfer rather than *building-level holdout*, and leakage-free evaluation is recommended by reviews but rarely implemented in BDG2 modelling papers.

### Cited findings
- **Classification gap**: among 337 works citing BDG2, **zero** have "classif*" in the title; the only explicit class label is Raymand et al.'s "performance class" pipeline (one of three tasks, evaluated by accuracy improvement over a benchmark, not by a leakage-free protocol) — [OpenAlex citing set](https://api.openalex.org/works?filter=cites:W3033693292,from_publication_date:2021-01-01) ; [Raymand et al. 2023](https://doi.org/10.1016/j.enbuild.2023.113327).
- **Adjacent classification work exists but on other datasets**: Norwegian smart-meter building-category classification (2,724 buildings, 84% accuracy) is *not* BDG2 — [doi](https://doi.org/10.1016/j.enbuild.2024.114954); discriminant analysis of residential smart meters is also non-BDG2 — [doi](https://doi.org/10.1016/j.enbuild.2021.111823). These show the method space is mature but the BDG2 instantiation is missing.
- **Cross-building generalisation**: the strongest BDG2 evidence is transfer from 327 source buildings with fine-tuning and explicit negative-transfer monitoring — [Yan et al. 2023](https://doi.org/10.1016/j.jobe.2023.108127); dataset-level TL across 16 BDG2 datasets gains up to 25.5% for small targets — [Spencer et al. 2025](https://doi.org/10.1016/j.enbuild.2025.115632). Neither reports a strict **unseen-site** holdout with no target leakage (Spencer's target dataset is unseen, but source/target preprocessing is shared).
- **Leakage-free evaluation**: recommended practice is backtesting or time-series CV — [Kazmi et al. 2023](https://doi.org/10.1016/j.buildenv.2023.110407); yet the BDG2 papers reviewed here use random meter splits (75/25) — [Fu et al. 2024](https://arxiv.org/abs/2404.00525) — or unspecified 70/20/10 and 80/10/10 splits — [Spencer et al. 2025](https://arxiv.org/abs/2410.14107); [Noorchenarboo & Grolinger 2024](https://arxiv.org/abs/2501.06099).
- **Pseudo-labels**: anomaly work evaluates against GEPIII top-5 preprocessing decisions rather than ground-truth faults, which is a label-leakage risk in its own right — [Quintana et al. 2022](https://arxiv.org/abs/2203.06618).

### Inferences
- A BDG2 study that (a) defines consumption-level classes from **held-out-year** EUI/load-factor quantiles, (b) evaluates with **site-holdout** (19 sites → leave-one-site-out) or **building-holdout** and (c) reports per-class precision/recall alongside regression baselines would occupy a genuinely empty niche.
- Boosted-tree baselines are defensible and under-used in the recent BDG2 literature relative to Transformers; a well-tuned GBDT with leakage-free splits is a credible strong baseline and a differentiator against the Transformer-heavy citation set.
- Because class boundaries are quantile-defined, class imbalance and boundary instability are the key methodological risks a new paper should pre-empt.

### Gaps
- No BDG2 paper was found that reports **leave-one-site-out** classification or consumption-level labels; this is the clearest publishable gap.
- Cannot confirm whether classification results exist in non-English venues or in 2026 preprints not yet indexed by OpenAlex (coverage lag ~weeks).
- No standard BDG2 benchmark leaderboard exists for either forecasting or classification, so no external numbers can be quoted as "state of the art".
