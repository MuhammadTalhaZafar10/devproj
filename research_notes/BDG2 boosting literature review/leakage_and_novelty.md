# Evaluation Pitfalls and Robustness in ML-Based Building-Energy Prediction (2020–2026): Evidence Base for Positioning a BDG2 Study

*Scope note: this note covers (a) leakage from lag/rolling features and random CV, (b) temporal and cross-building generalization, (c) drift/robustness diagnostics, and (d) how 2022–2026 papers frame novelty. Evidence was gathered 2026-11-13 via the OpenAlex, Crossref, Europe PMC and arXiv APIs plus full-text XML of the base paper (PMC12694449). The general web-search endpoint was unavailable (HTTP 402), and Elsevier journals do not deposit abstracts in Crossref/OpenAlex, so several building-energy citations below are verifiable at title/metadata level only and are labelled accordingly. Claims are tagged **[Established]** (multiple independent sources / replicated), **[Single-paper]**, **[Preprint]**, or **[Unverified detail]**.*

---

## KQ1. Does random k-fold cross-validation overestimate performance on building-energy/time-series data, and by how much?

### Takeaway

Yes. The strongest quantitative building-energy evidence is the base paper itself: on 10 BDG2 buildings, random CV gives R² = 0.90–0.99 while chronological validation gives R² = 0.24–0.34, a ΔR² of 0.63–0.74 for every one of six algorithms. Across fields, leakage is now documented as a systematic, taxonomy-level failure of ML-based science rather than an isolated bug. What is *not* yet established in the peer-reviewed building-energy literature is a multi-building, multi-algorithm *decomposition* of that gap into its separate causes.

### Cited Findings

- **Base paper, headline result.** Random CV yields inflated R² (0.90–0.99) "since temporally adjacent samples share nearly identical load patterns"; chronological evaluation drops true predictive power to 0.24–0.34, giving ΔR² = 0.63–0.74 across Random Forest, AdaBoost, GBM, XGBoost, LightGBM and CatBoost (energy, 1-h horizon, 10 BDG2 buildings). Per-model ΔR²: GBM 0.628 (best), AdaBoost 0.667, LightGBM 0.681, RF 0.699, CatBoost 0.699, XGBoost 0.738 (worst). — [Xie & Tummala 2025, *Sensors* 25(23):7294, §7.5.3 Table 17](https://doi.org/10.3390/s25237294) **[Single-paper, but internally consistent across 6 models]**
- **Same paper, cross-domain replication.** ΔR² = 0.12–0.26 (glucose), 0.42–1.21 (PM2.5), 0.63–0.74 (energy). The authors conclude "time-aware validation is not optional but mandatory for reliable sensor analytics deployment," and that PM2.5 models "appearing to achieve R² = 0.33–0.89 in development deliver negative R² in actual time-series deployment." — [Xie & Tummala 2025, §7.6.2](https://doi.org/10.3390/s25237294) **[Single-paper]**
- **Field-level evidence that leakage is pervasive and consequential.** A survey found leakage reported in **17 fields, collectively affecting 294 papers**, "in some cases leading to wildly overoptimistic conclusions"; the authors define a **taxonomy of eight leakage types** and propose model info sheets as a detection instrument. In their own re-analysis, correcting leakage removed the apparent superiority of complex ML over logistic regression for civil-war prediction. — [Kapoor & Narayanan 2023, *Patterns* 4(9):100804](https://doi.org/10.1016/j.patter.2023.100804) **[Established: highly cited, cross-field]**
- **Leakage is not uniform in magnitude.** In connectome-based prediction across four datasets and three phenotypes, leakage via **feature selection** and **repeated subjects** "drastically inflates" performance, whereas other leakage forms had minor effects; **small datasets exacerbate leakage effects**. — [Rosenblatt et al. 2024, *Nature Communications* 15:1829](https://doi.org/10.1038/s41467-024-46150-w) **[Single-paper, but large multi-dataset study]**
- **Leakage propagates through pipelines, including transfer learning.** A review categorises leakage types and how conditions propagate through the ML workflow, tying leakage to task type and to the "push the button" tooling problem. — [Apicella, Isgrò & Prevete 2025, *Artificial Intelligence Review* 58(11):339](https://doi.org/10.1007/s10462-025-11326-3) **[Review]**
- **Sequence-construction-before-splitting is a distinct, quantifiable leakage channel.** In an LSTM study comparing leaky (pre-split windowing) vs clean (split-then-window) pipelines, **10-fold CV showed RMSE inflation up to 20.5% at extended lag steps**, while 2-way and 3-way temporal splits stayed typically below 5%; smaller input windows and longer lags increased sensitivity. — [Albelali & Ahmed 2025, arXiv:2512.06932](https://arxiv.org/abs/2512.06932) **[Preprint — not peer reviewed; treat the 20.5% figure as indicative]**
- **Benchmark-integrity leakage is now recognised for time-series foundation models** — two kinds: train/test sample overlap from dataset reuse, and temporal overlap of correlated train/test series, both producing "overly optimistic performance estimates." — [Meyer et al. 2025, arXiv:2510.13654](https://arxiv.org/abs/2510.13654) **[Preprint / position paper]**
- **Important counter-evidence and nuance.** Cross-validation *can* be appropriate for purely autoregressive time-series predictors when the model has no access to future information and the dependence structure is handled; the original argument is about *how* CV is applied, not that CV is always invalid. — [Bergmeir & Benítez 2012, *Information Sciences* 191:192–213](https://doi.org/10.1016/j.ins.2011.12.028) **[Established, foundational; pre-2020 but load-bearing]**
- **Current best-practice recommendation in forecasting.** Rolling-origin / walk-forward evaluation with a gap between train and test is the standard protocol; random shuffling of temporally ordered data is explicitly discouraged. — [Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., §5.10](https://otexts.com/fpp3/tscv.html) **[Established textbook standard]**

### Inferences

- The building-energy ΔR² of 0.63–0.74 is large enough that the choice of validation scheme, not the choice of booster, determines the paper's conclusion. The base paper's own Table 19 shows modern GBDTs are statistically indistinguishable from each other on energy — so *validation design is the higher-leverage experimental variable*, which is precisely the space a small BDG2 study can occupy.
- The base paper's ΔR² is a **single composite number**: it conflates (i) temporal autocorrelation leakage, (ii) any target leakage in the rolling features, and (iii) building-identity leakage (10 buildings pooled, so the same building appears in train and test). Its design cannot separate these. That separation is a credible, bounded novelty claim.
- The connectome result (feature-selection leakage dominates; small data amplifies) predicts that in BDG2 the *feature-construction* channel may matter as much as the *split* channel — a testable hypothesis, not yet tested in building energy.

### Gaps

- **No peer-reviewed building-energy paper was found that reports the random-vs-temporal CV gap across many buildings with confidence intervals.** The quantitative anchor remains a single 2025 paper (10 buildings) plus a 2025 preprint. This is a genuine hole and simultaneously the opening for a replication/extension study.
- The base paper reports time-series R² as "0.287 ± 0.762", "0.247 ± 0.832", etc. — standard deviations of the same order as, or larger than, the means and inconsistent with an R² bounded in [0,1]. This is almost certainly a reporting error (mislabeled SD, or SD across folds rather than buildings). **[Unverified detail / internal inconsistency]** A new study should not inherit this ambiguity; reporting fold-level distributions is itself a small contribution.
- I could not locate a peer-reviewed study that isolates the *cost of pooling buildings* (same building in train and test) from the *cost of adjacent timestamps*. Absence of evidence here is based on API searches of OpenAlex/Crossref, not a systematic review.

---

## KQ2. What is documented about target leakage from rolling-window features that include the current timestep, and what leakage-free feature construction is recommended?

### Takeaway

The *general* principle is well established (a feature may only use information available at the prediction origin, and any statistic computed over a window must be causally aligned and lagged). The *specific* empirical question — how much inflation a rolling mean/std that includes time *t* adds when predicting *t+1* in building load — appears **not to have been quantified in the building-energy literature**. This is the least-covered of the four areas and the most defensible novelty slot.

### Cited Findings

- **Definitional anchor.** Leakage is formalised as the introduction of information about the target that would not be available at prediction time, with detection and avoidance strategies organised by pipeline stage; the treatment distinguishes legitimate from illegitimate use of contemporaneous variables. — [Kaufman, Rosset, Perlich & Stitelman 2012, *ACM TKDD* 6(4):1–21](https://doi.org/10.1145/2382577.2382579) **[Established, foundational; pre-2020]**
- **Taxonomy placement.** "Temporal leakage" (using information not available at prediction time) and "non-independence between train and test samples" are two of the eight leakage types; the recommended control is documenting, per feature, the timestamp at which it becomes available. — [Kapoor & Narayanan 2023](https://doi.org/10.1016/j.patter.2023.100804) **[Established]**
- **Non-causal preprocessing is an independently documented failure mode.** "Improper dataset processing with subtle future label leakage is unfortunately widely adopted," producing "abnormally superior but misleading results"; under strictly causal processing, the apparent benefit of signal-decomposition features largely disappears. — [Jiang, Wu & Chen 2024, arXiv:2405.06986](https://arxiv.org/abs/2405.06986) **[Preprint, but the mechanism is exactly the rolling/decomposition-window problem]**
- **Wind-power analogues show the fix is implementable and matters.** Cascade/rolling decomposition designed to avoid information leakage, and "moving front" decomposition that prevents future information from entering constituent series, both report that leakage-free variants change conclusions relative to conventional one-shot decomposition. — [Wang, Yuan & Zhang 2024, *Energy* 312:133513](https://doi.org/10.1016/j.energy.2024.133513) **[Single-paper]**; [Niyogi 2023, arXiv:2303.06394](https://arxiv.org/abs/2303.06394) **[Preprint]**
- **Empirical demonstration that the window/alignment choice changes measured leakage magnitude.** Leakage sensitivity depended on input window size and lag step: smaller windows and longer lags increased RMSE inflation. — [Albelali & Ahmed 2025](https://arxiv.org/abs/2512.06932) **[Preprint]**
- **The base paper's own feature set is exactly the ambiguous case.** It uses "lagged consumption (1–24 h), rolling statistics (6/12/24-h mean/std/min/max)" for 1-h-ahead prediction on BDG2, but the text does not state whether the rolling window ends at *t* (legitimate: known at origin) or is centred / includes *t+1* (leakage). Weather normalisation *is* handled correctly (z-score statistics computed on each building's training period only, then applied to test). — [Xie & Tummala 2025, §7.5](https://doi.org/10.3390/s25237294) **[Verified from full text]**
- **Review-level diagnosis that reporting is the weak link.** A 2016–2026 review of 119 PV forecasting studies names "insufficient reporting of data preprocessing procedures and temporal leakage prevention" as one of three major methodological gaps. — [Wedhasari & Castro 2026, *Energies* 19(12):2937](https://doi.org/10.3390/en19122937) **[Review, single-paper]**

### Inferences

- There are three distinct rolling-feature hazards, and the literature usually collapses them into one: (1) **centred windows** (uses *t+1…t+k*), a hard leak; (2) **windows ending at *t*** used to predict *t+1*, which is legitimate but makes the model a persistence model in disguise and inflates *apparent* skill relative to operational feature availability; (3) **normalisation/statistics fitted on the full series** (scaler leakage), which is common and easy to fix. Only (1) is normally called leakage; (2) and (3) are the ones a careful BDG2 audit can quantify.
- Because a 1-h-ahead load model with a 24-h rolling mean ending at *t* effectively re-supplies the target's recent level, part of the reported R² in this literature is persistence, not learned structure. A **feature-availability ablation** (drop all target-derived features; then add lags; then add rolling stats ending at *t*; then add the leaky centred variant) would isolate each contribution at low computational cost.
- This suggests a crisp, defensible framing: not "we detect leakage" (already done), but "we **price** each leakage channel in R²/RMSE terms on a large public panel, and provide a feature-availability protocol that reproduces the non-leaky number."

### Gaps

- **No building-energy paper was found that quantifies the inflation attributable specifically to rolling statistics including the current (or future) timestep.** Searches returned only general leakage taxonomies, non-energy domain studies, and renewable-generation analogues. This is the clearest literature gap identified in this task.
- The base paper does not document window alignment, so its own numbers cannot be classified as leaky or clean from the text. **[Unverified detail]** — treat as a reproducibility criticism, not an accusation of leakage.
- No standard checklist exists for "feature availability at prediction origin" in building-energy papers; the closest instruments are Kapoor & Narayanan's model info sheets and the guiding-questions approach from biology ([Bernett, Blumenthal & Grimm 2024, *Nature Methods* 21:1444–1453](https://doi.org/10.1038/s41592-024-02362-y)) — neither is building-specific.

---

## KQ3. What do papers report on cross-building (unseen building) generalization and transfer learning on BDG2-like panel data?

### Takeaway

Cross-building generalization is actively studied, but almost always as a *transfer-learning performance* question ("can we improve the target building?") rather than an *evaluation-validity* question ("how much of the reported skill is building-identity memorisation?"). Reported findings are consistently that transfer helps for data-scarce targets and that zero-shot performance on unseen buildings is materially lower and highly heterogeneous. Effect sizes are rarely expressed as a train/test protocol contrast.

### Cited Findings

- **The dataset and its intended use.** BDG2 contains **3,053 meters from 1,636 non-residential buildings, 19 sites, 2016–2017 hourly, ~53.6 million measurements**, covering electricity, heating/cooling water, steam, solar, water and irrigation; part was used for the ASHRAE GEPIII competition. — [Miller et al. 2020, *Scientific Data* 7:368](https://doi.org/10.1038/s41597-020-00712-x) **[Established primary source]**
- **Competition-scale evidence that unseen-building/future prediction is hard.** GEPIII drew **4,370 participants and 39,403 submissions**; the test set covered **2,380 meters in 1,448 buildings at 16 locations**. Aggregating the top 50 solutions, ML errors were "within a range of acceptability (RMSLE_scaled < 0.1) on **79.1%** of the test data", with moderate errors (0.1–0.3) on a further **16.1%**; errors were classified by timeframe, behaviour, magnitude and whether they were building-specific or campus-wide. — [Miller et al. 2022, *Science and Technology for the Built Environment* 28(5):610–627](https://doi.org/10.1080/23744731.2022.2067466) **[Established: the definitive post-hoc error analysis of GEPIII]**
- **Competition overview and design.** The GEPIII overview documents the competition design and results for long-term prediction aimed at measurement & verification. — [Miller et al. 2020, *Science and Technology for the Built Environment* 26(10):1427–1447](https://doi.org/10.1080/23744731.2020.1795514) **[Established]**
- **Cross-building with limited data is an explicitly posed benchmark problem.** The "Energy Detective" competition (195 participants, 89 teams) was designed as **cross-building prediction with limited physical parameters and historical data**; the overview analyses the submitted models and argues for hybrid and data-driven modelling improvements. — [Xiao et al. 2022, *Applied Energy* 305:117829](https://doi.org/10.1016/j.apenergy.2021.117829) **[Single-paper / competition report]**
- **Deep transfer learning improves cross-building accuracy relative to target-only, source-only, and pooled training.** An LSTM + domain-adversarial (DANN) strategy is reported to "significantly enhance" target-building prediction versus models trained on target-only data, source-only data, or both without transfer, attributing the gain to domain-invariant feature extraction under domain shift. — [Fang et al. 2021, *Energy* 215:119208](https://doi.org/10.1016/j.energy.2020.119208) **[Single-paper]**
- **Statistical characterisation of transfer-learning methodology for short-term prediction**, establishing the conditions (data volume, similarity) under which transfer helps. — [Fan, Sun & Xiao 2020, *Applied Energy* 262:114499](https://doi.org/10.1016/j.apenergy.2020.114499) **[Single-paper, highly cited (248)]**
- **Systematic comparison of TL strategies for cross-building prediction** finds performance depends on the transfer strategy and the source–target pairing, not on architecture alone. — [Li, Wu & Liu 2022, *Energy and Buildings* 275:112461](https://doi.org/10.1016/j.enbuild.2022.112461) **[Single-paper]**
- **BDG2-scale transformer TL study.** Six data-centric TL strategies on three Transformer architectures (vanilla, Informer, PatchTST) across **16 BDG2 datasets**; combining multiple source datasets under **zero-shot** transfer reduced MAE of the vanilla Transformer (exact magnitude not available in the accessible abstract). — [Spencer et al. 2025, *Energy and Buildings* 336:115632](https://doi.org/10.1016/j.enbuild.2025.115632) **[Single-paper]**
- **Domain-generalisation framing.** A multi-source domain-generalisation DNN is proposed for predicting energy across multiple office buildings, i.e. generalising to unseen buildings treated as unseen domains. — [Jiang et al. 2024, *Energy* 299:131467](https://doi.org/10.1016/j.energy.2024.131467) **[Single-paper; abstract not retrievable]**
- **Zero-shot and transfer are now formal benchmark tasks.** BuildingsBench (Buildings-900K simulated + >1,900 real buildings from 7 datasets) benchmarks **zero-shot STLF on unseen buildings** and **transfer learning via fine-tuning**; its main finding is that synthetically pretrained models "generalize surprisingly well to real commercial buildings," with dataset size/diversity showing a **power law with diminishing returns**. — [Emami et al. 2023, NeurIPS Datasets & Benchmarks](https://doi.org/10.52202/075280-0871) **[Established benchmark]**
- **Generalisation to unseen houses is achievable but conditional.** With sufficient historical data, four ML methods generalised "to a satisfactory level" on unseen UK house profiles using past consumption only. — [Mehdipour Pirbazar, Farmanbar & Chakravorty 2020, *Processes* 8(4):484](https://doi.org/10.3390/pr8040484) **[Single-paper]**

### Inferences

- The BDG2/GEPIII literature supplies the *ingredients* for a cross-building validity study (many buildings, known site structure, a published error analysis) but the papers ask "does transfer help?" rather than "how much reported skill survives when buildings are held out?" The base paper's own design pools 10 buildings and splits chronologically, so building identity is shared between train and test — i.e. its 0.24–0.34 temporal R² is still an *optimistic* estimate of unseen-building performance.
- A leave-one-site-out protocol is the sharper test on BDG2 because sites bundle weather, climate, utility, and operational conventions; leave-one-building-out controls only building identity. Reporting both would let a small study separate "new building, same site/weather" from "new site, new weather" — a distinction the reviewed papers do not make cleanly.
- The GEPIII error analysis (79.1% acceptable, but a long tail of building-specific failures) supports the argument that aggregate R² hides building-level failure, so per-building error distributions (not just means) should be reported.

### Gaps

- **No paper found that reports the *same* model under within-building, leave-one-building-out, and leave-one-site-out splits on BDG2 with a common metric.** This is likely the strongest available novelty claim for a new BDG2 study.
- Quantitative effect sizes for cross-building degradation (e.g., ΔR² or ΔnRMSE relative to within-building) could not be extracted: Elsevier abstracts are not deposited in Crossref/OpenAlex and the publisher pages returned HTTP 403. Exact numbers from [Energy 2020](https://doi.org/10.1016/j.energy.2020.119208), [Applied Energy 2020](https://doi.org/10.1016/j.apenergy.2020.114499), [E&B 2022](https://doi.org/10.1016/j.enbuild.2022.112461) and [E&B 2025](https://doi.org/10.1016/j.enbuild.2025.115632) must be taken from the PDFs before being quoted. **[Unverified detail]**
- Building count discrepancy: BDG2 reports 1,636 buildings / 3,053 meters, whereas the base paper states "1578 buildings." The base paper's filtering is undocumented, which matters for any replication claim. **[Internal inconsistency across sources]**

---

## KQ4. Which robustness diagnostics are used (noise injection, impulse corruption, calibration drift, SHAP stability, temporal holdout), and which papers establish them?

### Takeaway

The base paper is currently the most explicit *protocol* in this space for building energy: three perturbation families at four intensities, a temporal holdout, and cross-fold SHAP-rank stability. Its most important qualitative finding is that **rankings reverse across diagnostic axes** and that **stable explanations coexist with poor forward skill**. Outside that paper, robustness work in building energy is fragmented into concept-drift/continual-learning studies and general leakage studies; there is no agreed diagnostic suite.

### Cited Findings

- **Noise-injection protocol (three families, four levels).** Gaussian noise σ = {0.05, 0.10, 0.15, 0.20}×SD(feature); impulse corruption replacing p = {10%, 20%, 30%, 40%} of feature values with random extremes; "calibration drift" implemented as a cumulative bias with slope d = {0.5, 1.0, 1.5, 2.0} kWh/h. — [Xie & Tummala 2025, §7.5.2 Table 16](https://doi.org/10.3390/s25237294) **[Single-paper protocol, fully specified and therefore reusable]**
- **Energy-domain degradation magnitudes.** Gaussian: 2.6–27.0% RMSE increase (best CatBoost 14.5%, best overall AdaBoost 5.7%). Impulse: 29.3–157.4% (CatBoost 93.2%, AdaBoost 43.9%). Drift: **327–586%** RMSE inflation at maximum intensity, with the striking observation that degradation is "nearly constant across intensity levels," indicating a structural rather than magnitude-dependent failure. — [ibid., Table 16 / Table 21](https://doi.org/10.3390/s25237294) **[Single-paper]**
- **Ranking reversal across diagnostics — the key robustness result.** "No single algorithm dominates"; CatBoost wins under Gaussian/impulse noise but degrades badly under drift in energy (488%), while GBM/XGBoost are more drift-stable (551%/537%) and AdaBoost is most drift-robust (327%) despite the worst clean accuracy (RMSE 9.00 vs 5.67 kWh). — [ibid., §7.6.3](https://doi.org/10.3390/s25237294) **[Single-paper]**
- **SHAP/feature-importance stability is measured as Spearman ρ of SHAP rankings across CV folds.** Energy: RF 0.822 ± 0.034 (best), CatBoost 0.812, XGBoost 0.775, LightGBM 0.756, GBM 0.708, AdaBoost 0.650; PM2.5 0.726–0.779; glucose 0.488–0.679. — [ibid., §7.5.4 Table 18, §7.6.4](https://doi.org/10.3390/s25237294) **[Single-paper]**
- **Stability ≠ validity.** "Consistent feature rankings do not guarantee predictive reliability": ρ = 0.65–0.82 coexists with temporal R² = 0.24–0.34, and the paper concludes current interpretability methods "measure feature usage consistency rather than causal or predictive validity," recommending explanation methods be validated "on held-out temporal folds rather than random samples." — [ibid., §9.5](https://doi.org/10.3390/s25237294) **[Single-paper, but an unusually clear statement of the problem]**
- **Concept drift is an active building-energy research thread.** A large-scale comparison demonstrates continual learning for adaptive data-driven building energy prediction; results are framed around adaptation to distribution shift over time. — [Li et al. 2023, *Applied Energy* 347:121481](https://doi.org/10.1016/j.apenergy.2023.121481) **[Single-paper; abstract not retrievable, claim is title-level]**
- **Concept drift in load forecasting has an established online-learning literature.** Online ensemble learning combining RNN and ARIMA is evaluated explicitly "under concept drift." — [Jagait et al. 2021, *IEEE Access* 9:98992–99008](https://doi.org/10.1109/access.2021.3095420) **[Single-paper]**
- **Drift adaptation remains an open problem in the base paper's own framing**: continual boosting must "balance plasticity and stability," and deciding "when distribution shifts warrant model updates versus temporary anomalies" is unresolved; meta-learning and principled drift-detection tests are named as needed directions. — [Xie & Tummala 2025, §9.3](https://doi.org/10.3390/s25237294) **[Review/opinion]**
- **Temporal holdout as a diagnostic is standard but weakly standardised.** Rolling-origin evaluation with a gap is the textbook default ([Hyndman & Athanasopoulos](https://otexts.com/fpp3/tscv.html)); a 2026 review of 119 forecasting studies still finds "insufficient reporting of ... temporal leakage prevention" ([Energies 2026](https://doi.org/10.3390/en19122937)). **[Established gap]**

### Inferences

- **Terminology trap worth flagging in any new paper:** the base paper's "calibration drift" means *sensor* bias accumulation (a covariate shift on features), not *probability calibration* drift (reliability/ECE over time). Conflating the two is common. A study that reports both — sensor-drift stress tests *and* calibration/reliability diagnostics of the predictive distribution — would be doing something the reviewed literature does not.
- The rank-reversal finding implies that single-axis robustness claims ("CatBoost is robust") are not defensible; robustness must be reported as a profile across perturbation families. A small study can credibly deliver a **rank-stability matrix** (models × diagnostics) rather than a new SOTA claim.
- Because SHAP stability is high while forward skill is low, feature-importance stability should not be used as evidence of generalisation. This is a clean, citable "do not do this" that a new paper can operationalise by computing ρ on temporal folds vs random folds and showing the difference.

### Gaps

- No independent replication of the noise/drift protocol on a larger BDG2 subset was found; all magnitudes above come from one paper with 10 buildings. **[Single-paper]**
- SHAP-stability methodology in building energy has no dedicated peer-reviewed treatment located; the ρ-across-folds metric appears to originate here and is not yet a community standard.
- Impulse-corruption and drift-slope choices are ad hoc (p = 10–40%, d = 0.5–2.0 kWh/h); no paper was found that calibrates these to observed meter-failure statistics, which limits external validity.

---

## KQ5. What gaps do 2022–2026 reviews explicitly state as future work, and what could a small study credibly claim as novel?

### Takeaway

Reviews converge on four unmet needs: better public data and benchmarks, rigorous and *reported* validation/temporal-leakage control, drift adaptation, and interpretability that goes beyond feature importance. None of them asks for a new booster. The credible novelty for a small BDG2 study is therefore **methodological and diagnostic** — quantifying and decomposing evaluation inflation, and shipping a reusable protocol — not a new model.

### Cited Findings

- **Review-level gap statements (building energy).** A review of 113 HVAC prediction papers (2019–2024) finds neural models dominate, hourly granularity is most common, and weather/time/lag features are near-universal, then names three main challenges, the first being "limited access to high quality public datasets." — [Li, Arellano-Espitia & Aler 2025, *Journal of Building Engineering* 116:114612](https://doi.org/10.1016/j.jobe.2025.114612) **[Review; abstract truncated in accessible record]**
- **Review-level gap statements (forecasting methodology).** Of 119 studies (2016–2026), three major gaps: fragmented evaluation metrics limiting cross-study comparability; **insufficient reporting of preprocessing and temporal-leakage prevention**; and weak linkage of accuracy to operational/economic outcomes. — [Wedhasari & Castro 2026, *Energies* 19(12):2937](https://doi.org/10.3390/en19122937) **[Review]**
- **Open challenges named by the base paper**: uncertainty quantification and calibration; physics-informed/hybrid learning; nonstationarity and calibration drift (continual learning, drift detection); privacy-preserving distributed learning; and interpretability beyond feature importance (counterfactuals, causal inference, explanations validated on temporal folds). — [Xie & Tummala 2025, §9](https://doi.org/10.3390/s25237294) **[Review/opinion]**
- **Leakage-control instrument already exists but is not adopted in energy.** Model info sheets covering each of eight leakage types are proposed as a reporting standard. — [Kapoor & Narayanan 2023](https://doi.org/10.1016/j.patter.2023.100804) **[Established proposal]**
- **Benchmark infrastructure has moved toward seen/unseen splits.** BuildingsBench formalises zero-shot-on-unseen-buildings and transfer as separate tasks, i.e. the community already accepts that these are different questions. — [Emami et al. 2023](https://doi.org/10.52202/075280-0871) **[Established benchmark]**
- **Post-competition error analysis is accepted as a contribution type.** A dedicated GEPIII error analysis (top-50 solutions, error taxonomy by timeframe/behaviour/magnitude/incidence) was published in a journal two years after the competition — evidence that "careful re-analysis of an existing benchmark" is publishable in this field. — [Miller et al. 2022](https://doi.org/10.1080/23744731.2022.2067466) **[Established precedent]**
- **The base paper itself is a template for the "benchmark + robustness + stability" genre** and is explicit that its own cross-domain benchmark is small (n = 6 patients, 12 stations, 10 buildings) and that findings are "trade-offs," not a winner. — [Xie & Tummala 2025, §7.6, Table 19](https://doi.org/10.3390/s25237294) **[Single-paper]**

### Inferences

Candidate novelty statements, ordered by defensibility against the evidence above:

1. **Strongest — a leakage decomposition on the full BDG2 panel.** Factorial design: {random k-fold, blocked-by-time, leave-one-building-out, leave-one-site-out} × {target-free features, lags only, rolling stats ending at *t*, leaky centred/contaminated rolling stats} × a small set of boosters. Report ΔR²/ΔRMSE **per channel** (temporal autocorrelation vs building identity vs feature construction) with per-building distributions and CIs. Justification: the base paper reports only one composite ΔR² on 10 buildings ([§7.5.3](https://doi.org/10.3390/s25237294)); no paper found separates the channels (KQ1, KQ2, KQ3 gaps); BuildingsBench shows the community already treats seen/unseen as distinct tasks ([NeurIPS 2023](https://doi.org/10.52202/075280-0871)).
2. **Strong — a "leakage-free feature protocol" + reporting checklist for building-energy ML**, adapted from the eight-type taxonomy and model info sheets ([Kapoor & Narayanan 2023](https://doi.org/10.1016/j.patter.2023.100804)) and instantiated on BDG2 with per-feature availability timestamps. Novelty is the instantiation and the measured cost of compliance, not the taxonomy.
3. **Moderate — robustness as a rank-stability matrix rather than a winner.** Replicate the noise/impulse/drift protocol at BDG2 scale and report model-rank instability across diagnostics, testing the base paper's rank-reversal claim ([§7.6.3](https://doi.org/10.3390/s25237294)) on more buildings. A negative or partially confirming result is publishable and useful.
4. **Moderate — separate sensor drift from model calibration drift.** Report both covariate-level drift stress tests and temporal reliability/calibration of predictive intervals, which the reviewed literature conflates (KQ4 inference). Needs a probabilistic model or conformal wrapper, so scope grows.
5. **Weak / not novel — "CatBoost (or LightGBM) is best on BDG2."** Table 19 of the base paper already finds modern GBDTs statistically indistinguishable on energy (XGBoost p = 0.256, LightGBM p = 0.056, GBM p = 0.403 vs CatBoost), and the original GEPIII competition already established that top solutions cluster tightly. A new "winner" claim would be contradicted by existing evidence.

### Gaps

- **No review was found that explicitly lists "decompose the random-vs-temporal CV gap" as future work** — the gap is implied by the base paper's single composite ΔR² and by the PV review's "insufficient reporting of temporal leakage prevention," not stated as an agenda item. The novelty argument therefore has to be constructed, not cited; that is a weakness to acknowledge in the paper's introduction.
- The JOBE 2025 review's full list of three challenges could not be retrieved (abstract truncated); the second and third challenges are unknown. **[Unverified detail]**
- I found no 2022–2026 review that recommends a specific robustness-diagnostic suite for building energy, so claim (3) cannot be anchored to a review-level call.
- Whether a "decomposition" study is sufficiently novel for a mid-tier applied-energy journal (vs. requiring a methods venue) is a judgement call not answerable from the literature; the closest precedent is the GEPIII error-analysis paper published in *Science and Technology for the Built Environment*.

---

## Source-reliability summary (for the report writer)

| Claim | Status |
|---|---|
| Random CV inflates building-energy R² by 0.63–0.74 | Verified from base paper full text; **single paper** (10 buildings) |
| Leakage affects 294 papers across 17 fields; 8-type taxonomy | Verified abstract; **well established** (2,341 citations) |
| GEPIII: 4,370 participants; 79.1% of test data within RMSLE_scaled < 0.1 | Verified abstract; **well established** |
| BDG2 = 3,053 meters / 1,636 buildings / 19 sites | Verified abstract; **primary source** |
| Drift causes 327–586% RMSE inflation in energy | Verified from base paper; **single paper**, simulated drift |
| SHAP rank stability ρ = 0.65–0.82 while temporal R² = 0.24–0.34 | Verified from base paper; **single paper** |
| 10-fold CV → up to 20.5% RMSE inflation from pre-split windowing | Verified abstract; **preprint only** |
| Cross-building TL improves over target-only training | Verified abstract (LSTM-DANN); **single paper**, others title-level only |
| Rolling-window-including-*t* inflation in building energy is unquantified | **Absence of evidence** from API searches — not a systematic review |
