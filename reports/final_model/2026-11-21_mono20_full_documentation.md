# The submission model (mono20): full documentation, from raw data to frozen harness

Team Turquoise (Kiet Nguyen, Desmond Peng, Chloe (Yuxin) Li). DS-GA 3001, Banca Massiccia 1-year PD model.
Written 2026-11-21.

This is the single reference for how the submitted model was built. It follows the pipeline in order, from
Roger's raw file to the frozen harness bundle, and records every decision, the alternatives considered, the
evidence and the sources. It also covers what was tried and rejected along the way.

**Rules this document follows**
- Every number is copied from a file produced by code; the file is named next to it.
- `D-###` refers to `docs/DECISION_LOG.md`; `R-##` refers to `docs/REFERENCES.md` (VERIFIED sources only).
- "In-sample" means measured on the training rows. It describes the model; it is not evidence of
  out-of-sample quality. The out-of-sample evidence is the walk-forward in Section 8.

---

## Contents

- 0 One-page summary
- 1 The business problem and the hard constraints
- 2 The raw data
- 3 Labels: what "default in the next 12 months" means here
- 4 How the project got here: tournament history (Oct 30 - Nov 16)
- 5 Data cleaning: anomaly flag and missing values
- 6 Features: from 80 candidates to 75
- 7 Monotone sign table: which 20 features are constrained and why
- 8 Development: walk-forward grid, monotone vs unconstrained, ablation
- 9 Level calibration: from raw scores to PDs
- 10 The final fit
- 11 What the final model looks like
- 12 The quick-ratio audit and the choice of mono20 over the unconstrained model
- 13 The harness
- 14 Leakage controls: where every parameter is fitted
- 15 What did not work, what was rejected, what is still open
- 16 Known limitations and risks
- 17 Reproduction: scripts in order
- 18 Decision index
- 19 File index

---

## 0. One-page summary

| Item | Value | Where |
|---|---|---|
| Model | LightGBM gradient-boosted trees, binary log-loss objective, **20 monotone constraints** | Sections 7, 10 |
| Trees / leaves | 240 trees, up to 31 leaves (mean 30.6) | `reports/final_model/2026-11-21_mono20_summary.json` |
| Hyperparameters | learning rate 0.05, num_leaves 31, min_child_samples 3000, feature_fraction 0.5, no bagging, lambda_l2 10, monotone method "intermediate" | `experiments/final/2026-11-21_final_fit/variants/mono20/params.json` |
| Features | 75: 32 financial ratios and amounts, 19 flags, industry target encoding, 4 legal-form and 19 region dummies | Section 6 |
| Training data | all labelled FY2007-2011 firm-years: **835,016 rows, 11,336 defaults (1.3576%)** | D-151 |
| Label | default between 1 Jun of year t+1 and 31 May of year t+2, for a statement of year t | D-151 |
| Level | prior shift to **tau = 1.4641%**, PD floor 0.03%, cap 99% | D-072, D-073, D-075, D-150 |
| Out-of-sample evidence (development, same setting) | validation AUC 0.8519 (mean of FY2008-10), test AUC 0.8469 / 0.8394 / 0.8561 on FY2009 / 2010 / 2011 | Section 8 |
| Cost of the constraints | about 0.001 AUC vs unconstrained (0.00075 ablation; 0.00134 grid bootstrap) | Section 8 |
| In-sample (sanity only) | AUC 0.8784; mean PD 1.4505%; PD from 0.03% to 89.7% | Section 11 |
| Harness | one call `predict(df)` → n×1 PDs; 1,023,552 rows in 48 s; equals the training pipeline to 2.2e-16; 25/25 stress tests | Section 13 |
| Why monotone (D-163) | every constrained reason code follows credit logic; the unconstrained model reverses the quick ratio, against R-01 p. 13, R-33 p. 11, R-05 p. 54 | Section 12 |

---

## 1. The business problem and the hard constraints

**The problem.** Banca Massiccia lends to private, non-financial Italian firms with more than €1.5M of assets.
For each applicant it needs one number: the probability that the firm misses a payment in the next 12 months,
using only what is known when the lending decision is made (the latest annual statement and firm
characteristics).

The PD serves two uses, and the model must serve both:

| Use | What it needs from the PD | Metric family | Our selection metric |
|---|---|---|---|
| Approve / decline | correct **ordering** | discrimination (AUC, accuracy ratio = 2·AUC − 1) | AUC (D-158) |
| Risk-based pricing | correct **level** | calibration (log loss, observed/predicted) | log loss + level check (D-006, D-158) |

**Hard constraints** (assignment, Appendix A):
1. One function call returns an n×1 vector of PDs in (0,1), in input order, with a PD for every row and no
   errors on messy input.
2. The harness does not re-estimate anything and uses no holdout row other than the current one.
3. No LLMs inside the model; Python only.
4. Only information available at decision time.
5. Label-like variables (city, sector, legal form) are categorical.
6. External data only if cleared weeks ahead. We used none (D-004).
7. The harness's computational complexity is documented (Section 13).

**Grading weight that shaped choices.** Formulation and documentation are 25 of 30 points; the holdout is 5. This
is why explainability was allowed to cost about 0.001 AUC (Section 12).

---

## 2. The raw data

Source: `data/raw/train-2025.csv` (read-only). Profile: `reports/data/01_profile.md` (generated by
`src/data/audit.py`).

| Fact | Value |
|---|---|
| Rows | 1,023,552 |
| Columns | 44 (41 dictionary fields + `def_date` + `stmt_date` + an index column) |
| Firms | 237,711 |
| Fiscal years | 2007-2012; `stmt_date` is 31 December of `fs_year` on every row |
| Duplicate (id, fs_year) | 0 → one row = one firm's annual statement (D-001) |
| Firms in all 6 years / only once | 101,169 / 29,963 |

**Data issues found and how each was handled**

| Issue | Handling | Decision |
|---|---|---|
| Dictionary misspells 3 fields (`eqyt_tot`, `rev_operations`, `prof_operating`) | Code uses the file's names | D-010 |
| `eqty_corp_family_tot` is empty on every row | Dropped | D-010; `missing_rules.DROP_COLS` |
| `days_rec` 72% missing | Dropped | `missing_rules.DROP_COLS` |
| `def_date` contains dates in 2013 (after the dictionary's stated end, 2012) | Primary: keep dates as given, data end 31 May 2013 | D-009 |
| HQ_city codes | Treated as ISTAT province codes → 20 regions | D-012; `feature_rules._REGION_PROVINCES` |
| ATECO division codes | Mapped to the 21 A-U sections | D-013; `data/interim/ateco_sector_english.csv` |
| `def_date` | Used only to build the label; never a feature (dropped before preprocessing) | CLAUDE.md §2.4; `2026-11-21_01_build_features.py` |

**Missing values** (`reports/data/02_missingness.md`):
- `asst_tot`, `id`, `fs_year`, `stmt_date`, `legal_struct` and `ateco_sector` are never missing.
- The debt-detail block (bank / financial / trade debt, short and long term) is missing together on about 9-12% of
  rows.
- `roe` is missing on 7.1% and `margin_fin` on 4.0%.
- Every other field is missing on less than 0.1%.

---

## 3. Labels: what "default in the next 12 months" means here

**Definition (D-151, adopted by KN on 2026-11-08; built by `src/data/2026-11-08_build_labels_s5.py`):**
- For a statement of fiscal year t (dated 31 Dec t): **y = 1 if `def_date` is in [1 Jun t+1, 1 Jun t+2)**.
- A firm that defaulted before 1 Jun t+1 is dropped as "already defaulted": it was in default before the statement
  could be used.

**Why June.** The window opens when a compliant Italian firm's statement must be filed: approval within 120 days
of year-end, filing within 30 days of approval, so by about 30 May (R-28, R-29). This matches RiskCalc's 3-4 month
reporting lag (R-01 Sec. 3.2 p. 16). It also makes FY2011 a complete 12-month cohort.

| fs_year | rows | usable | defaults | default rate | already defaulted |
|---|---|---|---|---|---|
| 2007 | 144,528 | 144,251 | 1,260 | 0.87% | 277 |
| 2008 | 161,863 | 161,383 | 1,945 | 1.21% | 480 |
| 2009 | 171,150 | 170,549 | 2,687 | 1.58% | 601 |
| 2010 | 175,399 | 174,674 | 2,881 | 1.65% | 725 |
| 2011 | 184,973 | 184,159 | 2,563 | 1.39% | 814 |
| 2012 | 185,639 | 0 (window not observed) | — | — | 650 |
| **2007-2011** | | **835,016** | **11,336** | **1.3576%** | 2,897 |

Source: `reports/data/2026-11-08_labels_s5.md`.

**Label history and alternatives.**
- The first label (D-002, "start 6": default 6-18 months after the statement) was superseded by D-151 to align with
  Chloe's pipeline.
- Start 0 and start 3 were also considered.
- The level cost of switching windows was below 0.6e-4 log loss (D-076).

**Risk.** If Roger rules that the 2013 default dates are invalid (D-009), FY2011 cannot be labelled under any
window. The training set would then be FY2007-2010.

---

## 4. How the project got here: tournament history (Oct 30 - Nov 16)

mono20 is the end of a longer process. The final model inherits these lessons.

| Phase | What we built | Key result | Decision |
|---|---|---|---|
| Benchmarks | SC-01 single ratio, SC-02 Altman/Ohlson-style logit, SC-03 WoE scorecard, SC-04 Z'-style proxy | SC-03 pooled AUC 0.8166 vs 0.8027 (SC-02); all benchmarks under-predict the level | D-030, D-031, D-033 |
| RiskCalc replica | per-ratio nonparametric transforms → probit → final transform | built and scored; not promoted | D-040 - D-047 |
| GBM | **GB-01: monotone LightGBM**, 26 features, native missing values | pooled AUC 0.8375; beat SC-03 by +0.0209 [+0.0188, +0.0231] | D-050, D-065 |
| Constraints, first test | GB-02 unconstrained twin | +0.0065 AUC, but wc_ratio ran 85% against sign → **constraints kept** | **D-051** |
| Monotone method | "intermediate" vs "advanced" | no material gain from "advanced" (+0.0002) | D-057 |
| Glass box | GAM-01, EBM with monotone shapes | AUC 0.8254; −0.0120 vs GB-01 → transparent fallback | D-081, D-112 |
| Ensemble | GB-01 + EBM average | rejected | D-086 |
| Calibration round | prior shift, central-tendency rules, floor/cap | recency rules win; prior shift adopted; floor 0.03% | D-070 - D-075 |
| Round 2 features | full2 set (RiskCalc France ratios, sector percentiles, encodings) | GB-20 +0.0061 AUC vs GB-01; GB-25 promoted (restores the interest-to-sales sign) | D-090 - D-099, D-120, D-144 |
| Team alignment (Nov 8 →) | one label (D-151), the team's preprocessing pipeline (Chloe and Desmond), TA test harness | one shared pipeline for the final model | D-150 - D-156 |

**Lessons carried into mono20**
1. Monotone constraints were the default from the start (D-051). The unconstrained model was rejected then for the
   same reason as now: it uses ratios against their economic sign.
2. The level must be set separately with a prior shift. Raw model PDs under-predict in a rising-default period
   (D-033, D-071, D-072).
3. Every fitted statistic must come from the training window (D-061, D-141, D-153).

---

## 5. Data cleaning: anomaly flag and missing values

**Code.** The team's rules (Desmond and Chloe), schema 1.3.0, at commit `teammate/project-update @ 7949207`.
- Vendored unchanged in `experiments/final/2026-11-21_final_fit/vendor_7949207/`.
- Shipped as `anomaly_rules.py` and `missing_rules.py` in the harness.

**Order of operations:** numeric coercion → anomaly flag → missing-value fill → features.

### 5.1 Anomaly flag (`anomaly_rules.py`)
- `neg_bs_flag` = 1 if any balance-sheet item that cannot be negative is reported negative (fixed assets,
  long-term liabilities, bank / financial debt, `debt_st`, `debt_lt`).
- Reported values are never changed; the flag runs on raw values before imputation.
- AR and AP are excluded because offsetting can legitimately make them negative.

### 5.2 Missing values (`missing_rules.fill_missing`, applied in this order)

| Method | Rule | Fields |
|---|---|---|
| Drop | removed as unusable | `Unnamed: 0`, `eqty_corp_family_tot`, `days_rec` |
| A | missing HQ_city → its own category "Unknown" | HQ_city |
| C | flag before filling: `debt_detail_missing` = any debt-detail field missing | 6 debt-detail fields |
| D | accounting identity prof_operations = rev_operating − COGS; restore whichever one is missing | rev_operating, COGS, prof_operations |
| F | one missing fixed-asset item = eqty_tot − margin_fin − the other two (floored at 0); otherwise 0 | 3 fixed-asset fields |
| H | if a debt total is 0, its missing detail items are 0 | debt detail |
| B | **sector median of (field / total assets) × the firm's total assets.** Medians are by A-U section and fitted on training rows; the 3 sections with the fewest firms use the overall median; sections not seen in training use the overall median | 29 fields (`impute_columns`) |
| I | `roe` recomputed for every row as profit / equity × 100 (zero equity counts as 1) | roe |
| E | identities after B and D: margin_fin = eqty_tot − fixed assets; wc_net = asst_current − debt_st; roa = prof_operations / asst_tot × 100 | margin_fin, wc_net, roa |

**Method G removed.** Method G filled a firm's missing value from its previous year's statement. It was dropped
because it uses another row (another holdout record of the same firm), which the assignment forbids
(D-152, D-156; removed in `6bf455a`, schema 1.3.0). In mono20 it is absent in both training and inference.

**Where the medians come from.** `sector_medians(train)` is fitted once on the 835,016 final training rows and
frozen in `artifacts/preprocessing.json` (`medians`). The harness only looks them up.

---

## 6. Features: from 80 candidates to 75

Code: `feature_rules.make_period_features` (schema 1.3.0) and `preprocess.fit_preprocessor`.

### 6.1 Ratio construction
- **Ratios use `_safe_divide`:** a zero denominator becomes 1, so the ratio falls back to the numerator; negative
  denominators are divided as reported.
- The matching `zero_*` flag records the original zero.
- **Known side effect** (D-154 #3, open): with a zero denominator the "ratio" is a euro amount, which puts firms in
  extreme tails. Section 12 shows what this does to the quick ratio.

| Group (schema) | Features in the final 75 |
|---|---|
| Size | `size` = log(total assets) |
| Leverage / capital structure | equity_ratio, debt_to_eqty, st_debt_share, lt_debt_to_capital, bank_debt_conc, tangible_asset_ratio, goodwill_ratio, cur_asst_ratio, neg_bs_flag |
| Liquidity / activity | cash_ratio, AR_ratio (AR / sales), payables_ratio (AP / assets), wc_ratio, cur_debt_ratio (current assets / ST debt), quick_ratio ((cash + AR) / ST debt), asset_turnover |
| Profitability | roa, roe, op_margin, net_prof_margin, profit, ebitda, COGS, loss_flag, neg_pl_flag |
| Coverage / cash flow | exp_financing, int_coverage (EBITDA / interest), financing_expense_ratio, cash_to_financing_exp_ratio, ebitda_to_debt, operating_cash_flow_margin, cash_interest_coverage, operating_cash_flow_ratio, operating_cash_flow_to_debt |
| Data-quality flags | debt_detail_missing, 15 zero_* flags (zero_total_debt, zero_debt_st, zero_debt_lt, zero_bank_debt, zero_cash, zero_asst_current, zero_ar, zero_rev_operating, zero_AP_st, zero_wc_net, zero_ebitda, zero_prof_operations, zero_profit, zero_cf_operations, zero_exp_financing) |
| Industry | `ateco_sector_te`: smoothed default rate of the firm's A-U section |
| Legal form | legal_SAU, legal_SPA, legal_SRS, legal_SRU (reference: SRL) |
| Region | 19 region dummies including "Unknown" (reference: Lombardia) |

Operating cash flow (`cf_operations`) is profit + depreciation and amortisation (FEATURE_SPEC).

### 6.2 Industry target encoding (`preprocess._oof_target_encode`)
- **Frozen mapping:** each A-U section's default rate, smoothed toward the overall rate with weight 20:
  (defaults + 20 × overall rate) / (rows + 20).
- **Training rows** get **out-of-fold** values (5 folds grouped by firm), so a firm's own default never feeds its
  own feature.
- **The harness** uses the frozen full mapping, and the overall rate 1.3576% for unseen sections.
- Label-like variables are never used as numbers (CLAUDE.md §2.5). Industry enters as a rate; legal form and
  region enter as dummies.

### 6.3 Collinearity screen: 80 → 75 (Chloe)
- **Method:** iterative within-group VIF, threshold 10, fitted on round-1 training rows (FY2007, 144,251 rows).
  FY2007 is inside every training window, so the screen uses no evaluation year.
- **Dropped (5):** rev_operating, prof_operations, zero_asst_tot, legal_SRL, city_Lombardia. The last two become the
  reference categories.
- **Kept outside the VIF groups:** debt_detail_missing and the zero_* flags.
- Source: `vendor_7949207/config/vif_selected_features.json`.
- `roe_negative` is computed by the rules but is not one of the 80 inputs (D-156 note).

### 6.4 Features the model never uses
14 of the 75 features are never chosen for a split in mono20, so their SHAP is 0
(`2026-11-21_mono20_summary.json`):
- 9 zero_* flags (zero_ar, zero_asst_current, zero_cf_operations, zero_debt_st, zero_ebitda, zero_prof_operations,
  zero_profit, zero_total_debt, zero_wc_net)
- neg_bs_flag
- legal_SRS
- city_Molise, city_Unknown, city_Valle d'Aosta

They cost nothing at prediction time and were kept to preserve the frozen feature contract.

---

## 7. Monotone sign table: which 20 features are constrained and why

### 7.1 Rule (D-157, fixed before the grid ran)
A feature gets a constraint only if both of these hold:
1. It has an expected sign, from the old team bundle (19 signs) or from FEATURE_SPEC (8 mapped).
2. The FY2007 training data agree: the decile default-rate rank correlation has that sign with |rho| ≥ 0.7 (for
   flags: |z| ≥ 2).

FY2007 is inside every training window, so no evaluation year is used. Source:
`experiments/gbm/2026-11-20_grid75/sign_table.csv` (script `2026-11-20_01_sign_table.py`).

### 7.2 The 20 constraints in mono20
−1 means a higher value lowers PD; +1 means a higher value raises PD.

| Constraint | Features |
|---|---|
| −1 (higher → lower PD) | equity_ratio, tangible_asset_ratio, cash_ratio, cash_to_financing_exp_ratio, wc_ratio, cur_debt_ratio, quick_ratio, roa, op_margin, net_prof_margin, int_coverage, ebitda_to_debt, operating_cash_flow_margin, cash_interest_coverage, operating_cash_flow_ratio, operating_cash_flow_to_debt |
| +1 (higher → higher PD) | AR_ratio, payables_ratio, financing_expense_ratio, loss_flag |

### 7.3 Candidates left unconstrained because the data disagreed (D-157)

| Feature | Data check |
|---|---|
| size | rho −0.37 |
| st_debt_share | +0.13 |
| lt_debt_to_capital | +0.35 |
| goodwill_ratio | expected +, data −0.89 |
| neg_bs_flag | z −1.0 |
| debt_detail_missing | z −0.3 |
| ateco_sector_te | +0.68 (just under the bar) |

All other features have no expected sign and are free.

### 7.4 Literature support for the signs
- Full map: `docs/2026-11-16_feature_literature_claims.md` and
  `reports/features/2026-11-21_sign_table_monotone_decision.md`.
- **Direct statements:** quick_ratio (R-01 p. 13, R-33 p. 11, R-05 p. 54); liquidity and debt coverage in general
  (R-33 pp. 11-12).
- **By analogy plus our data:** operating_cash_flow_ratio (R-33 p. 12 states the sign for debt coverage, not for
  cash flow / assets; corrected in D-162).
- **What a constraint does:** it forbids any wrong-way step in every tree. It does not force a straight line.
- **How it is applied:** LightGBM `monotone_constraints`, method "intermediate" (D-057).

---

## 8. Development: walk-forward grid, monotone vs unconstrained, ablation

### 8.1 Data and design
- **Data:** Chloe's 11.20 per-round feature files (75 features). These were audited clean before use (D-156):
  - labels match the panel on all 835,016 rows;
  - no overlap between train, validation and test;
  - imputation medians and the industry encoding come from the training window only;
  - Method G is absent.
- **Design (`reports/final_model/2026-11-21_development_results.md`):**

| Round | Train | Validation (selection) | Test (report only) |
|---|---|---|---|
| 1 | FY2007 | FY2008 | FY2009 |
| 2 | FY2007-08 | FY2009 | FY2010 |
| 3 | FY2007-09 | FY2010 | FY2011 (lockbox) |

### 8.2 Selection rule (D-158, fixed before results)
1. Highest mean validation AUC over the 3 rounds.
2. Settings within 0.001 AUC count as tied; ties go to lower validation log loss, then to the simpler setting.
3. Early stopping on validation AUC.
4. Monotone is the default. Unconstrained only if its paired-bootstrap ΔAUC interval excludes 0 **and** no top-10
   driver moves more than 50% against the expected sign.

### 8.3 Grid
- 288 settings: num_leaves {7, 15, 31, 63} × min_child_samples {100, 300, 1000, 3000} × feature_fraction
  {0.5, 0.8, 1.0} × bagging {0.7, 1.0} × lambda_l2 {0, 10, 50}. Learning rate fixed at 0.05.
- 110 settings completed in both modes and all rounds (a uniform random sample within a 2.75-hour budget): 660 fits.
- **Winner in both modes:** num_leaves 31, min_child_samples 3000, feature_fraction 0.5, bagging 1.0, lambda_l2 10.
  This is the mono20 setting.
- **What mattered most:** larger min_child_samples (3000 best) and lambda_l2 (50 ≥ 10 ≫ 0). Both are regularisation,
  as expected for a rare-event target.

### 8.4 Round by round, winning setting
Raw PDs, before the level step. Source: development report Section 3.

| Mode | Round | Validation year | Val AUC | Val log loss | Obs/pred (raw) | Test year | Test AUC | Best iteration |
|---|---|---|---|---|---|---|---|---|
| monotone | 1 | 2008 | 0.85583 | 0.05019 | 1.22 | 2009 | 0.84694 | 183 |
| monotone | 2 | 2009 | 0.85282 | 0.06270 | 1.44 | 2010 | 0.83935 | 252 |
| monotone | 3 | 2010 | 0.84606 | 0.06738 | 1.47 | 2011 | 0.85612 | 232 |
| free | 1 | 2008 | 0.85620 | 0.05021 | 1.23 | 2009 | 0.84719 | 106 |
| free | 2 | 2009 | 0.85356 | 0.06262 | 1.43 | 2010 | 0.83952 | 174 |
| free | 3 | 2010 | 0.84898 | 0.06710 | 1.46 | 2011 | 0.85735 | 205 |

Obs/pred above 1 in every validation year means raw PDs, which are anchored to the training rate, under-predict the
rising default rate. This is why the level step exists (Section 9).

### 8.5 Monotone vs unconstrained
Source: `experiments/gbm/2026-11-20_grid75/bootstrap_free_minus_mono.csv`, `ablation_summary.csv`.

| Comparison | ΔAUC (free − mono) | Interval | Δlog loss |
|---|---|---|---|
| Grid bootstrap, mean of 3 rounds | +0.00134 | [+0.00044, +0.00232] | −0.00012 [−0.00020, −0.00004] |
| Round 1 / 2 / 3 | +0.00037 / +0.00074 / **+0.00291** | only round 3 excludes 0 | |
| Seed re-runs (9) | free ahead 9 of 9; mean +0.00081 | | |
| Constraint ablation (6 fits) | +0.00075 | | −0.00012 |

**Ablation, the key rows** (full 24-row table in `reports/features/2026-11-21_quick_ratio_monotone_verdict.md`
Section 8):

| Configuration | Validation AUC | vs mono20 |
|---|---|---|
| unconstrained | 0.85217 | +0.00075 |
| release quick_ratio only | 0.85215 | **+0.00074** |
| release operating_cash_flow_ratio only | 0.85197 | +0.00056 |
| **mono20** | **0.85142** | — |
| release roa / wc_ratio / AR_ratio / equity_ratio | 0.85131 / 0.85129 / 0.85127 / 0.85115 | **negative**: these constraints help |

**Reading.** Almost the whole accuracy cost of the constraints comes from one feature, the quick ratio. Several
constraints act as regularisation and improve AUC.

**Variants rejected:**
- lit13 (13 constraints, frees the quick ratio): withdrawn, because it contradicts direct literature (D-159
  correction).
- mono19 (cur_debt_ratio released): +0.0001, which is noise.
- mono_m3: its released features were chosen on validation, so its score is biased.

### 8.6 Tree count for the final fit
- No validation set exists once all years are training data, so early stopping is impossible.
- **Rule:** the mean of round-3 best iterations for this configuration over the refit and seed runs (232, 214, 245,
  260), rounded to 10 → **240 trees**.
- The unconstrained model got 210 by the same rule.
- Source: `2026-11-21_06_train_variants.py` docstring.

---

## 9. Level calibration: from raw scores to PDs

**Formula (frozen in `params.json`, applied in `inference.calibrated_predictions`):**

    PD = clip( expit( a + logit(p_raw) + delta ), 0.0003, 0.99 )
    a     = 0.00077508   solved so the mean in-sample PD equals the training default rate (1.3576%)
    delta = logit(tau) − logit(1.3576%) = 0.07661187
    tau   = 1.4641%

| Element | Choice | Why | Decision |
|---|---|---|---|
| Method | **prior shift**: one intercept move on the log-odds scale | It is the standard prior correction (R-22 eq. 4; R-07 pp. 88-89). It needs no data at prediction time and does not change the ranking (within-year ΔAUC = 0) | D-072 |
| Rejected methods | Platt fitted on a past year; isotonic; Saerens EM; Chloe's Platt on pooled out-of-time PDs | Platt copies one past year's level. Isotonic creates ties (AUC −0.0008). EM needs the holdout batch (forbidden). Pooled Platt applied to a refit model double-counts the level | D-072, D-150 |
| Central tendency tau | **1.4641%** = expected default rate two years after the FY2011 anchor (1.3917%) under a driftless log-odds random walk with σ = 0.230 per year | Pricing needs a point-in-time level (R-16 p. 11); recency rules beat long-run averages in 2008-10 (D-071); the holdout period is unknown, so we hedge toward the expected rate | D-073, D-150, D-151 |
| Floor / cap | 0.03% / 99% | Floor by analogy to the Basel corporate PD floor (R-15 para. 285; not a claim that the bank is IRB). In the walk-forward, 31 PDs fell below 0.03% with 0 defaults. A 50% cap was rejected: 65% of PDs above 50% defaulted | D-075 |
| Fallback PD | tau, for any row whose features are not finite | A neutral prior when the model cannot score a row | D-160 |

**Asymmetry.** Under-predicting by 30% costs about 1.5× as much log loss as over-predicting by 30% (D-074). This is
one reason tau sits slightly above the 2011 anchor.

**Source of the numbers:** `experiments/calibration/2026-11-08_chloe_lgbm_v1/2026-11-08_params_for_production.json`
(tau by horizon: h = 1 → 1.4275%, h = 2 → 1.4641%, h = 3 → 1.5014%).

---

## 10. The final fit

Scripts: `experiments/final/2026-11-21_final_fit/2026-11-21_01_build_features.py` (mode `final`), then
`2026-11-21_06_train_variants.py`.

1. **Data.** Raw file joined to the label panel (`data/interim/2026-11-08_panel_s5.parquet`, label status "ok"),
   FY2007-2011: 835,016 rows, 11,336 defaults. `def_date` is dropped before preprocessing.
2. **Preprocessing.** `fit_preprocessor` fitted once on all 835,016 rows: sector medians, industry encoding,
   category lists, the 75-feature order. Frozen in `final_preprocessor.joblib`, then in `params.json`.
3. **Parity check first** (mode `parity`). The vendored code, fitted on FY2007-09 and applied to FY2010,
   reproduces Chloe's development features on every validation row (75 of 75 features). Training rows match on
   74 of 75; `ateco_sector_te` differs only because out-of-fold values depend on fold assignment
   (`parity_round03.csv`). So the development numbers in Section 8 describe the same code that built the final
   model.
4. **Model.** LightGBM with the parameters in Section 0 plus the 20-feature constraint vector. Also: max_bin 255,
   binning sample 10^7 (full-data bins), seed 42, deterministic, row-wise. 240 trees, no early stopping.
5. **Level.** `a` solved by root-finding so the mean in-sample PD equals 1.3576%; delta from tau; floor and cap.
6. **Packaging.** Model file, params, manifest with SHA-256 checksums, README, and a zip
   (`harness/2026-11-21_variants/2026-11-21_Turquoise_harness_mono20.zip`).

**Software:** trained with lightgbm 4.7.0 / pandas 2.3.3. The harness uses the pinned lightgbm 4.6.0 / pandas
3.0.1 / numpy 2.1.3 / scipy 1.16.3 and scores identically.

---

## 11. What the final model looks like

All numbers in this section: `experiments/final/2026-11-21_final_fit/2026-11-21_08_mono20_profile.py` →
`reports/final_model/2026-11-21_mono20_*.csv` and `2026-11-21_mono20_summary.json`. They are in-sample, all 835,016
rows, using exact TreeSHAP.

### 11.1 PD distribution

| Percentile | 0% | 1% | 10% | 25% | 50% | 75% | 90% | 99% | 99.9% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| PD | 0.030% | 0.036% | 0.086% | 0.19% | **0.57%** | 1.32% | 2.79% | 15.9% | 66.3% | 89.7% |

- Mean PD 1.4505%.
- 0.47% of rows sit at the floor.
- The median firm's PD (0.57%) is well below the mean because risk is concentrated in a small group.

### 11.2 Ranking power in-sample (PD deciles)
Source: `2026-11-21_mono20_grade_table_insample.csv`.

| PD decile | PD range | Mean PD | Observed default rate | Share of all defaults |
|---|---|---|---|---|
| 1 (safest) | 0.03% - 0.09% | 0.06% | 0.02% | 0.1% |
| 5 | 0.38% - 0.57% | 0.47% | 0.31% | 2.3% |
| 8 | 1.11% - 1.62% | 1.34% | 1.11% | 8.2% |
| 9 | 1.62% - 2.79% | 2.10% | 1.99% | 14.6% |
| 10 (riskiest) | 2.79% - 89.7% | 8.30% | 8.56% | **63.0%** |

- The riskiest 10% of firms hold 63% of defaults.
- Mean PD sits above the observed rate in deciles 1-8 because tau (1.464%) is set above the training rate
  (1.358%). This is the deliberate level hedge.

### 11.3 Level by year (in-sample)
Source: `2026-11-21_mono20_level_by_year_insample.csv`.

| fs_year | Observed | Mean PD |
|---|---|---|
| 2007 | 0.87% | 1.36% |
| 2008 | 1.21% | 1.54% |
| 2009 | 1.58% | 1.56% |
| 2010 | 1.65% | 1.37% |
| 2011 | 1.39% | 1.41% |

- The model has no macro input, so year-to-year swings in the default rate are not tracked.
- The financial ratios move a little: mean PD is 1.36% in 2007 and 1.54-1.56% in 2008-09.
- The level is a single point-in-time hedge (Section 9). This is a known limitation (Section 16).

### 11.4 Global drivers
Source: `2026-11-21_mono20_shap_global.csv`. Mean |SHAP| is in log-odds.

| Rank | Feature | Mean \|SHAP\| | Constraint | Value-vs-SHAP rank correlation |
|---|---|---|---|---|
| 1 | tangible_asset_ratio | 0.313 | −1 | −0.94 |
| 2 | equity_ratio | 0.292 | −1 | −0.94 |
| 3 | AR_ratio | 0.179 | +1 | +0.94 |
| 4 | operating_cash_flow_to_debt | 0.146 | −1 | −0.94 |
| 5 | cur_asst_ratio | 0.134 | free | +0.94 |
| 6 | ateco_sector_te | 0.115 | free | +0.80 |
| 7 | payables_ratio | 0.114 | +1 | +0.91 |
| 8 | roa | 0.113 | −1 | −0.94 |
| 9 | st_debt_share | 0.097 | free | +0.77 |
| 10 | financing_expense_ratio | 0.086 | +1 | +0.87 |
| 11 | operating_cash_flow_ratio | 0.082 | −1 | −0.89 |
| 12 | net_prof_margin | 0.077 | −1 | −0.86 |
| 13 | int_coverage | 0.070 | −1 | −0.79 |
| 14 | cash_ratio | 0.070 | −1 | −0.93 |
| 15 | size | 0.065 | free | −0.89 |

- Eight of the top 10 drivers are constrained, and every constrained feature moves in its expected direction.
- The free drivers in the top 10 (cur_asst_ratio, industry, st_debt_share) also move in economically readable
  directions.

**By family** (`2026-11-21_mono20_shap_family.csv`, share of total |SHAP|):

| Family | Share |
|---|---|
| leverage / capital structure | 33% |
| coverage / cash flow | 19% |
| liquidity / activity | 18% |
| profitability | 15% |
| region | 5% |
| industry | 4% |
| legal form | 3% |
| size | 2% |
| data-quality flags | 1% |

Leverage, coverage and liquidity carry about 70% of the explanation. For comparison, RiskCalc France ranks debt coverage as its most important category (R-33 p. 14). Our ranking differs (leverage first), which is worth one sentence on a slide.

### 11.5 Reason codes for individual firms
Source: `2026-11-21_mono20_firm_examples.csv`. Top 5 drivers for firms at four points of the PD distribution.

| Firm (PD percentile) | PD | Top drivers (+ raises PD, − lowers PD) |
|---|---|---|
| 10th | 0.09% | tangible assets 93% of total (−0.42), equity 43% (−0.33), current assets 2% of total (−0.27), low-risk industry (−0.13), negative working capital (+0.12) |
| 50th | 0.57% | tangible assets 77% (−0.38), higher-risk industry (+0.31), AR 1.35× sales (+0.22), current assets 23% (−0.18), equity 8% (+0.16) |
| 90th | 2.79% | no tangible assets (+0.46), AR 75× sales (+0.41), all assets current (+0.38), negative cash flow to debt (+0.32), equity 99% (−0.25) |
| 99th | 15.9% | cash flow −30% of assets (+1.12), negative equity (+0.93), ROA −48% (+0.71), EBITDA −€0.8M (−0.36), negative cash flow to debt (+0.31) |

Note the 99th-percentile firm: raw EBITDA (an amount, unconstrained) gets a negative SHAP even though it is very
negative. Unconstrained amount features can still produce counter-intuitive reason codes. Present reason codes for
the constrained ratios only.

### 11.6 Agreement with the unconstrained model
- Spearman correlation of PDs: 0.988.
- Mean absolute PD difference: 0.22 percentage points.
- 66% of firms get a PD within ±20% of the unconstrained model's.
- Ranking is nearly the same; the differences are in individual firms' explanations and in the tails.

---

## 12. The quick-ratio audit and the choice of mono20 over the unconstrained model

The full record is `reports/features/2026-11-21_quick_ratio_monotone_verdict.md` (D-162, D-163). Short version:

1. **Before fitting, the quick ratio is not linear.**
   - The default rate rises from 1.35% to 2.13% over the three lowest deciles, then falls to 0.59%. The decline
     holds in each of FY2007-2011.
   - The low tail is 43% real estate and 31.5% construction, plus data errors (negative cash or debt).
   - The high tail is firms with short-term debt ≈ 0.
2. **The unconstrained model reverses the quick ratio** (value-vs-SHAP +0.38). Three RiskCalc documents state the
   opposite directly. R-05 p. 54 explains why a model can flip it: highly rated firms run lower quick ratios.
3. **"Only 2 features disagree" understates the problem.** Ten constrained features have at least 16% of their
   curve moving the wrong way in the free model; three have more than 70%.
4. **A monotone constraint does not require linearity.** It flattens non-monotone regions instead of reversing them.
5. **Decision D-163 (KN, 2026-11-21): submit mono20.**
   - Cost: about 0.001 AUC. The level is the same.
   - Benefit: correctly signed reason codes for every constrained ratio.
   - **Disclosure:** this overrides the pre-set D-158 rule. That rule picked the free model because it only checked
     the top 10 drivers, and the quick ratio is outside them.

---

## 13. The harness

Bundle: `harness/2026-11-21_variants/mono20/`, zip
`harness/2026-11-21_variants/2026-11-21_Turquoise_harness_mono20.zip`.

### 13.1 Interface
- `python3 harness.py --input_csv in.csv --output_csv out.csv`, or `from harness import predict; predict(df)`.
- Returns an n×1 array of PDs in input order. The CSV output is one column with no header.

### 13.2 Pipeline per row (`inference.py`)
1. Normalise column names. Treat absent columns as all-missing (warning, never an error).
2. Coerce to numbers; infinities become missing.
3. Anomaly flag.
4. Missing-value fill with the **frozen** medians.
5. Features.
6. **Frozen** industry encoding; unseen sections get the overall rate.
7. Legal-form and region dummies; unseen categories get all zeros.
8. Restore the original row order.
9. LightGBM (240 trees, 1 thread), then the frozen calibration.
10. If a row's features are not finite, it gets the fallback PD (tau).
11. If anything unexpected fails, every row gets tau and the error goes to stderr. One PD per row is always returned.

### 13.3 Compliance
- **No re-estimation:** every statistic is read from `artifacts/preprocessing.json`.
- **Only the current record:** no statistic is computed from the input. The one sort by firm and year (inherited
  from the team code) reorders rows but uses no value from another row.
- Method G is removed (Section 5).
- Checksums in `manifest.json`. A mismatch only warns, so a line-ending change on unzip cannot stop the run (D-152).

### 13.4 Tests (`reports/harness/2026-11-21_harness_test_report.md`)
Clean virtual environment with the pinned requirements, Python 3.13.7:
- **Full file:** 1,023,552 PDs, 0 fallbacks, 48.2 s.
- **Harness vs training pipeline:** maximum difference 2.2e-16.
- **25 stress tests, all passed.** They cover:
  - shuffled rows, reversed columns, a single row;
  - row independence (0 of 300 rows change when other rows are removed);
  - messy values, all-missing rows, 11 different dropped columns, extra columns, a header-only file, an index column;
  - unseen firm ids in a future year, duplicate rows, thousand separators;
  - `predict(df)` matching the command line, and same-firm multi-year rows scored alone vs together (no change).
- The main bundle was also run on Python 3.11.14 (identical output) and from a path containing spaces.

### 13.5 Computational complexity
- For n rows, p = 75 features, T = 240 trees of depth L ≤ 30:
  - preprocessing O(n·p) plus one O(n log n) sort;
  - scoring O(n·T·L);
  - memory O(n·p).
- Measured: about 48 s for 1,023,552 rows on an Apple M1 laptop, roughly linear in n.
- **Fix needed before submission:** the mono20 README's complexity paragraph still says "T = 210 trees" (copied from
  the unconstrained bundle). The variant header in that README says 240, but the paragraph should be corrected.

---

## 14. Leakage controls: where every parameter is fitted

| Parameter | Fitted on | Used at prediction | Check |
|---|---|---|---|
| Label | `def_date` of the same firm-year | never | `def_date` dropped before preprocessing |
| Sector medians (Method B) | training rows only | frozen lookup | D-156: development files 99.7-99.9% match training-window medians, 0% own-year |
| Industry encoding | training rows, out-of-fold by firm | frozen mapping | D-156: mean encoding in test files = training rate, not the test-year rate |
| VIF screen | FY2007 (inside every training window) | fixed feature list | `vif_selected_features.json` |
| Sign table | FY2007 | fixed constraint vector | D-157 |
| Hyperparameters | validation years FY2008-10 | fixed | D-158; test years reported only |
| Tree count | round-3 best iterations (validation FY2010) | fixed 240 | Section 8.6 |
| Level a, delta, tau | training rows; label history up to May 2013 | frozen constants | D-070 (what the bank knows when it sets the level) |
| Method G (previous-year fill) | — | **removed** | D-152, D-156 |

**Known caveats:**
- FY2011 was looked at before (D-118 correction, D-152), so the "lockbox" is clean for model choice but not for the
  level rule.
- The feature set was designed while looking at FY2007-09 (D-062).

---

## 15. What did not work, what was rejected, what is still open

### 15.1 Rejected (with the reason)

| Idea | Reason | Decision |
|---|---|---|
| Unconstrained LightGBM (GB-02 in round 1; the D-160 model in the final round) | Better by 0.001-0.007 AUC, but uses ratios against their economic sign (wc_ratio, quick_ratio) | D-051, D-163 |
| lit13 (constrain only the 13 features with direct sources) | Frees the quick ratio, which has direct literature support for "−" | D-159 correction |
| mono_m3 (release the 3 worst violators) | Chosen on validation, so biased; frees the quick ratio | D-159 |
| mono19 (release cur_debt_ratio) | +0.0001 AUC, within noise | D-161 |
| Platt calibration on a past year or on pooled out-of-time PDs | Copies one year's level or double-counts it | D-072, D-150 |
| Isotonic calibration; Saerens EM | Ties lower AUC; EM needs the holdout batch | D-072 |
| PD cap at 50% | 65% of PDs above 50% default; raises log loss | D-075 |
| Method G (previous-year imputation) | Uses another row of the same firm | D-152, D-156 |
| Growth / lagged features | Need the previous year's statement (decision-time and other-row concerns) | D-020, D-150 flag |
| Ensemble GB + EBM | Rejected on evidence | D-086 |
| Class re-weighting or resampling | Destroys the probability level and needs another calibration stage, with no expected AUC gain | D-054 |
| External macro data | Not cleared in time | D-004 |
| Basel II "one-sentence link" | Hidden white-ink text in the assignment, an AI-detection trap | D-017 |

### 15.2 Corrections made along the way
- The cash-flow ratio sign was labelled SUPPORTED by R-01 pages that state no direction. It is now "analogy to
  R-33 p. 12 plus data" (D-162).
- operating_cash_flow_ratio was first called "no direction". It actually has a correct cliff with a slight drift;
  severity low (D-162).
- 11.15 development files were fitted on their own year, so their metrics were not out-of-sample (D-153). Replaced
  by the clean 11.20 files (D-156).

### 15.3 Still open

| Item | Status | Decision |
|---|---|---|
| Feature construction fixes: zero-denominator rule, negative denominators, AR_ratio when sales ≤ 0, roe | Proposed, not in the frozen model | D-154 |
| Label window used by the holdout; validity of 2013 default dates | Asked Roger/TA (email Q1, Q2) | D-002a, D-009 |
| Holdout period (sets the tau horizon) | Asked (email Q3) | D-073 |
| Switch the TA / submission bundle from the unconstrained model to mono20 | Pending | D-163 |
| v3 split proposal (firm-random early stopping) | Proposal, not used for mono20 | D-155 |

---

## 16. Known limitations and risks

1. **No out-of-sample number for the final fit itself.** By design every labelled year is training data. The
   evidence is the walk-forward on the same code and setting (Sections 8, 10).
2. **Level depends on one hedge (tau).** If the holdout year is a stress year like 2010, PDs will be too low
   (D-073: a true 2.13% would cost 11.8e-4 log loss).
3. **No cycle input.** Year-level swings are not tracked (Section 11.3).
4. **Extreme tails from the zero-denominator rule** (D-154). Ratio features can hold euro amounts. LightGBM only
   uses the order of values, but some extreme firms are placed arbitrarily (quick-ratio audit, Section 2).
5. **Unconstrained amount features** (ebitda, profit, COGS, exp_financing) can give counter-intuitive reason codes
   (Section 11.5). Show reason codes for constrained ratios only.
6. **Sign table rests on one year** (FY2007, 1,260 defaults; D-157).
7. **Development-to-final parity is exact on validation rows**, but the final model was not separately re-tuned on
   all years (tree count by rule).
8. **Platform:** tested on macOS only. The runtime is plain Python and the model file is platform-independent.

---

## 17. Reproduction: scripts in order

| Step | Script | Output |
|---|---|---|
| Panel and profile | `src/data/build_panel.py`, `src/data/audit.py`, `src/data/missingness.py` | `data/interim/panel.parquet`, `reports/data/01_profile.md`, `02_missingness.md` |
| Labels (D-151) | `src/data/2026-11-08_build_labels_s5.py` | `data/interim/2026-11-08_panel_s5.parquet`, `reports/data/2026-11-08_labels_s5.md` |
| Development files | Chloe's pipeline (teammate/project-update @ 7949207), per round | `~/Downloads/non-aggregated feature data sets (75 features 11.20)/` |
| Sign table | `experiments/gbm/2026-11-20_grid75/2026-11-20_01_sign_table.py` | `sign_table.csv` |
| Grid, bootstrap, seeds | `experiments/gbm/2026-11-20_grid75/` scripts | `grid_summary.csv`, `bootstrap_free_minus_mono.csv`, `seed_check.csv` |
| Sign evidence, ablation, SHAP sign check | `2026-11-21_05..07_*.py` (grid75) | `sign_evidence*.csv`, `ablation_*.csv`, `shap_sign_check.csv` |
| Calibration constants | `experiments/calibration/2026-11-08_chloe_lgbm_v1/2026-11-08_r1_fit_and_calibrate.py` | `2026-11-08_params_for_production.json` |
| Final features + parity | `experiments/final/2026-11-21_final_fit/2026-11-21_01_build_features.py parity` then `final` | `parity_round03.csv`, `final_train_features.parquet`, `final_preprocessor.joblib` |
| Unconstrained final model (reference) | `2026-11-21_02_train_final.py`, `03_build_bundle.py` | `final_lightgbm.txt`, `final_params.json`, main bundle |
| **mono20** | `2026-11-21_06_train_variants.py` | `variants/mono20/`, `harness/2026-11-21_variants/mono20/`, zip |
| Harness parity and stress tests | `2026-11-21_04_parity_harness_vs_training.py`, `reports/harness/2026-11-21_stress_test.py` | `parity_harness_vs_training.json`, test report |
| Development report | `2026-11-21_07_development_results.py` | `reports/final_model/2026-11-21_development_results.md` |
| mono20 profile | `2026-11-21_08_mono20_profile.py` | `reports/final_model/2026-11-21_mono20_*` |
| Quick-ratio audit, constraint table | `experiments/gbm/2026-11-20_grid75/2026-11-21_09_*.py`, `2026-11-21_10_*.py` | `quick_ratio_audit/` |

---

## 18. Decision index (mono20 lineage)

| Decision | One line |
|---|---|
| D-001 | One row = one firm-year statement |
| D-004 | No external data |
| D-005, D-006 | Walk-forward by year; log loss as the level metric |
| D-009 | 2013 default dates kept (primary); sensitivity documented |
| D-010, D-012, D-013 | Field names, province → region, ATECO → section |
| D-017 | No Basel II one-sentence link (AI trap) |
| D-051 | Monotone constraints kept despite a cost in power (round 1) |
| D-054 | No re-weighting or resampling |
| D-057 | Monotone method "intermediate" |
| D-070 - D-075 | Level: leakage rule, recency, prior shift, tau, asymmetry, floor / cap |
| D-150 | Calibration recipe for the team's LightGBM; tau 1.4641% on the 5-month label |
| D-151 | Label: default in [1 Jun t+1, 1 Jun t+2) |
| D-152 | Harness robustness fixes; Method G off at inference |
| D-153, D-156 | 11.15 files rejected; 11.20 files audited clean |
| D-154 | Feature-construction fixes (open) |
| D-157 | Sign table: 20 constraints |
| D-158 | Selection rule and grid; same winning setting in both modes |
| D-159 | Constraint set analysis; lit13 withdrawn |
| D-160 | Unconstrained final model (superseded as the submission by D-163) |
| D-161 | mono20, lit13 and mono19 fitted and bundled |
| D-162 | Quick-ratio / cash-flow audit; literature correction |
| **D-163** | **Submit mono20** |

---

## 19. File index

| What | Path |
|---|---|
| **This document** | `reports/final_model/2026-11-21_mono20_full_documentation.md` |
| Submission bundle | `harness/2026-11-21_variants/mono20/`, zip `harness/2026-11-21_variants/2026-11-21_Turquoise_harness_mono20.zip` |
| Model + parameters | `experiments/final/2026-11-21_final_fit/variants/mono20/lightgbm.txt`, `params.json`, `insample_summary.json` |
| mono20 profile | `reports/final_model/2026-11-21_mono20_{summary.json, shap_global.csv, shap_family.csv, grade_table_insample.csv, level_by_year_insample.csv, firm_examples.csv}` |
| Development results | `reports/final_model/2026-11-21_development_results.md` |
| Monotone and quick-ratio record | `reports/features/2026-11-21_quick_ratio_monotone_verdict.md` |
| Sign table and literature conflicts | `reports/features/2026-11-21_sign_table_monotone_decision.md`, `docs/2026-11-16_feature_literature_claims.md` |
| Harness tests | `reports/harness/2026-11-21_harness_test_report.md` |
| Data profile, missingness, labels | `reports/data/01_profile.md`, `02_missingness.md`, `2026-11-08_labels_s5.md` |
| Calibration study | `reports/calibration/summary.md`, `experiments/calibration/2026-11-08_chloe_lgbm_v1/` |
| Team preprocessing code | `experiments/final/2026-11-21_final_fit/vendor_7949207/` |
| Decisions, references, AI use | `docs/DECISION_LOG.md`, `docs/REFERENCES.md`, `docs/AI_USE_LOG.md` |
