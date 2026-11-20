# Feature lineage: from Roger's raw file to the 81 model features (v4)

Date: 2026-11-20. Branch `final-consolidation`. Purpose: one page per stage so Kiet, Desmond and Chloe can
cross-check **what was done to the data, in what order, by whom, and where the code lives**.

How to read "Who": taken from the git author of the commit (`git log teammate/project-update -- <path>`) and from
the team docs. Lines marked **[confirm]** are inferred from a commit message or a doc, not from reading the
code line by line. Please confirm or correct them at 4 PM.

Detailed evidence (per-feature formulas, denominator row counts, univariate AUCs, parity test):
`reports/features/2026-11-20_feature_consistency_report.md` (§3 per-feature table, §4 imputation, §5 categoricals).

---

## 0. The pipeline at a glance

```
train-2025.csv (Roger, read-only)
  │  1. label s5 + label_status            (Chloe's window, Kiet's panel build)      → 835,016 labelled FY2007-11 rows
  │  2. anomaly flags on REPORTED values   (Desmond, notebook 01)                    → neg_bs_flag, neg_pl_flag, debt_detail_missing
  │  3. missing-value imputation A–I       (Desmond, notebook 02; fitted on train)   → filled amounts, roe recomputed, roe_negative
  │  4. outliers: no winsorisation         (Desmond, doc 03)                         → values kept as is
  │  5. ratios + raw amounts + zero flags  (Chloe/Desmond, preprocess.py)            → 34 numeric + 21 flags
  │  6. categoricals                       (Chloe: industry; Desmond: city)          → ateco_sector_te, 5 legal, 20 region dummies
  │  7. collinearity pruning               (Desmond, doc 05)                         → removed ebitda_ratio, cash_to_debt, ...
  │  8. fit on train / transform the rest  (Kiet: A1 fix, v4 builder D-156)          → round files, final_fit, frozen preprocessors
  │  9. model-side choices                 (LGB-v4-mono, GBv4-02/03)                 → 25 monotone constraints
  └ 10. harness                            (Chloe built; Kiet patched D-152; REBUILD needed for 1.3.0)
```

Code entry point for stages 2-8: `src/finance_project/data_processing/preprocess.py`. Note that `preprocess.py`
contains **no anomaly or imputation code of its own**: at import it executes the cells tagged `export` in
`notebooks/features/01_flag_anomalies.ipynb` and `02_missing_values.ipynb` (preprocess.py lines 209-222).
**Changing the notebook changes the training pipeline.** That is how Desmond's 6bf455a took effect without
touching any `.py` file.

---

## 1. Raw data and label

| Item | What | Who | Where |
|---|---|---|---|
| Raw file | `data/raw/train-2025.csv`, one row per firm-year statement; never modified | Roger | read-only |
| Label window (s5) | y = 1 if `def_date` in [1 Jun of fs_year+1, 1 Jun of fs_year+2) | Chloe defined; Kiet adopted for everyone (D-151, 2026-11-08) | `src/data/2026-11-08_build_labels_s5.py` → `data/interim/2026-11-08_panel_s5.parquet` |
| Usable rows | `status_s5 == "ok"` only (drops already-defaulted firms and censored windows) | Kiet (D-002/D-009 logic) | same |
| Result | 835,016 labelled rows FY2007-11, 11,336 defaults (1.3576%) | | `data/splits/2026-11-20_walkforward_v4_temporal_year.json` |
| Leakage rule | `def_date`, `y*`, `status*` are label-only, never features | all | CLAUDE.md §2.4 |

## 2. Anomaly flags (computed on the REPORTED values, before any imputation)

Owner: **Desmond** (doc `docs/project-update/features/01-financial-anomalies.md`, notebook `01_flag_anomalies`;
commits 4e8b7c8 "anomalies check", fc8d85e "adjust anomaly check rule").

| Flag in the model | Rule | Why |
|---|---|---|
| `neg_bs_flag` | any of 11 balance-sheet items < 0 (fixed assets, liab_lt, liab_lt_emp, bank/fin debt st/lt, debt_st, debt_lt) | balance-sheet items should not be negative; 5,117 rows, default rate 5.51% (doc 01) |
| `neg_pl_flag` | rev_operating < 0 or COGS < 0 or exp_financing < 0 (evaluated **after** imputation) | implausible P&L sign. Doc 01/04 also list `ebitda < prof_operations`, but the code omits it (D-154 (5)) |
| `debt_detail_missing` | any of the 6 debt-detail columns missing (method C, before filling) | missingness is itself informative |

Not flagged on purpose (values kept as reported): negative AR/AP, negative cash, negative revenue/COGS,
negative exp_financing, detail-sum > total, goodwill not matching its name, negative taxes, negative equity
(it is real insolvency). Dropped flags: `neg_goodwill_flag`, `fin_adj_flag` (no information). Full list and
reasons: doc 01.

## 3. Missing-value imputation (methods A–I)

Owner: **Desmond** (doc 02, notebook `02_missing_values`; commits ab90f8a, d77f849, 6bf455a). Parameters
(sector medians) are fitted **on the round's training rows only** (`fit_preprocessor`), then frozen.

| Method | What it does | Columns | State in v4 files (schema 1.3.0) |
|---|---|---|---|
| A | missing HQ_city → "Unknown" | HQ_city | on |
| B | industry-section (A–U) median of x / asst_tot, times the firm's asst_tot; unseen/missing section → overall median | 26 amount columns | on; **6bf455a: the 3 sections with the fewest firms (O, P, U) use the overall median**; liab_lt, liab_lt_emp, debt_lt now use B instead of G |
| C | set `debt_detail_missing` before filling | 6 debt details | on |
| D | prof_operations = rev_operating − COGS identity | 3 columns | on |
| E | identities after B/D | margin_fin, wc_net, roa | on |
| F | fixed-asset reconstruction (eqty_tot − margin_fin, else 0) | 3 fixed-asset columns | on |
| G | "previous year was 0 → 0" (reads the firm's other row) | liab_lt, liab_lt_emp, debt_lt | **removed in 6bf455a** (it broke Appendix A at prediction time; D-152) |
| H | detail = 0 when its total is 0 | bank/fin/AP details | on |
| I | `roe` | roe | **6bf455a: recomputed for every row as profit / eqty_tot × 100 (eqty 0 → /1) + new flag `roe_negative`** |
| – | dropped columns | eqty_corp_family_tot (100% missing), days_rec (72% missing, unclear definition) | dropped |

Cross-check points:
- `roe_negative` = 1 when profit < 0 **and** eqty_tot < 0 (a loss with negative equity, where ROE comes out
  *positive*). The name is misleading; proposed rename `loss_and_neg_equity` (consistency report M3).
- Recomputed `roe` is unbounded (up to about 48,706 on rows where the provider left it blank, which are the
  distressed firms). In LGB-v4-mono it is **unconstrained** (GBv4-03).

## 4. Outliers

Owner: **Desmond** (doc 03, commit 61f7bd0). Decision: **no winsorisation, no capping**. Extreme values are
real and informative, and trees use only the ordering, so values beyond the training range fall into the
last leaf. (Kiet's round-1/2 `full2` library winsorised at the train p1/p99; that library is reference only.)

## 5. Feature engineering (ratios, raw amounts, zero flags)

Owners: **Chloe and Desmond** (doc 04, `preprocess.make_period_features`; commits 3149c17 Chloe "continue
feature engineering", 2864270 Desmond "new features to add", cfe6cc2 Desmond "adjust feature list",
0fb3984 Desmond "deleted sales growth", 5a01749 Desmond "accruals bug" **[confirm what changed]**, 2ef695d Chloe
"cleaned feature schema + post-feature missing/anomaly check"). Mode: `firm_year`, so **every feature uses only
the current statement**; no prior-year features (sales growth was removed, which also settles the D-150 flag).

| Family | Features (34 numeric) |
|---|---|
| Size | `size` = ln(asst_tot) |
| Leverage / capital structure | `equity_ratio`, `debt_to_eqty`, `st_debt_share`, `lt_debt_to_capital`, `bank_debt_conc` |
| Asset structure | `tangible_asset_ratio`, `goodwill_ratio`, `cur_asst_ratio`, `asset_turnover` |
| Liquidity / working capital | `cash_ratio`, `cash_to_financing_exp_ratio`, `AR_ratio`, `payables_ratio`, `wc_ratio`, `cur_debt_ratio`, `quick_ratio` |
| Profitability | `roa`, `roe`, `op_margin`, `net_prof_margin` |
| Debt service / cash flow | `int_coverage`, `financing_expense_ratio`, `ebitda_to_debt`, `operating_cash_flow_margin`, `cash_interest_coverage`, `operating_cash_flow_ratio`, `operating_cash_flow_to_debt` |
| Raw euro amounts | `profit`, `rev_operating`, `ebitda`, `COGS`, `prof_operations`, `exp_financing` |

Flags (21): `neg_bs_flag`, `neg_pl_flag`, `debt_detail_missing`, `loss_flag` (profit < 0), `roe_negative`, and
16 `zero_*` flags (input exactly 0 after imputation: asst_tot, total_debt, debt_st, debt_lt, bank_debt, cash,
asst_current, ar, rev_operating, AP_st, wc_net, ebitda, prof_operations, profit, cf_operations, exp_financing).

Division rule (`_safe_divide`): **a zero denominator is replaced by 1**, so the ratio becomes the numerator
(in euros if the numerator is an amount). A negative denominator is divided as is (sign flips). The `zero_*`
flags mark the /1 cases. This rule is the source of D-154 items (1)-(3), still OPEN:
- `lt_debt_to_capital` with a negative denominator: 11,636 rows, default rate 18.3%.
- negative revenue pushes `AR_ratio` and the margins to the "safe" end: 2,341 rows, default rate 7.6%.
- `int_coverage` = EBITDA in euros when exp_financing = 0: 28,790 rows.

## 6. Categorical variables (labels, never numbers)

| Field | Treatment | Who | Unseen / missing at prediction |
|---|---|---|---|
| `ateco_sector` | 2-digit division → section A–U (lookup `data/interim/ateco_sector_english.csv`) → smoothed target encoding `ateco_sector_te` = (defaults + 20 × mean) / (n + 20); training rows get 5-fold out-of-fold values grouped by firm; other rows get the full-train map | Chloe (7d64a1b "industry grouping logic") | training global mean |
| `HQ_city` | ISTAT province code → 20 regions + Unknown, one-hot (reference Abruzzo) → 20 `city_*` dummies | Desmond (f76cc0b "city grouping logic") | `city_Unknown` = 1 |
| `legal_struct` | one-hot of training levels, reference SAA → 5 `legal_*` dummies | Chloe/Desmond **[confirm]** | all zeros (treated as SAA) |

Open housekeeping: the ATECO lookup CSV is git-ignored, so a fresh clone fails on import (consistency report M10).
Commit it.

## 7. Collinearity pruning

Owner: **Desmond** (doc 05, commit d152a2c). Spearman on all pairs plus VIF on WoE-transformed features.
Dropped: `ebitda_ratio`, `cash_to_debt`, and (per doc) `operating_cash_flow_to_debt`. **Mismatch: doc 05 says
`operating_cash_flow_to_debt` was dropped, but it is still in the schema and in the v4 files** (consistency
report M7). Desmond to confirm which is intended. For the GBM, collinearity affects interpretation (split
importance), not accuracy.

## 8. Fit on train, transform everything else (leakage control)

Owner: **Kiet** (audit finding A1 / D-153: the 11.15 validation/test files had been fitted on their own year;
builders `src/data/2026-11-18_build_colab_splits.py` (v3) and `src/data/2026-11-20_build_v4_temporal_year.py` (v4)).

- For each round: `fit_preprocessor(train)` learns the sector medians, the target encoding, the legal and region
  levels. `transform_with_preprocessor` applies the frozen artifact to validation, test and lockbox.
- v4 checks (`reports/data/2026-11-20_v4_checks.md`): rows in = rows out (0 dropped); 100% of encodings come from
  the training map; mean TE in validation/test equals the training level, not the test-year rate; identical 81
  features in every file; a row's features are the same alone or in a batch (max abs diff 0).

Final count: **81 features** = 34 numeric + 21 flags + 1 industry TE + 5 legal + 20 region.

## 9. Model-side choices (LGB-v4-mono)

Owner: challenger-gbm run orchestrated by Kiet (`experiments/gbm/v4/decisions.md` GBv4-02/03).
- 25 monotone constraints. Start: the team table `MONOTONE_DIRECTIONS` in `lightgbm_model.py` (19 features, Chloe).
  **Removed** `lt_debt_to_capital` (sign flips with negative equity). **Added** `size` −, `tangible_asset_ratio` −,
  `operating_cash_flow_margin` −, `cash_interest_coverage` −, `operating_cash_flow_to_debt` −,
  `debt_detail_missing` +, `roe_negative` +.
- Unconstrained on purpose: `roe`, `debt_to_eqty`, `bank_debt_conc`, `goodwill_ratio`, `cur_asst_ratio`,
  `asset_turnover`, the 6 raw amounts, `neg_bs_flag`, `neg_pl_flag`, all `zero_*`, legal and region dummies.

## 10. Harness

- Built by **Chloe** (4e0ba3a, `tests/harness_submission/`), schema 1.2.1, Method G on.
- Patched by **Kiet** for the TA test run (D-152, `harness/2026-11-16_turquoise_submission/`): G off,
  missing columns imputed, fallback PD for every row.
- Parity test: the harness computes exactly the training features (20,003 rows, 0 differences).
- **Must be rebuilt for schema 1.3.0** from the v4 final preprocessor
  (`data/processed/2026-11-20_v4_temporal_year/artifacts/final_preprocessor.joblib`) and the final LGB-v4-mono model.
  The current harness cannot read 1.3.0 artifacts (consistency report M1).

---

## Questions to settle with Desmond and Chloe

1. Desmond: is `operating_cash_flow_to_debt` meant to be in the model (doc 05 says dropped)?
2. Desmond: what did 5a01749 "accruals bug" change? There is no accruals feature in the 81.
3. Desmond: rename `roe_negative` → `loss_and_neg_equity`, and keep `roe` uncapped or cap / replace it?
4. Chloe and Desmond: D-154 (1)-(3) (denominator ≤ 0 rules): fix before the final fit, or list as known limitations?
5. Chloe: sign off the constraint changes in §9 (drop `lt_debt_to_capital`, add 7).
6. Both: rewrite docs 04/05 from the code (several doc/code contradictions; consistency report M7) before the
   deck's feature slide is written.
7. Whoever owns it: commit `data/interim/ateco_sector_english.csv`.
