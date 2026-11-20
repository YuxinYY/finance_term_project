# Feature pipeline audit: Chloe and Desmond's 11.15 feature set (80 features)

Date: 2026-11-16. Requested by KN. Code: `src/audit/2026-11-16_feature_pipeline_audit.py` (rounds 1-6, →
`audit_log.md`, `r1`-`r6` CSVs) and `src/audit/2026-11-16_eval_file_leakage_check.py` (§A1 → `r5_eval_file_leakage.csv`).
Segment calibration (`r7_segment_calibration.csv`) came from a session command using harness PDs; it is not yet
scripted. Every number below is in a CSV in this folder or in `audit_log.md`.

What was audited: the teammates' pipeline on branch `project-update` (docs `docs/project-update/features/01-05`,
`src/finance_project/data_processing/preprocess.py`, notebooks `01_flag_anomalies`, `02_missing_values`),
the feature files in `~/Downloads/non-aggregated feature data sets (80 features version 11.15)/round 1/`,
the schema `Feature Schema - 11 - 14.csv`, and the frozen round-03 pipeline in the harness.
Labels: s5 rule (D-151), identical to Chloe's `label` column (100.00% agreement, round 5).

## Verdict

The **feature arithmetic is correct**. All 28 ratio formulas reproduce exactly from raw data with independent code
(100% match on every compared row), labels agree 100%, and the documented raw-data facts check out.
The problems are in **how the evaluation files were produced** (one red flag) and in **edge-case definitions**
that hurt face validity and explanations more than accuracy.

| # | Severity | Finding | Fix |
|---|---|---|---|
| A1 | RED | Validation/test feature files were fit on their own year (imputation medians and target encoding from that year's labels) | Regenerate val/test with `transform` (frozen train artifacts); refit Platt; re-report metrics |
| A2 | RED (fixed) | Harness Method G read other holdout rows | Fixed in `harness/turquoise.zip` (D-152) |
| B1 | AMBER | `lt_debt_to_capital` monotone +1 conflicts with insolvent firms (negative denominator) | Floor the denominator or drop the constraint |
| B2 | AMBER | Negative revenue flips `AR_ratio`, `op_margin`, `net_prof_margin` to the "safe" end | Map denominator ≤ 0 to the risky end + flag |
| B3 | AMBER | Zero denominators replaced by 1 turn ratios into euro amounts | NaN-then-cap or explicit value + existing `zero_*` flag |
| B4 | AMBER | `roe` is blank exactly when extreme; median imputation hides distress | Add `roe_missing` flag or drop `roe` (team doc method I) |
| B5 | AMBER | Schema text and code disagree for 4 features; `goodwill` is probably not goodwill | Align schema to code; caveat goodwill |
| B6 | AMBER | Default rates in team docs are "ever defaulted" (4.37%), not the 12-month label (1.36%) | Recompute doc numbers with `label` |
| C1 | INFO | 46 of 80 features have univariate \|AUC − 0.5\| < 0.02; 3 are constant | Prune for a cleaner, explainable model |

Correction to my earlier message: I said Chloe's pipeline "drops 601 FY2009 and 277 FY2007 rows". Those are
unlabelled raw rows (firms that had already defaulted, excluded by the label rule). Among labelled rows **her
pipeline drops none** in round 1 (round 5).

---

## A1. Evaluation files are not out-of-sample (RED)

`preprocess.py` has a correct `transform` path (frozen medians + frozen industry mapping), but the round-1
`features_validation.csv` (FY2008) and `features_test.csv` (FY2009) were not produced by it:

| Test | FY2008 validation | FY2009 test |
|---|---|---|
| Imputed `bank_debt_conc` matches medians fitted on **training year FY2007** | 2.25% of 28,309 rows | 1.40% of 31,157 rows |
| … matches medians fitted on **the file's own year** | **100.00%** | **99.72%** |
| Distinct `ateco_sector_te` values per industry (frozen mapping gives 1) | 5 | 5 |
| Mean `ateco_sector_te` vs that year's realised default rate | 0.0121 vs 0.0121 | 0.0158 vs 0.0158 |
| corr(TE, own-year industry DR) vs corr(TE, FY2007 industry DR) | 0.955 vs 0.838 | 0.916 vs −0.305 |

So each evaluation file was built with `fit` on that year: 5-fold out-of-fold target encoding on **the evaluation
year's own labels** and imputation medians from the evaluation year. The industry feature therefore carries the
test year's realised default level. Consequences:
- Reported validation/test metrics (e.g. `tests/temporal_firm_year_v2_temporary_report.md`) are not clean
  out-of-time numbers. Calibration-in-the-large is the most affected, because the leaked TE mean equals the
  realised rate. Discrimination is less affected (industry is a weak ranker: univariate AUC 0.56).
- The harness's Platt calibrator was fitted on FY2010 rows with leaky TE values, while the harness feeds the
  frozen training mapping. The two inputs differ.
- The packaged harness itself is clean. On raw FY2011 data it gives AUC 0.854 and PD/DR 1.10
  (`reports/harness/2026-11-16_harness_test_report.md`), which is the honest number to quote today.

Fix: re-run each round as `fit` on train → `transform` on validation and test with the saved artifacts →
refit Platt on the transformed validation set → re-score.

## B1. `lt_debt_to_capital` and its monotone constraint (AMBER)

`lt_debt_to_capital = debt_lt / (debt_lt + eqty_tot)` is constrained "higher → riskier" (+1). When equity is
negative enough, the denominator is negative and the ratio becomes negative or flips sign.

- 11,636 labelled rows have `debt_lt + eqty_tot < 0`. Their default rate is **18.3%** (13× the 1.36% average),
  and they sit at the 14th percentile of the feature, so the constraint pushes them toward a *low* PD
  (`r4_constraint_conflicts.csv`).
- The decile curve is U-shaped (default rate 2.0%, 0.4%, 0.5% … 2.5%; `r6_decile_default_rates.csv`), and the
  univariate AUC is 0.484, the only constrained feature whose direction conflicts with the data (`r6_feature_profile.csv`).
- Damage to PDs is small. Other features (equity ratio, losses) compensate, and the segment's mean model PD is
  0.205 vs a realised 0.183 (PD/DR 1.12; `r7_segment_calibration.csv`). But a SHAP explanation for an insolvent
  firm will say "LT debt to capital *lowers* your PD", which a credit officer would reject.

## B2. Negative revenue (AMBER)

2,341 labelled rows have `rev_operating < 0` (default rate 7.6%). Dividing by a negative revenue moves `AR_ratio`
to the bottom of its range (constraint +1 → low PD) and `op_margin` / `net_prof_margin` to the top
(constraint −1 → low PD). Segment PD/DR is 1.02, so again the damage is to explanations, not level.

## B3. The "zero denominator → 1" rule (AMBER)

`_safe_divide` replaces a zero denominator with 1, so the ratio becomes the numerator **in euros**.
Affected labelled rows: revenue = 0 in 25,613 rows (`AR_ratio`, both margins, `operating_cash_flow_margin`);
financing expense = 0 in 28,790 (`int_coverage`, `cash_to_financing_exp_ratio`, `cash_interest_coverage`);
short-term debt = 0 in 1,384 (`cur_debt_ratio`, `quick_ratio`) (`r4_denominators.csv`).
A firm with no revenue and €2M of receivables gets `AR_ratio` = 2,000,000, i.e. "worst possible". Its realised
default rate is 1.9%. Trees cope because the `zero_*` flags exist, but the mixed units are hard to defend and
appear as a hump in `cur_debt_ratio` / `quick_ratio` (decile 1 = 1.5%, decile 2 = 2.7%).

## B4. `roe` (AMBER)

The provider leaves `roe` blank whenever |ROE| > 150 (team doc 02, method I). 56,624 labelled rows are blank, with
a default rate of **8.2%** vs 0.86% when present. The pipeline fills the blank with the industry median (about 3%),
which labels the most distressed firms as typical. A masking test confirms the fill is weak even for random
blanks (median error 6.8 points, Spearman 0.15; `r2_imputation_masking.csv`). The model still prices the segment
(PD/DR 1.13) through other features, but `roe` should get a `roe_missing` flag or be dropped, as the team's own
doc originally decided.

## B5. Schema vs code (AMBER)

| Feature | Schema (`Feature Schema - 11 - 14.csv`) | Code | Impact |
|---|---|---|---|
| `neg_pl_flag` | includes `ebitda < prof_operations` | omits it | 5 FY2009 rows (1,801 vs 1,796) |
| `cash_interest_coverage` | NaN when `exp_financing ≤ 0` | divides by 1 (zero) or by the negative value | 6,047 FY2009 rows |
| `operating_cash_flow_margin` | "cash generated per unit of profit" | `cf_operations / rev_operating` (per unit of **revenue**) | wording only |
| `debt_to_eqty` | "how non-equity funding is split between reported debt and other liabilities" | `(debt_st + debt_lt) / eqty_tot` | wording only |

`goodwill_ratio` is the 15th-strongest feature (AUC 0.362), but team doc 01 shows that `goodwill` is negative in
10.5% of rows, exceeds intangibles in 84%, and is negative only when operating profit is negative. It behaves like
a profit measure, so the deck must not call it goodwill without a caveat.

## B6. Default rates quoted in team docs (AMBER)

Docs 01-04 quote "overall default rate 4.37%" and segment rates such as `neg_bs_flag` 5.51% and
`debt_detail_missing` 5.29%. 4.37% is the share of raw rows whose firm has *any* `def_date` (4.3696%), not the
12-month label (1.3576%). Against the label, `neg_bs_flag` rows default at 1.54% vs 1.36% overall, a much
weaker signal than the doc suggests. These numbers should be recomputed before they go in the deck.

## C1. Weak and constant features (INFO)

46 of 80 features have univariate |AUC − 0.5| < 0.02: all 20 regions, most `zero_*` flags, `neg_bs_flag`,
`neg_pl_flag`, `debt_detail_missing`, `bank_debt_conc`, `asset_turnover`, `lt_debt_to_capital`. One never varies:
`zero_asst_tot` (assets are always ≥ €1.5M). Two are near-constant among the 835,016 labelled rows:
`zero_wc_net` (18 rows) and `legal_SRS` (8 rows) (`r6_decile_default_rates.csv`). Weak univariate
signal does not prove a feature is useless in a tree, but a shorter list is easier to explain and to monotone-audit.

---

## What checked out (GREEN)

**Raw identities** (`r1_identities.csv`; match rate over rows where all inputs exist):
`prof_operations = rev − COGS` 99.99%; `wc_net = asst_current − debt_st` 99.98%; `margin_fin = equity − fixed assets` 99.96%;
`roa` formula 99.99%; `roe = profit/equity` 99.99%; `cf_operations = profit + D&A` 99.86%; balance-sheet sum within 1% 80.05%;
asset sum within 1% 79.34%. All agree with docs 01/02.
One small discrepancy: docs say the short-term debt details sum to `debt_st` for 4% of rows; I get 1.32%
(long-term 59.7% vs the doc's 62%). This does not change any decision.

**Raw facts:** no duplicate (id, fs_year); minimum `asst_tot` is exactly €1,500,000; `neg_bs` rows = 5,117 (doc: 5,117);
negative equity 3.98%; `def_date` runs to 2013 (5,570 defaults dated 2013, the open question to Roger).

**Feature formulas:** an independent recompute of 28 ratios from raw data matches Chloe's FY2009 file on 100% of
compared rows (`r3_independent_recompute.csv`).

**Imputation** (`r2_imputation_masking.csv`, 3,000 known values masked per column in FY2010):
- Methods D and F are exact: `rev_operating` / `COGS` are restored by identity (Spearman 1.00), and fixed-asset zeros are kept.
- Method B (sector median × assets) is weak (Spearman 0.14-0.56), but those columns are missing in < 0.03% of rows, so the impact is negligible.
- The material case is the debt breakdown (9-12% missing). For those rows `bank_debt_conc` is essentially a sector
  constant, and the `debt_detail_missing` flag carries little signal (AUC 0.506).
- Unused reconciliation: `asst_current = wc_net + debt_st` holds for 99.98% and could restore either one exactly,
  but both are missing in < 0.001% of rows.

**Monotone constraints:** 18 of 19 agree with the univariate direction in the data (only `lt_debt_to_capital` conflicts).
The strongest features are stable by year (`audit_log.md`): cash-flow and EBITDA-to-debt AUC is 0.22-0.27 in every
year (equivalently 0.73-0.78 with the sign flipped).

**Calibration by segment and year** (harness PDs, `r7_segment_calibration.csv`): pooled PD/DR is 1.16, falling from
1.72 in FY2007 to 0.91 in FY2010 and 1.10 in FY2011. The model is not cycle-adjusted; its level mostly reflects the
calibration year.

## Appendix: reproducing A1
`src/audit/2026-11-16_eval_file_leakage_check.py` → `r5_eval_file_leakage.csv`. The script rebuilds the
FY2008/FY2009 features with `missing_rules.sector_medians` fitted on FY2007 and, separately, on the file's own
year. It compares the imputed `bank_debt_conc` with Chloe's file on rows with missing debt details, and compares
the mean industry TE per section with each year's industry default rate.
