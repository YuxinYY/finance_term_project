# Economically implausible values

All figures are computed on `data/train-2025.csv` (1,023,552 rows), excluding missing values, with a tolerance of €1. The reference is Riva, *Structure and Contents of (Italian) Financial Statements* (`docs/relevent reading/`). Its templates follow the post-2016 reform, while this data covers 2007–2012, so the mapping is indicative only. Missing-value handling: see [02-missing-value.md](02-missing-value.md).

Principle: **keep original values, only add flags, never change any value**. Rows that violate these checks default more often than the overall rate (4.37%), so they should not be dropped. Negative values are also already included in the data's totals (among rows with negatives, the asset and liability identities hold at about the same rate as for all rows, and the `wc_net` formula holds 100%), so changing them would break consistency between columns.

| Issue | Scale | Handling | Reason |
|---|---|---|---|
| 14 asset and liability columns are negative: `asst_intang_fixed`, `asst_tang_fixed`, `asst_fixed_fin`, `AR`, `liab_lt`, `liab_lt_emp`, `debt_bank_st/lt`, `debt_fin_st/lt`, `AP_st/lt`, `debt_st`, `debt_lt` | 70–1,285 rows per column, 6,662 rows in total, default rate 5.34% | `neg_bs_flag` | Assets are reported net (Riva 10.3.2) and offsetting is prohibited (10.2), so none should be negative; more likely the data vendor aggregated accounts without reclassifying. 5%–34% are also negative in the same firm's other years, which looks like a consistent reporting practice |
| `cash_and_equiv` is negative | 2,150 rows | None | `asst_current` already includes the negative cash; our inference is that overdrafts were netted into cash. **No support in the literature**: Riva reports overdrafts as bank debt (10.3.1 D.4) |
| `AR > asst_current` | 715 rows | None | 677 of these have negative cash; counting the negative cash, 676 rows hold (`asst_current` = inventory + receivables + financial assets + cash, 10.3.2) |
| `rev_operating`, `COGS` are negative | 3,270 / 7,655 rows, 9,762 rows in total, default rate 6.86% | `neg_pl_flag` | Cause unknown: inventory changes can move production value and costs away from sales (10.4.1–10.4.2), but cannot explain negative totals. Changing values would break `prof_operations = rev_operating − COGS` |
| `exp_financing` is negative | 720 rows | Merged into `neg_pl_flag` | Financial expenses (template item 17) should not be negative; see [Open questions](#open-questions) |
| Debt detail sum > `debt_st` / `debt_lt` | 5,086 / 9,013 rows (11,620 rows in total with no missing details, default rate 4.58%) | `debt_detail_inconsistent` | In 75% of these, the sum of all six details is still ≤ `debt_st + debt_lt`, so it is a short-/long-term classification mismatch between details and totals; the totals are more reliable (the `wc_net` formula holds) |
| Fixed + current > `asst_tot`; equity + liabilities > `asst_tot`; `cash_and_equiv > asst_current`; `eqty_tot > asst_tot` | 696 / 1,468 / 97 / 61 rows | None | Most exceed by less than 1% of `asst_tot`; rounding noise |
| `ebitda < prof_operations` (negative D&A) | 44 rows | Merged into `neg_pl_flag` | The implied D&A = `ebitda − prof_operations` should not be negative; the `cf_operations` formula still holds for these 44 rows, so the values are consistent, just implausible |
| Derivable columns do not match their formula: `prof_operations`, `wc_net`, `margin_fin`, `cf_operations = profit + (ebitda − prof_operations)` | 105 / 181 / 431 / 1,462 rows | None | Formulas hold for 99.85%–99.99% of rows, and we cannot tell which input is wrong; no overlap with the negative-D&A rows, and only 100 of the `cf_operations` mismatches also have another formula mismatch |
| `goodwill` does not match its name | 10.5% negative; only 15.7% have `≤ asst_intang_fixed` | Keep original values; add `neg_goodwill_flag` | Goodwill should be a non-negative component of intangible assets (10.3.2); see [Open questions](#open-questions) |
| `liab_lt_emp > liab_lt`, `debt_lt > liab_lt` | 57.7% / 68.6% | None (not a violation) | `eqty_tot + debt_st + debt_lt + liab_lt + liab_lt_emp ≈ asst_tot` holds for 80%, so we infer `liab_lt` = provisions for risks and `liab_lt_emp` = employee severance pay TFR (10.3.3), not long-term liabilities and a sub-item |
| `inc_financing` is negative | 81.7% | None (not a violation); see [Open questions](#open-questions) | Inferred to be net financial income/expense (template class C "15 + 16 − 17 ± 17bis", 10.4.1): `inc_financing + exp_financing ≥ 0` holds for 95.65% |
| `prof_financing` is 0 in 95.7% of rows and does not match `inc_financing`, `exp_financing` | 979,660 rows are 0 | Treat 0 as a real 0; add `fin_adj_flag` | See [Open questions](#open-questions) |
| `profit` cannot be derived from the other income-statement columns | Every combination matches < 0.2% | None | Residual correlates 0.78 with `taxes`; the definition still differs |
| `taxes` is negative | 29.7% | None (not a violation) | Prepaid taxes and similar are recorded with a negative sign (10.4.2 item 20) |
| `eqty_tot` is negative | 4.0% | None (not a violation) | Insolvency; a real situation |
| `COGS > rev_operating`; `profit > rev_operating`; `rev_operating == 0` | 23.8% / 2.0% / 3.05% | Deferred to the outlier stage | Operating losses, financial income of holding companies; ratios with revenue as the denominator break when revenue is 0 |
| `roa` ranges from −964 to 908 | | Deferred to the outlier stage | The formula matches exactly, so these are extreme values; `roa` is the return on operating profit (not net income) |
| `days_rec` has negative values and an unclear definition | 74,688 rows negative | Column dropped | See [02-missing-value.md](02-missing-value.md) |
| `roe` truncated at ±150 | 72,937 rows missing | Column dropped | See [02-missing-value.md](02-missing-value.md) method F |

Checked with no issues: no duplicate (`id`, `fs_year`); `fs_year` equals the year of `stmt_date`; every `def_date` is after `stmt_date`; every `asst_tot` > 0, **minimum €1.5M** (the sample seems to include only firms above €1.5M); no column in the data should fall between 0 and 1.

## Flag definitions

All flag rules use fixed values, so training and holdout use the same rules.

```
neg_bs_flag       = any of the 14 asset and liability columns < 0
neg_pl_flag       = rev_operating < 0 or COGS < 0 or exp_financing < 0 or ebitda < prof_operations
fin_adj_flag      = prof_financing != 0                                                        (43,236 rows)
neg_goodwill_flag = goodwill < 0                                                               (107,638 rows)
debt_detail_inconsistent = none of debt_bank_st, debt_bank_lt, debt_fin_st, debt_fin_lt, AP_st, AP_lt missing, and
                           (debt_bank_st + debt_fin_st + AP_st > debt_st + 1
                            or debt_bank_lt + debt_fin_lt + AP_lt > debt_lt + 1)               (11,620 rows)
```

`debt_detail_inconsistent` and `debt_detail_missing` from [02-missing-value.md](02-missing-value.md) method C (any of the six details missing, 134,145 rows) are mutually exclusive: one marks "details missing", the other "details present but inconsistent with the totals", so the two flags do not overlap and cause collinearity. Default rates: missing 5.29%, inconsistent 4.58%, neither 4.23%.

Later feature engineering: columns that keep negative values cannot be log-transformed directly; ratios with `asst_tot` (always > 0) as the denominator are unaffected.

## Open questions

> **The following are our hypotheses; no definition from the data provider or literature confirms them.** To be confirmed with the professor.

### The three financial income columns

One would expect `prof_financing = inc_financing − exp_financing`, but among the 43,235 rows where all three columns are present and `prof_financing ≠ 0`, `inc ± exp`, `inc` and `−exp` each match ≤ 0.02% of rows, and the pairwise correlations are only −0.06 to −0.21. We therefore assume the three columns map to different template items:

| Column | Assumed meaning | Evidence | Handling |
|---|---|---|---|
| `inc_financing` | Class C net financial income/expense (15 + 16 − 17 ± 17bis), **already net of** financial expenses | 81.7% negative; `inc_financing + exp_financing ≥ 0` holds for 95.65%, i.e. adding back the expense gives gross financial income that is almost always non-negative | Keep original values, negatives are normal; do not subtract `exp_financing` again |
| `exp_financing` | Item 17 interest and other financial expenses (gross) | | Keep original values; the 720 negative rows are merged into `neg_pl_flag` |
| `prof_financing` | Class D value adjustments to financial assets (revaluations − write-downs), unrelated to class C | Most non-financial firms have no such item, which explains 95.7% being 0 | Treat 0 as a real 0, not missing; add `fin_adj_flag = prof_financing != 0`; no formula correction |

Missing values are still filled with [02-missing-value.md](02-missing-value.md) method B; the sector median ratio of `prof_financing` is 0, so filled values are 0, consistent with the hypothesis above. If the hypothesis fails (e.g. the professor confirms that a 0 in `prof_financing` means missing), `prof_financing` and `fin_adj_flag` need to be reworked.

### `goodwill`

The column is named goodwill, but its values do not look like goodwill: 10.5% are negative (107,638 rows) and only 15.7% have `goodwill ≤ asst_intang_fixed`, whereas goodwill should be a non-negative component of intangible assets (10.3.2). It correlates 0.75 with `prof_operations`, and **100%** of rows with `goodwill < 0` have `prof_operations < 0`, so it looks more like some profit measure than an asset. Its true definition is unknown.

Handling: keep `goodwill` unchanged and add `neg_goodwill_flag = goodwill < 0`. These rows have a default rate of 8.87%, versus 3.84% for non-negative rows (4.37% overall). Note that this flag is a subset of `prof_operations < 0` (covering 44.2% of it) and overlaps heavily with operating losses. If the professor explains what `goodwill` means, we will decide whether to handle it differently.
