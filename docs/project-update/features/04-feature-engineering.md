# Feature engineering

Code: [notebooks/features/04_feature_engineering.ipynb](../../../notebooks/features/04_feature_engineering.ipynb), function `make_features(df)`. The input is data that has been through [01](01-financial-anomalies.md), [02](02-missing-value.md) and [03](03-outlier.md); the output has only the 31 features listed in the proposal. All figures are computed on the training data (1,023,552 rows); the overall default rate is 4.37%.

Left out: features based on `exp_financing` and `inc_financing` (interest coverage, financial expenses, financial income), the NLP signals, the raw amounts the proposal says are not used directly, and ROE (`roe` was dropped in 02).

## Problems with the features built as planned

### 1. Industry grouping (to discuss)

The proposal only says "about 7 groups, based on ATECO 2007 sections"; the grouping below is our own choice:

| Group | Sections (divisions) | Share | Default rate |
|---|---|---|---|
| `primary_utilities` | A, B, D, E (01–09, 35–39) | 4.2% | 2.34% |
| `manufacturing` | C (10–33) | 24.5% | 4.86% |
| `construction` | F (41–43) | 17.6% | 5.05% |
| `trade` | G (45–47) | 17.8% | 5.10% |
| `transport_hospitality` | H, I (49–56) | 5.7% | 3.45% |
| `real_estate` | L (68) | 19.5% | 3.04% |
| `services_other` | everything else | 10.8% | 4.61% |

Real estate has a large share, so it is its own group; `services_other` is heterogeneous. To discuss: the boundaries, whether to split `services_other` further, and whether to use the raw division as a categorical feature instead.

### 2. Handling invalid denominators

When the denominator is ≤ 0 the ratio is set to NaN, never `inf` (`RandomForest` raises an error on `inf`). No flag is added for now.

| Denominator ≤ 0 | Rows | Default rate | Affected features |
|---|---|---|---|
| `rev_operating` | 34,640 (3.38%) | 6.39% | `receivables_ratio`, `operating_margin` |
| `debt_st` | 2,077 (0.20%) | 4.96% | `current_ratio` |
| `debt_st + debt_lt` | 374 (0.04%) | | `st_debt_share`, `bank_debt_concentration`, `cash_flow_to_debt` |
| `ebitda` | 169,241 (16.53%) | 9.53% | `debt_to_ebitda` |

`asst_tot` is always > 0, so ratios on it are unaffected. Very large ratios are not handled separately (see [03](03-outlier.md)). To decide: whether to add flags for invalid denominators (e.g. `rev_zero_flag`), and how to fill the NaNs if the final model does not accept them.

### 3. `debt_to_ebitda`

The 16.53% of rows with `ebitda <= 0` are all NaN, yet this group's default rate is 9.53% (only 3.35% for `ebitda > 0`), so the missingness is itself a signal. Options: keep it as in the proposal; switch to `ebitda / (debt_st + debt_lt)` (only 374 rows have a denominator ≤ 0, and negative values are kept); or add an `ebitda <= 0` flag. To decide.

### 4. Definition of `receivables_ratio`

The proposal defines it as `AR / rev_operating`, which is what we implemented; the replacement for `days_rec` in [02](02-missing-value.md) is multiplied by 365. Tree models are unaffected, but the two documents should agree. Multiplying by 365 reads as days (median about 151 days), which is easier to explain to a bank. To decide.

### 5. Unseen categories

- `legal_struct`: an unseen value gives all zeros in `legal_*`. `SRS` has only 10 rows and `SAA` 248, too few to be meaningful; they could be merged into "other".
- `ateco_sector`: a value outside the 7 groups' ranges, or a missing one, goes to `services_other`.
- The features do not use the firm `id`, so firms never seen in training are not a problem.

## Suggested additional features

The flags produced by 01, 02 and 03 (not in the current output). Following the principle in [01](01-financial-anomalies.md) of keeping original values and only adding flags, the anomaly information lives only in the flags:

| Flag | From | Rows | Default rate |
|---|---|---|---|
| `neg_bs_flag` | 01 | 5,117 | 5.51% |
| `debt_detail_missing` | 02 | 134,145 | 5.29% |

`handle_outliers` in 03 does nothing for now, so it produces no flags. `fin_adj_flag` and `neg_goodwill_flag` are no longer used because they add no information (see [01](01-financial-anomalies.md)).
