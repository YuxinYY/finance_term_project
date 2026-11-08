# Missing-value handling

Code: [notebooks/features/02_missing_values.ipynb](../../../notebooks/features/02_missing_values.ipynb), function `fill_missing(df, medians)`.

## Definition
- Input: a raw-data DataFrame (training or holdout). `medians` is the sector-median table computed from the **training data** by `sector_medians(train)`. At prediction time, pass the table computed from the training data; never recompute it from the holdout.
- Output: the filled DataFrame. Compared with the input:
  - Dropped: `Unnamed: 0`, `eqty_corp_family_tot`, `days_rec`, `roe`.
  - Added: one 0/1 flag, `debt_detail_missing`.
  - `HQ_city` becomes a string category (missing → `"Unknown"`).
  - Numeric columns are first coerced with `pd.to_numeric(errors="coerce")`, so blanks, `"NA"` and other missing-value spellings all become NaN (the project instructions say the holdout may spell missing values differently from the training data).
- Only missing values (NaN) are handled here. Negative values and other economically implausible values are not, see [01-financial-anomalies.md](01-financial-anomalies.md).
- Column groups follow [data_dictionary.md](../../data_dictionary.md).

## Per-column handling (`data/train-2025.csv`, 1,023,552 rows)

| Group | Columns | Missing | Handling | Basis |
|---|---|---|---|---|
| 1 Identity & attributes | `id`, `stmt_date`, `legal_struct`, `ateco_sector`, `fs_year` | 0% | None | |
| | `HQ_city` | 0.027% | A: fill `"Unknown"` | Rows cannot be dropped |
| | `def_date` | 95.6% | None (NaN means no default, not missing) | |
| 2 Assets | `asst_intang_fixed`, `asst_tang_fixed`, `asst_fixed_fin`, `asst_current`, `AR`, `cash_and_equiv`, `goodwill` | 0.0001–0.023% | B | [1] |
| | `asst_tot` | 0% | None; it is the scale for method B (if a holdout row lacks it, fill with the sector median, see method B) | |
| 3 Equity & liabilities | `eqty_tot`, `liab_lt`, `liab_lt_emp`, `debt_st`, `debt_lt` | 0.0001–0.029% | B | [1] |
| | `debt_bank_st`, `debt_bank_lt`, `debt_fin_st`, `debt_fin_lt`, `AP_st`, `AP_lt` | 8.89–11.99% | B + C: flag `debt_detail_missing` | [1][3][4][5] |
| | `eqty_corp_family_tot` | 100% | Drop the column | Entirely missing |
| 4 Income statement | `rev_operating`, `COGS`, `prof_operations` | 0.002–0.017% | D: restore by formula; B if not restorable | Data check |
| | `inc_financing`, `exp_financing`, `prof_financing`, `inc_extraord`, `taxes`, `profit`, `ebitda` | 0.002–0.064% | B | [1] |
| 5 Cash flow & derived ratios | `cf_operations` | 0.009% | B | [1] |
| | `margin_fin` | 3.98% | E: restore by formula | Data check |
| | `wc_net` | 0.0006% | E: restore by formula | Data check |
| | `roa` | 0.003% | E: restore by formula | Data check |
| | `roe` | 7.13% | F: drop the column | Computable from `profit` and `eqty_tot`; problematic denominator |
| | `days_rec` | 72.31% | Drop the column | Too sparse, unclear definition |

## Methods

| Method | What it does | Why |
|---|---|---|
| A | Treat a missing `HQ_city` as its own category `"Unknown"` | The 277 rows come from 85 firms that are missing in every year, so other years cannot fill them; every holdout row needs a PD, so rows cannot be dropped |
| B | Sector (`ateco_sector`) median of column / `asst_tot`, times the row's `asst_tot`; sectors absent from training use the overall median | `asst_tot` is never missing, so filled values scale with firm size. Early Moody's RiskCalc filled missing financial data with region and sector means [1]. We do not use region (`HQ_city` has 110 values, too few rows per cell) and use the median instead of the mean (less sensitive to extreme values) |
| C | If any of the six columns is missing, set `debt_detail_missing = 1`, then fill with B | The gaps may come from small firms filing the short-form statement [5], and missingness may relate to financial distress [3][4]. The six columns cannot be restored from `debt_st` / `debt_lt` (the sums match for only 4% / 62% of rows). One flag and sector-only grouping for now; split into short/long flags or add a size grouping if results are poor |
| D | `prof_operations = rev_operating − COGS`; restore whichever of the three is missing | Holds within €1 for 99.99% of rows; no other income-statement identity holds |
| E | `margin_fin = eqty_tot − (asst_intang_fixed + asst_tang_fixed + asst_fixed_fin)`; `wc_net = asst_current − debt_st`; `roa = prof_operations / asst_tot × 100` | Match rates 99.96%, 99.98%, 99.99%. Runs after B and D, so every input to the formula is already filled and every row can be computed |
| F | Drop `roe` | `roe = profit / eqty_tot × 100` (99.99% match), so it carries no independent information. The data provider truncates at ±150 and sets values beyond that, or with `eqty_tot ≤ 0`, to missing (72,937 rows). Filling any value (e.g. −150) creates an artificial spike that would still need handling later. Replace it in feature engineering with features whose denominator is always positive, such as `eqty_tot / asst_tot` and `profit / asst_tot` |

Order of execution: unify missing spellings → drop columns → A → record the C flag (before filling) → D → B → E.

### Approaches not taken
- Backing out a missing asset from `asst_tot` minus the other three: the dataset has only fixed and current assets, and lacks "subscribed capital unpaid" and "accruals and deferrals" [2], so the result would be overstated.
- Filling with 0: `asst_current`, `AR` and `cash_and_equiv` are almost never 0.
- Dropping rows: every row needs a PD at prediction time.

### Why `days_rec` and `roe` are dropped
- `days_rec` is 72.31% missing, at a similar rate in every year (67–74%).
- `days_rec` matches `AR / rev_operating × 365` for only 0.6% of rows and has negative values; the definition is unclear and it cannot be restored. If needed later, compute `AR / rev_operating × 365` ourselves.
- `roe`: see method F.

### Safeguard added in code, not yet discussed
- `asst_tot` has no missing values in training, but if a holdout row lacks it, method B cannot compute. The code therefore fills it with the sector median of `asst_tot` (an amount, not a ratio). Training data never triggers this.

## References

Only the [1] web page, [2], and the Moody's whitepaper were read in full; [3] and [4] were read from search summaries only and must be checked against the originals before being cited formally.

| # | Source | Used for |
|---|---|---|
| [1] | [Moody's Credit Risk Insights](https://www.moodys.com/web/en/us/insights/credit-risk/advances-in-default-detection-and-early-warning.html): early RiskCalc filled missing data with region and sector means; a newer version moves to a nearest-neighbour approach (described as "forthcoming", unconfirmed) | Method B |
| [2] | Riva, *Structure and Contents of (Italian) Financial Statements*, 10.3.2 (`docs/relevent reading/`): assets are A) subscribed capital unpaid, B) fixed assets, C) current assets, D) accruals and deferrals | Why the four asset columns do not sum to `asst_tot` (within 1% for only 79.3% of rows) |
| [3] | [Zhou (2007), LSE](https://eprints.lse.ac.uk/24434/): distressed firms are more likely to have missing data; deleting them biases the sample toward healthy firms | Method C: keep rows, keep the missingness signal |
| [4] | [UK and Italian small-business comparison](https://arxiv.org/pdf/1412.5351): WoE transformation beats multiple imputation, suggesting missingness is not random | Method C: keep the missingness signal |
| [5] | Riva, section 10.9: small firms may opt for the short-form statement (Civil Code art. 2435-bis), listing only top-level items; eligible if at most one of total assets €4.4M, revenue €8.8M, 50 employees is exceeded | Method C: all six columns missing in 9.3% of rows under the thresholds, 5.4% otherwise, 0.9% for assets > €50M; consistent in direction but **not confirmed** |
| — | [Moody's RiskCalc v3.1 whitepaper](https://www.moodys.com/sites/products/ProductAttachments/RiskCalc%203.1%20Whitepaper.pdf): **does not discuss imputation**; covers data cleaning (p.8), zero inventories by sector (p.18), keeping rounded data improved performance (p.20–21) | Background; a 0 is not necessarily a missing value |
