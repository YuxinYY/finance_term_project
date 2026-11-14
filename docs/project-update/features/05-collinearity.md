# Collinearity check

The check covers the features currently produced by `preprocess.py` (`firm_year` mode). Data: `data/processed/walk_forward/round_01/train.csv` (FY2007, 115,400 rows); we ran `preprocess fit` and checked its output. Scope: 31 numeric features (including `ateco_sector_te`); `legal_*`, `city_*` and the flags are not included.

## Why check

- **WoE logistic**: most affected. Strong collinearity makes coefficients unstable and can flip their signs, so the scorecard becomes hard to explain.
- **EBM / LightGBM**: accuracy is largely unaffected, but importance is split between collinear features, which can mislead interpretation.

## Method

### 1. Spearman correlation, all pairs

We compute the Spearman correlation matrix of the 31 features and list every pair with |ρ| ≥ 0.7.

- **Spearman rather than Pearson**: Spearman uses only the ordering. Some ratios have extreme values (e.g. `AR_ratio` reaches about 93 million, see [03](03-outlier.md)), which would dominate Pearson. All three models also use only the ordering.
- **Across groups**: the built-in `preprocess collinearity` command only compares features within the same group (`FEATURE_GROUPS`), so it misses cross-group pairs such as `ebitda_ratio` and `ebitda_to_debt`.

### 2. VIF on WoE-transformed features

1. Fit the project's `WOEEncoder` (10 bins, the WoE logistic default) on train and transform the features into WoE values.
2. Compute the correlation matrix R of the WoE values; the VIF is the diagonal of R⁻¹ (the same formula as the built-in `calculate_vif`).
3. Drop candidate features and recompute to see how far the VIF falls.

The logistic model is fitted on WoE values, not raw ratios, so this is the space to check. Thresholds follow the usual rule of thumb: VIF > 10 is severe, 5–10 is moderate.

## Results

### Highly collinear pairs

| Pair | Groups | Spearman ρ | r after WoE | Reason |
|---|---|---|---|---|
| `cash_ratio` ~ `cash_to_debt` | liquidity / liquidity | +0.97 | +0.89 | Same numerator; the denominators are total assets and total liabilities, and total liabilities ≈ total assets × (1 − equity ratio) |
| `operating_cash_flow_ratio` ~ `operating_cash_flow_to_debt` | cash flow / cash flow | +0.96 | +0.93 | Same as above |
| `ebitda_ratio` ~ `ebitda_to_debt` | profitability / cash flow | +0.93 | +0.92 | Same as above, across groups |
| `wc_ratio` ~ `cur_debt_ratio` | liquidity / liquidity | +0.93 | < 0.8 | Both compare current assets with current liabilities (`wc_net` = `asst_current` − `debt_st`) |
| `financing_expense_ratio` ~ `financing_income_ratio` | cash flow / cash flow | −0.92 | −0.87 | `inc_financing` is negative in 82% of rows; it is in effect the net financial result, roughly minus the financing expense |
| `roa` ~ `ebitda_ratio` | profitability / profitability | +0.91 | < 0.8 | EBIT/TA vs EBITDA/TA; they differ only by depreciation |
| `cash_to_financing_exp_ratio` ~ `cash_to_debt` | cash flow / liquidity | +0.89 | < 0.8 | Across groups |
| `st_debt_share` ~ `lt_debt_to_capital` | leverage / leverage | −0.87 | < 0.8 | A high short-term share implies a low long-term share |

A further cluster of profitability features correlates at ρ 0.7–0.85: `roa`, `ebitda_ratio`, `tax_ratio`, `net_prof_margin`, `operating_cash_flow_ratio`, `int_coverage`.

### WoE VIF

| Feature | Before | After (adopted) |
|---|---|---|
| `ebitda_ratio` | **11.4** | (dropped) |
| `operating_cash_flow_to_debt` | **11.1** | (dropped) |
| `ebitda_to_debt` | **11.0** | 5.7 |
| `operating_cash_flow_ratio` | 9.7 | 3.9 |
| `cash_to_debt` | 7.8 | (dropped) |
| `cash_ratio` | 6.4 | < 3.5 |
| `financing_expense_ratio` | 5.0 | 4.9 |
| `financing_income_ratio` | 4.9 | 4.8 |
| Others | ≤ 4.3 | ≤ 4.2 |

After dropping `cash_to_debt`, `operating_cash_flow_to_debt` and `ebitda_ratio`, the highest VIF is 5.7.

## Decision

The following features are dropped for all three models (applied in `preprocess.py`, `MONOTONE_DIRECTIONS` in `lightgbm_model.py`, and `CORE_EBM_FEATURES` in `ebm_model.py`):

| Dropped | Kept | Reason |
|---|---|---|
| `cash_to_debt` | `cash_ratio` | Nearly identical to the total-assets version |
| `operating_cash_flow_to_debt` | `operating_cash_flow_ratio` | Same as above |
| `ebitda_ratio` | `ebitda_to_debt` | `ebitda_to_debt` has the cleanest monotone relation with default (lowest to highest quintile: 8.78% → 1.17%); `roa` already covers profit ÷ total assets |

The numeric features go from 30 to 27 (28 to 25 when excluding `ateco_sector_te` and the two lag features that are all NaN in round_01).

Pairs that remain with Spearman |ρ| ≥ 0.85: `wc_ratio` ~ `cur_debt_ratio` (+0.93), `financing_expense_ratio` ~ `financing_income_ratio` (−0.92), `st_debt_share` ~ `lt_debt_to_capital` (−0.87), `cash_ratio` ~ `cash_to_financing_exp_ratio` (+0.87). Their WoE VIFs are all below 5, so they are left as they are for now.

## Side finding: round_01 lag features are all NaN

The round_01 train set contains only FY2007, and `history.csv` also starts in 2007, so there are no earlier statements. As a result `annualized_change_in_roa` and `change_in_ar_turnover` are 100% NaN in the round_01 train set, and none of the three models can learn from them in that round (EBM drops all-missing columns automatically). They have values from round_02 onwards. This follows from where the data starts and is not a bug, but it should be noted when comparing results across rounds.

## Limitations

- Only round_01 was checked. The round_02 and round_03 train sets cover more years and include the lag features, so the numbers may differ; all three rounds should be run before a final decision.
- Categorical features and flags are not included.
- Spearman only looks at pairs; linear combinations of three or more features are reflected only in the WoE VIF.
- The check was run with an ad-hoc script that has not yet been merged into the `preprocess collinearity` command.

## Reproduction

```python
import numpy as np
import pandas as pd
from finance_project.data_processing.preprocess import _period_numeric_features
from finance_project.model_development.woe_logistic import WOEEncoder

d = pd.read_csv("features_train.csv")  # output of preprocess fit
num = [c for c in _period_numeric_features() if c in d and d[c].notna().any()] + ["ateco_sector_te"]

# Method 1: Spearman, all pairs
rho = d[num].corr(method="spearman")

# Method 2: WoE VIF
w = WOEEncoder(n_bins=10).fit(d[num], d["label"]).transform(d[num])
vif = pd.Series(np.diag(np.linalg.inv(w.corr().to_numpy())), index=num)
```
