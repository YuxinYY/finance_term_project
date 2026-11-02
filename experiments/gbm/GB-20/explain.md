# GB-20 explainability (fold-3 model, trained on 2007-2009)

Generated from `experiments/gbm/GB-22/*.csv` (`explain_r2b.py`). TreeSHAP (R-41) on the log-odds scale, 20,000 sampled 2010 rows; nothing is fitted on them. Plots: `GB-22/dep_*.png`. Decisions: `decisions_r2b.md`.

## Monotone check
Constrained features: 35. Violations: 0. With 0 splits: ['rev_nonpositive', 'neg_equity', 'n_core_imputed', 'debt_split_missing'].

## Top 15 drivers
| feature                    |   mean_abs_shap |   share | new_in_full2   |
|:---------------------------|----------------:|--------:|:---------------|
| tangible_ratio             |          0.3673 |  0.1240 | False          |
| asset_turnover             |          0.1680 |  0.0567 | False          |
| st_debt_ratio              |          0.1484 |  0.0501 | False          |
| ar_days                    |          0.1435 |  0.0484 | False          |
| interest_coverage_sect_pct |          0.1433 |  0.0484 | True           |
| equity_ratio_sect_pct      |          0.1384 |  0.0467 | True           |
| tfr_ratio                  |          0.1294 |  0.0437 | True           |
| roa_net                    |          0.1278 |  0.0431 | False          |
| payables_to_sales          |          0.1136 |  0.0383 | True           |
| roa_net_sect_pct           |          0.1080 |  0.0364 | True           |
| extraord_to_assets         |          0.1013 |  0.0342 | False          |
| province_te                |          0.0974 |  0.0329 | True           |
| equity_ratio               |          0.0911 |  0.0307 | False          |
| ateco_group                |          0.0904 |  0.0305 | False          |
| ateco_div_te               |          0.0806 |  0.0272 | True           |

SHAP splits credit among correlated features (D-122): report families, and do not read a high share as value.

## New features: unconstrained ones and the encodings
| feature           | expected   | constrained   |   mean_abs_shap |   rank |   pd_range |   n_splits |
|:------------------|:-----------|:--------------|----------------:|-------:|-----------:|-----------:|
| tfr_ratio         | none       | False         |           0.129 |      7 |      0.346 |        289 |
| province_te       | +          | True          |           0.097 |     12 |      0.394 |        107 |
| ateco_div_te      | +          | True          |           0.081 |     15 |      0.559 |        114 |
| interest_to_sales | none       | False         |           0.075 |     16 |      0.338 |        297 |
| region_te         | +          | True          |           0.054 |     20 |      0.426 |        103 |
| bs_gap            | none       | False         |           0.048 |     24 |      0.140 |        226 |
| fin_fixed_ratio   | none       | False         |           0.018 |     33 |      0.100 |        119 |

- **tfr_ratio**: PD contribution falls as the staff-severance fund grows, then rises slightly above about 7% of assets. Read as "a large fund for a given wage bill goes with lower risk, other things equal" [hypothesis, D-123].
- **interest_to_sales**: humped, and falling at the top where raw default rates stay high. Hardest shape to defend (D-125).
- **bs_gap, fin_fixed_ratio**: small effects, no firm economic reading [hypothesis].
- **Encodings**: rising by construction (constraint +), so the shape is not evidence (D-124).

## Reason codes
Top-3 positive contributions. 42% of firms repeat a family in the raw top 3, so a credit officer should see the family-merged view. 87% include a new feature.

| firm               |        pd | view          |   rank | driver             | plain_words                                                                              |      value |   shap_logodds |
|:-------------------|----------:|:--------------|-------:|:-------------------|:-----------------------------------------------------------------------------------------|-----------:|---------------:|
| low PD (5th pct)   | 0.0007467 | raw feature   |      1 | payables_to_sales  | large trade payables relative to sales                                                   |   0.5575   |        0.1552  |
| low PD (5th pct)   | 0.0007467 | raw feature   |      2 | ap_days            | the firm takes long to pay suppliers                                                     |   6.001    |        0.1403  |
| low PD (5th pct)   | 0.0007467 | raw feature   |      3 | tfr_ratio          | little staff-severance fund: few or short-tenured staff, or a holding/property structure |   0.004946 |        0.09766 |
| low PD (5th pct)   | 0.0007467 | family-merged |      1 | liquidity          | liquidity family, sum of its features                                                    | nan        |        0.07301 |
| low PD (5th pct)   | 0.0007467 | family-merged |      2 | geography          | geography family, sum of its features                                                    | nan        |        0.06325 |
| low PD (5th pct)   | 0.0007467 | family-merged |      3 | activity           | activity family, sum of its features                                                     | nan        |        0.04456 |
| median PD          | 0.004877  | raw feature   |      1 | tangible_ratio     | few fixed tangible assets to back the debt                                               |   0.009499 |        0.6073  |
| median PD          | 0.004877  | raw feature   |      2 | province_te        | province with a historically high default rate                                           |   0.2338   |        0.1753  |
| median PD          | 0.004877  | raw feature   |      3 | asset_turnover     | sales relative to assets                                                                 |   1.771    |        0.1279  |
| median PD          | 0.004877  | family-merged |      1 | asset structure    | asset structure family, sum of its features                                              | nan        |        0.5001  |
| median PD          | 0.004877  | family-merged |      2 | geography          | geography family, sum of its features                                                    | nan        |        0.2038  |
| median PD          | 0.004877  | family-merged |      3 | data quality       | data quality family, sum of its features                                                 | nan        |        0.1102  |
| high PD (99th pct) | 0.1425    | raw feature   |      1 | roa_net            | weak net profit on assets                                                                |  -0.1412   |        0.8086  |
| high PD (99th pct) | 0.1425    | raw feature   |      2 | roa_net_sect_pct   | net profitability low against sector peers                                               |   0.01991  |        0.3264  |
| high PD (99th pct) | 0.1425    | raw feature   |      3 | extraord_to_assets | extraordinary items (one-off gains or losses)                                            |  -0.01315  |        0.3179  |
| high PD (99th pct) | 0.1425    | family-merged |      1 | profitability      | profitability family, sum of its features                                                | nan        |        1.652   |
| high PD (99th pct) | 0.1425    | family-merged |      2 | asset structure    | asset structure family, sum of its features                                              | nan        |        0.5773  |
| high PD (99th pct) | 0.1425    | family-merged |      3 | leverage           | leverage family, sum of its features                                                     | nan        |        0.4707  |
