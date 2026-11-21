# Final model SHAP summary (2026-11-21)
Model: unconstrained LightGBM, 210 trees, 75 features, fit on all FY2007-2011 rows. Exact TreeSHAP on all 835,016 training rows (log-odds; calibration is a constant shift). Base value -5.2714.

## Top 15 drivers

|                             |   mean_abs_shap |   share_pct |   rank |
|:----------------------------|----------------:|------------:|-------:|
| tangible_asset_ratio        |          0.3136 |     11.5422 |      1 |
| equity_ratio                |          0.2725 |     10.0303 |      2 |
| operating_cash_flow_to_debt |          0.1349 |      4.964  |      3 |
| AR_ratio                    |          0.1299 |      4.7804 |      4 |
| st_debt_share               |          0.1137 |      4.184  |      5 |
| ateco_sector_te             |          0.1083 |      3.9841 |      6 |
| cur_asst_ratio              |          0.1012 |      3.7247 |      7 |
| roa                         |          0.0977 |      3.5956 |      8 |
| profit                      |          0.0953 |      3.5069 |      9 |
| payables_ratio              |          0.0895 |      3.2924 |     10 |
| net_prof_margin             |          0.0839 |      3.0892 |     11 |
| operating_cash_flow_ratio   |          0.0789 |      2.9047 |     12 |
| cash_ratio                  |          0.0694 |      2.5549 |     13 |
| int_coverage                |          0.0625 |      2.3007 |     14 |
| size                        |          0.0605 |      2.2271 |     15 |

## Direction vs expected sign (features with a candidate sign)

| feature                     |   rank |   mean_abs_shap |   expected_sign |   value_vs_shap |   agrees |
|:----------------------------|-------:|----------------:|----------------:|----------------:|---------:|
| tangible_asset_ratio        |      1 |         0.31361 |              -1 |          -0.936 |        1 |
| equity_ratio                |      2 |         0.27253 |              -1 |          -0.95  |        1 |
| operating_cash_flow_to_debt |      3 |         0.13487 |              -1 |          -0.921 |        1 |
| AR_ratio                    |      4 |         0.12989 |               1 |           0.869 |        1 |
| st_debt_share               |      5 |         0.11368 |               1 |           0.825 |        1 |
| ateco_sector_te             |      6 |         0.10825 |               1 |           0.813 |        1 |
| roa                         |      8 |         0.0977  |              -1 |          -0.436 |        1 |
| payables_ratio              |     10 |         0.08946 |               1 |           0.815 |        1 |
| net_prof_margin             |     11 |         0.08394 |              -1 |          -0.836 |        1 |
| operating_cash_flow_ratio   |     12 |         0.07892 |              -1 |           0.014 |        0 |
| cash_ratio                  |     13 |         0.06942 |              -1 |          -0.897 |        1 |
| int_coverage                |     14 |         0.06251 |              -1 |          -0.642 |        1 |
| size                        |     15 |         0.06051 |              -1 |          -0.882 |        1 |
| financing_expense_ratio     |     16 |         0.05685 |               1 |           0.849 |        1 |
| quick_ratio                 |     18 |         0.05159 |              -1 |           0.379 |        0 |
| goodwill_ratio              |     19 |         0.04837 |               1 |          -0.549 |        0 |
| cur_debt_ratio              |     20 |         0.04507 |              -1 |          -0.851 |        1 |
| op_margin                   |     22 |         0.04315 |              -1 |          -0.851 |        1 |
| wc_ratio                    |     25 |         0.03891 |              -1 |          -0.53  |        1 |
| ebitda_to_debt              |     26 |         0.03865 |              -1 |          -0.87  |        1 |
| cash_interest_coverage      |     27 |         0.03611 |              -1 |          -0.607 |        1 |
| operating_cash_flow_margin  |     33 |         0.02444 |              -1 |          -0.386 |        1 |
| cash_to_financing_exp_ratio |     36 |         0.02092 |              -1 |          -0.866 |        1 |
| lt_debt_to_capital          |     38 |         0.01713 |               1 |          -0.133 |        0 |
| debt_detail_missing         |     40 |         0.01268 |               1 |           0.051 |        1 |
| loss_flag                   |     47 |         0.0037  |               1 |           0.008 |        1 |
| neg_bs_flag                 |     65 |         0       |               1 |           0     |        0 |

Features whose SHAP direction is against the expected sign: ['operating_cash_flow_ratio', 'quick_ratio', 'goodwill_ratio', 'lt_debt_to_capital', 'neg_bs_flag']

## Families of near-twin features (summed SHAP vs anchor ratio)

| family        | anchor                    |   spearman_anchor_vs_family_shap |   mean_abs_family_shap |
|:--------------|:--------------------------|---------------------------------:|-----------------------:|
| cash_flow     | operating_cash_flow_ratio |                           -0.839 |                 0.2149 |
| profitability | roa                       |                           -0.752 |                 0.2298 |
| liquidity     | cur_debt_ratio            |                           -0.502 |                 0.1219 |

Per-firm reason codes: `2026-11-21_shap_firm_examples.csv`. Chart: `2026-11-21_shap_bar.png`.
