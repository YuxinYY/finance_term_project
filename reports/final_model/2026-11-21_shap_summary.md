# Final model SHAP summary (2026-11-21)
Model: unconstrained LightGBM, 210 trees, 75 features, fit on all FY2007-2011 rows. Exact TreeSHAP on 50,000 training rows (log-odds; calibration is a constant shift). Base value -5.2714.

## Top 15 drivers

|                             |   mean_abs_shap |   share_pct |   rank |
|:----------------------------|----------------:|------------:|-------:|
| tangible_asset_ratio        |          0.3128 |     11.5005 |      1 |
| equity_ratio                |          0.2732 |     10.045  |      2 |
| operating_cash_flow_to_debt |          0.1353 |      4.9744 |      3 |
| AR_ratio                    |          0.13   |      4.7812 |      4 |
| st_debt_share               |          0.1138 |      4.1826 |      5 |
| ateco_sector_te             |          0.1081 |      3.9762 |      6 |
| cur_asst_ratio              |          0.1009 |      3.7106 |      7 |
| roa                         |          0.0982 |      3.6109 |      8 |
| profit                      |          0.0956 |      3.5159 |      9 |
| payables_ratio              |          0.0896 |      3.2929 |     10 |
| net_prof_margin             |          0.0841 |      3.092  |     11 |
| operating_cash_flow_ratio   |          0.0792 |      2.9127 |     12 |
| cash_ratio                  |          0.0694 |      2.5529 |     13 |
| int_coverage                |          0.0627 |      2.3048 |     14 |
| size                        |          0.0603 |      2.2175 |     15 |

## Direction vs expected sign (features with a candidate sign)

| feature                     |   rank |   mean_abs_shap |   expected_sign |   value_vs_shap |   agrees |
|:----------------------------|-------:|----------------:|----------------:|----------------:|---------:|
| tangible_asset_ratio        |      1 |         0.31278 |              -1 |          -0.935 |        1 |
| equity_ratio                |      2 |         0.2732  |              -1 |          -0.95  |        1 |
| operating_cash_flow_to_debt |      3 |         0.13529 |              -1 |          -0.92  |        1 |
| AR_ratio                    |      4 |         0.13004 |               1 |           0.87  |        1 |
| st_debt_share               |      5 |         0.11376 |               1 |           0.823 |        1 |
| ateco_sector_te             |      6 |         0.10814 |               1 |           0.81  |        1 |
| roa                         |      8 |         0.09821 |              -1 |          -0.446 |        1 |
| payables_ratio              |     10 |         0.08956 |               1 |           0.816 |        1 |
| net_prof_margin             |     11 |         0.08409 |              -1 |          -0.836 |        1 |
| operating_cash_flow_ratio   |     12 |         0.07922 |              -1 |           0.009 |        0 |
| cash_ratio                  |     13 |         0.06943 |              -1 |          -0.897 |        1 |
| int_coverage                |     14 |         0.06268 |              -1 |          -0.641 |        1 |
| size                        |     15 |         0.06031 |              -1 |          -0.881 |        1 |
| financing_expense_ratio     |     16 |         0.05659 |               1 |           0.847 |        1 |
| quick_ratio                 |     18 |         0.05172 |              -1 |           0.383 |        0 |
| goodwill_ratio              |     19 |         0.0486  |               1 |          -0.545 |        0 |
| cur_debt_ratio              |     20 |         0.04503 |              -1 |          -0.85  |        1 |
| op_margin                   |     22 |         0.04314 |              -1 |          -0.853 |        1 |
| wc_ratio                    |     25 |         0.03895 |              -1 |          -0.529 |        1 |
| ebitda_to_debt              |     26 |         0.03872 |              -1 |          -0.869 |        1 |
| cash_interest_coverage      |     27 |         0.03614 |              -1 |          -0.604 |        1 |
| operating_cash_flow_margin  |     33 |         0.0244  |              -1 |          -0.387 |        1 |
| cash_to_financing_exp_ratio |     36 |         0.02092 |              -1 |          -0.867 |        1 |
| lt_debt_to_capital          |     38 |         0.01726 |               1 |          -0.127 |        0 |
| debt_detail_missing         |     40 |         0.01268 |               1 |           0.051 |        1 |
| loss_flag                   |     47 |         0.00372 |               1 |           0.008 |        1 |
| neg_bs_flag                 |     65 |         0       |               1 |           0     |        0 |

Features whose SHAP direction is against the expected sign: ['operating_cash_flow_ratio', 'quick_ratio', 'goodwill_ratio', 'lt_debt_to_capital', 'neg_bs_flag']

## Families of near-twin features (summed SHAP vs anchor ratio)

| family        | anchor                    |   spearman_anchor_vs_family_shap |   mean_abs_family_shap |
|:--------------|:--------------------------|---------------------------------:|-----------------------:|
| cash_flow     | operating_cash_flow_ratio |                           -0.84  |                 0.2157 |
| profitability | roa                       |                           -0.757 |                 0.2305 |
| liquidity     | cur_debt_ratio            |                           -0.501 |                 0.1219 |

Per-firm reason codes: `2026-11-21_shap_firm_examples.csv`. Chart: `2026-11-21_shap_bar.png`.
