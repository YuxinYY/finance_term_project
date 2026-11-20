# GB-08b explainability (fold-3 model: trained on fs_year 2007-2009, explained on 2010 test rows)

SHAP values are TreeSHAP contributions on the raw log-odds scale, computed by LightGBM (`pred_contrib=True`) on a random 40,000-row sample of 2010 test rows. The own calibration is a monotone map of the log-odds (Platt), so the ranking of drivers is the same on the PD scale. Shapley-value attribution properties: R-23 (Lundberg & Lee 2017, Sec. 3). The tree algorithm used to compute them is not in R-23 [no VERIFIED source for TreeSHAP yet].

Correlated features (e.g. roa_net / cfo_to_debt / roa_oper / ebitda_margin; equity_ratio / neg_equity) split their attribution between them, so read drivers by FAMILY (FEATURE_SPEC note).

## 1. Global importance and shape vs expected sign

`shap_spearman`: rank correlation between the feature value and its SHAP value (sign = direction the model uses). `pd_trend`: partial dependence (average log-odds, 3,000 rows) from the 2nd to the 98th percentile of the feature. `pd_violation_share`: share of the partial-dependence movement that goes AGAINST the expected sign (0 = perfectly monotone in the expected direction). Features with expected sign 'none' are unconstrained by design.

| feature            |   mean_abs_shap | expected           |   shap_spearman | pd_trend                                                                                                                        |   pd_violation_share | verdict        |
|:-------------------|----------------:|:-------------------|----------------:|:--------------------------------------------------------------------------------------------------------------------------------|---------------------:|:---------------|
| tangible_ratio     |           0.433 | -                  |          -0.973 | -4.36 -> -5.80 (range 1.43)                                                                                                     |                0.000 | agrees         |
| roa_net            |           0.223 | -                  |          -0.945 | -4.23 -> -5.56 (range 1.33)                                                                                                     |                0.000 | agrees         |
| equity_ratio       |           0.216 | -                  |          -0.868 | -4.85 -> -5.46 (range 0.61)                                                                                                     |                0.000 | agrees         |
| st_debt_ratio      |           0.184 | +                  |           0.956 | -5.49 -> -4.92 (range 0.57)                                                                                                     |                0.000 | agrees         |
| ar_days            |           0.165 | +                  |           0.969 | -5.47 -> -4.55 (range 0.92)                                                                                                     |                0.000 | agrees         |
| asset_turnover     |           0.159 | none               |           0.673 | -5.46 -> -4.72 (range 0.75)                                                                                                     |              nan     | no expectation |
| ap_days            |           0.148 | +                  |           0.946 | -5.46 -> -4.94 (range 0.52)                                                                                                     |                0.000 | agrees         |
| ateco_group        |           0.124 | none (categorical) |         nan     | levels: I -0.33, OTHER -0.28, A -0.25, L -0.09, J -0.06, F -0.04, M -0.04, R +0.00, E +0.01, H +0.01, G +0.02, N +0.09, C +0.25 |              nan     | n/a            |
| interest_coverage  |           0.115 | -                  |          -0.815 | -5.07 -> -5.36 (range 0.29)                                                                                                     |                0.000 | agrees         |
| cfo_to_debt        |           0.107 | -                  |          -0.923 | -5.01 -> -5.37 (range 0.36)                                                                                                     |                0.000 | agrees         |
| extraord_to_assets |           0.106 | none               |          -0.230 | -4.95 -> -4.95 (range 0.47)                                                                                                     |              nan     | no expectation |
| cash_ratio         |           0.079 | -                  |          -0.958 | -5.16 -> -5.41 (range 0.25)                                                                                                     |                0.000 | agrees         |
| roa_oper           |           0.077 | -                  |          -0.903 | -4.94 -> -5.37 (range 0.42)                                                                                                     |                0.000 | agrees         |
| interest_to_assets |           0.070 | +                  |           0.718 | -5.31 -> -5.04 (range 0.28)                                                                                                     |                0.000 | agrees         |
| ebitda_margin      |           0.050 | -                  |           0.100 | -5.34 -> -5.05 (range 0.36)                                                                                                     |                0.703 | CONTRADICTS    |
| legal_group        |           0.049 | none (categorical) |         nan     | levels: SAU -0.36, SPA -0.14, OTHER -0.01, SRL +0.01, SRU +0.11                                                                 |              nan     | n/a            |
| intangible_ratio   |           0.045 | +                  |           0.618 | -5.31 -> -5.14 (range 0.18)                                                                                                     |                0.000 | agrees         |
| st_debt_share      |           0.042 | +                  |           0.907 | -5.48 -> -5.24 (range 0.24)                                                                                                     |                0.000 | agrees         |
| log_assets         |           0.031 | -                  |          -0.952 | -5.15 -> -5.35 (range 0.20)                                                                                                     |                0.000 | agrees         |
| macro_area         |           0.015 | none (categorical) |         nan     | levels: North-East -0.03, Centre +0.00, North-West +0.01, South +0.01, OTHER +0.02, Islands +0.03                               |              nan     | n/a            |
| current_ratio      |           0.012 | -                  |          -0.779 | -5.26 -> -5.29 (range 0.03)                                                                                                     |                0.000 | agrees         |
| bank_debt_ratio    |           0.011 | +                  |           0.947 | -5.30 -> -5.26 (range 0.04)                                                                                                     |                0.000 | agrees         |
| wc_ratio           |           0.003 | -                  |          -0.472 | -5.27 -> -5.28 (range 0.01)                                                                                                     |                0.000 | agrees         |
| rev_nonpositive    |           0.000 | none               |          -0.730 | -5.28 -> -5.28 (range 0.00)                                                                                                     |              nan     | no expectation |
| neg_equity         |           0.000 | +                  |         nan     | -5.28 -> -5.28 (range 0.00)                                                                                                     |                0.000 | agrees         |
| debt_split_missing |           0.000 | +                  |         nan     | -5.28 -> -5.28 (range 0.00)                                                                                                     |                0.000 | agrees         |

Family importance (sum of mean |SHAP|):

|                   |   mean_abs_shap |
|:------------------|----------------:|
| asset structure   |           0.478 |
| activity          |           0.471 |
| profitability     |           0.456 |
| leverage          |           0.453 |
| coverage          |           0.291 |
| industry          |           0.124 |
| liquidity         |           0.094 |
| firm type         |           0.049 |
| size              |           0.031 |
| geography         |           0.015 |
| missing indicator |           0.000 |

Dependence plots for the top-8 features: `dep_<feature>.png` in this folder (left: SHAP dependence scatter; right: partial dependence).

## 2. Reason codes for three example 2010 applicants

Top-3 factors pushing the PD UP (largest positive SHAP) and the 2 strongest factors pulling it DOWN, in credit-officer wording. Percentiles are within the 40,000 sampled 2010 applicants. Values are on the natural scale (signed-log features back-transformed; winsorised at the training 1st/99th percentiles, so e.g. days outstanding are capped).

### low PD (5th pct): row_id 8912301532010, PD 0.08%, industry section H, observed default in window: no

Raises PD:
1. high share of intangibles: intangible assets / assets = 0.123 (percentile 94 of 2010 applicants; SHAP +0.17 log-odds)
2. little cash: cash / assets = 0.00214 (percentile 28 of 2010 applicants; SHAP +0.07 log-odds)
3. sales-to-assets in a riskier range: sales / assets = 1.41 (percentile 84 of 2010 applicants; SHAP +0.05 log-odds)

Lowers PD:
- favourable size (total assets): size (total assets) = 18.4 (percentile 100 of 2010 applicants; SHAP -0.38 log-odds)
- favourable tangible fixed assets / assets: tangible fixed assets / assets = 0.251 (percentile 58 of 2010 applicants; SHAP -0.35 log-odds)

### middle PD (median): row_id 19376006072010, PD 0.51%, industry section F, observed default in window: no

Raises PD:
1. little tangible collateral: tangible fixed assets / assets = 0 (percentile 10 of 2010 applicants; SHAP +0.67 log-odds)
2. slow collection of receivables: days receivables outstanding = 6e+04 (percentile 100 of 2010 applicants; SHAP +0.47 log-odds)
3. EBITDA negative or barely covers interest: EBITDA / interest expense = -35.8 (percentile 3 of 2010 applicants; SHAP +0.13 log-odds)

Lowers PD:
- favourable sales / assets: sales / assets = 0 (percentile 4 of 2010 applicants; SHAP -0.54 log-odds)
- favourable short-term liabilities / assets: short-term liabilities / assets = 0.00681 (percentile 3 of 2010 applicants; SHAP -0.26 log-odds)

### high PD (99th pct): row_id 27402101622010, PD 13.30%, industry section C, observed default in window: no

Raises PD:
1. losses / low net profitability: net profit / assets = -0.0447 (percentile 7 of 2010 applicants; SHAP +0.55 log-odds)
2. thin equity cushion: equity / assets = -0.0114 (percentile 2 of 2010 applicants; SHAP +0.47 log-odds)
3. industry (ATECO section) = C (raises PD, SHAP +0.43)

Lowers PD:
- favourable EBITDA margin: EBITDA margin = -0.114 (percentile 7 of 2010 applicants; SHAP -0.10 log-odds)
- favourable tangible fixed assets / assets: tangible fixed assets / assets = 0.124 (percentile 46 of 2010 applicants; SHAP -0.07 log-odds)

## 3. Harness-relevant cost

- Fold-3 booster: 542 trees, max 8 leaves (depth <= 3); 26 features.
- Prediction for 100,000 rows (feature transform + booster + calibration), single process, 8 threads, this laptop: 3.1 s total, of which feature transform 1.7 s.
- Complexity: O(n x (F + T x D)) for n rows, F features, T trees, depth D; linear in n, no dependence on other rows (each row is scored from its own record and frozen params).

