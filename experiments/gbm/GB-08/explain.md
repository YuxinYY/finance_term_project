# GB-08 explainability (fold-3 model: trained on fs_year 2007-2009, explained on 2010 test rows)

SHAP values are TreeSHAP contributions on the raw log-odds scale, computed by LightGBM (`pred_contrib=True`) on a random 40,000-row sample of 2010 test rows. The own calibration is a monotone map of the log-odds (Platt), so the ranking of drivers is the same on the PD scale. Shapley-value attribution properties: R-23 (Lundberg & Lee 2017, Sec. 3). The tree algorithm used to compute them is not in R-23 [no VERIFIED source for TreeSHAP yet].

Correlated features (e.g. roa_net / cfo_to_debt / roa_oper / ebitda_margin; equity_ratio / neg_equity) split their attribution between them, so read drivers by FAMILY (FEATURE_SPEC note).

## 1. Global importance and shape vs expected sign

`shap_spearman`: rank correlation between the feature value and its SHAP value (sign = direction the model uses). `pd_trend`: partial dependence (average log-odds, 3,000 rows) from the 2nd to the 98th percentile of the feature. `pd_violation_share`: share of the partial-dependence movement that goes AGAINST the expected sign (0 = perfectly monotone in the expected direction). Features with expected sign 'none' are unconstrained by design.

| feature            |   mean_abs_shap | expected           |   shap_spearman | pd_trend                                                                                                                        |   pd_violation_share | verdict        |
|:-------------------|----------------:|:-------------------|----------------:|:--------------------------------------------------------------------------------------------------------------------------------|---------------------:|:---------------|
| tangible_ratio     |           0.443 | -                  |          -0.971 | -4.24 -> -5.83 (range 1.59)                                                                                                     |                0.000 | agrees         |
| roa_net            |           0.262 | -                  |          -0.959 | -4.19 -> -5.64 (range 1.45)                                                                                                     |                0.000 | agrees         |
| equity_ratio       |           0.230 | -                  |          -0.894 | -4.79 -> -5.50 (range 0.71)                                                                                                     |                0.000 | agrees         |
| st_debt_ratio      |           0.180 | +                  |           0.974 | -5.54 -> -4.93 (range 0.60)                                                                                                     |                0.000 | agrees         |
| ar_days            |           0.160 | +                  |           0.975 | -5.50 -> -4.62 (range 0.88)                                                                                                     |                0.000 | agrees         |
| ap_days            |           0.149 | +                  |           0.936 | -5.48 -> -4.97 (range 0.51)                                                                                                     |                0.000 | agrees         |
| asset_turnover     |           0.133 | none               |           0.663 | -5.45 -> -4.87 (range 0.59)                                                                                                     |              nan     | no expectation |
| ateco_group        |           0.119 | none (categorical) |         nan     | levels: I -0.32, OTHER -0.26, A -0.25, L -0.09, J -0.06, M -0.06, F -0.03, H -0.01, R +0.00, G +0.00, E +0.02, N +0.10, C +0.23 |              nan     | n/a            |
| extraord_to_assets |           0.108 | none               |          -0.242 | -4.96 -> -4.97 (range 0.49)                                                                                                     |              nan     | no expectation |
| cfo_to_debt        |           0.097 | -                  |          -0.696 | -5.09 -> -4.96 (range 0.42)                                                                                                     |                0.580 | CONTRADICTS    |
| interest_to_assets |           0.091 | +                  |           0.817 | -5.37 -> -4.99 (range 0.37)                                                                                                     |                0.000 | agrees         |
| cash_ratio         |           0.085 | -                  |          -0.961 | -5.18 -> -5.44 (range 0.27)                                                                                                     |                0.000 | agrees         |
| interest_coverage  |           0.081 | -                  |          -0.301 | -5.15 -> -5.07 (range 0.29)                                                                                                     |                0.561 | CONTRADICTS    |
| roa_oper           |           0.067 | -                  |          -0.949 | -4.98 -> -5.39 (range 0.41)                                                                                                     |                0.000 | agrees         |
| log_assets         |           0.057 | -                  |          -0.939 | -5.15 -> -5.42 (range 0.26)                                                                                                     |                0.000 | agrees         |
| ebitda_margin      |           0.053 | -                  |          -0.756 | -5.26 -> -5.32 (range 0.19)                                                                                                     |                0.442 | mostly agrees  |
| intangible_ratio   |           0.049 | +                  |           0.608 | -5.34 -> -5.16 (range 0.18)                                                                                                     |                0.000 | agrees         |
| st_debt_share      |           0.048 | +                  |           0.894 | -5.51 -> -5.26 (range 0.25)                                                                                                     |                0.000 | agrees         |
| legal_group        |           0.047 | none (categorical) |         nan     | levels: SAU -0.35, SPA -0.14, OTHER -0.02, SRL +0.01, SRU +0.11                                                                 |              nan     | n/a            |
| bank_debt_ratio    |           0.019 | +                  |           0.942 | -5.35 -> -5.28 (range 0.07)                                                                                                     |                0.000 | agrees         |
| macro_area         |           0.015 | none (categorical) |         nan     | levels: North-East -0.02, Centre -0.01, North-West +0.01, South +0.02, OTHER +0.02, Islands +0.04                               |              nan     | n/a            |
| current_ratio      |           0.010 | -                  |          -0.771 | -5.29 -> -5.32 (range 0.03)                                                                                                     |                0.000 | agrees         |
| wc_ratio           |           0.005 | -                  |          -0.505 | -5.30 -> -5.31 (range 0.01)                                                                                                     |                0.000 | agrees         |
| neg_equity         |           0.000 | +                  |         nan     | -5.31 -> -5.31 (range 0.00)                                                                                                     |                0.000 | agrees         |
| debt_split_missing |           0.000 | +                  |         nan     | -5.31 -> -5.31 (range 0.00)                                                                                                     |                0.000 | agrees         |
| rev_nonpositive    |           0.000 | none               |         nan     | -5.31 -> -5.31 (range 0.00)                                                                                                     |              nan     | no expectation |

Family importance (sum of mean |SHAP|):

|                   |   mean_abs_shap |
|:------------------|----------------:|
| asset structure   |           0.493 |
| profitability     |           0.490 |
| leverage          |           0.476 |
| activity          |           0.441 |
| coverage          |           0.270 |
| industry          |           0.119 |
| liquidity         |           0.100 |
| size              |           0.057 |
| firm type         |           0.047 |
| geography         |           0.015 |
| missing indicator |           0.000 |

Dependence plots for the top-8 features: `dep_<feature>.png` in this folder (left: SHAP dependence scatter; right: partial dependence).

## 2. Reason codes for three example 2010 applicants

Top-3 factors pushing the PD UP (largest positive SHAP) and the 2 strongest factors pulling it DOWN, in credit-officer wording. Percentiles are within the 40,000 sampled 2010 applicants. Values are on the natural scale (signed-log features back-transformed; winsorised at the training 1st/99th percentiles, so e.g. days outstanding are capped).

### low PD (5th pct): row_id 33802206512010, PD 0.07%, industry section H, observed default in window: no

Raises PD:
1. sales-to-assets in a riskier range: sales / assets = 0.829 (percentile 63 of 2010 applicants; SHAP +0.15 log-odds)
2. region (macro-area) = South (raises PD, SHAP +0.02)
3. legal form = SRL (raises PD, SHAP +0.01)

Lowers PD:
- favourable tangible fixed assets / assets: tangible fixed assets / assets = 0.504 (percentile 74 of 2010 applicants; SHAP -0.56 log-odds)
- favourable net profit / assets: net profit / assets = 0.0306 (percentile 82 of 2010 applicants; SHAP -0.38 log-odds)

### middle PD (median): row_id 27775606532010, PD 0.50%, industry section C, observed default in window: no

Raises PD:
1. industry (ATECO section) = C (raises PD, SHAP +0.27)
2. slow collection of receivables: days receivables outstanding = 398 (percentile 79 of 2010 applicants; SHAP +0.21 log-odds)
3. stretched supplier payments: days payables outstanding = 148 (percentile 63 of 2010 applicants; SHAP +0.20 log-odds)

Lowers PD:
- favourable tangible fixed assets / assets: tangible fixed assets / assets = 0.321 (percentile 63 of 2010 applicants; SHAP -0.34 log-odds)
- favourable equity / assets: equity / assets = 0.3 (percentile 62 of 2010 applicants; SHAP -0.25 log-odds)

### high PD (99th pct): row_id 19222605402010, PD 13.44%, industry section C, observed default in window: no

Raises PD:
1. losses / low net profitability: net profit / assets = -0.0683 (percentile 4 of 2010 applicants; SHAP +1.12 log-odds)
2. thin equity cushion: equity / assets = -0.0389 (percentile 2 of 2010 applicants; SHAP +0.57 log-odds)
3. industry (ATECO section) = C (raises PD, SHAP +0.39)

Lowers PD:
- favourable extraordinary items / assets: extraordinary items / assets = 9.73e-07 (percentile 67 of 2010 applicants; SHAP -0.14 log-odds)
- favourable cash / assets: cash / assets = 0.0698 (percentile 78 of 2010 applicants; SHAP -0.13 log-odds)

## 3. Harness-relevant cost

- Fold-3 booster: 590 trees, max 8 leaves (depth <= 3); 26 features.
- Prediction for 100,000 rows (feature transform + booster + calibration), single process, 8 threads, this laptop: 5.7 s total, of which feature transform 2.5 s.
- Complexity: O(n x (F + T x D)) for n rows, F features, T trees, depth D; linear in n, no dependence on other rows (each row is scored from its own record and frozen params).

