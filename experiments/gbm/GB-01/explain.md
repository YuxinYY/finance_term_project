# GB-01 explainability (fold-3 model: trained on fs_year 2007-2009, explained on 2010 test rows)

SHAP values are TreeSHAP contributions on the raw log-odds scale, computed by LightGBM (`pred_contrib=True`) on a random 40,000-row sample of 2010 test rows. The own calibration is a monotone map of the log-odds (Platt), so the ranking of drivers is the same on the PD scale. Shapley-value attribution properties: R-23 (Lundberg & Lee 2017, Sec. 3). The tree algorithm used to compute them is not in R-23 [no VERIFIED source for TreeSHAP yet].

Correlated features (e.g. roa_net / cfo_to_debt / roa_oper / ebitda_margin; equity_ratio / neg_equity) split their attribution between them, so read drivers by FAMILY (FEATURE_SPEC note).

## 1. Global importance and shape vs expected sign

`shap_spearman`: rank correlation between the feature value and its SHAP value (sign = direction the model uses). `pd_trend`: partial dependence (average log-odds, 3,000 rows) from the 2nd to the 98th percentile of the feature. `pd_violation_share`: share of the partial-dependence movement that goes AGAINST the expected sign (0 = perfectly monotone in the expected direction). Features with expected sign 'none' are unconstrained by design.

| feature            |   mean_abs_shap | expected           |   shap_spearman | pd_trend                                                                                                                        |   pd_violation_share | verdict        |
|:-------------------|----------------:|:-------------------|----------------:|:--------------------------------------------------------------------------------------------------------------------------------|---------------------:|:---------------|
| tangible_ratio     |           0.412 | -                  |          -0.973 | -4.40 -> -5.73 (range 1.33)                                                                                                     |                0.000 | agrees         |
| equity_ratio       |           0.213 | -                  |          -0.859 | -4.82 -> -5.41 (range 0.59)                                                                                                     |                0.000 | agrees         |
| roa_net            |           0.207 | -                  |          -0.937 | -4.22 -> -5.50 (range 1.28)                                                                                                     |                0.000 | agrees         |
| st_debt_ratio      |           0.183 | +                  |           0.956 | -5.45 -> -4.88 (range 0.57)                                                                                                     |                0.000 | agrees         |
| ap_days            |           0.147 | +                  |           0.923 | -5.42 -> -4.93 (range 0.48)                                                                                                     |                0.000 | agrees         |
| ar_days            |           0.142 | +                  |           0.960 | -5.40 -> -4.81 (range 0.58)                                                                                                     |                0.000 | agrees         |
| ateco_group        |           0.121 | none (categorical) |         nan     | levels: I -0.29, A -0.25, OTHER -0.22, L -0.13, M -0.05, F -0.04, J -0.02, R -0.00, E +0.02, H +0.03, G +0.03, N +0.09, C +0.23 |              nan     | n/a            |
| asset_turnover     |           0.112 | none               |           0.664 | -5.36 -> -4.87 (range 0.49)                                                                                                     |              nan     | no expectation |
| interest_coverage  |           0.110 | -                  |          -0.813 | -5.05 -> -5.32 (range 0.27)                                                                                                     |                0.000 | agrees         |
| cfo_to_debt        |           0.101 | -                  |          -0.899 | -4.98 -> -5.33 (range 0.34)                                                                                                     |                0.000 | agrees         |
| extraord_to_assets |           0.096 | none               |          -0.220 | -4.94 -> -4.93 (range 0.41)                                                                                                     |              nan     | no expectation |
| cash_ratio         |           0.070 | -                  |          -0.956 | -5.14 -> -5.36 (range 0.22)                                                                                                     |                0.000 | agrees         |
| interest_to_assets |           0.069 | +                  |           0.623 | -5.27 -> -5.00 (range 0.27)                                                                                                     |                0.000 | agrees         |
| roa_oper           |           0.060 | -                  |          -0.849 | -4.95 -> -5.29 (range 0.34)                                                                                                     |                0.000 | agrees         |
| ebitda_margin      |           0.042 | -                  |          -0.932 | -5.15 -> -5.29 (range 0.14)                                                                                                     |                0.000 | agrees         |
| st_debt_share      |           0.041 | +                  |           0.848 | -5.42 -> -5.21 (range 0.21)                                                                                                     |                0.000 | agrees         |
| legal_group        |           0.033 | none (categorical) |         nan     | levels: SAU -0.36, SPA -0.08, OTHER -0.00, SRL +0.02, SRU +0.03                                                                 |              nan     | n/a            |
| intangible_ratio   |           0.026 | +                  |           0.621 | -5.26 -> -5.17 (range 0.10)                                                                                                     |                0.000 | agrees         |
| log_assets         |           0.022 | -                  |          -0.942 | -5.15 -> -5.30 (range 0.15)                                                                                                     |                0.000 | agrees         |
| current_ratio      |           0.013 | -                  |          -0.802 | -5.22 -> -5.26 (range 0.03)                                                                                                     |                0.000 | agrees         |
| bank_debt_ratio    |           0.009 | +                  |           0.943 | -5.26 -> -5.23 (range 0.03)                                                                                                     |                0.000 | agrees         |
| macro_area         |           0.008 | none (categorical) |         nan     | levels: North-East -0.01, OTHER +0.00, Centre +0.00, North-West +0.00, South +0.01, Islands +0.01                               |              nan     | n/a            |
| wc_ratio           |           0.003 | -                  |          -0.452 | -5.24 -> -5.25 (range 0.01)                                                                                                     |                0.000 | agrees         |
| neg_equity         |           0.000 | +                  |         nan     | -5.24 -> -5.24 (range 0.00)                                                                                                     |                0.000 | agrees         |
| debt_split_missing |           0.000 | +                  |         nan     | -5.24 -> -5.24 (range 0.00)                                                                                                     |                0.000 | agrees         |
| rev_nonpositive    |           0.000 | none               |         nan     | -5.24 -> -5.24 (range 0.00)                                                                                                     |              nan     | no expectation |

Family importance (sum of mean |SHAP|):

|                   |   mean_abs_shap |
|:------------------|----------------:|
| leverage          |           0.447 |
| asset structure   |           0.438 |
| profitability     |           0.406 |
| activity          |           0.400 |
| coverage          |           0.280 |
| industry          |           0.121 |
| liquidity         |           0.086 |
| firm type         |           0.033 |
| size              |           0.022 |
| geography         |           0.008 |
| missing indicator |           0.000 |

Dependence plots for the top-8 features: `dep_<feature>.png` in this folder (left: SHAP dependence scatter; right: partial dependence).

## 2. Reason codes for three example 2010 applicants

Top-3 factors pushing the PD UP (largest positive SHAP) and the 2 strongest factors pulling it DOWN, in credit-officer wording. Percentiles are within the 40,000 sampled 2010 applicants. Values are on the natural scale (signed-log features back-transformed; winsorised at the training 1st/99th percentiles, so e.g. days outstanding are capped).

### low PD (5th pct): row_id 32359301572010, PD 0.09%, industry section C, observed default in window: no

Raises PD:
1. industry (ATECO section) = C (raises PD, SHAP +0.18)
2. sales-to-assets in a riskier range: sales / assets = 0.758 (percentile 60 of 2010 applicants; SHAP +0.09 log-odds)
3. debt concentrated in the short term: short-term share of debt = 1 (percentile 100 of 2010 applicants; SHAP +0.03 log-odds)

Lowers PD:
- favourable tangible fixed assets / assets: tangible fixed assets / assets = 0.276 (percentile 60 of 2010 applicants; SHAP -0.37 log-odds)
- favourable net profit / assets: net profit / assets = 0.182 (percentile 98 of 2010 applicants; SHAP -0.30 log-odds)

### middle PD (median): row_id 11938908762010, PD 0.52%, industry section OTHER, observed default in window: no

Raises PD:
1. slow collection of receivables: days receivables outstanding = 6e+04 (percentile 100 of 2010 applicants; SHAP +0.36 log-odds)
2. EBITDA negative or barely covers interest: EBITDA / interest expense = -7.09 (percentile 6 of 2010 applicants; SHAP +0.19 log-odds)
3. low cash flow relative to debt: cash flow / debt = -0.0135 (percentile 12 of 2010 applicants; SHAP +0.12 log-odds)

Lowers PD:
- favourable tangible fixed assets / assets: tangible fixed assets / assets = 0.643 (percentile 80 of 2010 applicants; SHAP -0.38 log-odds)
- favourable equity / assets: equity / assets = 0.421 (percentile 73 of 2010 applicants; SHAP -0.18 log-odds)

### high PD (99th pct): row_id 21819906112010, PD 13.85%, industry section G, observed default in window: no

Raises PD:
1. losses / low net profitability: net profit / assets = -0.0665 (percentile 5 of 2010 applicants; SHAP +0.93 log-odds)
2. thin equity cushion: equity / assets = -0.0104 (percentile 3 of 2010 applicants; SHAP +0.50 log-odds)
3. heavy short-term liabilities: short-term liabilities / assets = 0.887 (percentile 91 of 2010 applicants; SHAP +0.41 log-odds)

Lowers PD:
- favourable EBITDA margin: EBITDA margin = 0.162 (percentile 67 of 2010 applicants; SHAP -0.04 log-odds)
- favourable operating profit / assets: operating profit / assets = 0.0349 (percentile 66 of 2010 applicants; SHAP -0.04 log-odds)

## 3. Harness-relevant cost

- Fold-3 booster: 386 trees, max 8 leaves (depth <= 3); 26 features.
- Prediction for 100,000 rows (feature transform + booster + calibration), single process, 8 threads, this laptop: 1.0 s total, of which feature transform 0.7 s.
- Complexity: O(n x (F + T x D)) for n rows, F features, T trees, depth D; linear in n, no dependence on other rows (each row is scored from its own record and frozen params).


## 4. Face validity: constrained (this model) vs unconstrained twin GB-02

GB-01: every feature with an expected sign has pd_violation_share = 0 (enforced by the monotone
constraints; table in section 1). The unconstrained twin GB-02 (fold-3 model, same diagnostics)
bends against the expected sign on many features; 'CONTRADICTS' = net partial-dependence movement
opposite to the expected sign:

| feature            | expected   |   mean_abs_shap |   shap_spearman | pd_trend                    |   pd_violation_share | verdict       |
|:-------------------|:-----------|----------------:|----------------:|:----------------------------|---------------------:|:--------------|
| tangible_ratio     | -          |           0.395 |          -0.933 | -4.46 -> -5.78 (range 1.39) |                0.053 | mostly agrees |
| equity_ratio       | -          |           0.264 |          -0.909 | -4.67 -> -5.59 (range 0.99) |                0.096 | mostly agrees |
| roa_net            | -          |           0.254 |          -0.945 | -4.41 -> -5.68 (range 1.34) |                0.066 | mostly agrees |
| st_debt_ratio      | +          |           0.203 |           0.978 | -5.71 -> -4.93 (range 0.78) |                0.007 | agrees        |
| ar_days            | +          |           0.170 |           0.922 | -5.50 -> -4.69 (range 0.91) |                0.125 | mostly agrees |
| ap_days            | +          |           0.129 |           0.742 | -5.40 -> -5.07 (range 0.51) |                0.298 | mostly agrees |
| cfo_to_debt        | -          |           0.121 |          -0.962 | -5.11 -> -5.50 (range 0.46) |                0.190 | mostly agrees |
| interest_to_assets | +          |           0.097 |           0.791 | -5.52 -> -5.13 (range 0.41) |                0.097 | mostly agrees |
| intangible_ratio   | +          |           0.088 |           0.067 | -5.37 -> -5.20 (range 0.36) |                0.362 | mostly agrees |
| interest_coverage  | -          |           0.085 |          -0.202 | -5.24 -> -5.13 (range 0.35) |                0.574 | CONTRADICTS   |
| cash_ratio         | -          |           0.083 |          -0.874 | -5.29 -> -5.55 (range 0.30) |                0.160 | mostly agrees |
| roa_oper           | -          |           0.080 |          -0.577 | -4.96 -> -5.33 (range 0.56) |                0.261 | mostly agrees |
| ebitda_margin      | -          |           0.069 |          -0.846 | -5.24 -> -5.47 (range 0.26) |                0.286 | mostly agrees |
| wc_ratio           | -          |           0.068 |           0.649 | -5.56 -> -5.22 (range 0.34) |                0.847 | CONTRADICTS   |
| log_assets         | -          |           0.062 |          -0.419 | -5.17 -> -5.46 (range 0.29) |                0.187 | mostly agrees |
| st_debt_share      | +          |           0.057 |           0.822 | -5.62 -> -5.34 (range 0.29) |                0.104 | mostly agrees |
| bank_debt_ratio    | +          |           0.039 |          -0.295 | -5.44 -> -5.49 (range 0.14) |                0.555 | CONTRADICTS   |
| current_ratio      | -          |           0.038 |          -0.815 | -5.37 -> -5.60 (range 0.26) |                0.152 | mostly agrees |
| neg_equity         | +          |           0.000 |           0.167 | -5.40 -> -5.40 (range 0.00) |                0.000 | agrees        |
| debt_split_missing | +          |           0.000 |         nan     | -5.40 -> -5.40 (range 0.00) |                0.000 | agrees        |

Reading: GB-02 makes `wc_ratio` (expected -) and `bank_debt_ratio` (expected +) work in the WRONG
direction and flattens `interest_coverage` (expected -) once the correlated liquidity / leverage /
profitability ratios are in the model (conditional suppressor effects among collinear ratios, see
univariate.md section 6). Its extra AUC (+0.0065, comparison.md) is bought partly with these
shapes, which a credit officer could not defend. Relaxing only the three decile-break features
(GB-06: current_ratio, tangible_ratio, st_debt_share) recovers just +0.0005 AUC.

## 5. Caveats for reason codes

- `ar_days` is winsorised at the training p99 (about 60,000 days): firms with near-zero revenue
  hit the cap (the median-PD example above). The reason code is correct (receivables are huge
  relative to sales), but the officer should read it as "almost no revenue", not as a literal
  collection period.
- `neg_equity` and `debt_split_missing` receive zero SHAP: their information is carried by
  `equity_ratio` (which goes below 0) and by native NaN handling of the debt-split ratios.
- `log_assets` is constrained to the expected '-' and has small importance (rank 19 of 26); the
  core-logit sign flip noted in D-022 does not appear as a contradiction here because the
  constraint forbids it.
