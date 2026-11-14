# Production LightGBM feature interpretation

## Method

TreeSHAP global importance was calculated on a stratified 100,000-row sample 
that preserves fiscal-year and default/non-default proportions.  The selected feature curves use 
all 835,016 labeled firm-year rows: continuous features are ranked into up to 50 
equal-frequency bins, then observed 12-month default rates are shrunk slightly toward the population 
rate and smoothed with a centered five-bin, exposure-weighted rolling mean.

This follows the RiskCalc-style default-frequency-curve idea, but the horizon here is 12 months. 
SHAP shows how the fitted model uses a feature; observed bin default rates are descriptive associations, not causal effects.

## Selected features

| Group | Rank | Feature | Mean absolute SHAP | Curve |
| --- | ---: | --- | ---: | --- |
| Business Structural | 1 | `ateco_sector_te` | 0.10349 | [CSV](curves/ateco_sector_te.csv) · [PNG](curves/ateco_sector_te.png) |
| Business Structural | 2 | `size` | 0.04943 | [CSV](curves/size.csv) · [PNG](curves/size.png) |
| Coverage Cash Flow | 1 | `operating_cash_flow_ratio` | 0.10607 | [CSV](curves/operating_cash_flow_ratio.csv) · [PNG](curves/operating_cash_flow_ratio.png) |
| Coverage Cash Flow | 2 | `operating_cash_flow_to_debt` | 0.10525 | [CSV](curves/operating_cash_flow_to_debt.csv) · [PNG](curves/operating_cash_flow_to_debt.png) |
| Leverage Capital Structure | 1 | `tangible_asset_ratio` | 0.33469 | [CSV](curves/tangible_asset_ratio.csv) · [PNG](curves/tangible_asset_ratio.png) |
| Leverage Capital Structure | 2 | `equity_ratio` | 0.21552 | [CSV](curves/equity_ratio.csv) · [PNG](curves/equity_ratio.png) |
| Liquidity Activity | 1 | `AR_ratio` | 0.12961 | [CSV](curves/AR_ratio.csv) · [PNG](curves/AR_ratio.png) |
| Liquidity Activity | 2 | `payables_ratio` | 0.09771 | [CSV](curves/payables_ratio.csv) · [PNG](curves/payables_ratio.png) |
| Profitability Growth | 1 | `net_prof_margin` | 0.11209 | [CSV](curves/net_prof_margin.csv) · [PNG](curves/net_prof_margin.png) |
| Profitability Growth | 2 | `tax_ratio` | 0.10027 | [CSV](curves/tax_ratio.csv) · [PNG](curves/tax_ratio.png) |

## Reading the charts

The top panel shows each bin's observed default rate, its 95% binomial confidence interval, and the smoothed default-frequency curve.  
The bottom panel shows the corresponding mean SHAP contribution. Positive SHAP values increase the model's raw log-odds of default; negative values decrease them.

One-hot legal-structure and city families are retained in `grouped_shap_importance.csv` as grouped fields, but are not selected for ratio-style curves.  High-collinearity feature pairs should be interpreted jointly.
