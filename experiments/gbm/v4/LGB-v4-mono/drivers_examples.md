# LGB-v4-mono: top-5 drivers for three FY2010 applicants (round 2 model, trained FY2007-08)

TreeSHAP contributions on the log-odds scale. A positive contribution raises the PD relative to the model's average applicant; negative lowers it. 'Percentile' = where the firm's value sits among all FY2010 applicants. Row ids are id*10000 + fs_year.

## low PD (10th percentile): row_id 23872802472010, PD 0.10% (raw 0.06%); defaulted within 12 months: no

Model baseline (training-average log-odds, after the level shift): 0.56%; the contributions below move the firm from this baseline to its PD.

| # | Driver | Firm value | Percentile | Effect on log-odds | In words |
|---|---|---|---|---|---|
| 1 | `tangible_asset_ratio` | 0.04515 | 34% | +0.309 | mid-range tangible fixed assets / assets; this raises the PD |
| 2 | `financing_expense_ratio` | 0.0004317 | 16% | -0.243 | low financing expense / assets; this lowers the PD |
| 3 | `legal_SPA` | 1 | 95% | -0.233 | legal form SPA: yes; this lowers the PD |
| 4 | `equity_ratio` | 0.2675 | 59% | -0.199 | mid-range equity as a share of assets; this lowers the PD |
| 5 | `AR_ratio` | 0.232 | 23% | -0.150 | low receivables / revenue; this lowers the PD |

## middle PD (median): row_id 15102602092010, PD 0.50% (raw 0.33%); defaulted within 12 months: no

Model baseline (training-average log-odds, after the level shift): 0.56%; the contributions below move the firm from this baseline to its PD.

| # | Driver | Firm value | Percentile | Effect on log-odds | In words |
|---|---|---|---|---|---|
| 1 | `tangible_asset_ratio` | 0.1813 | 52% | -0.238 | mid-range tangible fixed assets / assets; this lowers the PD |
| 2 | `ateco_sector_te` | 0.01365 | 83% | +0.199 | high industry default rate (training years); this raises the PD |
| 3 | `bank_debt_conc` | 0.6292 | 82% | -0.143 | high bank share of debt; this lowers the PD |
| 4 | `financing_expense_ratio` | 0.01035 | 58% | -0.106 | mid-range financing expense / assets; this lowers the PD |
| 5 | `net_prof_margin` | 0.003724 | 46% | +0.096 | mid-range net profit margin; this raises the PD |

## high PD (99th percentile): row_id 11816406302010, PD 15.43% (raw 10.78%); defaulted within 12 months: no

Model baseline (training-average log-odds, after the level shift): 0.56%; the contributions below move the firm from this baseline to its PD.

| # | Driver | Firm value | Percentile | Effect on log-odds | In words |
|---|---|---|---|---|---|
| 1 | `equity_ratio` | -0.1801 | 1% | +1.616 | low equity as a share of assets; this raises the PD |
| 2 | `operating_cash_flow_ratio` | -0.1416 | 1% | +0.952 | low operating cash flow / assets; this raises the PD |
| 3 | `roa` | -17.85 | 1% | +0.549 | low return on assets; this raises the PD |
| 4 | `goodwill_ratio` | -0.1079 | 1% | -0.306 | low goodwill field / assets; this lowers the PD |
| 5 | `cash_ratio` | 2.885e-06 | 2% | +0.263 | low cash / assets; this raises the PD |

Note: `roe_negative` = 1 means the firm made a net loss AND has negative equity (the name is misleading; consistency report M3). `roe` is profit/equity*100, so it is large and POSITIVE for loss-making firms with negative equity. `goodwill_ratio` uses the provider's goodwill field, which behaves like a profit measure in the data (team doc 01; consistency report), so read it with that caveat. A mid-range value can still raise or lower the PD: SHAP compares the firm with the model's baseline, and the shape of each driver is not linear.