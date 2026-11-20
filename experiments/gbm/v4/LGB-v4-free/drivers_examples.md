# LGB-v4-free: top-5 drivers for three FY2010 applicants (round 2 model, trained FY2007-08)

TreeSHAP contributions on the log-odds scale. A positive contribution raises the PD relative to the model's average applicant; negative lowers it. 'Percentile' = where the firm's value sits among all FY2010 applicants. Row ids are id*10000 + fs_year.

## low PD (10th percentile): row_id 21370709632010, PD 0.12% (raw 0.08%); defaulted within 12 months: no

Model baseline (training-average log-odds, after the level shift): 0.59%; the contributions below move the firm from this baseline to its PD.

| # | Driver | Firm value | Percentile | Effect on log-odds | In words |
|---|---|---|---|---|---|
| 1 | `tangible_asset_ratio` | 0.3885 | 67% | -0.373 | high tangible fixed assets / assets; this lowers the PD |
| 2 | `equity_ratio` | 0.5711 | 83% | -0.275 | high equity as a share of assets; this lowers the PD |
| 3 | `size` | 14.27 | 3% | +0.207 | low firm size (log total assets); this raises the PD |
| 4 | `cash_ratio` | 0.337 | 97% | -0.167 | high cash / assets; this lowers the PD |
| 5 | `ateco_sector_te` | 0.01365 | 83% | +0.144 | high industry default rate (training years); this raises the PD |

## middle PD (median): row_id 6781600862010, PD 0.51% (raw 0.34%); defaulted within 12 months: no

Model baseline (training-average log-odds, after the level shift): 0.59%; the contributions below move the firm from this baseline to its PD.

| # | Driver | Firm value | Percentile | Effect on log-odds | In words |
|---|---|---|---|---|---|
| 1 | `tangible_asset_ratio` | 0.06366 | 38% | +0.214 | mid-range tangible fixed assets / assets; this raises the PD |
| 2 | `AR_ratio` | 0.141 | 14% | -0.213 | low receivables / revenue; this lowers the PD |
| 3 | `equity_ratio` | 0.1957 | 50% | -0.210 | mid-range equity as a share of assets; this lowers the PD |
| 4 | `operating_cash_flow_ratio` | 0.01738 | 45% | -0.114 | mid-range operating cash flow / assets; this lowers the PD |
| 5 | `profit` | 2,042 | 39% | +0.111 | mid-range net profit (EUR); this raises the PD |

## high PD (99th percentile): row_id 21972801632010, PD 15.35% (raw 10.72%); defaulted within 12 months: no

Model baseline (training-average log-odds, after the level shift): 0.59%; the contributions below move the firm from this baseline to its PD.

| # | Driver | Firm value | Percentile | Effect on log-odds | In words |
|---|---|---|---|---|---|
| 1 | `equity_ratio` | -0.03863 | 2% | +1.124 | low equity as a share of assets; this raises the PD |
| 2 | `operating_cash_flow_ratio` | -0.03339 | 6% | +0.440 | low operating cash flow / assets; this raises the PD |
| 3 | `AR_ratio` | 1.346 | 83% | +0.237 | high receivables / revenue; this raises the PD |
| 4 | `roa` | -3.03 | 8% | +0.232 | low return on assets; this raises the PD |
| 5 | `payables_ratio` | 0.547 | 94% | +0.227 | high short-term trade payables / assets; this raises the PD |

Note: `roe_negative` = 1 means the firm made a net loss AND has negative equity (the name is misleading; consistency report M3). `roe` is profit/equity*100, so it is large and POSITIVE for loss-making firms with negative equity. `goodwill_ratio` uses the provider's goodwill field, which behaves like a profit measure in the data (team doc 01; consistency report), so read it with that caveat. A mid-range value can still raise or lower the PD: SHAP compares the firm with the model's baseline, and the shape of each driver is not linear.