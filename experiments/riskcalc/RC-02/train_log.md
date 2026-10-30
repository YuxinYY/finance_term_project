# RC-02 train log

RC-01 + debt-split block ratios bank_debt_ratio, ap_days (15.7% missing as a block; missing gets its own empirical default rate inside T_i) -> tests H2

## Per-fold fit

|   fold | train_years        |   n_train |   train_defaults |   base_rate |   n_bins | converged   |         llf |   F_blocks |   seconds |
|-------:|:-------------------|----------:|-----------------:|------------:|---------:|:------------|------------:|-----------:|----------:|
|      1 | [2007]             |    144214 |             1379 |      0.0096 |       45 | True        |  -6566.4304 |         27 |   10.5000 |
|      2 | [2007, 2008]       |    305475 |             3443 |      0.0113 |       50 | True        | -15406.5078 |         32 |   10.5000 |
|      3 | [2007, 2008, 2009] |    475861 |             6195 |      0.0130 |       50 | True        | -26895.4813 |         40 |   22.2000 |

## Probit coefficients by fold (coef, SE, z)

Transformed ratios enter as Phi^-1(T_i(x)), a PD-scale measure: expected sign + for every T_i (the ratio's own direction is inside T_i). Indicators are vs the reference level (largest training level).

| term                 |   coef_f1 |   coef_f2 |   coef_f3 |    se_f1 |    se_f2 |   se_f3 |
|:---------------------|----------:|----------:|----------:|---------:|---------:|--------:|
| const                |   10.4383 |    9.6523 |    8.5663 |   1.355  |   0.5115 |  0.4021 |
| T_log_assets         |    0.693  |    0.8776 |    0.6656 |   0.516  |   0.1601 |  0.1416 |
| T_equity_ratio       |    0.4742 |    0.4293 |    0.4391 |   0.071  |   0.0393 |  0.0282 |
| T_st_debt_ratio      |    0.3253 |    0.3113 |    0.3035 |   0.0488 |   0.0276 |  0.0202 |
| T_cash_ratio         |    0.2891 |    0.4762 |    0.3534 |   0.1668 |   0.0897 |  0.069  |
| T_roa_net            |    0.6798 |    0.6502 |    0.6399 |   0.0451 |   0.0256 |  0.019  |
| T_interest_coverage  |    0.2089 |    0.2463 |    0.2409 |   0.0579 |   0.0328 |  0.0238 |
| T_interest_to_assets |    0.6759 |    0.5504 |    0.5005 |   0.0612 |   0.0356 |  0.0274 |
| T_ar_days            |    0.5926 |    0.4302 |    0.4041 |   0.1061 |   0.0711 |  0.0477 |
| T_tangible_ratio     |    1.0737 |    0.8969 |    0.789  |   0.0893 |   0.0566 |  0.0408 |
| T_neg_equity         |   -0.064  |   -0.0648 |   -0.0551 |   0.0703 |   0.04   |  0.0286 |
| T_bank_debt_ratio    |   -0.1814 |   -0.2573 |   -0.1496 |   0.1468 |   0.0811 |  0.0584 |
| T_ap_days            |    0.6243 |    0.6142 |    0.6543 |   0.1194 |   0.0775 |  0.0498 |
| ateco_group=A        |  nan      |   -0.2732 |   -0.3108 | nan      |   0.0722 |  0.057  |
| ateco_group=E        |  nan      |  nan      |   -0.1175 | nan      | nan      |  0.063  |
| ateco_group=F        |   -0.2469 |   -0.2675 |   -0.2125 |   0.0348 |   0.0229 |  0.0174 |
| ateco_group=G        |   -0.1306 |   -0.1355 |   -0.11   |   0.0327 |   0.0215 |  0.0167 |
| ateco_group=H        |   -0.1295 |   -0.1393 |   -0.0921 |   0.0719 |   0.0473 |  0.0354 |
| ateco_group=I        |  nan      |   -0.3837 |   -0.3128 | nan      |   0.0608 |  0.0436 |
| ateco_group=J        |   -0.1783 |   -0.122  |   -0.119  |   0.074  |   0.0466 |  0.0366 |
| ateco_group=L        |   -0.312  |   -0.3206 |   -0.2822 |   0.0378 |   0.0246 |  0.0187 |
| ateco_group=M        |   -0.2744 |   -0.2293 |   -0.1743 |   0.0609 |   0.0382 |  0.0284 |
| ateco_group=N        |   -0.1524 |   -0.096  |   -0.0547 |   0.0798 |   0.05   |  0.0376 |
| ateco_group=R        |  nan      |   -0.1819 |   -0.1142 | nan      |   0.0795 |  0.0592 |
| ateco_group=OTHER    |   -0.2275 |   -0.2803 |   -0.2618 |   0.0488 |   0.0484 |  0.0418 |

Sign violations among T_i (coef <= 0 in any fold): ['T_neg_equity', 'T_bank_debt_ratio']

Ratio selection: fixed economically balanced set from FEATURE_SPEC (same in every fold; only bins, rates and coefficients are re-estimated per fold).
