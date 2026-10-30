# RC-03 train log

RC-02 candidate set + in-window sign selection: any T_i with probit coef <= 0 in the fold's own training fit is dropped and the probit refitted

## Per-fold fit

|   fold | train_years        |   n_train |   train_defaults |   base_rate |   n_bins | converged   |         llf | dropped_in_window                                   |   F_blocks |   seconds |
|-------:|:-------------------|----------:|-----------------:|------------:|---------:|:------------|------------:|:----------------------------------------------------|-----------:|----------:|
|      1 | [2007]             |    144214 |             1379 |      0.0096 |       45 | True        |  -6567.5233 | T_bank_debt_ratio (z=-1.24), T_neg_equity (z=-0.81) |         31 |   11.8000 |
|      2 | [2007, 2008]       |    305475 |             3443 |      0.0113 |       50 | True        | -15412.6130 | T_bank_debt_ratio (z=-3.17), T_neg_equity (z=-1.43) |         34 |   14.8000 |
|      3 | [2007, 2008, 2009] |    475861 |             6195 |      0.0130 |       50 | True        | -26900.3804 | T_bank_debt_ratio (z=-2.56), T_neg_equity (z=-1.78) |         39 |   17.0000 |

## Probit coefficients by fold (coef, SE, z)

Transformed ratios enter as Phi^-1(T_i(x)), a PD-scale measure: expected sign + for every T_i (the ratio's own direction is inside T_i). Indicators are vs the reference level (largest training level).

| term                 |   coef_f1 |   coef_f2 |   coef_f3 |    se_f1 |    se_f2 |   se_f3 |
|:---------------------|----------:|----------:|----------:|---------:|---------:|--------:|
| const                |   10.8341 |   10.2108 |    8.9076 |   1.3291 |   0.4864 |  0.3869 |
| T_log_assets         |    0.6905 |    0.8844 |    0.6723 |   0.5166 |   0.1604 |  0.1417 |
| T_equity_ratio       |    0.4165 |    0.3691 |    0.391  |   0.0446 |   0.0258 |  0.0187 |
| T_st_debt_ratio      |    0.3338 |    0.3164 |    0.308  |   0.0484 |   0.0275 |  0.0201 |
| T_cash_ratio         |    0.265  |    0.4488 |    0.3414 |   0.1641 |   0.0888 |  0.0684 |
| T_roa_net            |    0.6833 |    0.6552 |    0.6423 |   0.045  |   0.0256 |  0.0189 |
| T_interest_coverage  |    0.2041 |    0.2421 |    0.2374 |   0.058  |   0.0328 |  0.0238 |
| T_interest_to_assets |    0.6485 |    0.5062 |    0.4751 |   0.0559 |   0.0325 |  0.0251 |
| T_ar_days            |    0.5936 |    0.4276 |    0.3999 |   0.1059 |   0.071  |  0.0477 |
| T_tangible_ratio     |    1.0851 |    0.9126 |    0.8005 |   0.089  |   0.0563 |  0.0406 |
| T_ap_days            |    0.64   |    0.6429 |    0.671  |   0.119  |   0.0771 |  0.0495 |
| ateco_group=F        |   -0.2479 |   -0.2707 |   -0.2148 |   0.0347 |   0.0229 |  0.0173 |
| ateco_group=G        |   -0.128  |   -0.1321 |   -0.1079 |   0.0327 |   0.0215 |  0.0166 |
| ateco_group=H        |   -0.1249 |   -0.1321 |   -0.0876 |   0.0718 |   0.0472 |  0.0353 |
| ateco_group=J        |   -0.1738 |   -0.1145 |   -0.1148 |   0.0739 |   0.0465 |  0.0365 |
| ateco_group=L        |   -0.3151 |   -0.3275 |   -0.2871 |   0.0377 |   0.0245 |  0.0186 |
| ateco_group=M        |   -0.269  |   -0.2222 |   -0.171  |   0.0607 |   0.0381 |  0.0284 |
| ateco_group=N        |   -0.1466 |   -0.0867 |   -0.0494 |   0.0797 |   0.0499 |  0.0375 |
| ateco_group=OTHER    |   -0.2256 |   -0.2751 |   -0.2592 |   0.0487 |   0.0484 |  0.0418 |
| ateco_group=A        |  nan      |   -0.2729 |   -0.3107 | nan      |   0.0722 |  0.057  |
| ateco_group=I        |  nan      |   -0.3841 |   -0.3136 | nan      |   0.0607 |  0.0435 |
| ateco_group=R        |  nan      |   -0.1762 |   -0.111  | nan      |   0.0793 |  0.0592 |
| ateco_group=E        |  nan      |  nan      |   -0.1131 | nan      | nan      |  0.0629 |

Sign violations among T_i (coef <= 0 in any fold): none

Ratio selection: fixed economically balanced set from FEATURE_SPEC (same in every fold; only bins, rates and coefficients are re-estimated per fold).
