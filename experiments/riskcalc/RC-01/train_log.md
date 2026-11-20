# RC-01 train log

RiskCalc v1: 10 core ratios (T_i = monotone shrunk binned default rate, probit scale) + ateco_group indicators; probit; F = PAV on index

## Per-fold fit

|   fold | train_years        |   n_train |   train_defaults |   base_rate |   n_bins | converged   |         llf |   F_blocks |   seconds |
|-------:|:-------------------|----------:|-----------------:|------------:|---------:|:------------|------------:|-----------:|----------:|
|      1 | [2007]             |    144214 |             1379 |      0.0096 |       45 | True        |  -6581.2357 |         34 |   14.2000 |
|      2 | [2007, 2008]       |    305475 |             3443 |      0.0113 |       50 | True        | -15445.5886 |         30 |   12.2000 |
|      3 | [2007, 2008, 2009] |    475861 |             6195 |      0.0130 |       50 | True        | -26988.5542 |         35 |   19.3000 |

## Probit coefficients by fold (coef, SE, z)

Transformed ratios enter as Phi^-1(T_i(x)), a PD-scale measure: expected sign + for every T_i (the ratio's own direction is inside T_i). Indicators are vs the reference level (largest training level).

| term                 |   coef_f1 |   coef_f2 |   coef_f3 |    se_f1 |    se_f2 |   se_f3 |
|:---------------------|----------:|----------:|----------:|---------:|---------:|--------:|
| const                |    9.7828 |    9.0404 |    7.7829 |   1.3225 |   0.4704 |  0.38   |
| T_log_assets         |    0.7866 |    0.9138 |    0.7167 |   0.5166 |   0.1597 |  0.141  |
| T_equity_ratio       |    0.4908 |    0.4387 |    0.4636 |   0.0697 |   0.0386 |  0.0277 |
| T_st_debt_ratio      |    0.3552 |    0.3359 |    0.331  |   0.0481 |   0.0273 |  0.0199 |
| T_cash_ratio         |    0.2379 |    0.4372 |    0.3429 |   0.1643 |   0.0889 |  0.0684 |
| T_roa_net            |    0.666  |    0.6398 |    0.6214 |   0.0448 |   0.0254 |  0.0188 |
| T_interest_coverage  |    0.2166 |    0.2504 |    0.2446 |   0.0579 |   0.0328 |  0.0238 |
| T_interest_to_assets |    0.6098 |    0.4748 |    0.4403 |   0.0556 |   0.0324 |  0.025  |
| T_ar_days            |    0.7649 |    0.5857 |    0.5716 |   0.1008 |   0.0682 |  0.0456 |
| T_tangible_ratio     |    1.0633 |    0.8926 |    0.7769 |   0.0891 |   0.0564 |  0.0406 |
| T_neg_equity         |   -0.0797 |   -0.0772 |   -0.076  |   0.0699 |   0.0397 |  0.0285 |
| ateco_group=A        |  nan      |   -0.2675 |   -0.3017 | nan      |   0.0723 |  0.057  |
| ateco_group=E        |  nan      |  nan      |   -0.103  | nan      | nan      |  0.0626 |
| ateco_group=F        |   -0.2458 |   -0.2677 |   -0.2083 |   0.0347 |   0.0229 |  0.0173 |
| ateco_group=G        |   -0.122  |   -0.1266 |   -0.1011 |   0.0326 |   0.0215 |  0.0166 |
| ateco_group=H        |   -0.1367 |   -0.1451 |   -0.1008 |   0.0718 |   0.0472 |  0.0353 |
| ateco_group=I        |  nan      |   -0.3805 |   -0.3101 | nan      |   0.0605 |  0.0434 |
| ateco_group=J        |   -0.1819 |   -0.1231 |   -0.1259 |   0.0737 |   0.0464 |  0.0364 |
| ateco_group=L        |   -0.3215 |   -0.3325 |   -0.2913 |   0.0376 |   0.0245 |  0.0186 |
| ateco_group=M        |   -0.2719 |   -0.2242 |   -0.1721 |   0.0606 |   0.038  |  0.0283 |
| ateco_group=N        |   -0.1588 |   -0.1005 |   -0.0641 |   0.0797 |   0.0498 |  0.0375 |
| ateco_group=R        |  nan      |   -0.1838 |   -0.1155 | nan      |   0.0795 |  0.0592 |
| ateco_group=OTHER    |   -0.2249 |   -0.2751 |   -0.2591 |   0.0487 |   0.0483 |  0.0417 |

Sign violations among T_i (coef <= 0 in any fold): ['T_neg_equity']

Ratio selection: fixed economically balanced set from FEATURE_SPEC (same in every fold; only bins, rates and coefficients are re-estimated per fold).
