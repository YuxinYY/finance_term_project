# GBM experiments: comparison (pooled over test years 2008-2010, 506,159 rows, 7,784 defaults)

Own calibration in every row (Platt slope from inner slice B, level anchored to the training default rate). AR = 2*AUC - 1. Brier = overall probability accuracy.

| exp   | desc                                                                                                  |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   cal_slope |   auc_2008 |   auc_2009 |   auc_2010 |   auc_seen |   auc_new |
|:------|:------------------------------------------------------------------------------------------------------|--------:|--------:|-----------:|---------:|----------:|--------------:|------------:|-----------:|-----------:|-----------:|-----------:|----------:|
| GB-01 | monotone LightGBM, full feature set, native NaN, Platt-slope calibration                              | 0.83748 | 0.67496 |   0.064155 | 0.013676 |  0.011451 |      0.015379 |     0.9763  |    0.84414 |    0.83593 |    0.83065 |    0.84382 |   0.78651 |
| GB-02 | GB-01 without monotone constraints (face-validity cost diagnostic)                                    | 0.844   | 0.68801 |   0.063321 | 0.013584 |  0.011423 |      0.015379 |     0.97893 |    0.84848 |    0.84194 |    0.84019 |    0.85046 |   0.79199 |
| GB-03 | GB-01 on the 13 core features only                                                                    | 0.82981 | 0.65963 |   0.064921 | 0.013781 |  0.011345 |      0.015379 |     0.98132 |    0.8381  |    0.82742 |    0.82259 |    0.83643 |   0.77663 |
| GB-04 | GB-01 with the shared build.py imputation instead of native NaN                                       | 0.83758 | 0.67515 |   0.064148 | 0.01368  |  0.011475 |      0.015379 |     0.97439 |    0.8448  |    0.83637 |    0.83022 |    0.84406 |   0.78561 |
| GB-05 | GB-01 with LightGBM's 'advanced' monotone method (still fully monotone, less restrictive splits)      | 0.83766 | 0.67531 |   0.064136 | 0.013676 |  0.011459 |      0.015379 |     0.97672 |    0.84466 |    0.83577 |    0.83089 |    0.84395 |   0.78723 |
| GB-06 | diagnostic: GB-01 with constraints dropped on the 3 decile-break features (feature-architect warning) | 0.83799 | 0.67598 |   0.064045 | 0.013664 |  0.011488 |      0.015379 |     0.97643 |    0.84559 |    0.83492 |    0.83213 |    0.84449 |   0.78575 |

## Paired firm-clustered bootstrap vs GB-01 (200 draws; d = new - GB-01; negative d_log_loss favours new)

| new   | old   |       d_auc | d_auc_ci           |   share_auc_favours_new |   d_log_loss | d_ll_ci              |   share_ll_favours_new |
|:------|:------|------------:|:-------------------|------------------------:|-------------:|:---------------------|-----------------------:|
| GB-02 | GB-01 |  0.0065264  | [+0.0052, +0.0079] |                   1     |  -0.0008338  | [-0.00096, -0.00072] |                  1     |
| GB-03 | GB-01 | -0.0076646  | [-0.0087, -0.0066] |                   0     |   0.0007659  | [+0.00067, +0.00085] |                  0     |
| GB-04 | GB-01 |  9.7359e-05 | [-0.0003, +0.0006] |                   0.695 |  -7.405e-06  | [-0.00004, +0.00003] |                  0.68  |
| GB-05 | GB-01 |  0.00017801 | [-0.0000, +0.0004] |                   0.94  |  -1.9125e-05 | [-0.00004, +0.00000] |                  0.965 |
| GB-06 | GB-01 |  0.00051199 | [+0.0000, +0.0010] |                   0.98  |  -0.00010984 | [-0.00014, -0.00007] |                  1     |

Round-trip check: GB-01 fold-3 model saved with save_model and reloaded with load_model reproduces the fold-3 OOT PDs; max |diff| = 0.00e+00.
