# GAM-02 training log

GAM-01 + 10 pairwise interaction terms (FAST-selected inside the training window)

BASE: {"max_bins": 128, "max_interaction_bins": 32, "learning_rate": 0.03, "smoothing_rounds": 75, "interaction_smoothing_rounds": 75, "early_stopping_rounds": 100, "max_rounds": 30000, "inner_bags": 0, "validation_size": 0.15, "n_jobs": 8, "random_state": 0}
inner bags 4, final bags 8
GRID (folds 2-3): [{'max_leaves': 2, 'min_samples_leaf': 200}, {'max_leaves': 2, 'min_samples_leaf': 1000}, {'max_leaves': 3, 'min_samples_leaf': 200}, {'max_leaves': 3, 'min_samples_leaf': 1000}]
DEFAULT (fold 1): {'max_leaves': 2, 'min_samples_leaf': 500}
Fixed cfg from GAM-01: {1: {'max_leaves': 2, 'min_samples_leaf': 500}, 2: {'max_leaves': 3, 'min_samples_leaf': 1000}, 3: {'max_leaves': 3, 'min_samples_leaf': 200}}


## Fold 1: train [2007] (144,214 rows, 1,379 def) -> test 2008 (161,261)
  [GAM-02] cfg={'max_leaves': 2, 'min_samples_leaf': 500} A_logloss=0.04917 A_auc=0.8143 (46.5s)
  chosen {'max_leaves': 2, 'min_samples_leaf': 500}; Platt on B a=-0.035 b=0.994; B LL raw 0.04508 platt 0.04508
  test 2008: AUC 0.8353 LL 0.05503 mean pd 0.0104 vs DR 0.0128; after rule-e shift (tau 0.0117, delta 0.200) LL 0.05474; raw AUC 0.8353 LL 0.05503; predict 0.83s/100k; final fit 69.6s; wall 130.8s

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 def) -> test 2009 (170,386)
  [GAM-02] cfg={'max_leaves': 3, 'min_samples_leaf': 1000} A_logloss=0.05318 A_auc=0.8450 (223.7s)
  chosen {'max_leaves': 3, 'min_samples_leaf': 1000}; Platt on B a=0.079 b=0.986; B LL raw 0.05610 platt 0.05600
  test 2009: AUC 0.8239 LL 0.06719 mean pd 0.0117 vs DR 0.0162; after rule-e shift (tau 0.0150, delta 0.291) LL 0.06638; raw AUC 0.8239 LL 0.06721; predict 0.91s/100k; final fit 142.8s; wall 384.0s

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 def) -> test 2010 (174,512)
  [GAM-02] cfg={'max_leaves': 3, 'min_samples_leaf': 200} A_logloss=0.06773 A_auc=0.8160 (143.5s)
  chosen {'max_leaves': 3, 'min_samples_leaf': 200}; Platt on B a=0.077 b=0.961; B LL raw 0.06587 platt 0.06555
  test 2010: AUC 0.8192 LL 0.07235 mean pd 0.0118 vs DR 0.0170; after rule-e shift (tau 0.0161, delta 0.217) LL 0.07148; raw AUC 0.8192 LL 0.07249; predict 0.94s/100k; final fit 264.3s; wall 434.0s

## Own calibration
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.82624 | 0.65248 |   0.065095 | 0.013786 |  0.011294 |      0.015379 |              1.3616 |    8.1146e-150 |         0.3459  |     0.96988 |
| 2008    | 161261 |       2064 | 0.83534 | 0.67067 |   0.055025 | 0.011475 |  0.010353 |      0.012799 |              1.2363 |    7.3687e-21  |         0.23502 |     0.99774 |
| 2009    | 170386 |       2752 | 0.82393 | 0.64786 |   0.067193 | 0.01425  |  0.011711 |      0.016152 |              1.3791 |    1.8432e-58  |         0.36382 |     0.9677  |
| 2010    | 174512 |       2968 | 0.81925 | 0.6385  |   0.072352 | 0.015468 |  0.011757 |      0.017007 |              1.4466 |    3.5644e-81  |         0.41347 |     0.94973 |

## With rule-e prior shift
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.82636 | 0.65272 |   0.064428 | 0.013738 |  0.013979 |      0.015379 |              1.1001 |     6.7257e-17 |        0.10759  |     0.96973 |
| 2008    | 161261 |       2064 | 0.83534 | 0.67067 |   0.054737 | 0.011444 |  0.012399 |      0.012799 |              1.0323 |     0.14664    |        0.035356 |     0.99794 |
| 2009    | 170386 |       2752 | 0.82393 | 0.64786 |   0.066381 | 0.014183 |  0.015156 |      0.016152 |              1.0657 |     0.00086338 |        0.072551 |     0.96781 |
| 2010    | 174512 |       2968 | 0.81925 | 0.6385  |   0.071477 | 0.015422 |  0.01429  |      0.017007 |              1.1901 |     1.5029e-20 |        0.19598  |     0.94919 |
