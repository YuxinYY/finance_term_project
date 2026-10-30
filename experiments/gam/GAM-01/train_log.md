# GAM-01 training log

EBM main effects only, monotone constraints from the expected-sign table

BASE: {"max_bins": 128, "max_interaction_bins": 32, "learning_rate": 0.03, "smoothing_rounds": 75, "interaction_smoothing_rounds": 75, "early_stopping_rounds": 100, "max_rounds": 30000, "inner_bags": 0, "validation_size": 0.15, "n_jobs": 8, "random_state": 0}
inner bags 4, final bags 8
GRID (folds 2-3): [{'max_leaves': 2, 'min_samples_leaf': 200}, {'max_leaves': 2, 'min_samples_leaf': 1000}, {'max_leaves': 3, 'min_samples_leaf': 200}, {'max_leaves': 3, 'min_samples_leaf': 1000}]
DEFAULT (fold 1): {'max_leaves': 2, 'min_samples_leaf': 500}
Fixed cfg from GAM-01: {1: {'max_leaves': 2, 'min_samples_leaf': 500}, 2: {'max_leaves': 3, 'min_samples_leaf': 1000}, 3: {'max_leaves': 3, 'min_samples_leaf': 200}}


## Fold 1: train [2007] (144,214 rows, 1,379 def) -> test 2008 (161,261)
  [GAM-01] cfg={'max_leaves': 2, 'min_samples_leaf': 500} A_logloss=0.04929 A_auc=0.8132 (35.6s)
  chosen {'max_leaves': 2, 'min_samples_leaf': 500}; Platt on B a=0.076 b=1.022; B LL raw 0.04501 platt 0.04501
  test 2008: AUC 0.8348 LL 0.05507 mean pd 0.0104 vs DR 0.0128; after rule-e shift (tau 0.0117, delta 0.200) LL 0.05479; raw AUC 0.8348 LL 0.05508; predict 0.84s/100k; final fit 54.9s; wall 105.5s

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 def) -> test 2009 (170,386)
  [GAM-01] cfg={'max_leaves': 3, 'min_samples_leaf': 1000} A_logloss=0.05319 A_auc=0.8451 (66.2s)
  chosen {'max_leaves': 3, 'min_samples_leaf': 1000}; Platt on B a=0.083 b=0.987; B LL raw 0.05613 platt 0.05603
  test 2009: AUC 0.8233 LL 0.06725 mean pd 0.0117 vs DR 0.0162; after rule-e shift (tau 0.0150, delta 0.291) LL 0.06644; raw AUC 0.8233 LL 0.06727; predict 0.83s/100k; final fit 114.2s; wall 197.7s

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 def) -> test 2010 (174,512)
  [GAM-01] cfg={'max_leaves': 3, 'min_samples_leaf': 200} A_logloss=0.06775 A_auc=0.8160 (106.5s)
  chosen {'max_leaves': 3, 'min_samples_leaf': 200}; Platt on B a=0.077 b=0.961; B LL raw 0.06589 platt 0.06557
  test 2010: AUC 0.8177 LL 0.07247 mean pd 0.0118 vs DR 0.0170; after rule-e shift (tau 0.0161, delta 0.217) LL 0.07160; raw AUC 0.8177 LL 0.07260; predict 0.79s/100k; final fit 173.9s; wall 305.3s

## Own calibration
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.82545 | 0.65089 |   0.06517  | 0.013788 |  0.011309 |      0.015379 |              1.3598 |    1.2198e-148 |         0.34476 |     0.96433 |
| 2008    | 161261 |       2064 | 0.83479 | 0.66958 |   0.055072 | 0.011464 |  0.010395 |      0.012799 |              1.2313 |    3.9428e-20  |         0.23217 |     0.97405 |
| 2009    | 170386 |       2752 | 0.82327 | 0.64653 |   0.067254 | 0.014254 |  0.011705 |      0.016152 |              1.3799 |    1.199e-58   |         0.36406 |     0.96866 |
| 2010    | 174512 |       2968 | 0.81771 | 0.63542 |   0.072467 | 0.015481 |  0.011767 |      0.017007 |              1.4454 |    7.8575e-81  |         0.41155 |     0.95239 |

## With rule-e prior shift
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.82554 | 0.65107 |   0.064506 | 0.013742 |  0.013995 |      0.015379 |              1.0989 |     1.5657e-16 |        0.1064   |     0.96347 |
| 2008    | 161261 |       2064 | 0.83479 | 0.66958 |   0.054791 | 0.011443 |  0.012434 |      0.012799 |              1.0294 |     0.18487    |        0.032516 |     0.97411 |
| 2009    | 170386 |       2752 | 0.82327 | 0.64653 |   0.066441 | 0.014185 |  0.015152 |      0.016152 |              1.066  |     0.00080291 |        0.072794 |     0.96867 |
| 2010    | 174512 |       2968 | 0.81771 | 0.63542 |   0.071595 | 0.015432 |  0.014309 |      0.017007 |              1.1886 |     2.8031e-20 |        0.19405  |     0.95113 |
