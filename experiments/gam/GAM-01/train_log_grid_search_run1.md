# GAM-01 training log

EBM main effects only, monotone constraints from the expected-sign table

BASE: {"max_bins": 128, "max_interaction_bins": 32, "learning_rate": 0.03, "smoothing_rounds": 75, "interaction_smoothing_rounds": 75, "early_stopping_rounds": 100, "max_rounds": 30000, "inner_bags": 0, "validation_size": 0.15, "n_jobs": 8, "random_state": 0}
inner bags 4, final bags 8
GRID (folds 2-3): [{'max_leaves': 2, 'min_samples_leaf': 200}, {'max_leaves': 2, 'min_samples_leaf': 1000}, {'max_leaves': 3, 'min_samples_leaf': 200}, {'max_leaves': 3, 'min_samples_leaf': 1000}]
DEFAULT (fold 1): {'max_leaves': 2, 'min_samples_leaf': 500}


## Fold 1: train [2007] (144,214 rows, 1,379 def) -> test 2008 (161,261)
  [GAM-01] cfg={'max_leaves': 2, 'min_samples_leaf': 500} A_logloss=0.04929 A_auc=0.8132 (36.7s)
  chosen {'max_leaves': 2, 'min_samples_leaf': 500}; Platt on B a=0.076 b=1.022; B LL raw 0.04501 platt 0.04501
  test 2008: AUC 0.8345 LL 0.05508 mean pd 0.0104 vs DR 0.0128; after rule-e shift (tau 0.0117, delta 0.200) LL 0.05479; raw AUC 0.8345 LL 0.05508; predict 0.89s/100k; final fit 59.0s; wall 112.0s

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 def) -> test 2009 (170,386)
  [GAM-01] cfg={'max_leaves': 2, 'min_samples_leaf': 200} A_logloss=0.05324 A_auc=0.8435 (70.4s)
  [GAM-01] cfg={'max_leaves': 2, 'min_samples_leaf': 1000} A_logloss=0.05324 A_auc=0.8435 (71.4s)
  [GAM-01] cfg={'max_leaves': 3, 'min_samples_leaf': 200} A_logloss=0.05319 A_auc=0.8451 (66.3s)
  [GAM-01] cfg={'max_leaves': 3, 'min_samples_leaf': 1000} A_logloss=0.05319 A_auc=0.8451 (64.3s)
  chosen {'max_leaves': 3, 'min_samples_leaf': 1000}; Platt on B a=0.083 b=0.987; B LL raw 0.05613 platt 0.05603
  test 2009: AUC 0.8233 LL 0.06725 mean pd 0.0117 vs DR 0.0162; after rule-e shift (tau 0.0150, delta 0.291) LL 0.06644; raw AUC 0.8233 LL 0.06726; predict 0.83s/100k; final fit 112.6s; wall 402.8s

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 def) -> test 2010 (174,512)
  [GAM-01] cfg={'max_leaves': 2, 'min_samples_leaf': 200} A_logloss=0.06797 A_auc=0.8155 (114.9s)
  [GAM-01] cfg={'max_leaves': 2, 'min_samples_leaf': 1000} A_logloss=0.06797 A_auc=0.8155 (115.8s)
  [GAM-01] cfg={'max_leaves': 3, 'min_samples_leaf': 200} A_logloss=0.06775 A_auc=0.8160 (110.7s)
  [GAM-01] cfg={'max_leaves': 3, 'min_samples_leaf': 1000} A_logloss=0.06775 A_auc=0.8160 (109.5s)
  chosen {'max_leaves': 3, 'min_samples_leaf': 200}; Platt on B a=0.077 b=0.961; B LL raw 0.06589 platt 0.06557
  test 2010: AUC 0.8177 LL 0.07247 mean pd 0.0118 vs DR 0.0170; after rule-e shift (tau 0.0161, delta 0.217) LL 0.07159; raw AUC 0.8177 LL 0.07259; predict 0.83s/100k; final fit 176.2s; wall 652.2s

## Own calibration
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.8254  | 0.65079 |   0.065168 | 0.013787 |  0.011313 |      0.015379 |              1.3594 |    2.3184e-148 |         0.34442 |     0.96396 |
| 2008    | 161261 |       2064 | 0.83446 | 0.66891 |   0.055075 | 0.01146  |  0.010395 |      0.012799 |              1.2312 |    3.9496e-20  |         0.2322  |     0.9729  |
| 2009    | 170386 |       2752 | 0.82335 | 0.64669 |   0.067248 | 0.014254 |  0.011717 |      0.016152 |              1.3784 |    2.8252e-58  |         0.36289 |     0.96859 |
| 2010    | 174512 |       2968 | 0.81771 | 0.63542 |   0.072465 | 0.01548  |  0.011766 |      0.017007 |              1.4455 |    7.6772e-81  |         0.41165 |     0.95245 |

## With rule-e prior shift
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.82548 | 0.65096 |   0.064506 | 0.013741 |  0.014    |      0.015379 |              1.0985 |     2.0326e-16 |        0.10603  |     0.96306 |
| 2008    | 161261 |       2064 | 0.83446 | 0.66891 |   0.054795 | 0.01144  |  0.012433 |      0.012799 |              1.0294 |     0.18486    |        0.03255  |     0.97281 |
| 2009    | 170386 |       2752 | 0.82335 | 0.64669 |   0.066438 | 0.014186 |  0.015167 |      0.016152 |              1.0649 |     0.0009992  |        0.071621 |     0.96861 |
| 2010    | 174512 |       2968 | 0.81771 | 0.63542 |   0.071593 | 0.015432 |  0.014307 |      0.017007 |              1.1887 |     2.786e-20  |        0.19416  |     0.95118 |
