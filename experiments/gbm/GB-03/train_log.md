# GB-03 training log

GB-01 on the 13 core features only

BASE params: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 2, "verbose": -1}
GRID (folds 2-3): [{'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}, {'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}]
DEFAULT (fold 1): {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}
max rounds 4000, early stopping 150 on slice A log loss; final trees = round(1.2 x best_iter)


## Fold 1: train [2007] (144,214 rows, 1,379 defaults) -> test 2008 (161,261 rows)
  [GB-03] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=286 A_logloss=0.04864 A_auc=0.8208 (4.6s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=286 final_trees=343
  slices: inner-train 101,261 (946 def), A 21,563 (230 def), B 21,390 (203 def; isotonic eligible=False)
  calibration on B (2-fold firm cross-fit log loss): raw 0.04457, platt 0.04460, isotonic 0.04513; Platt on B a=-0.049 b=0.985
  own calibrator: b=0.985, a anchored=-0.058 (train rate 0.0096)
  test 2008: AUC 0.8381, log loss 0.05488, mean pd 0.0105 vs DR 0.0128; raw AUC 0.8381 log loss 0.05488; predict 0.88s/100k rows

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 defaults) -> test 2009 (170,386 rows)
  [GB-03] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=593 A_logloss=0.05338 A_auc=0.8470 (6.8s)
  [GB-03] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=551 A_logloss=0.05340 A_auc=0.8466 (8.5s)
  [GB-03] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=318 A_logloss=0.05323 A_auc=0.8486 (7.9s)
  [GB-03] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=364 A_logloss=0.05326 A_auc=0.8482 (7.7s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=318 final_trees=382
  slices: inner-train 144,214 (1379 def), A 81,167 (1028 def), B 80,094 (1036 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.05651, platt 0.05623, isotonic 0.05641; Platt on B a=0.031 b=0.943
  own calibrator: b=0.943, a anchored=-0.207 (train rate 0.0113)
  test 2009: AUC 0.8274, log loss 0.06708, mean pd 0.0117 vs DR 0.0162; raw AUC 0.8274 log loss 0.06710; predict 1.44s/100k rows

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 defaults) -> test 2010 (174,512 rows)
  [GB-03] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=284 A_logloss=0.06837 A_auc=0.8183 (6.7s)
  [GB-03] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=284 A_logloss=0.06836 A_auc=0.8182 (7.4s)
  [GB-03] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=198 A_logloss=0.06816 A_auc=0.8199 (8.5s)
  [GB-03] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=198 A_logloss=0.06817 A_auc=0.8202 (7.1s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=198 final_trees=238
  slices: inner-train 305,475 (3443 def), A 85,773 (1395 def), B 84,613 (1357 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.06607, platt 0.06533, isotonic 0.06600; Platt on B a=0.456 b=1.030
  own calibrator: b=1.030, a anchored=0.104 (train rate 0.0130)
  test 2010: AUC 0.8226, log loss 0.07209, mean pd 0.0118 vs DR 0.0170; raw AUC 0.8226 log loss 0.07203; predict 0.88s/100k rows

## Pooled / per-year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.82981 | 0.65963 |   0.064921 | 0.013781 |  0.011345 |      0.015379 |              1.3555 |    1.0558e-145 |         0.34155 |     0.98132 |
| 2008    | 161261 |       2064 | 0.8381  | 0.6762  |   0.054884 | 0.011469 |  0.010478 |      0.012799 |              1.2215 |    8.9804e-19  |         0.22321 |     0.97997 |
| 2009    | 170386 |       2752 | 0.82742 | 0.65484 |   0.067078 | 0.014288 |  0.011692 |      0.016152 |              1.3815 |    4.9397e-59  |         0.36188 |     1.0111  |
| 2010    | 174512 |       2968 | 0.82259 | 0.64519 |   0.07209  | 0.015425 |  0.011809 |      0.017007 |              1.4402 |    2.3526e-79  |         0.41261 |     0.95832 |
