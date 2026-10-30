# GB-01 training log

monotone LightGBM, full feature set, native NaN, Platt-slope calibration

BASE params: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 4, "verbose": -1}
GRID (folds 2-3): [{'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}, {'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}]
DEFAULT (fold 1): {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}
max rounds 4000, early stopping 150 on slice A log loss; final trees = round(1.2 x best_iter)


## Fold 1: train [2007] (144,214 rows, 1,379 defaults) -> test 2008 (161,261 rows)
  [GB-01] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=282 A_logloss=0.04878 A_auc=0.8230 (2.9s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=282 final_trees=338
  slices: inner-train 101,261 (946 def), A 21,563 (230 def), B 21,390 (203 def; isotonic eligible=False)
  calibration on B (2-fold firm cross-fit log loss): raw 0.04398, platt 0.04401, isotonic 0.04553; Platt on B a=0.082 b=1.017
  own calibrator: b=1.017, a anchored=0.065 (train rate 0.0096)
  test 2008: AUC 0.8441, log loss 0.05423, mean pd 0.0106 vs DR 0.0128; raw AUC 0.8441 log loss 0.05423; predict 0.95s/100k rows

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 defaults) -> test 2009 (170,386 rows)
  [GB-01] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=663 A_logloss=0.05277 A_auc=0.8512 (8.6s)
  [GB-01] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=601 A_logloss=0.05272 A_auc=0.8519 (8.0s)
  [GB-01] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=408 A_logloss=0.05259 A_auc=0.8533 (8.3s)
  [GB-01] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=408 A_logloss=0.05261 A_auc=0.8528 (7.5s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=408 final_trees=490
  slices: inner-train 144,214 (1379 def), A 81,167 (1028 def), B 80,094 (1036 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.05579, platt 0.05549, isotonic 0.05573; Platt on B a=0.039 b=0.943
  own calibrator: b=0.943, a anchored=-0.203 (train rate 0.0113)
  test 2009: AUC 0.8359, log loss 0.06617, mean pd 0.0119 vs DR 0.0162; raw AUC 0.8359 log loss 0.06623; predict 1.28s/100k rows

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 defaults) -> test 2010 (174,512 rows)
  [GB-01] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=456 A_logloss=0.06754 A_auc=0.8266 (12.6s)
  [GB-01] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=698 A_logloss=0.06756 A_auc=0.8266 (16.9s)
  [GB-01] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=341 A_logloss=0.06733 A_auc=0.8279 (17.8s)
  [GB-01] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=322 A_logloss=0.06728 A_auc=0.8282 (10.6s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=322 final_trees=386
  slices: inner-train 305,475 (3443 def), A 85,773 (1395 def), B 84,613 (1357 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.06518, platt 0.06447, isotonic 0.06476; Platt on B a=0.336 b=0.998
  own calibrator: b=0.998, a anchored=-0.008 (train rate 0.0130)
  test 2010: AUC 0.8307, log loss 0.07136, mean pd 0.0119 vs DR 0.0170; raw AUC 0.8307 log loss 0.07137; predict 1.03s/100k rows

## Pooled / per-year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.83748 | 0.67496 |   0.064155 | 0.013676 |  0.011451 |      0.015379 |              1.343  |    1.5815e-137 |         0.33507 |     0.9763  |
| 2008    | 161261 |       2064 | 0.84414 | 0.68828 |   0.054229 | 0.011347 |  0.010564 |      0.012799 |              1.2116 |    1.8031e-17  |         0.21797 |     0.96898 |
| 2009    | 170386 |       2752 | 0.83593 | 0.67186 |   0.066169 | 0.014158 |  0.011855 |      0.016152 |              1.3625 |    1.8837e-54  |         0.35107 |     1.005   |
| 2010    | 174512 |       2968 | 0.83065 | 0.6613  |   0.071361 | 0.015359 |  0.011876 |      0.017007 |              1.4321 |    3.4628e-77  |         0.40824 |     0.95729 |
