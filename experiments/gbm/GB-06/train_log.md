# GB-06 training log

diagnostic: GB-01 with constraints dropped on the 3 decile-break features (feature-architect warning)

BASE params: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 8, "verbose": -1}
GRID (folds 2-3): [{'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}, {'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}]
DEFAULT (fold 1): {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}
max rounds 4000, early stopping 150 on slice A log loss; final trees = round(1.2 x best_iter)


## Fold 1: train [2007] (144,214 rows, 1,379 defaults) -> test 2008 (161,261 rows)
  [GB-06] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=288 A_logloss=0.04876 A_auc=0.8232 (1.8s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=288 final_trees=346
  slices: inner-train 101,261 (946 def), A 21,563 (230 def), B 21,390 (203 def; isotonic eligible=False)
  calibration on B (2-fold firm cross-fit log loss): raw 0.04414, platt 0.04416, isotonic 0.04571; Platt on B a=0.100 b=1.022
  own calibrator: b=1.022, a anchored=0.087 (train rate 0.0096)
  test 2008: AUC 0.8456, log loss 0.05411, mean pd 0.0106 vs DR 0.0128; raw AUC 0.8456 log loss 0.05412; predict 0.77s/100k rows

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 defaults) -> test 2009 (170,386 rows)
  [GB-06] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=846 A_logloss=0.05257 A_auc=0.8529 (5.5s)
  [GB-06] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=972 A_logloss=0.05252 A_auc=0.8536 (6.3s)
  [GB-06] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=544 A_logloss=0.05242 A_auc=0.8544 (4.5s)
  [GB-06] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=662 A_logloss=0.05243 A_auc=0.8538 (5.5s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=544 final_trees=653
  slices: inner-train 144,214 (1379 def), A 81,167 (1028 def), B 80,094 (1036 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.05558, platt 0.05530, isotonic 0.05580; Platt on B a=0.033 b=0.944
  own calibrator: b=0.944, a anchored=-0.200 (train rate 0.0113)
  test 2009: AUC 0.8349, log loss 0.06618, mean pd 0.0119 vs DR 0.0162; raw AUC 0.8349 log loss 0.06623; predict 0.98s/100k rows

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 defaults) -> test 2010 (174,512 rows)
  [GB-06] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=729 A_logloss=0.06749 A_auc=0.8256 (8.1s)
  [GB-06] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=523 A_logloss=0.06752 A_auc=0.8261 (6.2s)
  [GB-06] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=341 A_logloss=0.06738 A_auc=0.8268 (5.4s)
  [GB-06] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=388 A_logloss=0.06734 A_auc=0.8267 (5.7s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=388 final_trees=466
  slices: inner-train 305,475 (3443 def), A 85,773 (1395 def), B 84,613 (1357 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.06506, platt 0.06435, isotonic 0.06472; Platt on B a=0.327 b=0.996
  own calibrator: b=0.996, a anchored=-0.013 (train rate 0.0130)
  test 2010: AUC 0.8321, log loss 0.07115, mean pd 0.0119 vs DR 0.0170; raw AUC 0.8321 log loss 0.07117; predict 0.83s/100k rows

## Pooled / per-year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.83799 | 0.67598 |   0.064045 | 0.013664 |  0.011488 |      0.015379 |              1.3387 |    9.9417e-135 |         0.33126 |     0.97643 |
| 2008    | 161261 |       2064 | 0.84559 | 0.69117 |   0.054105 | 0.011339 |  0.010627 |      0.012799 |              1.2044 |    1.5897e-16  |         0.21102 |     0.97335 |
| 2009    | 170386 |       2752 | 0.83492 | 0.66984 |   0.066179 | 0.01415  |  0.011885 |      0.016152 |              1.359  |    1.2338e-53  |         0.34797 |     1.0001  |
| 2010    | 174512 |       2968 | 0.83213 | 0.66425 |   0.071146 | 0.015337 |  0.011896 |      0.017007 |              1.4297 |    1.5908e-76  |         0.40637 |     0.95911 |
