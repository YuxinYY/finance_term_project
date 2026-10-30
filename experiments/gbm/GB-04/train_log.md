# GB-04 training log

GB-01 with the shared build.py imputation instead of native NaN

BASE params: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 4, "verbose": -1}
GRID (folds 2-3): [{'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}, {'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}]
DEFAULT (fold 1): {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}
max rounds 4000, early stopping 150 on slice A log loss; final trees = round(1.2 x best_iter)
Fixed cfg per fold reused from GB-01: {1: {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}, 2: {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}, 3: {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}}


## Fold 1: train [2007] (144,214 rows, 1,379 defaults) -> test 2008 (161,261 rows)
  [GB-04] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=282 A_logloss=0.04871 A_auc=0.8237 (6.9s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=282 final_trees=338
  slices: inner-train 101,261 (946 def), A 21,563 (230 def), B 21,390 (203 def; isotonic eligible=False)
  calibration on B (2-fold firm cross-fit log loss): raw 0.04395, platt 0.04398, isotonic 0.04587; Platt on B a=0.076 b=1.015
  own calibrator: b=1.015, a anchored=0.060 (train rate 0.0096)
  test 2008: AUC 0.8448, log loss 0.05418, mean pd 0.0107 vs DR 0.0128; raw AUC 0.8448 log loss 0.05418; predict 1.09s/100k rows

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 defaults) -> test 2009 (170,386 rows)
  [GB-04] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=388 A_logloss=0.05252 A_auc=0.8546 (6.5s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=388 final_trees=466
  slices: inner-train 144,214 (1379 def), A 81,167 (1028 def), B 80,094 (1036 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.05578, platt 0.05550, isotonic 0.05590; Platt on B a=0.033 b=0.943
  own calibrator: b=0.943, a anchored=-0.201 (train rate 0.0113)
  test 2009: AUC 0.8364, log loss 0.06617, mean pd 0.0119 vs DR 0.0162; raw AUC 0.8364 log loss 0.06621; predict 1.08s/100k rows

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 defaults) -> test 2010 (174,512 rows)
  [GB-04] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=341 A_logloss=0.06737 A_auc=0.8280 (13.7s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=341 final_trees=409
  slices: inner-train 305,475 (3443 def), A 85,773 (1395 def), B 84,613 (1357 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.06510, platt 0.06439, isotonic 0.06466; Platt on B a=0.336 b=0.998
  own calibrator: b=0.998, a anchored=-0.007 (train rate 0.0130)
  test 2010: AUC 0.8302, log loss 0.07139, mean pd 0.0119 vs DR 0.0170; raw AUC 0.8302 log loss 0.07139; predict 0.98s/100k rows

## Pooled / per-year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.83758 | 0.67515 |   0.064148 | 0.01368  |  0.011475 |      0.015379 |              1.3402 |     1.147e-135 |         0.33262 |     0.97439 |
| 2008    | 161261 |       2064 | 0.8448  | 0.6896  |   0.054182 | 0.011347 |  0.01066  |      0.012799 |              1.2007 |     4.8215e-16 |         0.20762 |     0.96866 |
| 2009    | 170386 |       2752 | 0.83637 | 0.67274 |   0.066166 | 0.014163 |  0.011855 |      0.016152 |              1.3624 |     1.8916e-54 |         0.35063 |     1.0062  |
| 2010    | 174512 |       2968 | 0.83022 | 0.66045 |   0.071385 | 0.015365 |  0.011857 |      0.017007 |              1.4343 |     9.1528e-78 |         0.41045 |     0.95183 |
