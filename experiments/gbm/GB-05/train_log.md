# GB-05 training log

GB-01 with LightGBM's 'advanced' monotone method (still fully monotone, less restrictive splits)

BASE params: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 4, "verbose": -1}
GRID (folds 2-3): [{'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}, {'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}]
DEFAULT (fold 1): {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}
max rounds 4000, early stopping 150 on slice A log loss; final trees = round(1.2 x best_iter)


## Fold 1: train [2007] (144,214 rows, 1,379 defaults) -> test 2008 (161,261 rows)
  [GB-05] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=273 A_logloss=0.04868 A_auc=0.8243 (6.4s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=273 final_trees=328
  slices: inner-train 101,261 (946 def), A 21,563 (230 def), B 21,390 (203 def; isotonic eligible=False)
  calibration on B (2-fold firm cross-fit log loss): raw 0.04401, platt 0.04404, isotonic 0.04565; Platt on B a=0.088 b=1.019
  own calibrator: b=1.019, a anchored=0.073 (train rate 0.0096)
  test 2008: AUC 0.8447, log loss 0.05423, mean pd 0.0106 vs DR 0.0128; raw AUC 0.8447 log loss 0.05423; predict 1.28s/100k rows

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 defaults) -> test 2009 (170,386 rows)
  [GB-05] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=611 A_logloss=0.05277 A_auc=0.8511 (5.8s)
  [GB-05] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=611 A_logloss=0.05272 A_auc=0.8518 (5.9s)
  [GB-05] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=410 A_logloss=0.05259 A_auc=0.8536 (5.5s)
  [GB-05] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=433 A_logloss=0.05260 A_auc=0.8533 (4.1s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=410 final_trees=492
  slices: inner-train 144,214 (1379 def), A 81,167 (1028 def), B 80,094 (1036 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.05578, platt 0.05548, isotonic 0.05580; Platt on B a=0.035 b=0.942
  own calibrator: b=0.942, a anchored=-0.207 (train rate 0.0113)
  test 2009: AUC 0.8358, log loss 0.06617, mean pd 0.0119 vs DR 0.0162; raw AUC 0.8358 log loss 0.06622; predict 0.79s/100k rows

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 defaults) -> test 2010 (174,512 rows)
  [GB-05] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=524 A_logloss=0.06753 A_auc=0.8265 (5.0s)
  [GB-05] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=524 A_logloss=0.06755 A_auc=0.8266 (4.9s)
  [GB-05] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=341 A_logloss=0.06732 A_auc=0.8278 (4.9s)
  [GB-05] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=342 A_logloss=0.06728 A_auc=0.8282 (4.5s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=342 final_trees=410
  slices: inner-train 305,475 (3443 def), A 85,773 (1395 def), B 84,613 (1357 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.06516, platt 0.06445, isotonic 0.06484; Platt on B a=0.317 b=0.992
  own calibrator: b=0.992, a anchored=-0.027 (train rate 0.0130)
  test 2010: AUC 0.8309, log loss 0.07131, mean pd 0.0119 vs DR 0.0170; raw AUC 0.8309 log loss 0.07133; predict 0.78s/100k rows

## Pooled / per-year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.83766 | 0.67531 |   0.064136 | 0.013676 |  0.011459 |      0.015379 |              1.3421 |    6.2474e-137 |         0.33419 |     0.97672 |
| 2008    | 161261 |       2064 | 0.84466 | 0.68932 |   0.054227 | 0.011349 |  0.01056  |      0.012799 |              1.2121 |    1.5953e-17  |         0.21843 |     0.96691 |
| 2009    | 170386 |       2752 | 0.83577 | 0.67154 |   0.066166 | 0.014156 |  0.011868 |      0.016152 |              1.3609 |    4.3544e-54  |         0.34974 |     1.0058  |
| 2010    | 174512 |       2968 | 0.83089 | 0.66179 |   0.07131  | 0.015357 |  0.01189  |      0.017007 |              1.4304 |    1.1654e-76  |         0.40662 |     0.95833 |
