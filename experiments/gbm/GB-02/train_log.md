# GB-02 training log

GB-01 without monotone constraints (face-validity cost diagnostic)

BASE params: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 4, "verbose": -1}
GRID (folds 2-3): [{'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}, {'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}]
DEFAULT (fold 1): {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}
max rounds 4000, early stopping 150 on slice A log loss; final trees = round(1.2 x best_iter)


## Fold 1: train [2007] (144,214 rows, 1,379 defaults) -> test 2008 (161,261 rows)
  [GB-02] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=380 A_logloss=0.04849 A_auc=0.8279 (3.5s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=380 final_trees=456
  slices: inner-train 101,261 (946 def), A 21,563 (230 def), B 21,390 (203 def; isotonic eligible=False)
  calibration on B (2-fold firm cross-fit log loss): raw 0.04386, platt 0.04386, isotonic 0.04431; Platt on B a=0.202 b=1.045
  own calibrator: b=1.045, a anchored=0.172 (train rate 0.0096)
  test 2008: AUC 0.8485, log loss 0.05374, mean pd 0.0106 vs DR 0.0128; raw AUC 0.8485 log loss 0.05377; predict 1.12s/100k rows

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 defaults) -> test 2009 (170,386 rows)
  [GB-02] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=1180 A_logloss=0.05223 A_auc=0.8564 (14.1s)
  [GB-02] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=1645 A_logloss=0.05216 A_auc=0.8573 (20.5s)
  [GB-02] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=838 A_logloss=0.05215 A_auc=0.8573 (15.2s)
  [GB-02] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=838 A_logloss=0.05210 A_auc=0.8574 (13.4s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=838 final_trees=1006
  slices: inner-train 144,214 (1379 def), A 81,167 (1028 def), B 80,094 (1036 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.05511, platt 0.05484, isotonic 0.05533; Platt on B a=0.202 b=0.988
  own calibrator: b=0.988, a anchored=-0.044 (train rate 0.0113)
  test 2009: AUC 0.8419, log loss 0.06540, mean pd 0.0118 vs DR 0.0162; raw AUC 0.8419 log loss 0.06540; predict 1.69s/100k rows

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 defaults) -> test 2010 (174,512 rows)
  [GB-02] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=1665 A_logloss=0.06681 A_auc=0.8313 (28.7s)
  [GB-02] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=1084 A_logloss=0.06682 A_auc=0.8314 (20.3s)
  [GB-02] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=810 A_logloss=0.06667 A_auc=0.8326 (20.0s)
  [GB-02] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=810 A_logloss=0.06661 A_auc=0.8331 (16.5s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=810 final_trees=972
  slices: inner-train 305,475 (3443 def), A 85,773 (1395 def), B 84,613 (1357 def; isotonic eligible=True)
  calibration on B (2-fold firm cross-fit log loss): raw 0.06418, platt 0.06343, isotonic 0.06370; Platt on B a=0.416 b=1.018
  own calibrator: b=1.018, a anchored=0.063 (train rate 0.0130)
  test 2010: AUC 0.8402, log loss 0.07015, mean pd 0.0119 vs DR 0.0170; raw AUC 0.8402 log loss 0.07011; predict 1.64s/100k rows

## Pooled / per-year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.844   | 0.68801 |   0.063321 | 0.013584 |  0.011423 |      0.015379 |              1.3462 |    1.1729e-139 |         0.33795 |     0.97893 |
| 2008    | 161261 |       2064 | 0.84848 | 0.69696 |   0.053737 | 0.011296 |  0.01059  |      0.012799 |              1.2086 |    4.5378e-17  |         0.21462 |     0.98375 |
| 2009    | 170386 |       2752 | 0.84194 | 0.68387 |   0.065399 | 0.01406  |  0.011764 |      0.016152 |              1.373  |    5.9448e-57  |         0.35997 |     1.0014  |
| 2010    | 174512 |       2968 | 0.84019 | 0.68038 |   0.070149 | 0.015232 |  0.011861 |      0.017007 |              1.4338 |    1.2022e-77  |         0.4109  |     0.95388 |
