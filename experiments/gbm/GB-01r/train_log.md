# GB-01r training log

GB-01 rebuilt with r2_lib (reproduction check)
feature set: full (26 features)
relaxed (unconstrained) features: []
monotone vector: {'log_assets': -1, 'equity_ratio': -1, 'st_debt_ratio': 1, 'bank_debt_ratio': 1, 'st_debt_share': 1, 'wc_ratio': -1, 'current_ratio': -1, 'cash_ratio': -1, 'roa_net': -1, 'roa_oper': -1, 'ebitda_margin': -1, 'interest_coverage': -1, 'cfo_to_debt': -1, 'interest_to_assets': 1, 'asset_turnover': 0, 'ar_days': 1, 'ap_days': 1, 'tangible_ratio': -1, 'intangible_ratio': 1, 'extraord_to_assets': 0, 'neg_equity': 1, 'debt_split_missing': 1, 'rev_nonpositive': 0, 'ateco_group': 0, 'legal_group': 0, 'macro_area': 0}
BASE params: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 4, "verbose": -1}
Lockbox: panel read with fs_year < 2011; max fs_year read = 2010


## Fold 1: train [2007] (144,214 rows) -> test 2008; 1 candidate cfgs
  [GB-01r] #0 best_iter=282 A_ll=0.04878 A_auc=0.8230 B_ll=0.04398 B_auc=0.8297 (2.6s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}
  CHOSEN #0 {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=282 final_trees=338
  inner: A_ll 0.04878 A_auc 0.8230 | B_ll 0.04398 B_auc 0.8297
  Platt on B b=1.017; in-sample AUC 0.8476
  OUTER test 2008 (scored once): AUC 0.8441 log loss 0.05423 mean pd 0.0106 vs DR 0.0128 (15.0s)

## Fold 2: train [2007, 2008] (305,475 rows) -> test 2009; 4 candidate cfgs
  [GB-01r] #0 best_iter=663 A_ll=0.05277 A_auc=0.8512 B_ll=0.05589 B_auc=0.8353 (6.4s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}
  [GB-01r] #1 best_iter=601 A_ll=0.05272 A_auc=0.8519 B_ll=0.05590 B_auc=0.8351 (3.5s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}
  [GB-01r] #2 best_iter=408 A_ll=0.05259 A_auc=0.8533 B_ll=0.05579 B_auc=0.8364 (3.2s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}
  [GB-01r] #3 best_iter=408 A_ll=0.05261 A_auc=0.8528 B_ll=0.05571 B_auc=0.8372 (3.1s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}
  CHOSEN #2 {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=408 final_trees=490
  inner: A_ll 0.05259 A_auc 0.8533 | B_ll 0.05579 B_auc 0.8364
  Platt on B b=0.943; in-sample AUC 0.8590
  OUTER test 2009 (scored once): AUC 0.8359 log loss 0.06617 mean pd 0.0119 vs DR 0.0162 (34.2s)

## Fold 3: train [2007, 2008, 2009] (475,861 rows) -> test 2010; 4 candidate cfgs
  [GB-01r] #0 best_iter=456 A_ll=0.06754 A_auc=0.8266 B_ll=0.06545 B_auc=0.8406 (4.7s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}
  [GB-01r] #1 best_iter=698 A_ll=0.06756 A_auc=0.8266 B_ll=0.06534 B_auc=0.8428 (6.5s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}
  [GB-01r] #2 best_iter=341 A_ll=0.06733 A_auc=0.8279 B_ll=0.06512 B_auc=0.8431 (4.4s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}
  [GB-01r] #3 best_iter=322 A_ll=0.06728 A_auc=0.8282 B_ll=0.06518 B_auc=0.8433 (4.3s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}
  CHOSEN #3 {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=322 final_trees=386
  inner: A_ll 0.06728 A_auc 0.8282 | B_ll 0.06518 B_auc 0.8433
  Platt on B b=0.998; in-sample AUC 0.8498
  OUTER test 2010 (scored once): AUC 0.8307 log loss 0.07136 mean pd 0.0119 vs DR 0.0170 (40.4s)

## Pooled / per-year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.83748 | 0.67496 |   0.064155 | 0.013676 |  0.011451 |      0.015379 |              1.343  |    1.5815e-137 |         0.33507 |     0.9763  |
| 2008    | 161261 |       2064 | 0.84414 | 0.68828 |   0.054229 | 0.011347 |  0.010564 |      0.012799 |              1.2116 |    1.8031e-17  |         0.21797 |     0.96898 |
| 2009    | 170386 |       2752 | 0.83593 | 0.67186 |   0.066169 | 0.014158 |  0.011855 |      0.016152 |              1.3625 |    1.8837e-54  |         0.35107 |     1.005   |
| 2010    | 174512 |       2968 | 0.83065 | 0.6613  |   0.071361 | 0.015359 |  0.011876 |      0.017007 |              1.4321 |    3.4628e-77  |         0.40824 |     0.95729 |
