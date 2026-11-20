# GB-10 training log

GB-01 + recency weights, half-life chosen on inner slice A from {none,4,2,1,0.5} yrs
feature set: full (26 features)
relaxed (unconstrained) features: []
monotone vector: {'log_assets': -1, 'equity_ratio': -1, 'st_debt_ratio': 1, 'bank_debt_ratio': 1, 'st_debt_share': 1, 'wc_ratio': -1, 'current_ratio': -1, 'cash_ratio': -1, 'roa_net': -1, 'roa_oper': -1, 'ebitda_margin': -1, 'interest_coverage': -1, 'cfo_to_debt': -1, 'interest_to_assets': 1, 'asset_turnover': 0, 'ar_days': 1, 'ap_days': 1, 'tangible_ratio': -1, 'intangible_ratio': 1, 'extraord_to_assets': 0, 'neg_equity': 1, 'debt_split_missing': 1, 'rev_nonpositive': 0, 'ateco_group': 0, 'legal_group': 0, 'macro_area': 0}
BASE params: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 2, "verbose": -1}
Lockbox: panel read with fs_year < 2011; max fs_year read = 2010


## Fold 1: train [2007] (144,214 rows) -> test 2008; 5 candidate cfgs
  [GB-10] #0 best_iter=282 A_ll=0.04878 A_auc=0.8230 B_ll=0.04398 B_auc=0.8297 (3.6s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000, '_hl': None}
  [GB-10] #1 best_iter=282 A_ll=0.04878 A_auc=0.8230 B_ll=0.04398 B_auc=0.8297 (3.5s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000, '_hl': 4.0}
  [GB-10] #2 best_iter=282 A_ll=0.04878 A_auc=0.8230 B_ll=0.04398 B_auc=0.8297 (3.8s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000, '_hl': 2.0}
  [GB-10] #3 best_iter=282 A_ll=0.04878 A_auc=0.8230 B_ll=0.04398 B_auc=0.8297 (3.6s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000, '_hl': 1.0}
  [GB-10] #4 best_iter=282 A_ll=0.04878 A_auc=0.8230 B_ll=0.04398 B_auc=0.8297 (3.6s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000, '_hl': 0.5}
  CHOSEN #0 {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000, '_hl': None} best_iter=282 final_trees=338
  inner: A_ll 0.04878 A_auc 0.8230 | B_ll 0.04398 B_auc 0.8297
  Platt on B b=1.017; in-sample AUC 0.8476
  OUTER test 2008 (scored once): AUC 0.8441 log loss 0.05423 mean pd 0.0106 vs DR 0.0128 (33.4s)

## Fold 2: train [2007, 2008] (305,475 rows) -> test 2009; 20 candidate cfgs
  [GB-10] #0 best_iter=663 A_ll=0.05277 A_auc=0.8512 B_ll=0.05589 B_auc=0.8353 (9.3s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500, '_hl': None}
  [GB-10] #1 best_iter=601 A_ll=0.05272 A_auc=0.8519 B_ll=0.05590 B_auc=0.8351 (8.4s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000, '_hl': None}
  [GB-10] #2 best_iter=408 A_ll=0.05259 A_auc=0.8533 B_ll=0.05579 B_auc=0.8364 (7.4s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500, '_hl': None}
  [GB-10] #3 best_iter=408 A_ll=0.05261 A_auc=0.8528 B_ll=0.05571 B_auc=0.8372 (6.2s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000, '_hl': None}
  [GB-10] #4 best_iter=663 A_ll=0.05277 A_auc=0.8512 B_ll=0.05589 B_auc=0.8353 (8.0s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500, '_hl': 4.0}
  [GB-10] #5 best_iter=601 A_ll=0.05272 A_auc=0.8519 B_ll=0.05590 B_auc=0.8351 (6.9s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000, '_hl': 4.0}
  [GB-10] #6 best_iter=408 A_ll=0.05259 A_auc=0.8533 B_ll=0.05579 B_auc=0.8364 (6.5s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500, '_hl': 4.0}
  [GB-10] #7 best_iter=408 A_ll=0.05261 A_auc=0.8528 B_ll=0.05571 B_auc=0.8372 (6.0s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000, '_hl': 4.0}
  [GB-10] #8 best_iter=663 A_ll=0.05277 A_auc=0.8512 B_ll=0.05589 B_auc=0.8353 (7.8s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500, '_hl': 2.0}
  [GB-10] #9 best_iter=601 A_ll=0.05272 A_auc=0.8519 B_ll=0.05590 B_auc=0.8351 (6.8s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000, '_hl': 2.0}
  [GB-10] #10 best_iter=408 A_ll=0.05259 A_auc=0.8533 B_ll=0.05579 B_auc=0.8364 (12.5s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500, '_hl': 2.0}
  [GB-10] #11 best_iter=408 A_ll=0.05261 A_auc=0.8528 B_ll=0.05571 B_auc=0.8372 (13.4s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000, '_hl': 2.0}
  [GB-10] #12 best_iter=663 A_ll=0.05277 A_auc=0.8512 B_ll=0.05589 B_auc=0.8353 (15.1s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500, '_hl': 1.0}
  [GB-10] #13 best_iter=601 A_ll=0.05272 A_auc=0.8519 B_ll=0.05590 B_auc=0.8351 (14.0s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000, '_hl': 1.0}
  [GB-10] #14 best_iter=408 A_ll=0.05259 A_auc=0.8533 B_ll=0.05579 B_auc=0.8364 (15.6s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500, '_hl': 1.0}
  [GB-10] #15 best_iter=408 A_ll=0.05261 A_auc=0.8528 B_ll=0.05571 B_auc=0.8372 (15.0s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000, '_hl': 1.0}
  [GB-10] #16 best_iter=663 A_ll=0.05277 A_auc=0.8512 B_ll=0.05589 B_auc=0.8353 (19.0s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500, '_hl': 0.5}
  [GB-10] #17 best_iter=601 A_ll=0.05272 A_auc=0.8519 B_ll=0.05590 B_auc=0.8351 (23.0s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000, '_hl': 0.5}
  [GB-10] #18 best_iter=408 A_ll=0.05259 A_auc=0.8533 B_ll=0.05579 B_auc=0.8364 (15.7s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500, '_hl': 0.5}
  [GB-10] #19 best_iter=408 A_ll=0.05261 A_auc=0.8528 B_ll=0.05571 B_auc=0.8372 (21.9s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000, '_hl': 0.5}
  CHOSEN #2 {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500, '_hl': None} best_iter=408 final_trees=490
  inner: A_ll 0.05259 A_auc 0.8533 | B_ll 0.05579 B_auc 0.8364
  Platt on B b=0.943; in-sample AUC 0.8590
  OUTER test 2009 (scored once): AUC 0.8359 log loss 0.06617 mean pd 0.0119 vs DR 0.0162 (291.6s)

## Fold 3: train [2007, 2008, 2009] (475,861 rows) -> test 2010; 20 candidate cfgs
  [GB-10] #0 best_iter=456 A_ll=0.06754 A_auc=0.8266 B_ll=0.06545 B_auc=0.8406 (26.2s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500, '_hl': None}
  [GB-10] #1 best_iter=698 A_ll=0.06756 A_auc=0.8266 B_ll=0.06534 B_auc=0.8428 (43.6s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000, '_hl': None}
  [GB-10] #2 best_iter=341 A_ll=0.06733 A_auc=0.8279 B_ll=0.06512 B_auc=0.8431 (23.8s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500, '_hl': None}
  [GB-10] #3 best_iter=322 A_ll=0.06728 A_auc=0.8282 B_ll=0.06518 B_auc=0.8433 (25.8s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000, '_hl': None}
  [GB-10] #4 best_iter=496 A_ll=0.06753 A_auc=0.8265 B_ll=0.06538 B_auc=0.8411 (29.8s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500, '_hl': 4.0}
  [GB-10] #5 best_iter=496 A_ll=0.06751 A_auc=0.8266 B_ll=0.06544 B_auc=0.8411 (29.0s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000, '_hl': 4.0}
  [GB-10] #6 best_iter=341 A_ll=0.06730 A_auc=0.8277 B_ll=0.06509 B_auc=0.8430 (24.4s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500, '_hl': 4.0}
  [GB-10] #7 best_iter=316 A_ll=0.06727 A_auc=0.8284 B_ll=0.06520 B_auc=0.8427 (23.2s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000, '_hl': 4.0}
  [GB-10] #8 best_iter=496 A_ll=0.06747 A_auc=0.8263 B_ll=0.06531 B_auc=0.8415 (27.3s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500, '_hl': 2.0}
  [GB-10] #9 best_iter=497 A_ll=0.06752 A_auc=0.8262 B_ll=0.06539 B_auc=0.8413 (27.2s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000, '_hl': 2.0}
  [GB-10] #10 best_iter=341 A_ll=0.06729 A_auc=0.8278 B_ll=0.06507 B_auc=0.8429 (31.2s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500, '_hl': 2.0}
  [GB-10] #11 best_iter=322 A_ll=0.06733 A_auc=0.8272 B_ll=0.06522 B_auc=0.8421 (22.2s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000, '_hl': 2.0}
  [GB-10] #12 best_iter=497 A_ll=0.06747 A_auc=0.8260 B_ll=0.06534 B_auc=0.8409 (23.0s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500, '_hl': 1.0}
  [GB-10] #13 best_iter=497 A_ll=0.06748 A_auc=0.8261 B_ll=0.06539 B_auc=0.8408 (23.7s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000, '_hl': 1.0}
  [GB-10] #14 best_iter=322 A_ll=0.06727 A_auc=0.8271 B_ll=0.06509 B_auc=0.8421 (20.1s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500, '_hl': 1.0}
  [GB-10] #15 best_iter=298 A_ll=0.06726 A_auc=0.8274 B_ll=0.06518 B_auc=0.8419 (20.0s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000, '_hl': 1.0}
  [GB-10] #16 best_iter=496 A_ll=0.06742 A_auc=0.8253 B_ll=0.06531 B_auc=0.8402 (22.2s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500, '_hl': 0.5}
  [GB-10] #17 best_iter=496 A_ll=0.06747 A_auc=0.8252 B_ll=0.06542 B_auc=0.8397 (23.8s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000, '_hl': 0.5}
  [GB-10] #18 best_iter=284 A_ll=0.06720 A_auc=0.8268 B_ll=0.06505 B_auc=0.8416 (23.1s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500, '_hl': 0.5}
  [GB-10] #19 best_iter=287 A_ll=0.06716 A_auc=0.8269 B_ll=0.06513 B_auc=0.8414 (21.8s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000, '_hl': 0.5}
  CHOSEN #19 {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000, '_hl': 0.5} best_iter=287 final_trees=344
  inner: A_ll 0.06716 A_auc 0.8269 | B_ll 0.06513 B_auc 0.8414
  Platt on B b=0.992; in-sample AUC 0.8457
  OUTER test 2010 (scored once): AUC 0.8311 log loss 0.07137 mean pd 0.0118 vs DR 0.0170 (603.7s)

## Pooled / per-year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.83767 | 0.67533 |   0.064157 | 0.013676 |  0.011415 |      0.015379 |              1.3473 |    2.8115e-140 |         0.33879 |     0.97651 |
| 2008    | 161261 |       2064 | 0.84414 | 0.68828 |   0.054229 | 0.011347 |  0.010564 |      0.012799 |              1.2116 |    1.8031e-17  |         0.21797 |     0.96898 |
| 2009    | 170386 |       2752 | 0.83593 | 0.67186 |   0.066169 | 0.014158 |  0.011855 |      0.016152 |              1.3625 |    1.8837e-54  |         0.35107 |     1.005   |
| 2010    | 174512 |       2968 | 0.83113 | 0.66226 |   0.071367 | 0.015359 |  0.011771 |      0.017007 |              1.4448 |    1.0499e-80  |         0.41874 |     0.95873 |
