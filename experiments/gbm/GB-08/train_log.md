# GB-08 training log

minimal defensible monotone constraints (economically ambiguous constraints dropped)
feature set: full (26 features)
relaxed (unconstrained) features: ['interest_coverage', 'cfo_to_debt', 'ebitda_margin']
monotone vector: {'log_assets': -1, 'equity_ratio': -1, 'st_debt_ratio': 1, 'bank_debt_ratio': 1, 'st_debt_share': 1, 'wc_ratio': -1, 'current_ratio': -1, 'cash_ratio': -1, 'roa_net': -1, 'roa_oper': -1, 'ebitda_margin': 0, 'interest_coverage': 0, 'cfo_to_debt': 0, 'interest_to_assets': 1, 'asset_turnover': 0, 'ar_days': 1, 'ap_days': 1, 'tangible_ratio': -1, 'intangible_ratio': 1, 'extraord_to_assets': 0, 'neg_equity': 1, 'debt_split_missing': 1, 'rev_nonpositive': 0, 'ateco_group': 0, 'legal_group': 0, 'macro_area': 0}
BASE params: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 2, "verbose": -1}
Lockbox: panel read with fs_year < 2011; max fs_year read = 2010


## Fold 1: train [2007] (144,214 rows) -> test 2008; 1 candidate cfgs
  [GB-08] #0 best_iter=407 A_ll=0.04853 A_auc=0.8238 B_ll=0.04382 B_auc=0.8329 (9.3s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}
  CHOSEN #0 {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=407 final_trees=488
  inner: A_ll 0.04853 A_auc 0.8238 | B_ll 0.04382 B_auc 0.8329
  Platt on B b=1.015; in-sample AUC 0.8630
  OUTER test 2008 (scored once): AUC 0.8465 log loss 0.05396 mean pd 0.0105 vs DR 0.0128 (49.7s)

## Fold 2: train [2007, 2008] (305,475 rows) -> test 2009; 4 candidate cfgs
  [GB-08] #0 best_iter=953 A_ll=0.05258 A_auc=0.8538 B_ll=0.05566 B_auc=0.8372 (29.9s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}
  [GB-08] #1 best_iter=967 A_ll=0.05252 A_auc=0.8541 B_ll=0.05564 B_auc=0.8379 (29.0s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}
  [GB-08] #2 best_iter=552 A_ll=0.05235 A_auc=0.8553 B_ll=0.05548 B_auc=0.8391 (22.9s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}
  [GB-08] #3 best_iter=684 A_ll=0.05233 A_auc=0.8554 B_ll=0.05544 B_auc=0.8399 (27.3s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}
  CHOSEN #3 {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=684 final_trees=821
  inner: A_ll 0.05233 A_auc 0.8554 | B_ll 0.05544 B_auc 0.8399
  Platt on B b=0.954; in-sample AUC 0.8702
  OUTER test 2009 (scored once): AUC 0.8386 log loss 0.06587 mean pd 0.0118 vs DR 0.0162 (189.4s)

## Fold 3: train [2007, 2008, 2009] (475,861 rows) -> test 2010; 4 candidate cfgs
  [GB-08] #0 best_iter=731 A_ll=0.06738 A_auc=0.8274 B_ll=0.06504 B_auc=0.8446 (39.3s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}
  [GB-08] #1 best_iter=1005 A_ll=0.06738 A_auc=0.8271 B_ll=0.06495 B_auc=0.8458 (48.9s) cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}
  [GB-08] #2 best_iter=462 A_ll=0.06718 A_auc=0.8285 B_ll=0.06477 B_auc=0.8460 (31.4s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}
  [GB-08] #3 best_iter=492 A_ll=0.06710 A_auc=0.8296 B_ll=0.06479 B_auc=0.8465 (30.6s) cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}
  CHOSEN #3 {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=492 final_trees=590
  inner: A_ll 0.06710 A_auc 0.8296 | B_ll 0.06479 B_auc 0.8465
  Platt on B b=1.005; in-sample AUC 0.8577
  OUTER test 2010 (scored once): AUC 0.8354 log loss 0.07089 mean pd 0.0119 vs DR 0.0170 (244.2s)

## Pooled / per-year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.8407  | 0.6814  |   0.063807 | 0.013655 |  0.011423 |      0.015379 |              1.3462 |    1.1683e-139 |         0.33712 |     0.9789  |
| 2008    | 161261 |       2064 | 0.84655 | 0.6931  |   0.053959 | 0.011318 |  0.010545 |      0.012799 |              1.2137 |    9.9346e-18  |         0.21933 |     0.97827 |
| 2009    | 170386 |       2752 | 0.83864 | 0.67729 |   0.065869 | 0.014148 |  0.011797 |      0.016152 |              1.3691 |    5.1129e-56  |         0.35516 |     1.0053  |
| 2010    | 174512 |       2968 | 0.83542 | 0.67084 |   0.070893 | 0.015333 |  0.01187  |      0.017007 |              1.4328 |    2.5152e-77  |         0.40916 |     0.95625 |
