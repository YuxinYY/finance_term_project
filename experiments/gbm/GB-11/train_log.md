# GB-11 training log

random-forest benchmark (sklearn RandomForest, 300 trees, sqrt features, 50% rows/tree, min_samples_leaf chosen on inner A; one-hot categoricals; D-053 calibration)
feature set: full (26 features)
relaxed (unconstrained) features: ['log_assets', 'equity_ratio', 'st_debt_ratio', 'bank_debt_ratio', 'st_debt_share', 'wc_ratio', 'current_ratio', 'cash_ratio', 'roa_net', 'roa_oper', 'ebitda_margin', 'interest_coverage', 'cfo_to_debt', 'interest_to_assets', 'asset_turnover', 'ar_days', 'ap_days', 'tangible_ratio', 'intangible_ratio', 'extraord_to_assets', 'neg_equity', 'debt_split_missing', 'rev_nonpositive', 'ateco_group', 'legal_group', 'macro_area']
monotone vector: {'log_assets': 0, 'equity_ratio': 0, 'st_debt_ratio': 0, 'bank_debt_ratio': 0, 'st_debt_share': 0, 'wc_ratio': 0, 'current_ratio': 0, 'cash_ratio': 0, 'roa_net': 0, 'roa_oper': 0, 'ebitda_margin': 0, 'interest_coverage': 0, 'cfo_to_debt': 0, 'interest_to_assets': 0, 'asset_turnover': 0, 'ar_days': 0, 'ap_days': 0, 'tangible_ratio': 0, 'intangible_ratio': 0, 'extraord_to_assets': 0, 'neg_equity': 0, 'debt_split_missing': 0, 'rev_nonpositive': 0, 'ateco_group': 0, 'legal_group': 0, 'macro_area': 0}
BASE params: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 4, "verbose": -1}
Lockbox: panel read with fs_year < 2011; max fs_year read = 2010


## Fold 1: train [2007] (144,214 rows) -> test 2008; 3 candidate cfgs
  [GB-11] #0 best_iter=None A_ll=0.04891 A_auc=0.8245 B_ll=0.04420 B_auc=0.8266 (66.9s) cfg={'n_estimators': 300, 'min_samples_leaf': 50, 'max_features': 'sqrt', 'max_samples': 0.5}
  [GB-11] #1 best_iter=None A_ll=0.04923 A_auc=0.8309 B_ll=0.04474 B_auc=0.8252 (55.1s) cfg={'n_estimators': 300, 'min_samples_leaf': 200, 'max_features': 'sqrt', 'max_samples': 0.5}
  [GB-11] #2 best_iter=None A_ll=0.05113 A_auc=0.8182 B_ll=0.04643 B_auc=0.8113 (39.2s) cfg={'n_estimators': 300, 'min_samples_leaf': 1000, 'max_features': 'sqrt', 'max_samples': 0.5}
  CHOSEN #0 {'n_estimators': 300, 'min_samples_leaf': 50, 'max_features': 'sqrt', 'max_samples': 0.5} best_iter=None final_trees=300
  inner: A_ll 0.04891 A_auc 0.8245 | B_ll 0.04420 B_auc 0.8266
  Platt on B b=1.122; in-sample AUC 0.9620
  OUTER test 2008 (scored once): AUC 0.8449 log loss 0.05424 mean pd 0.0107 vs DR 0.0128 (256.8s)

## Fold 2: train [2007, 2008] (305,475 rows) -> test 2009; 3 candidate cfgs
  [GB-11] #0 best_iter=None A_ll=0.05330 A_auc=0.8491 B_ll=0.05562 B_auc=0.8407 (72.8s) cfg={'n_estimators': 300, 'min_samples_leaf': 50, 'max_features': 'sqrt', 'max_samples': 0.5}
  [GB-11] #1 best_iter=None A_ll=0.05397 A_auc=0.8517 B_ll=0.05635 B_auc=0.8402 (56.6s) cfg={'n_estimators': 300, 'min_samples_leaf': 200, 'max_features': 'sqrt', 'max_samples': 0.5}
  [GB-11] #2 best_iter=None A_ll=0.05617 A_auc=0.8438 B_ll=0.05839 B_auc=0.8295 (47.4s) cfg={'n_estimators': 300, 'min_samples_leaf': 1000, 'max_features': 'sqrt', 'max_samples': 0.5}
  CHOSEN #0 {'n_estimators': 300, 'min_samples_leaf': 50, 'max_features': 'sqrt', 'max_samples': 0.5} best_iter=None final_trees=300
  inner: A_ll 0.05330 A_auc 0.8491 | B_ll 0.05562 B_auc 0.8407
  Platt on B b=1.108; in-sample AUC 0.9573
  OUTER test 2009 (scored once): AUC 0.8396 log loss 0.06596 mean pd 0.0118 vs DR 0.0162 (422.0s)

## Fold 3: train [2007, 2008, 2009] (475,861 rows) -> test 2010; 3 candidate cfgs
  [GB-11] #0 best_iter=None A_ll=0.06679 A_auc=0.8340 B_ll=0.06520 B_auc=0.8454 (256.1s) cfg={'n_estimators': 300, 'min_samples_leaf': 50, 'max_features': 'sqrt', 'max_samples': 0.5}
  [GB-11] #1 best_iter=None A_ll=0.06732 A_auc=0.8330 B_ll=0.06583 B_auc=0.8442 (235.3s) cfg={'n_estimators': 300, 'min_samples_leaf': 200, 'max_features': 'sqrt', 'max_samples': 0.5}
  [GB-11] #2 best_iter=None A_ll=0.06912 A_auc=0.8258 B_ll=0.06774 B_auc=0.8343 (170.1s) cfg={'n_estimators': 300, 'min_samples_leaf': 1000, 'max_features': 'sqrt', 'max_samples': 0.5}
  CHOSEN #0 {'n_estimators': 300, 'min_samples_leaf': 50, 'max_features': 'sqrt', 'max_samples': 0.5} best_iter=None final_trees=300
  inner: A_ll 0.06679 A_auc 0.8340 | B_ll 0.06520 B_auc 0.8454
  Platt on B b=1.090; in-sample AUC 0.9527
  OUTER test 2010 (scored once): AUC 0.8335 log loss 0.07088 mean pd 0.0119 vs DR 0.0170 (1133.6s)

## Pooled / per-year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.83945 | 0.67891 |   0.063924 | 0.013673 |  0.01146  |      0.015379 |              1.3419 |    7.6914e-137 |         0.32993 |     0.96914 |
| 2008    | 161261 |       2064 | 0.84489 | 0.68977 |   0.054241 | 0.011403 |  0.010678 |      0.012799 |              1.1986 |    9.255e-16   |         0.1997  |     1.0061  |
| 2009    | 170386 |       2752 | 0.83961 | 0.67921 |   0.065961 | 0.014134 |  0.011758 |      0.016152 |              1.3736 |    3.9559e-57  |         0.35943 |     0.9687  |
| 2010    | 174512 |       2968 | 0.83348 | 0.66697 |   0.070883 | 0.015322 |  0.011891 |      0.017007 |              1.4302 |    1.2036e-76  |         0.40354 |     0.94062 |
