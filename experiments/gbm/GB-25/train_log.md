# GB-25 training log

GB-20 with interest_to_sales constrained to +1 (R-33 sign; pre-specified fix of D-125); all else identical

feature_set=full2 (45 features), threads=4
BASE: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 4, "verbose": -1}
GRID (folds 2-3): [{'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}, {'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}]
DEFAULT (fold 1): {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}
monotone: {'log_assets': -1, 'equity_ratio': -1, 'st_debt_ratio': 1, 'bank_debt_ratio': 1, 'st_debt_share': 1, 'wc_ratio': -1, 'current_ratio': -1, 'cash_ratio': -1, 'roa_net': -1, 'roa_oper': -1, 'ebitda_margin': -1, 'interest_coverage': -1, 'cfo_to_debt': -1, 'interest_to_assets': 1, 'asset_turnover': 0, 'ar_days': 1, 'ap_days': 1, 'tangible_ratio': -1, 'intangible_ratio': 1, 'extraord_to_assets': 0, 'neg_equity': 1, 'debt_split_missing': 1, 'rev_nonpositive': 0, 'ateco_group': 0, 'legal_group': 0, 'macro_area': 0, 'operating_margin': -1, 'interest_to_sales': 1, 'payables_to_sales': 1, 'personnel_to_sales': 1, 'equity_ratio_sect_pct': -1, 'roa_net_sect_pct': -1, 'interest_coverage_sect_pct': -1, 'cash_ratio_sect_pct': -1, 'st_debt_ratio_sect_pct': 1, 'tangible_ratio_sect_pct': -1, 'ateco_div_te': 1, 'province_te': 1, 'region_te': 1, 'n_core_imputed': 1, 'bs_gap': 0, 'nonbank_fin_debt': 1, 'ap_lt_ratio': 1, 'fin_fixed_ratio': 0, 'tfr_ratio': 0}


## Fold 1: train [2007] (144,214 rows, 1,379 def) -> test 2008
  [GB-25] inner features fitted (6s)
  [GB-25] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=384 A_ll=0.04806 A_auc=0.8322 B_ll=0.04404 (4.3s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=384 final_trees=461
  B (203 def; isotonic eligible=False): cross-fit log loss raw 0.04404 platt 0.04405 isotonic 0.04709; Platt b=0.951; anchored a=-0.178
  test 2008: AUC 0.8468 log loss 0.05411 (raw AUC 0.8468; common-cal log loss 0.05383); 2.16s/100k; wall 36.0s

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 def) -> test 2009
  [GB-25] inner features fitted (10s)
  [GB-25] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=798 A_ll=0.05259 A_auc=0.8526 B_ll=0.05589 (10.5s)
  [GB-25] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=798 A_ll=0.05261 A_auc=0.8521 B_ll=0.05584 (10.2s)
  [GB-25] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=482 A_ll=0.05249 A_auc=0.8536 B_ll=0.05579 (9.0s)
  [GB-25] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=480 A_ll=0.05249 A_auc=0.8540 B_ll=0.05570 (8.5s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=480 final_trees=576
  B (1036 def; isotonic eligible=True): cross-fit log loss raw 0.05570 platt 0.05534 isotonic 0.05580; Platt b=0.931; anchored a=-0.244
  test 2009: AUC 0.8441 log loss 0.06563 (raw AUC 0.8441; common-cal log loss 0.06484); 2.49s/100k; wall 92.7s

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 def) -> test 2010
  [GB-25] inner features fitted (19s)
  [GB-25] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=817 A_ll=0.06695 A_auc=0.8346 B_ll=0.06474 (18.3s)
  [GB-25] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=857 A_ll=0.06694 A_auc=0.8348 B_ll=0.06475 (20.6s)
  [GB-25] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=452 A_ll=0.06678 A_auc=0.8350 B_ll=0.06449 (14.5s)
  [GB-25] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=457 A_ll=0.06673 A_auc=0.8362 B_ll=0.06461 (14.4s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=457 final_trees=548
  B (1357 def; isotonic eligible=True): cross-fit log loss raw 0.06461 platt 0.06388 isotonic 0.06424; Platt b=0.977; anchored a=-0.080
  test 2010: AUC 0.8348 log loss 0.07098 (raw AUC 0.8348; common-cal log loss 0.07010); 2.48s/100k; wall 150.4s

## Pooled / per year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.84239 | 0.68479 |   0.063804 | 0.013674 |  0.011348 |      0.015379 |              1.3551 |    1.6255e-145 |         0.34395 |     0.99091 |
| 2008    | 161261 |       2064 | 0.84683 | 0.69366 |   0.054107 | 0.011375 |  0.010417 |      0.012799 |              1.2286 |    8.5777e-20  |         0.23033 |     1.0052  |
| 2009    | 170386 |       2752 | 0.84405 | 0.6881  |   0.065634 | 0.014161 |  0.011781 |      0.016152 |              1.371  |    1.7511e-56  |         0.35767 |     1.0144  |
| 2010    | 174512 |       2968 | 0.8348  | 0.6696  |   0.070977 | 0.015322 |  0.011787 |      0.017007 |              1.4429 |    3.8078e-80  |         0.41816 |     0.95738 |
