# GB-23 training log

GB-20 with GB-09's tuned cfg per fold (chosen by phase A on v1 features, inner slice A)

feature_set=full2 (45 features), threads=4
BASE: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 4, "verbose": -1}
GRID (folds 2-3): [{'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}, {'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}]
DEFAULT (fold 1): {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}
monotone: {'log_assets': -1, 'equity_ratio': -1, 'st_debt_ratio': 1, 'bank_debt_ratio': 1, 'st_debt_share': 1, 'wc_ratio': -1, 'current_ratio': -1, 'cash_ratio': -1, 'roa_net': -1, 'roa_oper': -1, 'ebitda_margin': -1, 'interest_coverage': -1, 'cfo_to_debt': -1, 'interest_to_assets': 1, 'asset_turnover': 0, 'ar_days': 1, 'ap_days': 1, 'tangible_ratio': -1, 'intangible_ratio': 1, 'extraord_to_assets': 0, 'neg_equity': 1, 'debt_split_missing': 1, 'rev_nonpositive': 0, 'ateco_group': 0, 'legal_group': 0, 'macro_area': 0, 'operating_margin': -1, 'interest_to_sales': 0, 'payables_to_sales': 1, 'personnel_to_sales': 1, 'equity_ratio_sect_pct': -1, 'roa_net_sect_pct': -1, 'interest_coverage_sect_pct': -1, 'cash_ratio_sect_pct': -1, 'st_debt_ratio_sect_pct': 1, 'tangible_ratio_sect_pct': -1, 'ateco_div_te': 1, 'province_te': 1, 'region_te': 1, 'n_core_imputed': 1, 'bs_gap': 0, 'nonbank_fin_debt': 1, 'ap_lt_ratio': 1, 'fin_fixed_ratio': 0, 'tfr_ratio': 0}
fixed cfg per fold from GB-09: {1: {'bagging_fraction': 0.6487, 'feature_fraction': 0.9069, 'lambda_l1': 0.1, 'lambda_l2': 10.6247, 'learning_rate': 0.0241, 'max_depth': 3, 'min_data_in_leaf': 500, 'min_gain_to_split': 0.0, 'num_leaves': 8}, 2: {'bagging_fraction': 0.6577, 'feature_fraction': 0.6819, 'lambda_l1': 0.1, 'lambda_l2': 0.1304, 'learning_rate': 0.05, 'max_depth': 4, 'min_data_in_leaf': 4000, 'min_gain_to_split': 0.5, 'num_leaves': 16}, 3: {'bagging_fraction': 0.7343, 'feature_fraction': 0.9884, 'lambda_l1': 10.0, 'lambda_l2': 2.2567, 'learning_rate': 0.0156, 'max_depth': 4, 'min_data_in_leaf': 1000, 'min_gain_to_split': 0.5, 'num_leaves': 8}}


## Fold 1: train [2007] (144,214 rows, 1,379 def) -> test 2008
  [GB-23] inner features fitted (11s)
  [GB-23] cfg={'bagging_fraction': 0.6487, 'feature_fraction': 0.9069, 'lambda_l1': 0.1, 'lambda_l2': 10.6247, 'learning_rate': 0.0241, 'max_depth': 3, 'min_data_in_leaf': 500, 'min_gain_to_split': 0.0, 'num_leaves': 8} best_iter=562 A_ll=0.04804 A_auc=0.8341 B_ll=0.04388 (13.6s)
  chosen {'bagging_fraction': 0.6487, 'feature_fraction': 0.9069, 'lambda_l1': 0.1, 'lambda_l2': 10.6247, 'learning_rate': 0.0241, 'max_depth': 3, 'min_data_in_leaf': 500, 'min_gain_to_split': 0.0, 'num_leaves': 8} best_iter=562 final_trees=674
  B (203 def; isotonic eligible=False): cross-fit log loss raw 0.04388 platt 0.04387 isotonic 0.04634; Platt b=0.950; anchored a=-0.183
  test 2008: AUC 0.8476 log loss 0.05400 (raw AUC 0.8476; common-cal log loss 0.05372); 2.9s/100k; wall 62.2s

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 def) -> test 2009
  [GB-23] inner features fitted (12s)
  [GB-23] cfg={'bagging_fraction': 0.6577, 'feature_fraction': 0.6819, 'lambda_l1': 0.1, 'lambda_l2': 0.1304, 'learning_rate': 0.05, 'max_depth': 4, 'min_data_in_leaf': 4000, 'min_gain_to_split': 0.5, 'num_leaves': 16} best_iter=211 A_ll=0.05245 A_auc=0.8539 B_ll=0.05537 (8.8s)
  chosen {'bagging_fraction': 0.6577, 'feature_fraction': 0.6819, 'lambda_l1': 0.1, 'lambda_l2': 0.1304, 'learning_rate': 0.05, 'max_depth': 4, 'min_data_in_leaf': 4000, 'min_gain_to_split': 0.5, 'num_leaves': 16} best_iter=211 final_trees=253
  B (1036 def; isotonic eligible=True): cross-fit log loss raw 0.05537 platt 0.05502 isotonic 0.05545; Platt b=0.939; anchored a=-0.212
  test 2009: AUC 0.8442 log loss 0.06566 (raw AUC 0.8442; common-cal log loss 0.06488); 4.03s/100k; wall 89.9s

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 def) -> test 2010
  [GB-23] inner features fitted (44s)
  [GB-23] cfg={'bagging_fraction': 0.7343, 'feature_fraction': 0.9884, 'lambda_l1': 10.0, 'lambda_l2': 2.2567, 'learning_rate': 0.0156, 'max_depth': 4, 'min_data_in_leaf': 1000, 'min_gain_to_split': 0.5, 'num_leaves': 8} best_iter=1059 A_ll=0.06658 A_auc=0.8386 B_ll=0.06435 (102.6s)
  chosen {'bagging_fraction': 0.7343, 'feature_fraction': 0.9884, 'lambda_l1': 10.0, 'lambda_l2': 2.2567, 'learning_rate': 0.0156, 'max_depth': 4, 'min_data_in_leaf': 1000, 'min_gain_to_split': 0.5, 'num_leaves': 8} best_iter=1059 final_trees=1271
  B (1357 def; isotonic eligible=True): cross-fit log loss raw 0.06435 platt 0.06363 isotonic 0.06398; Platt b=0.972; anchored a=-0.096
  test 2010: AUC 0.8389 log loss 0.07064 (raw AUC 0.8389; common-cal log loss 0.06977); 6.9s/100k; wall 365.9s

## Pooled / per year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.84424 | 0.68848 |   0.063661 | 0.013668 |  0.011371 |      0.015379 |              1.3525 |    9.4844e-144 |         0.3426  |     0.98204 |
| 2008    | 161261 |       2064 | 0.84762 | 0.69523 |   0.053995 | 0.01136  |  0.010452 |      0.012799 |              1.2246 |    3.3604e-19  |         0.22801 |     0.98917 |
| 2009    | 170386 |       2752 | 0.84415 | 0.68831 |   0.065658 | 0.014182 |  0.011814 |      0.016152 |              1.3672 |    1.4822e-55  |         0.3542  |     1.0048  |
| 2010    | 174512 |       2968 | 0.83889 | 0.67778 |   0.070643 | 0.015299 |  0.011787 |      0.017007 |              1.4429 |    3.8609e-80  |         0.4193  |     0.95535 |
