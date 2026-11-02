# GB-20 training log

GB-01 settings (grid, inner scheme, Platt-slope calibration, constraints from expected_signs; sign 0 unconstrained) on feature set full2 (45 features)

feature_set=full2 (45 features), threads=4
BASE: {"objective": "binary", "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "min_sum_hessian_in_leaf": 1.0, "max_bin": 255, "cat_smooth": 50.0, "cat_l2": 10.0, "min_data_per_group": 500, "max_cat_threshold": 16, "monotone_constraints_method": "intermediate", "monotone_penalty": 0.0, "seed": 0, "deterministic": true, "force_col_wise": true, "num_threads": 4, "verbose": -1}
GRID (folds 2-3): [{'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500}, {'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500}, {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000}]
DEFAULT (fold 1): {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000}
monotone: {'log_assets': -1, 'equity_ratio': -1, 'st_debt_ratio': 1, 'bank_debt_ratio': 1, 'st_debt_share': 1, 'wc_ratio': -1, 'current_ratio': -1, 'cash_ratio': -1, 'roa_net': -1, 'roa_oper': -1, 'ebitda_margin': -1, 'interest_coverage': -1, 'cfo_to_debt': -1, 'interest_to_assets': 1, 'asset_turnover': 0, 'ar_days': 1, 'ap_days': 1, 'tangible_ratio': -1, 'intangible_ratio': 1, 'extraord_to_assets': 0, 'neg_equity': 1, 'debt_split_missing': 1, 'rev_nonpositive': 0, 'ateco_group': 0, 'legal_group': 0, 'macro_area': 0, 'operating_margin': -1, 'interest_to_sales': 0, 'payables_to_sales': 1, 'personnel_to_sales': 1, 'equity_ratio_sect_pct': -1, 'roa_net_sect_pct': -1, 'interest_coverage_sect_pct': -1, 'cash_ratio_sect_pct': -1, 'st_debt_ratio_sect_pct': 1, 'tangible_ratio_sect_pct': -1, 'ateco_div_te': 1, 'province_te': 1, 'region_te': 1, 'n_core_imputed': 1, 'bs_gap': 0, 'nonbank_fin_debt': 1, 'ap_lt_ratio': 1, 'fin_fixed_ratio': 0, 'tfr_ratio': 0}


## Fold 1: train [2007] (144,214 rows, 1,379 def) -> test 2008
  [GB-20] inner features fitted (16s)
  [GB-20] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=370 A_ll=0.04804 A_auc=0.8323 B_ll=0.04409 (21.5s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 1000} best_iter=370 final_trees=444
  B (203 def; isotonic eligible=False): cross-fit log loss raw 0.04409 platt 0.04410 isotonic 0.04700; Platt b=0.953; anchored a=-0.174
  test 2008: AUC 0.8476 log loss 0.05405 (raw AUC 0.8476; common-cal log loss 0.05377); 4.74s/100k; wall 100.7s

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 def) -> test 2009
  [GB-20] inner features fitted (21s)
  [GB-20] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=676 A_ll=0.05261 A_auc=0.8519 B_ll=0.05582 (19.9s)
  [GB-20] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=728 A_ll=0.05259 A_auc=0.8518 B_ll=0.05570 (19.6s)
  [GB-20] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=602 A_ll=0.05245 A_auc=0.8533 B_ll=0.05562 (24.8s)
  [GB-20] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=482 A_ll=0.05249 A_auc=0.8537 B_ll=0.05559 (23.8s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=602 final_trees=722
  B (1036 def; isotonic eligible=True): cross-fit log loss raw 0.05562 platt 0.05522 isotonic 0.05585; Platt b=0.919; anchored a=-0.289
  test 2009: AUC 0.8445 log loss 0.06551 (raw AUC 0.8445; common-cal log loss 0.06475); 5.32s/100k; wall 220.3s

## Fold 3: train [2007, 2008, 2009] (475,861 rows, 6,195 def) -> test 2010
  [GB-20] inner features fitted (37s)
  [GB-20] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 500} best_iter=817 A_ll=0.06688 A_auc=0.8349 B_ll=0.06476 (45.5s)
  [GB-20] cfg={'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000} best_iter=872 A_ll=0.06690 A_auc=0.8346 B_ll=0.06476 (49.2s)
  [GB-20] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 500} best_iter=659 A_ll=0.06672 A_auc=0.8358 B_ll=0.06440 (51.6s)
  [GB-20] cfg={'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=614 A_ll=0.06668 A_auc=0.8365 B_ll=0.06453 (49.8s)
  chosen {'num_leaves': 8, 'max_depth': 3, 'min_data_in_leaf': 2000} best_iter=614 final_trees=737
  B (1357 def; isotonic eligible=True): cross-fit log loss raw 0.06453 platt 0.06380 isotonic 0.06414; Platt b=0.961; anchored a=-0.133
  test 2010: AUC 0.8369 log loss 0.07077 (raw AUC 0.8369; common-cal log loss 0.06991); 5.29s/100k; wall 387.7s

## Pooled / per year (own calibration)
| group   |      n |   defaults |     auc |      ar |   log_loss |    brier |   mean_pd |   observed_dr |   ratio_obs_to_pred |   binom_pvalue |   cal_intercept |   cal_slope |
|:--------|-------:|-----------:|--------:|--------:|-----------:|---------:|----------:|--------------:|--------------------:|---------------:|----------------:|------------:|
| ALL     | 506159 |       7784 | 0.84353 | 0.68705 |   0.063673 | 0.013665 |  0.011384 |      0.015379 |              1.3509 |    1.0751e-142 |         0.34019 |     0.99573 |
| 2008    | 161261 |       2064 | 0.84755 | 0.69511 |   0.054051 | 0.011376 |  0.010386 |      0.012799 |              1.2323 |    2.7028e-20  |         0.23313 |     1.0165  |
| 2009    | 170386 |       2752 | 0.84451 | 0.68902 |   0.065511 | 0.014145 |  0.011875 |      0.016152 |              1.3601 |    6.6416e-54  |         0.34891 |     1.0188  |
| 2010    | 174512 |       2968 | 0.83688 | 0.67377 |   0.070769 | 0.01531  |  0.011827 |      0.017007 |              1.438  |    8.9408e-79  |         0.41406 |     0.95679 |
