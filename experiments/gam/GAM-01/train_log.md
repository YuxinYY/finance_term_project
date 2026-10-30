# GAM-01 training log

EBM main effects only, monotone constraints from the expected-sign table

BASE: {"max_bins": 128, "max_interaction_bins": 32, "learning_rate": 0.03, "smoothing_rounds": 75, "interaction_smoothing_rounds": 75, "early_stopping_rounds": 100, "max_rounds": 30000, "inner_bags": 0, "validation_size": 0.15, "n_jobs": 8, "random_state": 0}
inner bags 4, final bags 8
GRID (folds 2-3): [{'max_leaves': 2, 'min_samples_leaf': 200}, {'max_leaves': 2, 'min_samples_leaf': 1000}, {'max_leaves': 3, 'min_samples_leaf': 200}, {'max_leaves': 3, 'min_samples_leaf': 1000}]
DEFAULT (fold 1): {'max_leaves': 2, 'min_samples_leaf': 500}


## Fold 1: train [2007] (144,214 rows, 1,379 def) -> test 2008 (161,261)
  [GAM-01] cfg={'max_leaves': 2, 'min_samples_leaf': 500} A_logloss=0.04929 A_auc=0.8132 (36.7s)
  chosen {'max_leaves': 2, 'min_samples_leaf': 500}; Platt on B a=0.076 b=1.022; B LL raw 0.04501 platt 0.04501
  test 2008: AUC 0.8345 LL 0.05508 mean pd 0.0104 vs DR 0.0128; after rule-e shift (tau 0.0117, delta 0.200) LL 0.05479; raw AUC 0.8345 LL 0.05508; predict 0.89s/100k; final fit 59.0s; wall 112.0s

## Fold 2: train [2007, 2008] (305,475 rows, 3,443 def) -> test 2009 (170,386)
  [GAM-01] cfg={'max_leaves': 2, 'min_samples_leaf': 200} A_logloss=0.05324 A_auc=0.8435 (70.4s)
  [GAM-01] cfg={'max_leaves': 2, 'min_samples_leaf': 1000} A_logloss=0.05324 A_auc=0.8435 (71.4s)
