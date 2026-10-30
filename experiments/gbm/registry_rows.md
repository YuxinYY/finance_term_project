Rows for docs/EXPERIMENTS.md (orchestrator merges). Point metrics pooled over 2008-2010 (own calibration),
from experiments/gbm/comparison.md. CIs columns are left for the referee; the GBM agent's bootstrap
(dX vs GB-01) is in comparison.md.

| exp_id | agent | date | description (what changed vs parent) | parent | predictions path | leakage verdict | pooled AUC [95% CI] | pooled log loss [95% CI] | promoted? |
|---|---|---|---|---|---|---|---|---|---|
| GB-01 | challenger-gbm | 2026-10-31 | Monotone LightGBM, 26 features + categoricals, native NaN, Platt slope from inner slice B with level anchored to the train rate; grid of 4 configs on the last training year (fold 1: default config) | — | experiments/gbm/GB-01/oot_predictions.parquet | | 0.8375 | 0.06416 | |
| GB-02 | challenger-gbm | 2026-10-31 | GB-01 without monotone constraints (face-validity cost diagnostic; contradicts the expected sign on wc_ratio, bank_debt_ratio, interest_coverage) | GB-01 | experiments/gbm/GB-02/oot_predictions.parquet | | 0.8440 | 0.06332 | not for submission |
| GB-03 | challenger-gbm | 2026-10-31 | GB-01 on the 13 core features only | GB-01 | experiments/gbm/GB-03/oot_predictions.parquet | | 0.8298 | 0.06492 | |
| GB-04 | challenger-gbm | 2026-10-31 | GB-01 with build.py imputation (impute=True) instead of native NaN; GB-01's per-fold configs reused | GB-01 | experiments/gbm/GB-04/oot_predictions.parquet | | 0.8376 | 0.06415 | |
| GB-05 | challenger-gbm | 2026-10-31 | GB-01 with monotone_constraints_method='advanced' | GB-01 | experiments/gbm/GB-05/oot_predictions.parquet | | 0.8377 | 0.06414 | |
| GB-06 | challenger-gbm | 2026-10-31 | Diagnostic: GB-01 with constraints dropped on current_ratio, tangible_ratio, st_debt_share | GB-01 | experiments/gbm/GB-06/oot_predictions.parquet | | 0.8380 | 0.06405 | diagnostic only |
