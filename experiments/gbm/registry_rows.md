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

Round 2 phase A rows (same columns; AUC and log loss are point values plus paired difference vs GB-01, from comparison.md; leakage verdict pending referee):

| GB-01r | challenger-gbm | 2026-11-02 | Reproduction of GB-01 through r2_lib (max PD diff 0.0) | GB-01 | experiments/gbm/GB-01r/oot_predictions.parquet | pending referee | 0.8375 | 0.06416 | no: check only |
| GB-07 | challenger-gbm | 2026-11-02 | Overfitting diagnostics for GB-01 with GB-02 as contrast; no predictions | GB-01 | experiments/gbm/GB-07/summary.md | n/a | n/a | n/a | no: diagnostic |
| GB-08 | challenger-gbm | 2026-11-02 | Constraints dropped on interest_coverage, cfo_to_debt, ebitda_margin; set informed by inner splits of later folds | GB-01 | experiments/gbm/GB-08/oot_predictions.parquet | pending referee; 2008-09 not strictly clean (D-100) | 0.8407; dAUC +0.0032 [+0.0024, +0.0042] | 0.06381; dLL -0.00035 [-0.00043, -0.00027] | no: fails face validity (D-103) |
| GB-08b | challenger-gbm | 2026-11-02 | Constraint dropped on ebitda_margin only; conceived after GB-08's outer score | GB-01 | experiments/gbm/GB-08b/oot_predictions.parquet | pending referee; post hoc (D-100) | 0.8390; dAUC +0.0015 [+0.0009, +0.0021] | 0.06400; dLL -0.00016 [-0.00019, -0.00011] | open: human decision, agent recommends against (D-103) |
| GB-09 | challenger-gbm | 2026-11-02 | Random search over 37 configs on inner A, constraints kept | GB-01 | experiments/gbm/GB-09/oot_predictions.parquet | pending referee | 0.8384; dAUC +0.0009 [+0.0004, +0.0014] | 0.06410; dLL -0.00006 [-0.00011, -0.00000] | no: gain almost only 2010 (D-104) |
| GB-10 | challenger-gbm | 2026-11-02 | Recency weights, half-life chosen on inner A; only fold 3 can differ | GB-01 | experiments/gbm/GB-10/oot_predictions.parquet | pending referee | 0.8377; dAUC +0.0002 [-0.0003, +0.0006] | 0.06416; dLL +0.000002 [-0.00003, +0.00004] | no: null (D-105) |
| GB-11 | challenger-gbm | 2026-11-02 | sklearn random-forest benchmark, unconstrained, D-053 calibration | GB-01 | experiments/gbm/GB-11/oot_predictions.parquet | pending referee; chosen after the LightGBM RF's outer score | 0.8395; dAUC +0.0020 [+0.0001, +0.0038] | 0.06392; dLL -0.00023 [-0.00039, -0.00007] | no: benchmark only (D-106) |
| GB-11lgbrf | challenger-gbm | 2026-11-02 | LightGBM boosting='rf' benchmark attempt, rejected implementation | GB-01 | experiments/gbm/GB-11lgbrf/oot_predictions.parquet | pending referee | 0.7896; dAUC -0.0479 [-0.0516, -0.0445] | 0.06920; dLL +0.00504 [+0.00478, +0.00529] | no: broken probabilities (D-106) |
