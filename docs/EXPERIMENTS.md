# EXPERIMENT REGISTRY

One row per submitted experiment. Challengers append; the referee fills the verdict columns.

| exp_id | agent | date | description (what changed vs parent) | parent | predictions path | leakage verdict | pooled AUC [95% CI] | pooled log loss [95% CI] | promoted? |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| SC-01 | challenger-scorecard | 2026-10-31 | single best core ratio (roa_net chosen in-window, every fold) -> logit | none | experiments/scorecard/SC-01/oot_predictions.parquet | pending referee (fits only on training window, by construction) | 0.7455 [pending] | 0.0713 [pending] | benchmark floor |
| SC-02 | challenger-scorecard | 2026-10-31 | unpenalised logit on 10 core features + 3 categoricals, winsorised/imputed by feature layer | SC-01 | experiments/scorecard/SC-02/oot_predictions.parquet | pending referee | 0.8027 [pending] | 0.0672 [pending] | transparent logit reference |
| SC-03 | challenger-scorecard | 2026-10-31 | monotone WoE binning (20 quantile bins, PAV, min bin 2%) + logit; points-scaled scorecard | SC-02 | experiments/scorecard/SC-03/oot_predictions.parquet | pending referee | 0.8166 [pending] | 0.0666 [pending] | provisional benchmark to beat |
| SC-04 | challenger-scorecard | 2026-10-31 | Z'-style proxy logit (wc_ratio, equity_ratio, roa_oper, asset_turnover); lineage only | SC-01 | experiments/scorecard/SC-04/oot_predictions.parquet | pending referee | 0.7704 [pending] | 0.0699 [pending] | lineage reference |
| RC-01 | challenger-riskcalc | 2026-10-31 | RiskCalc-style: 10 core ratios → monotone shrunk binned default-rate transforms T_i (probit scale, missing = own rate) → probit + ateco_group indicators → final monotone transform F (PAV, interpolated) | - | experiments/riskcalc/RC-01/oot_predictions.parquet | | | | |
| RC-02 | challenger-riskcalc | 2026-10-31 | RC-01 + debt-split block ratios bank_debt_ratio, ap_days (missing priced inside T_i) — tests H2 | RC-01 | experiments/riskcalc/RC-02/oot_predictions.parquet | | | | |
| RC-03 | challenger-riskcalc | 2026-10-31 | RC-02 + in-window sign rule (T_i with probit coef <= 0 dropped, refit); dropped neg_equity and bank_debt_ratio in all 3 folds → 10 ratios. Best RiskCalc candidate (explain.md, reusable fit_and_save) | RC-02 | experiments/riskcalc/RC-03/oot_predictions.parquet | | | | |
| GB-01 | challenger-gbm | 2026-10-31 | Monotone LightGBM, 26 features + categoricals, native NaN, Platt slope from inner slice B with level anchored to the train rate; grid of 4 configs on the last training year (fold 1: default config) | — | experiments/gbm/GB-01/oot_predictions.parquet | | 0.8375 | 0.06416 | |
| GB-02 | challenger-gbm | 2026-10-31 | GB-01 without monotone constraints (face-validity cost diagnostic; contradicts the expected sign on wc_ratio, bank_debt_ratio, interest_coverage) | GB-01 | experiments/gbm/GB-02/oot_predictions.parquet | | 0.8440 | 0.06332 | not for submission |
| GB-03 | challenger-gbm | 2026-10-31 | GB-01 on the 13 core features only | GB-01 | experiments/gbm/GB-03/oot_predictions.parquet | | 0.8298 | 0.06492 | |
| GB-04 | challenger-gbm | 2026-10-31 | GB-01 with build.py imputation (impute=True) instead of native NaN; GB-01's per-fold configs reused | GB-01 | experiments/gbm/GB-04/oot_predictions.parquet | | 0.8376 | 0.06415 | |
| GB-05 | challenger-gbm | 2026-10-31 | GB-01 with monotone_constraints_method='advanced' | GB-01 | experiments/gbm/GB-05/oot_predictions.parquet | | 0.8377 | 0.06414 | |
| GB-06 | challenger-gbm | 2026-10-31 | Diagnostic: GB-01 with constraints dropped on current_ratio, tangible_ratio, st_debt_share | GB-01 | experiments/gbm/GB-06/oot_predictions.parquet | | 0.8380 | 0.06405 | diagnostic only |
