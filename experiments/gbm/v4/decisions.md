# Decisions: LightGBM on the v4 temporal_year split (GBv4-xx)

Experiments `LGB-v4-mono` and `LGB-v4-free`, 2026-11-20, challenger-gbm (Claude Code).
Code: `experiments/gbm/v4/train_v4.py`, `analyse_v4.py`, `compare_v4.py`. Numbers are in the files named
in each entry. All entries need human review; none is in `docs/DECISION_LOG.md` yet (the orchestrator
decides whether to copy them).

### GBv4-01 Fresh search, validation years only, one config per variant
- Date / phase / agent: 2026-11-20 / final consolidation / challenger-gbm
- Status: ADOPTED (for these experiments)
- Decision: 13 configs per variant (config 0 = a hand-set conservative default, 12 random draws, seed 20261120,
  from num_leaves {7,15,31}, max_depth {3,4,5} (leaves capped at 2^depth), learning_rate {0.02,0.03,0.05},
  min_child_samples {200,500,1000,2000}, min_sum_hessian_in_leaf {0.5,2,5}, feature_fraction {0.5,0.7,0.9},
  bagging_fraction {0.7,0.8,1.0} (freq 1 if < 1), lambda_l1 {0,1,5}, lambda_l2 {0,5,20}). The same 13 configs for both
  variants. Objective binary, no class weights, no resampling. Early stopping on the round's validation year (AUC,
  patience 200, max 5,000 trees). Choice: highest mean validation AUC over rounds 1-2 (FY2008, FY2009); tie-break mean
  validation log-loss. Test years FY2009/FY2010 scored only after `config.json` was written. Chloe's hyper-parameters,
  Platt step and code were not used.
- Alternatives considered: Optuna (rejected: no time, and with 2 selection years a larger search mostly fits noise);
  early stopping on log-loss (rejected: the validation year's default rate differs from training, so log-loss stopping
  mixes level drift into the tree count; AUC is level-invariant).
- Business reason: one tuning rule for both variants, so the constraint cost is not confounded with tuning effort.
- Evidence: `LGB-v4-mono/selection_ledger.csv`, `LGB-v4-free/selection_ledger.csv`. Chosen: mono config 1
  (31 leaves, depth 5, lr 0.03, min_child_samples 200, min_sum_hessian 2, feature_fraction 0.9, bagging 0.7, l1 1, l2 20),
  mean validation AUC 0.8541; free config 5 (31 leaves, depth 5, lr 0.03, min_child_samples 200, min_sum_hessian 5,
  feature_fraction 0.9, bagging 0.7, l1 5, l2 0), 0.8551. Best iterations mono 370 / 583 / 727, free 202 / 398 / 467
  (rounds 1/2/3; `best_iteration.json`).
- Source(s): assumption (validation-only selection per D-156).
- Risks / what would change our mind: both winners sit at the deep / small-leaf edge of the grid (31 leaves, depth 5,
  min_child_samples 200), so a deeper grid might do slightly better; the top four configs per variant are within
  0.0010 validation AUC, i.e. the choice among them is close to noise. Validation AUC is optimistic (the same year
  set the tree count); quote test-year numbers only.
- Human reviewer (initials) and date:

### GBv4-02 Monotone constraint set for LGB-v4-mono (25 features)
- Date / phase / agent: 2026-11-20 / final consolidation / challenger-gbm
- Status: ADOPTED (for LGB-v4-mono)
- Decision: start from the team table `MONOTONE_DIRECTIONS` in `src/finance_project/model_development/lightgbm_model.py`
  (19 features), drop one, add seven:
  - kept from the team table (18): equity_ratio -, st_debt_share +, cash_ratio -, cash_to_financing_exp_ratio -,
    AR_ratio +, payables_ratio +, wc_ratio -, cur_debt_ratio -, quick_ratio -, roa -, op_margin -, net_prof_margin -,
    int_coverage -, financing_expense_ratio +, operating_cash_flow_ratio -, ebitda_to_debt -, loss_flag +, ateco_sector_te +.
  - REMOVED: lt_debt_to_capital (team +1). debt_lt / (debt_lt + eqty_tot) turns negative when equity is very negative
    (11,636 rows, DR 18.3%; consistency report §3, D-154 (1) open), so a + constraint would push those insolvent firms
    to a low PD. Left unconstrained.
  - ADDED (decided here): size - (experiments/gbm table log_assets -1; consistency report "-");
    tangible_asset_ratio - (experiments/gbm table tangible_ratio -1; collateral; report "could be constrained -1");
    operating_cash_flow_margin -, cash_interest_coverage - (report "-"; same family as op_margin / int_coverage);
    operating_cash_flow_to_debt - (experiments/gbm table cfo_to_debt -1); debt_detail_missing + (experiments/gbm table
    debt_split_missing +1); roe_negative + (see GBv4-03).
  - LEFT UNCONSTRAINED on purpose (ambiguous or sign flips): roe (GBv4-03), debt_to_eqty (U-shaped, sign flips with
    negative equity), bank_debt_conc, goodwill_ratio (field behaves like a profit measure), cur_asst_ratio, asset_turnover,
    the raw euro amounts profit, rev_operating, ebitda, COGS, prof_operations, exp_financing (scale-confounded with size,
    consistency report M8), the data-anomaly flags neg_bs_flag and neg_pl_flag, all 16 zero_* flags (mark /1 cases, no
    economic direction), the 5 legal and 20 region dummies (categorical).
  Method: LightGBM `monotone_constraints_method = intermediate`.
- Alternatives considered: the team table unchanged (rejected: lt_debt_to_capital conflict); constraining the raw euro
  amounts to the report's expected sign (rejected: they proxy size, and the ratio versions already carry the direction).
- Business reason: a credit officer must see every ratio with a clear economic direction move the PD the right way.
- Evidence: `LGB-v4-mono/face_validity.csv`: all 25 constrained features (23 used in the trees, 2 never split on) have 0% of ICE rows moving against the sign
  at any step in both rounds; the face-validity flags in the mono model fall only on unconstrained features (GBv4-05).
- Source(s): team table (code); `reports/features/2026-11-20_feature_consistency_report.md` §3; experiments/gbm/GB-25/gb25_monotone.csv; assumption.
- Risks / what would change our mind: if D-154 (1) is fixed (risky end for a negative denominator), lt_debt_to_capital can be constrained +.
- Human reviewer (initials) and date:

### GBv4-03 roe unconstrained; roe_negative constrained + and described by its real meaning
- Date / phase / agent: 2026-11-20 / final consolidation / challenger-gbm (orchestrator correction, consistency report M2/M3)
- Status: ADOPTED
- Decision: `roe` is unconstrained. Under schema 1.3.0 roe = profit / eqty_tot * 100, so a loss with negative equity gives a
  large POSITIVE roe (distressed firms; max about 48,706 per the consistency report); a "-" constraint would give those
  firms a low PD. `roe_negative` = 1 iff profit < 0 AND eqty_tot < 0 (loss plus negative equity; the name is misleading);
  expected sign +, constrained +. `face_validity.csv` and `drivers_examples.md` describe it as "net loss AND negative equity".
  No run ever used a "-" constraint on roe (both constraints were in the first and only training run), so no rerun was needed.
- Alternatives considered: roe -1 (team docs' generic sign; rejected for the reason above); renaming the column (out of scope: no feature changes).
- Business reason: avoids a wrong-signed explanation for the riskiest firms.
- Evidence: `LGB-v4-mono/config.json` (`monotone_constraints`: roe 0, roe_negative 1); consistency report rows `roe`, `roe_negative`.
- Source(s): `notebooks/features/02_missing_values.ipynb` method I (via the consistency report); assumption.
- Risks / what would change our mind: Desmond confirms a different definition.
- Human reviewer (initials) and date:

### GBv4-04 Level: prior shift to the validation-year rate (PROVISIONAL), no Platt
- Date / phase / agent: 2026-11-20 / final consolidation / challenger-gbm
- Status: OPEN (PROVISIONAL; needs team sign-off)
- Decision: pd_shift = expit(logit(pd_raw) + delta), delta = logit(tau) - logit(training default rate), tau = the round's
  validation-year default rate (delta +0.325 / +0.412 / +0.292 for rounds 1/2/3; `best_iteration.json`).
- Alternatives considered: Platt on the validation year (excluded by the brief; D-150); raw PDs.
- Business reason: pricing needs the level; raw PDs carry the training years' rate.
- Evidence (`LGB-v4-*/metrics_test.csv`): obs/pred on test years, raw -> shift: mono 1.640 -> 1.233 (FY2009),
  1.730 -> 1.203 (FY2010); free 1.628 -> 1.219, 1.721 -> 1.196. Log-loss improves (mono 0.06412 -> 0.06262, 0.06932 -> 0.06727).
  The shift removes about two thirds of the shortfall but both test years are still about 20% under-predicted, because the
  default rate kept rising from the validation year to the test year (FY2008 1.21% -> FY2009 1.58%; FY2009 1.58% -> FY2010 1.65%).
- Source(s): D-072 (R-22 eq. (4)); D-073 for the final tau choice.
- Risks / what would change our mind: the final harness level must come from D-073's tau logic, not this rule; in a falling
  year (FY2011) this rule would over-predict.
- Human reviewer (initials) and date:

### GBv4-05 Keep the monotone variant (LGB-v4-mono) as the GBM entry; LGB-v4-free is a diagnostic
- Date / phase / agent: 2026-11-20 / final consolidation / challenger-gbm
- Status: OPEN (recommendation; team decision)
- Decision: recommend LGB-v4-mono. See `comparison.md` for the bootstrap CIs and violation counts.
- Alternatives considered: LGB-v4-free (higher AUC, but wrong-signed drivers a credit officer would see).
- Business reason: approval needs ranking, but every per-firm reason must have the right direction (CLAUDE.md rule 8).
- Evidence: `comparison.md`, `comparison_bootstrap.csv`, `comparison_violations.json`. Free minus mono AUC
  +0.0022 [+0.0005, +0.0039] (FY2009), +0.0015 [-0.0001, +0.0032] (FY2010); log-loss -0.00025, -0.00024. Free has 2 / 3
  SHAP-direction and 11 / 11 ICE violations on the 25 features mono constrains; mono has 0 on them (its 2 SHAP flags per year
  are the unconstrained raw amounts ebitda and exp_financing).
- Source(s): D-126 (constraints kept when the cost is small); assumption.
- Risks / what would change our mind: if the team weights the out-of-sample AUC points above face validity.
- Human reviewer (initials) and date:
