# Tournament scorecard, round 1 (referee, 2026-10-31)

Scope: 13 experiments (SC-01..SC-04, RC-01..RC-03, GB-01..GB-06) on the frozen walk-forward folds
(`data/splits/walkforward.json` v1: test years 2008, 2009, 2010; 506,159 test rows; 7,784 defaults).
Label: panel `y` (start 6 months, 12-month horizon, D-002/D-009).

Every number below is produced by code and read from a file:
`reports/tournament/score_r1.py` (metrics, bootstrap, face validity, figures) writes `r1_*.csv`;
`reports/tournament/build_scorecard_r1.py` renders the tables; `reports/tournament/repro_check_r1.py`
refits fold 3 of SC-03, RC-03 and GB-01. All metrics come from `src/common/metrics.py`.

Conventions. AR = 2·AUC − 1 carries the same information as AUC (it is the CAP-curve view of the
same ranking; we report AUC as the headline). Log loss is the primary probability-accuracy metric
(D-006). Brier is reported as *overall probability accuracy*, not as a calibration measure (R-16
p. 32). Calibration = mean PD vs observed default rate (obs/pred), and the logistic recalibration
intercept/slope (perfect: 0 / 1). Decile tables with Clopper-Pearson bands are in
`r1_deciles_pooled.csv` and `fig/r1_calibration_deciles.png` (diagnostic only).

## 1. Decision rule (published with every scorecard)

A challenger replaces the incumbent only if:
1. its leakage / process verdict is CLEAN;
2. the 95% CI of Δlog-loss excludes zero in its favour, OR (the ΔAUC CI excludes zero in its favour
   AND log loss is not worse);
3. it has no more face-validity violations than the incumbent (or each violation is justified);
4. the per-year results show no year in which it is clearly worse (no hidden fragility).

Otherwise the simpler / more explainable model stays (statistical ties go to transparency).
When several challengers pass, the one that passes the same rule against every other passing
challenger is promoted; a passing challenger that is a statistical tie with a simpler one loses.

Round-1 specifics: incumbent at the start = **SC-03** (WoE scorecard, D-031). Bootstrap = paired,
firm-clustered (whole firms resampled with replacement, so a firm's 2008-2010 rows stay together),
**200 draws** (D-007), seed 0, the same draws for every model; percentile CIs. Because of D-062
(below), condition 4 is also checked on fold 3 (test 2010) alone, the only test year no design
choice has seen.

## 2. Result in one paragraph ("what changed and why")

**GB-01 (monotone-constrained LightGBM, 26 features, native NaN) replaces SC-03 as incumbent.**
Against SC-03 it is better on both power and probability accuracy with CIs far from zero (T4),
in every test year and for both seen and new firms (T5), including the clean 2010 fold; it has
zero face-validity violations (T8), the same as SC-03. It also beats the best transparent challenger
RC-03 in every year (T6, T7), so this is not a tie that should go to transparency. The unconstrained
twin GB-02 is better again but contradicts the expected sign on `wc_ratio` and `interest_coverage`
(and is borderline on `bank_debt_ratio`), so it fails condition 3. The RiskCalc models beat SC-03 on
log loss, but RC-01/RC-02 carry wrong-signed transforms (neg_equity; bank_debt_ratio) and RC-03's
fix was designed after seeing outer results (CONCERN, D-063); RC-03 is kept as the transparent
fallback and goes to calibration with GB-01. Every model under-predicts the level by about a third
(obs/pred 1.34-1.42 pooled, rising each year); a common, leakage-free logit shift (L1) and an oracle
per-year shift (L2, diagnostic) leave the ranking unchanged, so the comparison is not an artefact of
shared level error. The level itself is the calibration-specialist's decision.

## 3. Figures

- `fig/r1_roc_cap.png`: ROC and CAP overlays (pooled) for SC-02, SC-03, RC-01, RC-03, GB-01, GB-02.
- `fig/r1_calibration_deciles.png`: observed vs predicted default rate by PD decile and test year.
- `fig/r1_by_year.png`: AUC and log loss by test year, all 13 models.

## 4. Contract and reproduction checks

All 13 files pass the contract (one PD per test row and fold, all in (0,1), no NaN, no duplicates,
row ids identical to `walkforward_test_rows.parquet`; T11). The referee refitted fold 3 of SC-03,
RC-03 and GB-01 from the challengers' current code on the fold-3 training rows only, and predicted
on a test frame with every label column (`y*`, `status*`, `def_date`, `label_status`, `win_*`)
removed: the submitted fold-3 PDs are reproduced exactly (max |diff| = 0; T15).

{{TABLES_A}}

## 5. Level adjustment used for comparison (diagnostic, not a level choice)

- **L1, leakage-free**: one common logit shift per fold, δ_k = logit(default rate of the last training
  year) − logit(default rate of the whole training window), from training labels only, applied to
  every model (T14: fold 1 δ = 0 because it has one training year; fold 2 +0.129; fold 3 +0.219).
  It moves each model toward the most recent in-window year. It is a comparison device; it does
  not pick a central tendency.
- **L2, oracle (uses test labels)**: each model's own calibration-in-the-large intercept per test
  year, i.e. the log loss each model would have if its level were exactly right. This bounds what
  any level fix can achieve and isolates shape/ranking; it is not a model result.

Under both, the pooled order of the face-valid models is unchanged (T1) and every Δlog-loss CI
keeps its sign (T4, T6 column "dLogLoss L1"). The level gap is shared: obs/pred grows from
about 1.2 (2008) to about 1.4-1.5 (2010) for every serious candidate (SC-02/03, RC-x, GB-x; the weak SC-01/SC-04 benchmarks reach about 1.3; T2). Calibration slopes are close to 1
pooled (0.98-1.02 for the main candidates), so the miss is a level shift, not a shape problem; the
exception is new firms, where the main candidates (SC-03, RC-x, GB-x) have slopes 0.84-0.91 (PDs too spread out; T3).

## 6. Leakage and process verdicts

Verdict scale used by the caller: CLEAN / CONCERN / FAIL (= the agent definition's CLEAN /
SUSPECT / LEAKING). Code paths read: `experiments/scorecard/{run_all,scorecard_lib}.py`,
`experiments/riskcalc/{run,riskcalc,inner_validation}.py`, `experiments/gbm/{run_experiment,gbm_lib,compare}.py`,
`src/features/{build,core_selection,univariate}.py`, `src/common/{metrics,labels}.py`.

**Common to all 13 experiments (shared CONCERN, D-062).** Every fold fit is correctly confined to
the fold's training rows: `B.fit` (winsor bounds, sector medians, debt-share imputation, category
pooling), WoE bins, RiskCalc T_i / probit / F, GBM inner split, early stopping, grid choice,
Platt slope and level anchor are all computed from the frame `fs_year <= t, label_status == ok`
(asserted in each runner, confirmed by the fold-3 refits). No `def_date`, label or future-year field
is a feature; no firm-history feature exists (D-020). **However, the feature set itself
(`docs/FEATURE_SPEC.md`: the core set, which ratios exist, the `sign = none` decisions for
asset_turnover and extraord_to_assets) was chosen by the feature-architect on fs_year 2007-2009,
including an inner split that scored 2009 (`src/features/core_selection.py`).** 2008 and 2009 are
the test years of folds 1 and 2. That is feature selection using test-fold labels for folds 1-2. It
is common to incumbent and challengers (all use the same library), so it biases the paired
comparisons much less than the absolute levels, but fold-1/2 AUCs are somewhat optimistic for
everyone. Fold 3 (test 2010) is clean of it. The promotion decision below holds on 2010 alone.

| exp | verdict | reasons |
|---|---|---|
| SC-01 | CLEAN | Feature choice (max in-window |AUC−0.5|, roa_net every fold) inside the fold; unpenalised logit; nothing tuned. |
| SC-02 | CLEAN | Unpenalised logit on the fixed core set; no tuning. |
| SC-03 | CLEAN | 20 quantile bins, PAV, min bin 2%: fixed settings, identical in every fold, no inner or outer selection; bins/WoE/betas from training rows (refit reproduced exactly). |
| SC-04 | CLEAN | Fixed 4-ratio logit; no tuning. |
| RC-01 | CLEAN | Bin rule, prior, PAV, F settings are code defaults used unchanged in every fold. The inner validation (train 2007-08, validate 2009) was run *after* RC-01..03 outer results existed (file times 00:00-00:02 vs 23:57-23:59) and changed no setting, so it did not select anything; it must not be quoted as tuning evidence for folds 1-2 (2009 is fold 2's test year). |
| RC-02 | CLEAN | As RC-01; the two added ratios were declared as an H2 test. |
| RC-03 | CONCERN | The sign-drop rule runs inside each fold on training coefficients only, but the *rule itself* was added after RC-01/RC-02 outer metrics and coefficients were seen (`run.py` modified 23:59:03, after RC-02 ran at 23:58). It is a face-validity rule, not a performance search, and its measured effect vs RC-02 is nil (T6: dAUC −0.0002 [−0.0004, +0.0001]), so the optimism is small; but RC-03's numbers are post hoc and the RC-03 vs RC-01/RC-02 differences are not evidence. Report it as "RC-02 with a post-hoc sign rule". |
| GB-01 | CLEAN | Folds 2-3: time-ordered inner split (inner-train = earlier years; last training year split by firm into A = early stopping + grid, B = Platt slope); fold 1: random 70/15/15 firm split inside 2007 and the pre-declared DEFAULT config (a deviation from "time-ordered" in PLAYBOOK, unavoidable with one training year, no test data involved). Final trees = round(1.2 × best_iter): an untested heuristic, but fixed in advance and applied in-window. Platt slope from B, level anchored to the training-window rate: in-window. Variant results (GB-02..06) did not change GB-01, which was the first-declared spec. |
| GB-02 | CLEAN | Same pipeline without constraints (diagnostic). |
| GB-03 | CLEAN | Same pipeline, core features. |
| GB-04 | CLEAN | Reuses GB-01's per-fold configs, which were themselves chosen in-window. |
| GB-05 | CLEAN | Same pipeline, 'advanced' monotone method. |
| GB-06 | CLEAN | Same pipeline; `gbm_lib.py` was edited (00:06) after GB-01..05 ran to add this variant; the shared code path is unchanged (GB-01 fold 3 reproduced exactly with the current file). |

## 7. Face validity

{{TABLES_FV}}

Reading:
- **SC-03, RC-03, GB-01, GB-03, GB-04, GB-05: zero violations.** SC-03 and RC-x enforce the shape
  by PAV (all 111 WoE/T_i monotonicity checks pass, `r1_monotone_shape_checks.json`); the GBMs by
  monotone constraints. For the GBMs only the fold-3 model is saved, so folds 1-2 are covered by
  construction (constraint vector in the code), not by measurement.
- **SC-02**: `log_assets` positive in folds 1 and 3 (+0.049, +0.012; not significant, D-032).
  Justified as a near-zero conditional effect, but it is a reason-code risk.
- **RC-01**: T_neg_equity negative in all folds (z −1.1, −1.9, −2.7). **RC-02**: neg_equity and
  bank_debt_ratio negative in all folds. Explained in D-041 (redundancy with T_equity_ratio;
  collinearity within leverage) — a fair explanation, but a credit officer would see a wrong-signed
  driver, so we count them.
- **GB-02** contradicts `wc_ratio` (83% of PD movement against the sign) and `interest_coverage`
  (57%); `bank_debt_ratio` is borderline (45% here on a 25-point grid; 56% in the GBM agent's
  check): two to three violations plus partial non-monotonicity on most ratios (T9). Confirms the
  challenger's own finding.
- **GB-06** (constraints dropped on three features): `current_ratio` 87% against the sign. This is
  the cost of the "decile break" relaxation: the model uses current_ratio upward once wc_ratio is
  constrained.
- **tangible_ratio as GB-01's #1 driver: plausible.** The expected sign (−, pledgeable collateral)
  holds univariately (fold-3 training rows: AUC 0.616 with PD falling as the ratio rises, T16) and,
  importantly, **within every ATECO group with >= 30 defaults** (within-group AUC 0.56 to 0.73, T16),
  so it is not just a proxy for asset-heavy, low-default sectors such as real estate (L). Its large
  mean |SHAP| (0.41, twice the next driver) is partly an attribution artefact: GBM splits shared
  leverage / asset-structure signal across correlated ratios and can load it on one of them. Explain
  it to a credit officer as "asset structure / collateral" and report SHAP by family (D-058). A human
  should still read the GB-01 dependence plot (`experiments/gbm/GB-01/dep_tangible_ratio.png`).

## 8. Promotion decision (rule applied)

| candidate | cond. 1 process | cond. 2 vs SC-03 | cond. 3 face-validity violations, features (SC-03: 0) | cond. 4 per year (incl. 2010) | outcome |
|---|---|---|---|---|---|
| GB-01 | CLEAN | LL and AUC both better, CIs exclude 0 | 0 | better in 2008, 2009, 2010, seen, new | **PROMOTED** (new incumbent) |
| GB-04 | CLEAN | passes | 0 | passes | tie with GB-01 (T6) → GB-01 kept (no imputation step) |
| GB-05 | CLEAN | passes | 0 | passes | tie with GB-01 → GB-01 kept (simpler default method) |
| GB-03 | CLEAN | passes | 0 | passes | dominated by GB-01 (T6) |
| RC-03 | CONCERN | LL better; AUC better pooled, tie in 2010 | 0 | no year worse | fails cond. 1; dominated by GB-01 in every year; **transparent fallback** |
| RC-01 | CLEAN | LL better; AUC tie | 1 feature (neg_equity, all 3 folds) | 2010 AUC tie | fails cond. 3 |
| RC-02 | CLEAN | passes | 2 features (neg_equity, bank_debt_ratio; all folds) | passes | fails cond. 3 |
| GB-02 | CLEAN | passes | 2 features (+1 borderline) | passes | fails cond. 3 (not for submission) |
| GB-06 | CLEAN | passes | 1 feature (current_ratio) | passes | fails cond. 3 |
| SC-01, SC-02, SC-04 | CLEAN | worse | 0 / 1 (log_assets) / 0 | - | not promoted |

**Incumbent after round 1: GB-01.** Hand-off to the calibration-specialist: **GB-01** (champion) and
**RC-03** (transparent fallback; if the humans decide explainability outweighs ~0.018 AUC and
0.0016 log loss, this is the model). Inputs available in-window: GB-01 `oot_predictions_raw.parquet`
and `inner_B_predictions.parquet`; RC-03 `params_fold*.json` (final transform F).

## 9. Notes for the next steps

- Calibration: all models under-predict by ~25-45% depending on the year (T2); the level gap grows
  with the default-rate trend. New-firm slopes ≈ 0.88 suggest a new-firm adjustment may help (H5).
- The new-firm AUC gap (seen − new ≈ 0.06 for every model with CIs ≈ [0.04, 0.08], T3) is a property
  of the population, not of a model; GB-01 has the same gap as SC-03.
- Round-2 ideas (if time): partial monotonicity only where the economic sign is weak; re-run the
  core-set selection per fold to remove the shared D-062 caveat; test the 1.2× tree heuristic on the
  inner split.

{{TABLES_B}}
