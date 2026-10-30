# challenger-gbm decisions (D-050..D-059), for the orchestrator to merge into docs/DECISION_LOG.md

All numbers come from `experiments/gbm/<exp>/metrics.md`, `train_log.md`, `config.json`,
`experiments/gbm/comparison.md` and `experiments/gbm/GB-01/explain.md` (code: `gbm_lib.py`,
`run_experiment.py`, `compare.py`, `explain.py` in `experiments/gbm/`). Pooled = test years
2008-2010, 506,159 rows, 7,784 defaults.

### D-050 GBM challenger = monotone-constrained LightGBM, full feature set, native NaN (GB-01)
- Date / phase / agent: 2026-10-31 / Phase 3 (tournament round 1) / challenger-gbm
- Status: ADOPTED (as the GBM entry to the tournament; promotion is the referee's call)
- Decision: LightGBM binary log-loss objective; monotone constraints from `B.expected_signs()` on every
  feature with sign +1/-1 (asset_turnover, extraord_to_assets, rev_nonpositive and the 3 categoricals
  unconstrained); categoricals as LightGBM categoricals (never numeric codes); NaN left to LightGBM's
  native missing-value handling; no class re-weighting or resampling.
- Alternatives considered: unconstrained twin (GB-02, D-051), core features only (GB-03, D-055), shared
  imputation (GB-04, D-056), 'advanced' monotone method (GB-05, D-057), partial constraints (GB-06,
  D-051). XGBoost was not tested (team decision: not installed). So the past-cohort tip that
  "calibrated XGBoost did well" is tested only in its LightGBM form.
- Business reason: one model that ranks well (approval) and whose drivers all move PD in the direction
  a credit officer expects (defensible declines, reason codes).
- Evidence: GB-01 pooled AUC 0.8375 (AR 0.675), log loss 0.06416, Brier 0.01368; per year AUC
  0.8441 / 0.8359 / 0.8307 (2008/2009/2010); seen firms AUC 0.8438, new firms 0.7865. All 20
  sign-expected features have zero partial-dependence movement against the expected sign (explain.md §1).
- Source(s): monotone constraints = FEATURE_SPEC expected signs (D-018, D-022); R-19 (abstract) for the
  need to check boosted-tree probabilities post hoc.
- Risks / what would change our mind: referee bootstrap shows the RiskCalc/scorecard challengers equal on
  AUC (then the simpler model wins under the tie rule); the new-firm AUC gap (0.787 vs 0.844) also
  appears in the other challengers or is specific to GBM.
- Human reviewer (initials) and date:

### D-051 Monotone constraints kept despite a significant cost in power
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED (constraints kept). Unconstrained model REJECTED for submission ("what didn't work" /
  trade-off row)
- Decision: keep the constraints. The unconstrained twin GB-02 is better on both metrics, but it gets
  there partly by using ratios against their economic direction.
- Alternatives considered: GB-02 (no constraints); GB-06 (constraints dropped only on the three
  decile-break features current_ratio, tangible_ratio, st_debt_share flagged by feature-architect).
- Business reason: a model that says "more working capital raises PD" or "more bank debt lowers PD"
  cannot be defended to a credit officer or a validator, and its reason codes would be wrong-signed.
- Evidence (comparison.md, firm-clustered paired bootstrap, 200 draws, d = variant - GB-01):
  GB-02 dAUC +0.0065 [+0.0052, +0.0079], d log loss -0.00083 [-0.00096, -0.00072] (-1.3%), both
  significant; per year AUC 0.8485 / 0.8419 / 0.8402. GB-02 fold-3 partial dependence CONTRADICTS the
  expected sign for wc_ratio (84.7% of movement against), bank_debt_ratio (55.5%), interest_coverage
  (57.4%), and partly violates nearly every other ratio (GB-01/explain.md §4). GB-06 recovers only
  dAUC +0.0005 [+0.0000, +0.0010], d log loss -0.00011, so the gap is not caused by the three decile-
  break features. It is spread over collinear ratios used conditionally against sign.
- Source(s): assumption (face validity is a course grading criterion and a credit-officer requirement).
- Risks / what would change our mind: if the referee's face-validity test also accepts limited
  non-monotonicity, a partially constrained model could be revisited in round 2. The gap is about
  0.0065 AUC, about the same size as the extra power the full feature set adds over the core set (D-055).
- Human reviewer (initials) and date:

### D-052 Inner validation scheme, tuning grid and number of trees
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED
- Decision: folds 2-3: inner-train = training years except the last; the last training year is
  split by firm (crc32 hash of id) into A (early stopping + grid choice by A log loss) and B
  (calibration). Fold 1 (one training year): random firm split 70/15/15 with the pre-declared DEFAULT
  config (num_leaves 8, depth 3, min_data_in_leaf 1000), no grid. Pre-declared grid (folds 2-3):
  num_leaves/max_depth in {4/2, 8/3} x min_data_in_leaf in {500, 2000}; fixed lr 0.03,
  feature/bagging fraction 0.8, lambda_l2 10, early stopping 150 rounds (max 4000). Final booster
  refit on all training rows with round(1.2 x best_iter) trees (heuristic: more rows support
  slightly more trees). Feature params (`B.fit`) refit on inner-train for the inner model and on the
  full window for the final model.
- Alternatives considered: Optuna search (rejected: time box, and the grid differences are tiny);
  no refit (final model = inner model; rejected because it would drop the most recent training year).
- Business reason: tuning never sees the test year; the latest year is used for validation, which
  mimics deployment.
- Evidence: chosen per fold (config.json): fold 1 DEFAULT, 282 -> 338 trees; fold 2 8 leaves/min 500,
  408 -> 490; fold 3 8 leaves/min 2000, 322 -> 386. The grid spread in A log loss is <= 0.0003
  (train_log.md), so the choice matters little.
- Source(s): assumption (standard practice). The 1.2 factor is a heuristic, not tuned.
- Risks / what would change our mind: the 1.2 factor is untested; the harness refit on 2007-2010
  should use the same rule (encoded in `fit_pd_model`).
- Human reviewer (initials) and date:

### D-053 Own calibration: Platt slope from held-out slice B, level anchored to the training default rate
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED (own calibration); final level-setting is DEFERRED to calibration-specialist
- Decision: Platt map logit(PD) = a + b*logit(p_raw). Slope b is fitted on slice B with the inner model
  (B was used neither for fitting nor for early stopping). The intercept a is re-anchored so that the
  mean calibrated PD over the full training window equals the training default rate. Isotonic was
  compared but not adopted.
- Alternatives considered: (a) raw booster probabilities; (b) Platt as fitted on B (its intercept
  mixes in the year-on-year rise in the default rate, then would be applied to a refit model that
  already contains that year, so it double-counts); (c) isotonic (eligible only if B has >= 500
  defaults: fold 1 B has 203 defaults, so not eligible there).
- Business reason: pricing needs the right level. A shape-only fix from held-out data, plus an
  explicit, documented level choice (training-window average), keeps the level decision with one
  owner (calibration-specialist) for all challengers.
- Evidence: 2-fold firm-cross-fitted log loss on B (raw / Platt / isotonic): fold 1 0.04398 / 0.04401
  / 0.04553; fold 2 0.05579 / 0.05549 / 0.05573; fold 3 0.06518 / 0.06447 / 0.06476. Platt slope b on B
  = 1.017 / 0.943 / 0.998. So the LightGBM log-loss booster (no re-weighting) is close to slope 1. In
  this data, the Platt gain on B is almost all intercept (level drift between years), not the sigmoid
  distortion R-19 describes for boosting. OOT effect of our map vs raw: pooled AUC unchanged, log loss
  0.064155 vs raw within 1e-4 (metrics.md). OOT level: mean PD 1.145% vs observed 1.538% (obs/pred
  1.34; 1.21 / 1.36 / 1.43 by year), calibration slope 0.976. The level gap grows as default rates rise,
  so it is a central-tendency problem, not a shape problem.
- Source(s): R-19 (abstract only: boosted-tree probabilities can be distorted and need post-hoc
  calibration; we tested this and found little distortion here). Platt (R-20) is TO-VERIFY, so
  "Platt scaling" is cited as a method name only [UNVERIFIED].
- Risks / what would change our mind: the calibration-specialist's chosen central tendency. Raw OOT
  predictions (`oot_predictions_raw.parquet`) and inner-B predictions (`inner_B_predictions.parquet`:
  row_id, y, pd_raw_inner, fold) are provided so the common method can be fitted without touching
  test years.
- Human reviewer (initials) and date:

### D-054 No class re-weighting or resampling
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED
- Decision: train on the natural ~1-1.3% default rate with strong min_data_in_leaf (500-2000) and
  shallow trees; no scale_pos_weight, no up/down-sampling.
- Alternatives considered: scale_pos_weight plus re-calibration (rejected: it destroys the probability
  level and needs another calibration stage, with no expected gain in AUC).
- Business reason: the PD level is used for pricing.
- Evidence: calibration slope 0.976 pooled without any re-weighting (metrics.md).
- Source(s): assumption.
- Risks / what would change our mind: none expected.
- Human reviewer (initials) and date:

### D-055 Full feature set vs core set for the GBM
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED full set (core-only GB-03 REJECTED)
- Decision: the GBM uses all 26 features from FEATURE_SPEC.
- Alternatives considered: the 13 core features only (GB-03).
- Business reason: more power for approval ranking at no cost in face validity (all constrained).
- Evidence: GB-03 vs GB-01: dAUC -0.0077 [-0.0087, -0.0066], d log loss +0.00077 [+0.00067, +0.00085];
  GB-03 per year AUC 0.8381 / 0.8274 / 0.8226. It is also a like-for-like reference for the probit /
  logit challengers that use the core set: GB-03 pooled AUC 0.8298.
- Source(s): assumption.
- Risks / what would change our mind: if the referee rates the GBM vs probit gap as mostly a feature-set
  effect, the probit challengers could add non-core features in round 2.
- Human reviewer (initials) and date:

### D-056 Native NaN handling vs shared imputation (build.py impute=True)
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED native NaN (imputation GB-04 REJECTED as unnecessary, not as harmful)
- Decision: keep NaN for LightGBM; the `debt_split_missing` flag stays in the feature set.
- Alternatives considered: GB-04 = same per-fold configs as GB-01 with sector-median / block-share
  imputation.
- Business reason: simpler harness (no imputation step for this model); identical results.
- Evidence: GB-04 vs GB-01 dAUC +0.0001 [-0.0003, +0.0006], d log loss -0.000007 [-0.00004, +0.00003]:
  no difference.
- Source(s): assumption.
- Risks / what would change our mind: none. Either is acceptable for the harness.
- Human reviewer (initials) and date:

### D-057 LightGBM monotone method 'intermediate' (vs 'advanced')
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED 'intermediate' ('advanced' GB-05 REJECTED: no material gain)
- Evidence: GB-05 vs GB-01 dAUC +0.0002 [-0.0000, +0.0004], d log loss -0.00002 [-0.00004, +0.00000].
- Business reason: same face validity, simpler and faster; the difference is below any business relevance.
- Source(s): assumption.
- Human reviewer (initials) and date:

### D-058 Explainability: TreeSHAP reason codes and dependence checks
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED
- Decision: per-firm reason codes = top-3 positive TreeSHAP contributions (LightGBM `pred_contrib`, log-odds
  scale) in plain words with natural-scale values and percentiles; global = mean |SHAP| + partial
  dependence vs expected sign; report driver FAMILIES because correlated ratios split attribution.
- Evidence: GB-01/explain.md. Top drivers (mean |SHAP|): tangible_ratio 0.41, equity_ratio 0.21, roa_net
  0.21, st_debt_ratio 0.18, ap_days 0.15, ar_days 0.14, ateco_group 0.12. Every sign-expected feature
  agrees. asset_turnover (unconstrained by design) rises with PD over its range in the fold-3 model
  (-5.36 -> -4.87 log-odds).
- Source(s): R-23 (Shapley-value properties, Sec. 3). The TreeSHAP algorithm is NOT in R-23, so it is
  cited as [UNVERIFIED] until literature-scout verifies a source.
- Risks: tangible_ratio being the #1 driver should be sanity-checked by a human (sector mix: real
  estate / asset-heavy firms; FEATURE_SPEC notes decile breaks). ar_days hits the p99 cap (~60,000
  days) for near-zero-revenue firms, so the reason code must be worded "almost no revenue".
- Human reviewer (initials) and date:

### D-059 Reusable estimation for the harness; prediction cost
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED
- Decision: `experiments/gbm/gbm_lib.py`: `fit_pd_model(train_df, "GB-01")` fits feature params +
  inner model + calibrator + final booster on any training frame (needs columns id, fs_year, y, raw
  fields); `save_model` / `load_model` write and read booster.txt, feature_params.json,
  calibrator.json, meta.json; `predict_pd(art, df)` is frozen (no re-estimation, row-wise).
- Evidence: round trip of the saved fold-3 model reproduces the fold-3 OOT PDs exactly (max |diff| 0.0,
  comparison.md). Cost: 100,000 rows in 1.0 s (0.8 s of it is the shared feature transform), 8
  threads, laptop. Complexity O(n x (F + T x D)): 386 trees of depth <= 3 in fold 3.
- Source(s): n/a.
- Risks: the harness must call `predict_pd` with the same LightGBM major version (4.7.0 used).
- Human reviewer (initials) and date:
