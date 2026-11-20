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


# Round 2, phase A (D-100..D-109), for the orchestrator to merge into docs/DECISION_LOG.md

All numbers come from `experiments/gbm/comparison.md` (round-2 section), `comparison_r2_*.csv`, and the `GB-07/`, `GB-08/`, `GB-08b/`, `GB-09/` csv files named below. Pooled = test years 2008-2010, 506,159 rows, 7,784 defaults. Own calibration only (D-053). d = variant - GB-01; 200 firm-clustered draws, percentile CI; no multiplicity correction.

### D-100 Round-2 anti-overfitting protocol, and where it was not fully met
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: ADOPTED
- Decision: every choice is made on the fold's inner slices (fold 3: train 2007-08, validate 2009, firm halves A/B; fold 2: train 2007, validate 2008; fold 1: firm split 70/15/15 inside 2007). Outer test years are scored once per final configuration. fs_year >= 2011 is excluded by a pyarrow read filter in `r2_lib.load_panel`, `compare_r2.py`, and the round-1 scripts `explain.py`, `run_experiment.py`, `compare.py` (those three were patched; only the read changes).
- Alternatives considered: a single choice made once on the fold-3 inner split and applied to all folds. It leaks: 2009 labels would inform the fold-2 score.
- Business reason: a level or ranking gain that exists only because the test year was peeked at would disappear on the holdout.
- Evidence: GB-01 rebuilt through `r2_lib` (GB-01r) matches GB-01 row by row, max |PD diff| = 0.0. The 15:12 edit of src/features/build.py did not change the 26 round-1 features (`check_feature_stability.py`: same list, identical fitted params, fold-3 OOT diff 0.0). Known departures: (1) the GB-08 and GB-08b constraint sets are fixed across folds and were informed by inner splits of later folds (D-102), so their 2008 and 2009 scores are not strictly clean; 2010 is. (2) GB-08b was conceived after GB-08's outer score was seen. (3) The sklearn RF replaced the LightGBM RF after the latter's outer score was seen (D-106). (4) Round-1 outer results, including GB-02 and GB-06, were known before round 2 started.
- Source(s): assumption (protocol set by the orchestrator); R-07 and R-01 for walk-forward as the preferred design.
- Risks / what would change our mind: the lockbox check later on is the real test of any promotion.
- Human reviewer (initials) and date:

### D-101 GB-07 overfitting diagnostics: GB-01 shows no harmful overfitting
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: ADOPTED (finding)
- Decision: keep GB-01's regularisation and the 1.2 x best_iter rule (D-052). The in-sample vs out-of-time gap is mostly not reducible by tuning.
- Alternatives considered: more shrinkage or fewer trees, tested through GB-09 (D-104).
- Business reason: a model that memorises the training window would rank new applicants worse than its validation suggests.
- Evidence (`GB-07/gap_by_fold.csv`, `learning_curve_flatness.csv`, `calibration_by_year.csv`, `learning_curve.png`, `gap.png`): GB-01 in-sample vs next-year AUC 0.8476/0.8441, 0.8590/0.8359, 0.8498/0.8307, an AR gap of 0.7, 4.6, 3.8 points. GB-02 AR gap 5.2, 8.7, 6.5 points, yet GB-02 scores higher out of time, so a bigger gap is not by itself worse. Inner log loss at 1.2 x best_iter is +0.00009, +0.00004, +0.00003 above the minimum; at 2x +0.00033, +0.00022, +0.00016. Calibration slope 0.969, 1.005, 0.957. Firm-halves A and B of the same year differ by up to 0.017 AUC (fold 2: 0.8533 vs 0.8364), far more than any variant gap here, which is why only paired tests are used.
- Source(s): R-33 Sec. 4.7-4.8 pp. 22-23 (in- vs out-of-sample AR gap -0.5 points, N-33). Not like for like: France, different model and design, and our gap includes a move across years and into new firms.
- Risks / what would change our mind: the split of the gap between drift and overfitting is inferred, not measured. Evidence for drift: fold 2's inner model shows no 2007 to 2008 decay (0.8528 vs A 0.8533, though B is 0.8364).
- Human reviewer (initials) and date:

### D-102 GB-08 ablation: the cost of the constraints is real but overlapping, and the "denominator" rule was post hoc
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: ADOPTED (finding); the denominator rule itself is REJECTED as an economic justification
- Decision: report the ablation as a measure of cost. Do not use "denominator near zero" as the economic reason for dropping constraints.
- Alternatives considered: dropping the costliest constraints regardless of economics (roa_net, roa_oper); rejected because these are primary reason codes.
- Business reason: each dropped constraint must carry a reason a credit officer would accept.
- Evidence (`GB-08/ablation_summary.csv`, `checks_additivity.csv`, `checks_tail_dr.csv`, `checks_sector_mix.csv`, `shapes_costly.png`, `freed_shapes.png`): mean A+B log-loss cost, largest first: roa_net 0.00024, cfo_to_debt 0.00022, roa_oper 0.00019, interest_coverage 0.00018, ebitda_margin 0.00010. All dropped 0.00052. Sum of single costs 0.00069, 0.00177, 0.00161 vs all-dropped 0.00021, 0.00052, 0.00084, so the costs overlap and do not add. No CI on single-drop costs. Raw default rates do rise in the top tail, in both years checked and also for the assets-denominated ROA ratios: e.g. cfo_to_debt 2007 lowest ventile 0.21% [0.13, 0.34] vs top 0.40% [0.28, 0.58]; roa_net 2009 0.46% [0.34, 0.63] vs 0.89% [0.71, 1.12]. The lowest ventile is picked after the fact, so these contrasts flatter the rise. Because ROA shows the same rise, a near-zero denominator does not explain it. The property-vehicle story holds only for ebitda_margin (section L 46% of the top quintile vs 10% of the rest, 2007); for interest_coverage and cfo_to_debt section L is under-represented (13% vs 19%, 14% vs 19%).
- Source(s): assumption. No verified source on the top-tail rise. Cause unresolved.
- Risks / what would change our mind: a verified source or a data explanation for why very strong ratios default more often.
- Human reviewer (initials) and date:

### D-103 GB-08 and GB-08b: statistical winners, not adopted on face validity
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: REJECTED for GB-08; OPEN for GB-08b (human decision requested, agent recommends against)
- Decision: keep all 20 constraints. GB-08 drops interest_coverage, cfo_to_debt, ebitda_margin. GB-08b drops ebitda_margin only.
- Alternatives considered: GB-02 (D-051); reclassifying ebitda_margin as "no expected sign" like asset_turnover, which would be a post-hoc FEATURE_SPEC change.
- Business reason: reason codes such as "strong cash cover raises PD" cannot be defended in a decline conversation.
- Evidence: GB-08 dAUC +0.0032 [+0.0024, +0.0042], d log loss -0.00035 [-0.00043, -0.00027]; AUC 0.8465, 0.8386, 0.8354. GB-08b dAUC +0.0015 [+0.0009, +0.0021], d log loss -0.00016 [-0.00019, -0.00011]; AUC 0.8451, 0.8374, 0.8328; every per-year CI excludes zero on both metrics. Both meet the win rule. Face validity: GB-08 cfo_to_debt CONTRADICTS (58%), interest_coverage CONTRADICTS (56%), ebitda_margin mostly agrees (44%); at the 99th percentile of cfo_to_debt the log-odds are -4.66, as high as cash burners at -4.64; 5.2% of 40,000 sampled 2009 firms get a freed ratio in their top-3 PD-raising reasons while above its PD minimum (`GB-08/checks_wrong_way_reason_codes.csv`). GB-08b ebitda_margin CONTRADICTS (70%): PD multiple vs median margin is 0.87-0.90x for margins of -6% and below, and 1.18-1.22x for margins of 56% and above; 5.7% of firms get it as a top-3 raiser with a top-quintile margin (`GB-08b/checks_ebitda_shape.csv`, `checks_ebitda_reason_codes.csv`). The top-tail side has a supported story (section L, median turnover 0.06). The loss-making side does not. Inner B agrees with GB-08 in all folds but is not independent: A+B costs informed the set.
- Source(s): assumption (credit-officer face validity, D-051, D-064).
- Risks / what would change our mind: a referee face-validity rule that accepts limited non-monotonicity, or Kiet deciding that a margin effect capped at about +/-20% PD is acceptable. An untested alternative for feature-architect: give the model the business-model signal directly so ebitda_margin can stay constrained.
- Human reviewer (initials) and date:

### D-104 GB-09 random search: formally passes on AUC, not recommended for promotion
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: REJECTED as a replacement (agent judgment stricter than the win rule; orchestrator may overrule)
- Decision: keep GB-01's pre-declared small grid.
- Alternatives considered: 37 configs (GB-01's 5 + 32 random, sklearn ParameterSampler, seed 0): learning_rate log-uniform 0.015-0.10; max_depth 2, 3, 4; num_leaves 4-16 capped at 2^depth; min_data_in_leaf 200-4000; feature and bagging fraction 0.5-1.0; lambda_l1 0-10; lambda_l2 log-uniform 0.1-100; min_gain_to_split 0-0.5. Same lightgbm early stopping on A.
- Business reason: a more complex config should earn its place on the holdout, not on one year.
- Evidence: dAUC +0.0009 [+0.0004, +0.0014]; by year +0.0001 [-0.0008, +0.0010], +0.00003 [-0.0009, +0.0010], +0.0021 [+0.0012, +0.0028]. d log loss -0.00006 [-0.00011, -0.00000]; by year +0.000003, +0.00011, -0.00028, so WIN_ll is false. Slope 0.968 vs 0.976. Chosen configs differ widely (lr 0.024, 0.050, 0.016; depth 3, 4, 4; final trees 322, 198, 1052). Selection check (`GB-09/selection_check.csv`): Spearman A vs B log loss 0.31, 0.69, 0.68; the fold-1 A gain reverses on B; the top 10 configs lie within 0.00008-0.00015 A log loss.
- Source(s): assumption.
- Risks / what would change our mind: if the 2010 gain were confirmed on the lockbox, the fold-3 config might be worth testing as a fixed config for the final refit.
- Human reviewer (initials) and date:

### D-105 GB-10 recency weights: no gain in ranking
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: REJECTED
- Decision: no recency weighting of training rows.
- Alternatives considered: half-life none, 4, 2, 1, 0.5 years, crossed with GB-01's candidates.
- Business reason: test whether recent years describe next year better.
- Evidence: dAUC +0.0002 [-0.0003, +0.0006], d log loss +0.000002 [-0.00003, +0.00004]. Folds 1-2 are identical to GB-01 by construction: with one inner-train year weights cannot be tested, so the tie goes to no weighting. Hence this experiment tests only fold 3 and cannot meet the all-years rule. Fold 3 chose half-life 0.5, the edge of the grid, on an A gain of 0.00012; B log loss improved 0.00005 but B AUC fell (0.8414 vs 0.8433); 2010 AUC +0.0005 [-0.0009, +0.0017]. The intercept stays anchored to the unweighted training rate, so any level benefit is excluded by design; level belongs to calibration (D-071).
- Source(s): assumption.
- Risks / what would change our mind: one fold is thin evidence; a longer window at the final refit could be retested.
- Human reviewer (initials) and date:

### D-106 GB-11 random-forest benchmark: GB-01 does not clear it on ranking
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: ADOPTED as a benchmark, not a candidate. LightGBM boosting='rf' implementation (GB-11lgbrf) REJECTED.
- Decision: report sklearn RandomForest (300 trees, sqrt features, 50% rows per tree, one-hot categoricals, D-053 calibration) as the benchmark and state plainly that the monotone GB-01 does not beat it.
- Alternatives considered: LightGBM RF: per-year AUC 0.8429, 0.8340, 0.8229 but pooled 0.7896 and raw inner log loss 0.09-0.17, with raw 99th-percentile PD 0.95-0.998. Attribution to specific LightGBM settings (lambda_l2=0 etc.) is untested.
- Business reason: R-17 says beating logistic regression alone no longer signals progress, so the model needs a stronger reference point.
- Evidence: GB-11 AUC 0.8395 (0.8449, 0.8396, 0.8335), log loss 0.06392. vs GB-01: dAUC +0.0020 [+0.0001, +0.0038], d log loss -0.00023 [-0.00039, -0.00007]; WIN_auc true (lower bound barely above zero), WIN_ll false (2008 +0.00001). GB-02 vs GB-11: +0.0045 [+0.0028, +0.0064]. GB-08 vs GB-11: +0.0012 [-0.0005, +0.0030]. So boosting beats RF when both are unconstrained; GB-01's shortfall is the price of monotonicity. RF in-sample AUC 0.953-0.962 is not a useful overfit signal because leaves memorise rows. min_samples_leaf 50 was the grid edge in all folds, so the benchmark may be understated.
- Source(s): R-17 Conclusions, MS p. 25 (VERIFIED; retail credit scoring, not corporate).
- Risks / what would change our mind: RF face validity was not measured and no fold-3 model was saved, so GB-11 has no explain output. RF is unconstrained and is not eligible for submission on current rules.
- Human reviewer (initials) and date:

### D-107 GB-12 combination not run
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: REJECTED (not run; no outer scoring used)
- Decision: skip GB-12. The spec remains in `run_r2.py` for later use.
- Alternatives considered: GB-08 constraints + GB-09 top-5 configs per fold + weights.
- Business reason: a combination is only useful if it could be submitted.
- Evidence: of the pieces, GB-08 and GB-08b fail face validity (D-103), GB-10 is null (D-105), GB-09 is not recommended (D-104). No two eligible pieces exist to combine, and another outer look would not change any decision.
- Source(s): n/a.
- Risks / what would change our mind: if Kiet accepts GB-08b (D-103), run GB-08b + GB-09 once.
- Human reviewer (initials) and date:

### D-108 Pipeline accepts a feature-set name; round-1 scripts carry the lockbox filter
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: ADOPTED
- Decision: `run_r2.py <exp> --features full|core|<list name in build.py>`; `r2_lib.py` holds the general fitter, `gbm_lib.py` is untouched.
- Alternatives considered: editing gbm_lib, rejected because GB-01 must stay reproducible for the harness.
- Business reason: a new feature set can be tested without code changes.
- Evidence: the feature-set route was run only with "full". FULL2_FEATURES now exists in build.py but was not run from this pipeline. One limit to check: `B.transform` returns `ALL_FEATURES`, so a list outside it would need the caller to confirm columns exist.
- Source(s): n/a.
- Risks / what would change our mind: not exercised with FULL2.
- Human reviewer (initials) and date:

### D-109 Recommendation, round 2 phase A: GB-01 stays the GBM incumbent
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: OPEN (recommendation; promotion belongs to the referee, GB-08b to the team)
- Decision: keep GB-01 as the constrained GBM entry. Look for further gains in information (phase B, FULL2), not tuning or weighting.
- Alternatives considered: GB-02, GB-08, GB-08b, GB-09, GB-10, GB-11 (above).
- Business reason: approval needs ordering and pricing needs level; none of these variants moves the level (mean PD 1.14-1.15% vs observed 1.54%), which stays with calibration.
- Evidence: comparison.md round-2 section. Tuning plus weighting yields at most +0.0009 AUC. Constraints cost up to +0.0065 and are the only large lever, but using it breaks face validity.
- Source(s): R-17 (benchmark); otherwise assumption.
- Risks / what would change our mind: the win rule counts a +0.00003 year as "better" and has no multiplicity correction across six outer-scored variants. Suggest the referee also require per-year CIs or a minimum effect size.
- Human reviewer (initials) and date:
