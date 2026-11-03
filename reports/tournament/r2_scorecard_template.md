# Referee scorecard, round 2 (referee, 2026-11-03)

Code: `r2_score.py`, `r2_boot_worker.py`, `r2_audit.py`, `r2_interest_check.py`, `r2_build_md.py`. Tables are rendered from
`r2_*.csv`; metrics come from `src/common/metrics.py` only. **Lockbox:** the panel was read with fs_year < 2011 and asserted;
no fs_year 2011+ row or value was opened. 2010 rows enter the audit as feature rows with labels removed. Referee fitted no
new configuration: the refits only reproduce submitted files.

## Decision rule (published with every scorecard)

A challenger replaces the incumbent only if:
1. its leakage verdict is CLEAN;
2. the 95% CI of Δlog-loss excludes zero in its favour, OR (the ΔAUC CI excludes zero in its favour AND log loss is not worse);
3. it has no more face-validity violations than the incumbent (or each violation is justified);
4. the per-year results show no year in which it is clearly worse (no hidden fragility).

Otherwise the simpler / more explainable model stays (statistical ties go to transparency). D-062 adds: a promotion must also hold on 2010 alone.

## 0. Verdict

**GB-25 becomes the GBM incumbent, replacing GB-01.** Leakage CLEAN, with a process note on model choice (section 3).

| Question | Answer | Numbers |
|---|---|---|
| Better than GB-01? | Yes, on ranking and on probability accuracy | Pooled ΔAUC +0.0049 [+0.0037, +0.0061], Δlog loss -0.00035 [-0.00044, -0.00025]. 2010 alone: +0.0042 [+0.0022, +0.0059], -0.00038 [-0.00055, -0.00021] |
| Any year worse? | No. 2008 is the weak year | 2008 ΔAUC +0.0027 [+0.0001, +0.0053], Δlog loss -0.00012 [-0.00029, +0.00006] (tie) |
| Why GB-25 over GB-20? | Face validity, at a measured cost | GB-20 is statistically better: GB-25 minus GB-20 pooled ΔAUC -0.0011 [-0.0017, -0.0006], 2010 -0.0021 [-0.0029, -0.0012]. But in GB-20, 20% of firms get a lower PD when their interest expense doubles (GB-25: 1.7%, all loss-making firms; section 5) |
| Number to quote | 2010, not pooled | GB-25 2010 AUC 0.8348 [0.8272, 0.8415], AR 0.670, log loss 0.07098; gain over GB-01 as above. Replace with the lockbox result once it exists |
| Does it fix weak segments? | Small firms yes, construction only partly, new firms no | Section 4 |
| Does it fix 2010 under-prediction? | No | 2010 obs/pred 1.19 (GB-25+CAL) vs 1.18 (CAL-01); grades 6-14 obs/pred 1.33 vs 1.32 |
| Lockbox finalists | GB-25, GB-01, GAM-01 | Section 6; needs human sign-off before anyone opens 2011 |

**What changed and why.** The gain comes from new information in full2, not tuning: tuning and recency weights on v1 add at most
+0.0009 AUC (GB-09), while full2 with GB-01's own recipe adds +0.0060 (GB-20). GB-25 then gives back +0.0011 of it to remove a
direction a credit officer could not accept. Plain meaning: the bank ranks borrowers slightly better and prices slightly more
accurately, but in the 2010 riskiest-10% cut GB-25 catches about the same share of defaults as GB-01 (table in section 4), and
the level problem is untouched. This is a modest upgrade to the ordering, not a new model of risk.

## 1. Contract and metrics

All 15 vectors PASS (`r2_contract.csv`). Hash = first 12 hex digits of SHA-256 at scoring time.

{{CONTRACT}}

Level check: the +CAL files equal a recomputed prior shift to rule e within 2e-15, with shift, training rate and target recomputed
from training rows and `ct_targets.csv` lines for test years 2008-2010 only (`r2_level_shift_check.csv`).

### 1.1 Leaderboard, pooled 2008-2010 (AR = 2·AUC − 1: same information as AUC, CAP view)

GB-02 and GB-11 are references, not candidates: GB-02 fails face validity and GB-11 is unconstrained. +CAL rows carry the shipped level logic.

{{LEAD_POOLED}}

### 1.2 Fold 3, 2010, reported separately (D-062)

{{Y2010}}

### 1.3 2008 and 2009 (feature design saw these years; mildly optimistic)

2008:

{{Y2008}}

2009:

{{Y2009}}

Seen/new and all other cells: `r2_metrics_long.csv`. Deciles: `r2_deciles.csv`. Figures: `fig/r2_roc_cap.png`, `fig/r2_calibration_deciles.png`, `fig/r2_by_year.png`, `fig/r2_grade_level.png`, `fig/r2_segments.png`.

### 1.4 Paired firm-clustered bootstrap (positive ΔAUC and negative Δlog loss favour "new")

Draws: primary models (GB-25, GB-20, GB-01, GB-02, GB-11, GAM-01) use 1,002 draws (3 × 334, seeds 0-2, pooled). Table rows, +CAL pairs and
segments use 200 draws (seed 3 for pairs, seed 0 for segments), as the coordinator allowed under the deadline; at 200 draws the percentile bounds are noisier.
The fast estimator matches `metrics.paired_bootstrap` to 1.1e-16 on a 15,000-firm subsample (`r2_bootstrap_equivalence.csv`).

{{PAIR_MAIN}}

Table rows:

{{PAIR_TABLE}}

Shipped level logic:

{{PAIR_CAL}}

Readings:
- **GB-25 vs GB-11 (random forest benchmark):** pooled ΔAUC favours GB-25; Δlog loss is a tie; 2010 is a tie on both. The monotone model now matches the benchmark that GB-01 could not beat on log loss.
- **Correction to D-106 / comparison.md:** with 1,002 draws, GB-11 vs GB-01 pooled ΔAUC is +0.0020 [-0.0002, +0.0042], so the AUC "win" at 200 draws does not hold. Its log-loss gain does (-0.00023 [-0.00040, -0.00007]).
- **GB-25 vs GB-02:** the price of constraints remains: Δlog loss +0.00048 [+0.00034, +0.00062], 2010 ΔAUC -0.0054 [-0.0081, -0.0027].
- **GB-23:** better than GB-20 on pooled AUC and on 2010, worse in 2009 on log loss, and it carries the same free interest_to_sales. Its tuned settings differ widely by fold (D-127). Not promoted.

## 2. Leakage audit of the full2 pipeline: CLEAN

Code read: `src/features/build.py`, `experiments/gbm/run_round2b.py`, helpers in `gbm_lib.py`.

| Check | Result | Evidence |
|---|---|---|
| OOF encodings on training rows only | Pass. `oof=True` appears only on `df[m_tr]` and `df`; instrumented refits show no oof=True row outside the training window | `r2_repro_refit.csv`, `r2_refit_call_log.csv`, lines 112, 137 of `run_round2b.py` |
| A row's own label cannot reach its encoding | Pass. Flipping all labels of one firm-hash fold leaves that fold's OOF encodings unchanged (0.0), while other folds move up to 0.95 | `r2_oof_label_flip.csv` |
| Scoring rows use oof=False | Pass. A/B slices and test rows use the default; instrumented scoring shows only oof=False | lines 113-114, 161; `r2_repro_refit.csv` |
| Grouped by firm | Pass. No firm spans two OOF folds (max 1 fold per firm, folds 1 and 3 checked) | `r2_oof_label_flip.csv` |
| Params fitted in window, every fold | Pass. Saved fold-3 params are byte-equal to a refit on training rows; fit calls never include a test-fold row; max year in fit calls equals the last training year | `r2_param_provenance.csv`, `r2_repro_refit.csv` |
| Inner slices and in-sample files | Pass. All rows inside the training window, none in the test fold, all three folds, both models | `r2_inner_rows_check.csv` |
| Calibration | Pass. Slope from slice B of the training window; level anchored to the training rate on the OOF matrix | `run_round2b.py` lines 133, 142-145 |
| No 2010+ rows in designing features | Pass in code. `univariate_v2.py` reads fs_year <= 2009 with a parquet filter and asserts; `ablation_v2.py` uses that loader. See the people channel in section 3 | code read |
| Decision-time information | Pass. Features are ratios of the current statement plus category lookups; no `def_date`, `stmt_date`, `fs_year`, id or label in the feature list | `r2_param_provenance.csv` |
| Reproduction | Saved fold-3 models reproduce GB-25 and GB-20 with labels removed and rows shuffled, max diff 0.0. Refits from training rows alone reproduce the submitted files | tables below |
| Single-row vs batch | Identical on 999 rows (800 random, 199 awkward): feature diff 0.0, PD diff 0.0, also with every field as a string and blanks as "NA" | `r2_single_vs_batch.csv` |
| Other holdout rows (CLAUDE.md §2.2) | Compliant. `transform` is a function of one row and frozen params; it needs no id column and refits nothing | above; `build.py` lines 473-518, 829-870 |

Harness notes (not leakage): (a) `feature_params.json` also stores OOF tables the harness does not need. They hold training statistics only and could be stripped; (b) a fully blank row gets PD 4.0% and contract-valid output; (c) cost measured at 6.4 s per 100k rows, 3 threads, while other jobs ran, against 1.0 s for GB-01 (D-059), so the harness-engineer must re-measure; (d) encodings are rate lookups on category labels, which respects CLAUDE.md §2.5; their rising shapes are imposed by the + constraint, not evidence (D-124).

Saved models:

{{REPRO_SAVED}}

Refits, training rows only in the fit, test rows scored with labels removed and shuffled, 4 threads:

{{REPRO_REFIT}}

Instrumented call log, GB-25 fold 3 (all folds in `r2_refit_call_log.csv`):

{{CALLS}}

Label flip:

{{FLIP}}

Single-row vs batch:

{{SB}}

Parameter provenance:

{{PROV}}

Inner rows:

{{INNER}}

**Verdict: GB-20 CLEAN; GB-25 CLEAN; GB-23 CLEAN on the shared code path (its outer files were not reproduced).** GB-08, GB-08b, GB-09, GB-10, GB-11 were not re-audited: their process notes stand as disclosed in D-100, and none is promoted.

## 3. Process verdict and selection ledger

Leakage asks whether test rows reached a fit; they did not. Process asks whether people chose after seeing results; they partly did.

1. **Feature design (D-062).** full2 was designed on 2007-2009, so 2008 and 2009 are mildly optimistic. 2010 was not loaded by the design code.
2. **People channel.** Code cannot exclude what people knew. Round-1 scorecards and `r1b_diagnostics.md` (written 14:52, showing 2010 cells) existed before the round-2 feature files (15:12). I found no reference to them in the feature evidence file, but "2010 is clean" means clean in code.
3. **interest_to_sales sign: + (a priori, R-33) → none (post hoc on 2007-09 deciles, D-099) → + again (GB-25).** D-130 calls the fix pre-specified, which is true of GB-25's configuration before its own outer score. **It was not blind:** the proposal followed GB-20's outer score, and `GB-22/dependence_bins.csv` (15:57, before the GB-25 run at 16:34) already showed 2010 default rates by interest_to_sales decile (decile 10: 2.36%). 2010 is therefore not fully clean for this choice.
4. **Direction of the bias.** GB-25 is the lowest of the three full2 configurations on 2010 (GB-23 0.8389, GB-20 0.8369, GB-25 0.8348). Choosing it is not a pick of the luckiest result, so this choice does not inflate its number. Family-level optimism from reusing 2010 remains unquantified.
5. **Fold-2 configuration.** GB-25 re-chose min_data_in_leaf 2000 vs GB-20's 500 by the same recipe, so that fold's comparison also includes a settings change (D-130).
6. **Disclosures in D-100, D-129, D-130 are accepted.** GB-08b conceived after GB-08's score; RF implementation swapped after an outer score; GB-08/08b constraint sets informed by later folds' inner splits; TFR readings developed after seeing training data and 2010 cells.

**Process verdict: ACCEPTABLE WITH DISCLOSURE.** Nothing disqualifies, and the 2008-2010 numbers cannot serve as a clean final estimate. Only the lockbox can do that.

**Selection ledger update.** Prediction files covering 2008-2010: 30 → 51 (+21). Adding the 62 calibration configurations gives 113 looks (r1b: about 90). New configurations scored on the outer years in round 2: 9 (GB-08, GB-08b, GB-09, GB-10, GB-11, GB-11lgbrf, GB-20, GB-23, GB-25), 3 of them on full2, plus 3 common-level vectors. Inner looks at 2008/2009 slices: below. The referee added 0 configurations.

{{LEDGER}}

**Headline number.** Quote **GB-25 on 2010: AUC 0.8348 [0.8272, 0.8415], AR 0.670, log loss 0.07098 own level, 0.07010 at the shipped level; gain over GB-01 +0.0042 AUC [+0.0022, +0.0059], -0.00038 log loss [-0.00055, -0.00021].** Pooled (AUC 0.8424, +0.0049) goes next to it, labelled as including years the design saw. Do not quote the architect's +0.0096 or GB-20's +0.0060 as the expected gain. Once run, the lockbox replaces 2010 as the headline.

## 4. Segment check for GB-25

AR with paired difference vs GB-01 (200 draws). Cells pool 2008-2010 unless the name carries a year. Size edges come from each fold's training rows (r1b).

{{SEG}}

| Weak segment | GB-01 → GB-25 pooled | Clean year 2010 | Reading |
|---|---|---|---|
| Small firms (Q1) | AR 0.626 → 0.638, ΔAR CI excludes 0 | +0.012, CI excludes 0 | **Improves.** Still the weakest size band |
| Construction (F) | AR 0.559 → 0.568, ΔAR CI barely excludes 0 | +0.001, CI includes 0 | **Not shown on the clean year.** The gain is 2009 (+0.026) |
| New firms | AR 0.573 → 0.569, tie | +0.013, CI includes 0 | **No improvement.** 2008 is significantly worse (-0.020 [-0.034, -0.006], log loss +0.0008, 387 defaults); that fold has one training year. Point estimates are positive in 2009 and 2010, both ties |

Other sections: C, G, H, L, M and every size quintile improve pooled; J and N are positive ties; A, E, I point negative, all ties; thin cells (D, Q, R, S, pooled) are not resolvable.

Share of defaults in the riskiest 10% of firms (CAP point, diagnostic, used in no decision):

{{CAP}}

### 4.1 Observed vs predicted at the shipped level logic (prior shift to rule e)

{{LEVEL}}

Bands on the 16-grade grid (`r1b_grade_edges.json`; each model's own grade assignment):

{{BANDS}}

The same firms (CAL-01's grades 6-14 in 2010), so that grade migration cannot flatter a model:

{{SAME}}

2010 by grade:

{{GRADE10}}

2008 and 2009, GB-25+CAL:

{{GRADEOTHER}}

**The 2010 middle-book under-prediction remains.** GB-25+CAL grades 6-14 (55% of firms): 2,108 observed vs 1,590 expected, obs/pred 1.33, +518 defaults; CAL-01 was 1.32, +514. On the same firms 1.325 vs 1.319. This is a level (cycle) miss, and a better ranking does not touch it; it stays with the calibration round. No GB-25 grade holds more than about 11% of firms in any year.

## 5. Face validity

### 5.1 Matrix

{{FVMAT}}

### 5.2 Constrained features: zero violations

The monotone settings written in each `booster.txt` equal the config (asserted). Fold 3 saved models, 1,000 rows, about 40 grid points per feature: GB-25 36 features, GB-20 35, GB-01 20, all 0 violating rows (`r2_face_validity.csv`). Refit GB-25 models, all folds:

{{ICE}}

neg_equity, debt_split_missing and n_core_imputed have 0 splits in GB-25 fold 3, so the zero violations there are vacuous. The expected-sign flags add nothing beyond the ratios. NaN is outside the monotone order, but no constrained feature's NaN value falls outside its curve.

### 5.3 The decisive test: raise only interest expense

Interest expense × 0.25 ... 4 on 3,000 firms with positive interest, all derived ratios recomputed, no labels used:

{{RAW}}

Split by EBITDA sign:

{{RAWE}}

- **GB-20:** PD steps down somewhere on the path for 78% of firms, and falls for 20% when interest doubles. That is the reversal D-125 worried about, measured directly.
- **GB-25:** 0 step-downs for firms with non-negative EBITDA. **Interest monotonicity holds there exactly.**
- **New finding, shared with the incumbent:** among negative-EBITDA firms (15% of the sample), PD can fall when interest rises: GB-25 40%, GB-01 30%. `interest_coverage = ebitda / exp_financing` moves toward 0 as interest rises when EBITDA is negative, so the ratio improves mechanically. This is a feature-definition issue from v1, not a violation of a constraint. Worst step -0.13 log-odds in GB-25. Not fixed here. Sent to the feature-architect, with the fix tested only on inner splits or at the next refit.

GB-20 and GB-25 interest_to_sales partial dependence (log-odds; training default rates 2007-09):

{{INTEREST}}

### 5.4 Unconstrained features in GB-25: are the shapes defensible?

GB-25 leaves 9 unconstrained: 3 categoricals, asset_turnover, extraord_to_assets, rev_nonpositive (all as GB-01), and bs_gap, fin_fixed_ratio, tfr_ratio (new).

| Feature | Shape in GB-25 | Verdict |
|---|---|---|
| asset_turnover | Rises almost steadily (about -5.50 to -4.99 log-odds across deciles), while raw default rates are hump-shaped. Largest free driver, top-3 PD raiser for 29% of firms | **Defensible with caveat.** "Thin-margin trading" is an assumption, and it can absorb part of the interest effect since interest_to_sales = interest_to_assets / turnover (D-130). Humans should review; same issue exists in GB-01 |
| tfr_ratio | Falls then plateaus; raw rates are hump-shaped, so the effect is conditional. Top-3 raiser for 21% of firms | **Defensible only as a hypothesis** (D-123, [UNVERIFIED]). Needs human sign-off and reason-code wording; the lockbox should confirm it |
| extraord_to_assets | U-shape, also in raw rates (2.81% and 1.88% in the tails vs about 1%) | **Defensible.** Matches the two-sided reason in the spec |
| bs_gap | Small (range 0.13); slightly lower PD with larger gap | **Harmless, weakly explained** ("unclassified liabilities"; hypothesis) |
| fin_fixed_ratio | Nearly flat (range 0.08), top-3 raiser for 0.04% of firms | **Harmless; earns nothing** (D-125). Candidate to drop at a later refit, not now |
| rev_nonpositive | 0 splits | Vacuous |
| ateco_group | Real estate ties for the highest conditional effect with one of the lowest raw rates; probably an offset to low turnover elsewhere in the model (not tested) | **Defensible as an offset, awkward as a reason code.** Merge reason codes by family |
| legal_group | SAU lowest, SRU/SRL highest; code meanings [UNVERIFIED] | Acceptable, with labels to verify |
| macro_area | Flat (range 0.015), probably because province and region encodings took over (not tested) | Harmless. OTHER is a thin level |

Reason-code exposure (fold 3, 20,000 2010 rows):

{{RC}}

Partial-dependence detail, GB-25 (2010 feature rows; default rates from training rows 2007-09):

{{SHAPES}}

Levels:

{{LEVELS}}

## 6. Verdict, finalists, revised rule

| Model | Leakage | Rule 2 vs GB-01 | Face validity | No clearly worse year | Decision |
|---|---|---|---|---|---|
| **GB-25** | CLEAN | Pass, both metrics, holds on 2010 | 0 violations | Pass (2008 log-loss tie; new firms 2008 flagged) | **Promoted: GBM incumbent** |
| GB-20 | CLEAN | Pass, stronger | 0 by spec, but the spec's "none" was post hoc; 1 against the a priori sign, and fails the raw-interest test | Pass | Not promoted. Face validity outweighs -0.0011 AUC / +0.00013 log loss (human may overrule) |
| GB-23 | CLEAN (shared path) | Pass | Same free feature | 2009 log loss worse than GB-20 | Not promoted |
| GB-08, GB-08b | Notes in D-100 | Pass | Fail (D-103) | Pass | Not promoted (D-103 stands; GB-08b needs the humans' answer only if they reopen it) |
| GB-09, GB-10 | Notes in D-100 | GB-09 passes narrowly; GB-10 fails | 0 | GB-09 fragile | Not promoted |
| GB-11, GB-02 | Benchmarks | n/a | Not eligible | n/a | References |
| GAM-01 | CLEAN (D-110) | Loses to GB-25 by +0.0169 AUC | 0 | n/a | Remains transparent fallback |

**The gap to GAM-01 is not a statistical tie**, so the transparency tie-break does not apply: GB-25 vs GAM-01 is ΔAUC +0.0169 [+0.0149, +0.0188], Δlog loss -0.00137 [-0.00151, -0.00122].

**Lockbox finalists for D-119: GB-25, GB-01, GAM-01.**
- GB-25: primary. Never scored on 2011 (no GB-2x name found in `reports/calibration`, `reports/final` or `harness`).
- GB-01: the incumbent it must beat. Its 2011 AUC already exists on file (D-118), so it is not blind; say so.
- GAM-01: transparent fallback, never scored on 2011.
- GB-20 and GB-23 stay out: the GB-25 vs GB-20 choice rests on face validity, which 2011 cannot change, and adding them only spends the lockbox.
- Before opening: freeze code hashes (`build.py` 499d81aeeaca, `run_round2b.py` c24f2ef2af5b, `gbm_lib.py` cd1526eb2dce), seeds and the GB-25 recipe. Pre-register the new-firm cell and interest_to_sales, tfr_ratio and asset_turnover shapes as named read-outs. 2011 still cannot test the level.

### Revised win rule (proposed, D-149)

A challenger wins on statistics only if **all** hold against the incumbent, on identical rows, firm-clustered, at least 1,000 draws:
- **R1 minimum effect:** ΔAUC ≥ +0.002 AND Δlog loss ≤ -0.00015, both pooled and on the clean year.
- **R2 clean year:** both 95% CIs exclude zero in its favour on the year the design did not see.
- **R3 no bad year:** no test year has a CI excluding zero against it on either metric.
- **R4 multiplicity:** the pooled 99.5% CIs exclude zero on both metrics (Bonferroni for about 10 outer-scored configurations in a round).

Then leakage CLEAN, face validity, and the transparency tie-break apply as before. Basis: 0.002 is about 3 s.d. of seed noise (0.0006, D-090); 0.00015 is that gap scaled by GB-25's Δlog-loss/ΔAUC ratio (about 0.07). **Disclosure: proposed after seeing these results**, so it governs the lockbox and later rounds and does not retroactively justify this promotion, which uses the rule published above. The 99.5% bound at 200 draws rests on about half a draw per tail, so R4 for table rows is indicative only.

Applied to round 2 vs GB-01:

{{WIN}}

Statistics alone pass GB-25, GB-20, GB-23, GB-08 and GB-02; of these only GB-25 also passes face validity.

Sensitivity of R1 to the thresholds (`r2_win_rule_sensitivity.csv`):

{{SENS}}

## 7. What was skipped or reduced

- Bootstrap: the single-stream 1,000-draw job was too slow and was stopped. Primary models use 3 × 334 = 1,002 draws across seeds 0-2; table rows, +CAL pairs and segments use 200, which is below the standing 1,000 minimum and allowed by the coordinator for this deadline.
- Refits: GB-25 all folds, GB-20 fold 1 only (fold 3 via saved model; fold 2 not reproduced). GB-23 and phase-A models not reproduced.
- Face validity not re-measured for GB-23, GB-08, GB-08b, GB-09, GB-10, GAM-01; GB-11 has no saved model. Folds 1-2 ICE only for GB-25.
- The wrong-way reason-code check of D-103 was not run for GB-25; only top-3 exposure of free features was.
- Decisions are at the end of this file for the orchestrator to merge; `docs/DECISION_LOG.md` was not edited, per the round's write restriction. The scorecard file name also follows that instruction, `r2_scorecard.md`.
- Figures exist, but no human reviewed them.

## Decision entries for the orchestrator to merge (DECISION_LOG template)

### D-140 Referee scoring method, round 2
- Date / phase / agent: 2026-11-03 / round 2 / referee
- Status: ADOPTED
- Decision: score 11 models plus 4 shipped-level vectors with `metrics.py`, pooled and by year, with 2010 reported separately. Firm-clustered paired bootstrap: 1,002 draws (3 seeds) for primary models, 200 for table rows, +CAL and segments.
- Alternatives considered: one 1,000-draw stream for all models (too slow under the deadline; stopped).
- Business reason: comparisons must reflect that one firm contributes several years.
- Evidence: all 15 contracts PASS (`r2_contract.csv`); fast estimator matches `metrics.paired_bootstrap` to 1.1e-16 (`r2_bootstrap_equivalence.csv`). GB-11 vs GB-01 pooled ΔAUC is +0.0020 [-0.0002, +0.0042] at 1,002 draws, correcting the 200-draw "win" in D-106.
- Source(s): D-005, D-006, D-060; R-16 p. 32 (Brier = overall probability accuracy).
- Risks / what would change our mind: 200-draw bounds are noisy; re-run before quoting any table-row CI near zero.
- Human reviewer (initials) and date:

### D-141 Leakage verdict on the full2 pipeline: CLEAN; harness compliant
- Date / phase / agent: 2026-11-03 / round 2 / referee
- Status: ADOPTED
- Decision: GB-20 and GB-25 CLEAN. GB-23 CLEAN on the shared path. Full2 features comply with CLAUDE.md §2.2.
- Alternatives considered: SUSPECT because target encodings use labels. Rejected: tables come from training labels only, cross-fitted by firm.
- Business reason: a PD built on later or other applicants' data would fail on the holdout and break the assignment's constraints.
- Evidence: see scorecard section 2: saved models and refits reproduce submitted PDs (`r2_repro_saved.csv`, `r2_repro_refit.csv`); label flip 0.0 own-fold change (`r2_oof_label_flip.csv`); single-row vs batch 0.0 on 999 rows, also as strings (`r2_single_vs_batch.csv`); fit calls include 0 test rows (`r2_refit_call_log.csv`); params byte-equal to refit (`r2_param_provenance.csv`).
- Source(s): CLAUDE.md §2.2, §2.4, §2.5; R-47 (cross-fitted encoding, per D-124).
- Risks / what would change our mind: the harness must call with oof=False and re-measure cost (6.4 s per 100k rows under contention, vs 1.0 s for GB-01).
- Human reviewer (initials) and date:

### D-142 Process verdict and selection ledger, round 2
- Date / phase / agent: 2026-11-03 / round 2 / referee
- Status: ADOPTED (reporting rule)
- Decision: process ACCEPTABLE WITH DISCLOSURE. 2010 is clean in code for feature design, but not fully clean for the GB-25 choice. Ledger: 51 prediction files plus 62 calibration configurations = 113 looks; 9 new configurations, 3 on full2.
- Alternatives considered: treating 2010 as fully clean (rejected: `GB-22/dependence_bins.csv` showed 2010 default rates by interest_to_sales decile before GB-25 ran; r1b 2010 cells predate the feature files).
- Business reason: the deck must not present a reused year as an untouched test.
- Evidence: `r2_selection_ledger.json`; file times 14:52 (r1b), 15:12 (feature code), 15:57 (dependence bins), 16:34 (GB-25). GB-25 is lowest of the three full2 configurations on 2010 (0.8348 vs 0.8369, 0.8389), so this choice does not inflate its number.
- Source(s): D-062, D-099, D-100, D-118, D-125, D-129, D-130.
- Risks / what would change our mind: selection optimism remains unquantified; only the lockbox addresses it.
- Human reviewer (initials) and date:

### D-143 Headline number for the deck
- Date / phase / agent: 2026-11-03 / round 2 / referee
- Status: ADOPTED until the lockbox runs
- Decision: quote GB-25 on 2010: AUC 0.8348 [0.8272, 0.8415], AR 0.670, log loss 0.07098 (0.07010 at the shipped level), with gain over GB-01 of +0.0042 [+0.0022, +0.0059] and -0.00038 [-0.00055, -0.00021]. Pooled AUC 0.8424 and +0.0049 go next to it, labelled.
- Alternatives considered: pooled alone; GB-20's +0.0060; the architect's inner +0.0096 (all use years the design saw, or a model not chosen).
- Business reason: the bank should expect the gain a new year is likely to deliver.
- Evidence: `r2_metrics_long.csv`, `r2_marginal_ci.csv`, `r2_paired_bootstrap.csv`.
- Source(s): D-062, D-121.
- Risks / what would change our mind: 2010 is one cycle point; the lockbox replaces this number.
- Human reviewer (initials) and date:

### D-144 GB-25 promoted to GBM incumbent, replacing GB-01
- Date / phase / agent: 2026-11-03 / round 2 / referee
- Status: ADOPTED (referee); human sign-off requested on the trade against GB-20
- Decision: GB-25 is the GBM incumbent. GB-20 and GB-23 are not promoted, and GB-01 stays as the comparison model.
- Alternatives considered: GB-20 (statistically better, fails the raw-interest test); GB-23; GB-08/08b; GB-09; GB-10; GB-01 stays.
- Business reason: better ordering and accuracy than GB-01, without any firm with non-negative EBITDA getting a lower PD for more interest.
- Evidence: vs GB-01 pooled ΔAUC +0.0049 [+0.0037, +0.0061], Δlog loss -0.00035 [-0.00044, -0.00025]; 2008 +0.0027 [+0.0001, +0.0053], -0.00012 [-0.00029, +0.00006]; 2009 +0.0081 [+0.0064, +0.0098], -0.00053 [-0.00070, -0.00038]; 2010 +0.0042 [+0.0022, +0.0059], -0.00038 [-0.00055, -0.00021]. Vs GB-20 -0.0011 [-0.0017, -0.0006], +0.00013 [+0.00009, +0.00018]. Doubling interest lowers PD for 20% of firms in GB-20, 1.7% in GB-25 (`r2_raw_interest_test.csv`).
- Source(s): D-060, D-062, D-065 (rule); R-33 Table 3 p. 11, p. 12 (direction, via D-130); R-39.
- Risks / what would change our mind: humans may value the 2010 cost (-0.0021 AUC vs GB-20) above the shape; a lockbox reversal; new firms in a 2008-like year.
- Human reviewer (initials) and date:

### D-145 Segment check: small firms improve, construction not on the clean year, new firms do not
- Date / phase / agent: 2026-11-03 / round 2 / referee
- Status: ADOPTED (finding)
- Decision: report weak segments as unresolved except small firms; keep the manual-review buffer idea for builders and first-time borrowers (D-113).
- Alternatives considered: claiming a general segment gain from pooled cells (rejected: construction's 2010 gain is +0.001, CI includes 0).
- Business reason: cutoffs for these firms still rest on weaker rankings.
- Evidence: `r2_segments.csv` (200 draws): Q1 AR 0.626 → 0.638; F 0.559 → 0.568, 2010 +0.001; new 0.573 → 0.569, 2008 -0.020 [-0.034, -0.006]. Riskiest-10% capture in 2010: `r2_cap_points.csv`.
- Source(s): D-113, D-128; R-33 Tables 9-14 pp. 19-21.
- Risks / what would change our mind: the final model trains on more years, which may repair the one-training-year new-firm result; untested.
- Human reviewer (initials) and date:

### D-146 Level: the 2010 middle-book under-prediction remains
- Date / phase / agent: 2026-11-03 / round 2 / referee
- Status: ADOPTED (finding); level stays with the calibration round
- Decision: do not claim any level improvement from full2.
- Alternatives considered: a group-level correction on these years (rejected, as in D-078).
- Business reason: in a 2010-like year, middle-book pricing would still be about a quarter too low.
- Evidence: GB-25+CAL obs/pred 1.03 / 1.06 / 1.19 by year; grades 6-14 in 2010: 2,108 vs 1,590 expected, 1.33; CAL-01 1.32; same firms 1.325 vs 1.319 (`r2_level_by_year.csv`, `r2_grade_bands.csv`, `r2_grade_same_firms.csv`).
- Source(s): D-070, D-071, D-116, D-129.
- Risks / what would change our mind: rule e inherits selection optimism (D-071) and 2011 cannot test the level (D-118).
- Human reviewer (initials) and date:

### D-147 Face validity: zero violations; free-feature review; a negative-EBITDA quirk found
- Date / phase / agent: 2026-11-03 / round 2 / referee
- Status: zero violations ADOPTED; tfr_ratio and asset_turnover wording OPEN for humans; interest_coverage quirk OPEN for feature-architect
- Decision: count GB-25 as 0 violations, and record GB-20 as 0 by spec but 1 against the a priori sign. Add the raw-input test to the referee's standard checks.
- Alternatives considered: counting only constraint checks (rejected: they miss a free near-duplicate undoing a constrained direction).
- Business reason: reason codes must make sense to a credit officer.
- Evidence: ICE 0 violating rows in all folds for GB-25 (`r2_ice_refit_gb25.csv`), fold 3 for GB-20 and GB-01 (`r2_face_validity.csv`). Raw-interest test: non-negative EBITDA step-downs 0% GB-25, 0% GB-01, 77% GB-20; negative EBITDA (15% of firms) 40% GB-25, 30% GB-01, 86% GB-20 (`r2_raw_interest_by_ebitda.csv`). Unconstrained: 9 in GB-25; asset_turnover top-3 raiser 29%, tfr_ratio 21% (`r2_unconstrained_reason_codes.csv`).
- Source(s): D-022, D-064, D-123 ([UNVERIFIED] TFR readings), D-125.
- Risks / what would change our mind: the interest_coverage rule applies to the incumbent too; changing it means a feature change, so it needs its own inner-only test before any outer look.
- Human reviewer (initials) and date:

### D-148 Lockbox finalists (amends D-119): GB-25, GB-01, GAM-01
- Date / phase / agent: 2026-11-03 / round 2 / referee
- Status: OPEN (needs orchestrator and human sign-off before anyone opens 2011)
- Decision: replace D-119's finalists with GB-25 (primary), GB-01 (incumbent to beat), GAM-01 (transparent fallback). No other model is scored on 2011. Pre-register the new-firm cell and three shape read-outs, and freeze the code hashes listed in section 6.
- Alternatives considered: adding GB-20/GB-23 (rejected: face-validity decision; would spend the lockbox); GB-25 alone (rejected: no comparison).
- Business reason: one sealed check of the model the bank would actually use.
- Evidence: scorecard sections 3 and 6; no GB-2x name in `reports/calibration`, `reports/final` or `harness` by file search.
- Source(s): D-087, D-118, D-119; R-33 Sec. 4.8 p. 23.
- Risks / what would change our mind: GB-01's 2011 AUC is already on file, so that side is not blind; one year, one cycle point; 2011 cannot test the level.
- Human reviewer (initials) and date:

### D-149 Revised win rule: minimum effect, clean-year CI, no bad year, multiplicity
- Date / phase / agent: 2026-11-03 / round 2 / referee
- Status: OPEN (proposal; applies to the lockbox and later rounds if signed off; not used for D-144)
- Decision: adopt R1-R4 in scorecard section 6: ΔAUC ≥ +0.002 and Δlog loss ≤ -0.00015 pooled and on the clean year; clean-year 95% CIs exclude zero on both metrics; no year's CI against; pooled 99.5% CIs exclude zero on both.
- Alternatives considered: the current rule (a +0.00003 year counts as better, per D-109); per-year CIs required in every year (too strict: one fold of 2,064 defaults decides; GB-25 itself ties on 2008 log loss).
- Business reason: a difference too small to matter to a lending decision should not change the model.
- Evidence: `r2_win_rule_test.csv`, `r2_win_rule_sensitivity.csv`. Threshold basis: seed s.d. 0.0006 (D-090) × about 3, scaled by GB-25's ratio. Set after seeing results; disclosed.
- Source(s): D-100, D-109; assumption for thresholds.
- Risks / what would change our mind: 2011 has fewer usable months under y11, so thresholds may need restating before opening, never after.
- Human reviewer (initials) and date:
