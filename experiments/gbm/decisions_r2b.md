# Round 2 phase B decisions (challenger-gbm, 2026-11-02): D-120 .. D-130

For the orchestrator to merge into `docs/DECISION_LOG.md`. Every number is read from a file written by
`experiments/gbm/run_round2b.py`, `explain_r2b.py` or `boot_r2b.py`, all collected in
`experiments/gbm/comparison_r2b.md`. Proposals only: promotion is the referee's call and sign-off is the humans'.
Bootstrap = referee's firm-clustered paired bootstrap; 1,000 draws for GB-20 and GB-23 vs GB-01
(`comparison_r2b_bootstrap_1000.csv`), 200 draws elsewhere. d = first model minus second.
"Inner" = slice B of a fold's own training window (fold-2 inner: fit 2007, early stop 2008 A, score 2008 B;
fold-3 inner: fit 2007-08, early stop 2009 A, score 2009 B). 3 seeds, GB-01's cfg for that fold.

### D-120 Feature set full2 beats v1 in the GBM (GB-20 meets the win rule vs GB-01)
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED by the GBM challenger as its round-2 entry; promotion over GB-01 is OPEN for the referee
- Decision: GB-20 = GB-01's grid, inner scheme, calibration and constraint rule on the 45 full2 features.
  Training rows use out-of-fold encodings; slices A/B and test rows use the fold's full-window table.
- Alternatives considered: stay on v1 (GB-01); GB-23 (D-127); partial drops (D-122).
- Business reason: better ranking for approve/decline and better log loss for pricing, with no monotone violation.
- Evidence: pooled AUC 0.8435 vs 0.8375, dAUC +0.0061 [+0.0047, +0.0074]; d log loss -0.00048 [-0.00059, -0.00038].
  Per year dAUC: 2008 +0.0034 [+0.0006, +0.0062], 2009 +0.0086 [+0.0067, +0.0109], 2010 +0.0062 [+0.0040, +0.0084].
  d log loss: 2008 -0.00018 [-0.00036, +0.00001], 2009 -0.00066 [-0.00085, -0.00048], 2010 -0.00059 [-0.00078, -0.00041].
  Calibration slope pooled 0.996 vs 0.976; 2010 0.957 vs 0.957. Common calibration (CAL-01 method): GB-20+CAL vs CAL-01
  dAUC +0.0060 [+0.0049, +0.0075], d log loss -0.00050 [-0.00060, -0.00038] (200 draws).
  Recipe check: run on v1 features, the driver reproduces GB-01's best_iter (408, 322) and B log loss (0.05579, 0.06518).
- Source(s): R-38 (algorithm), R-39 (monotone_constraints option); D-062 reporting rule.
- Risks / what would change our mind: the referee's refit check; a reversal on the lockbox (D-119 currently freezes only
  GB-01 and GAM-01, so GB-20 must be added before anyone opens 2011).
- Human reviewer (initials) and date:

### D-121 The feature-architect's +0.0096 is confirmed in direction, smaller on the clean year
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (reporting)
- Decision: quote +0.0062 (2010) next to the pooled +0.0061; do not quote +0.0096 as the expected gain.
- Alternatives considered: quoting the inner number, or the 2009 number (+0.0086). Both use a year the features were designed on (D-062).
- Business reason: the deck must state the gain a new year is likely to deliver.
- Evidence: fold-3 inner on the frozen scheme +0.0093 [+0.0066, +0.0119], close to the architect's check (D-090). Outer: 2009 +0.0086,
  2010 +0.0062. The feature-architect's concern that the gain is untuned capacity is not supported: v1 and full2 each stop early
  on their own best tree count (fold-3 inner mean 350 vs 595), and the gain survives GB-09's tuning (D-127).
- Source(s): D-062, D-090.
- Risks / what would change our mind: one clean year only; 2010 is also one point in the cycle.
- Human reviewer (initials) and date:

### D-122 Which groups carry the gain: F1 and F5 are robust, F3 only in the longer window, F2 and F4 add little
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (finding); the set stays as designed, so that GB-20 stays a test of the pre-declared full2
- Decision: report by group. Do not read SHAP share as value.
- Alternatives considered: pruning F2 now. Not done: inner gain from pruning is not established, and pruning after seeing the outer result needs its own run.
- Business reason: fewer, explainable drivers are easier to defend to a credit officer.
- Evidence (GB-21, dAUC = drop group minus full2, fold-2 inner; fold-3 inner): F1 -0.0025 [-0.0041, -0.0006]; -0.0016 [-0.0030, -0.0006].
  F5 -0.0020 [-0.0041, -0.0003]; -0.0022 [-0.0033, -0.0011]. F3 -0.0003 [-0.0030, +0.0025]; -0.0028 [-0.0044, -0.0013].
  F4 +0.0001 [-0.0012, +0.0012]; -0.0011 [-0.0018, -0.0003]. F2 +0.0004 [-0.0011, +0.0019]; -0.0008 [-0.0019, +0.0004]
  (log loss +0.00003, +0.00014). All 19 together: -0.0053, -0.0093. The single drops sum to far less than the total, so the groups
  substitute for each other. F2 is the largest SHAP family (sum of mean |SHAP| 0.56) with no measurable AUC value: its attribution is
  taken from equity_ratio (0.212 to 0.091) and roa_net (0.208 to 0.128).
- Source(s): R-41 (TreeSHAP); D-025 (attribution among correlated features).
- Risks / what would change our mind: inner slices are 2008 and 2009, which the feature design saw (D-062).
- Human reviewer (initials) and date:

### D-123 tfr_ratio: conditional effect is "more fund, lower PD"; D-097's interaction wording is corrected
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (reading); stays unconstrained
- Decision: describe TFR to a credit officer as "a large staff-severance fund for a given wage bill goes with lower risk, other things equal", not as "labour intensity".
  We do not impose a minus sign: that would be chosen after seeing the shape (the D-063 pattern).
- Alternatives considered: D-097's reading (labour intensity; "the tree uses it in interaction"); sign minus; dropping it.
- Business reason: this is the most robust new single feature, so Kiet must be able to explain it.
- Evidence: drop tfr_ratio alone -0.0023 [-0.0042, -0.0007] and -0.0020 [-0.0030, -0.0010], the only single drop whose CI excludes zero and that is negative in both inner windows among those tested.
  SHAP by band (2010 sample): 0 +0.112, 0-2% +0.094, 2-5% -0.096, 5-10% -0.283, >10% -0.199.
  (1) Sector mix hides it: 48% of zero-TFR rows in the 2010 sample are real estate. Training-row AUC 0.510 all, 0.473 excluding L, 0.448 in manufacturing.
  (2) Suppression: standardised logit coefficient (excl. L) -0.043 alone, -0.197 next to personnel_to_sales (+0.325).
  (3) Fund / wage bill: default rate falls from 2.3% (decile 1) to 1.0% (decile 9), then 1.7% in decile 10.
  (4) Equity: tfr SHAP is the same across equity terciles (5-10% band: -0.27, -0.29, -0.28), so no equity interaction, although raw rates rise with TFR among low-equity firms (2.1% to 4.5%).
  (5) Turnover: a real interaction. Low-turnover, TFR > 5% firms: raw default 5.7-11.5% on 880 rows, 62 defaults; high-turnover 0.7-1.1%. SHAP in the 5-10% band: -0.09 low, -0.22 mid, -0.32 high turnover.
- Source(s): assumption. No VERIFIED source for TFR. The readings "tenure / maturity" and "patient funding owed to staff" are hypotheses [UNVERIFIED]. The wage bill uses D-091's goodwill hypothesis.
- Risks / what would change our mind: the bank's field definition; TFR rules are Italy-specific; found after looking at the data, so the lockbox should confirm it.
- Human reviewer (initials) and date:

### D-124 Encodings: division persists, province only partly, region is erratic; the group's value needs more training years
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (kept in full2); province_te flagged as the weakest of the three on persistence
- Decision: keep all three. Say plainly that the evidence for province alone is not significant.
- Alternatives considered: dropping province_te and region_te (inner: +0.0003 [-0.0023, +0.0027]; -0.0013 [-0.0026, +0.0001]). Not significant either way, so no post-hoc removal.
- Business reason: industry and location add context that a firm's own statement lacks, with unseen levels mapped to the prior.
- Evidence: single drops, fold-2; fold-3 inner: division -0.0000; -0.0009 [-0.0016, -0.0002]. Province +0.0011 [-0.0017, +0.0034]; -0.0007 [-0.0018, +0.0002]. Region +0.0005; -0.0001 (ns).
  Group: -0.0003 (one fit year); -0.0028 (two fit years). Slope of next-year default on the encoding (1 = taken at face value, diagnostic, `encoding_persistence.csv`):
  division 1.22 / 0.87 / 0.85; province 0.50 / 0.78 / 0.48; region 0.06 / 1.31 / 0.95. Rank correlation with next-year level rates: division 0.65 / 0.64 / 0.72; province 0.20 / 0.27 / 0.40.
  Correction to the brief: the encodings carry expected sign + (D-094), so under the "signs from expected_signs" rule they ARE constrained in GB-20; their rising shapes are imposed, not evidence.
  The unconstrained new features are interest_to_sales, bs_gap, fin_fixed_ratio and tfr_ratio.
- Source(s): R-47 Sec. 1.2, 2.7, 4.2-5 (regularised, cross-fitted target encoding; generic ML datasets, not credit). R-46 abstract only (origin of the idea; our m is our own design, R-46's formula is not verified).
- Risks / what would change our mind: geography was fitted on 2007-09 conditions.
- Code mapping (status corrected 2026-11-02): province and division labels are VERIFIED against R-53 and R-54. All 110 province-to-region pairs in `src/data/codes.py` match R-53's file; 100 = Prato (Toscana), 63 = Napoli, 21 = Bolzano. Divisions per R-54 pp. 51-53: 13 textiles, 14 clothing, 15 leather, 29 motor vehicles, 31 furniture (all section C), 55 accommodation (I), 61 telecommunications (J), 68 real estate (L). The lists are verified, but that the data provider uses these codes is still an inference from the data (D-012, N-53). Linking Prato to its textile activity is our own reading, not a claim from these sources.
- Human reviewer (initials) and date:

### D-125 Face validity: zero violations, but interest_to_sales can undo a constraint (decision needed)
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: zero-violation finding ADOPTED; interest_to_sales sign resolved by D-130 (the about -0.001 cost below is the inner estimate; the outer cost on 2010 is about double)
- Decision: propose restoring R-33's + sign on interest_to_sales in the final configuration, subject to one outer scoring.
- Alternatives considered: leave free (GB-20); drop it. Inner: constrained + -0.0012 [-0.0025, -0.0001] and -0.0010 [-0.0021, -0.0000]; d log loss +0.00017, +0.00004. Drop -0.0008 and -0.0023.
- Business reason: "more interest per euro of sales lowers PD" cannot be defended to a credit officer.
- Evidence: GB-22, fold-3 model: 32 constrained features agree, 3 have 0 splits, 0 violations; ICE 500 rows x 35 features 0 violating rows (folds 1-2 by construction). A first run flagged n_core_imputed; that was a bug in our check (a one-point grid gave NaN). The grid now spans the observed min-max.
  interest_to_sales SHAP: deciles 3-4 about -0.09, 7-8 about +0.10, 10 -0.15, while raw default in decile 10 is 2.4%. Spearman 0.64 with interest_to_assets (constrained +), so this free near-duplicate can partly reverse a direction we impose elsewhere; D-099 item 6 already flags the sign as post hoc.
  bs_gap (+0.06 at exact 0 to -0.08) and fin_fixed_ratio (+0.02 to -0.04) are small; the first is "hypothesis only", and drop fin_fixed_ratio is +0.0016 [+0.0005, +0.0027] and -0.0004 (ns), so it earns nothing.
  n_core_imputed has 0 splits in fold 3: D-095's "adds back risk" does not happen in this model (native NaN already handles it). Reason codes: 42% of firms' top 3 repeat a family, so codes must be merged by family.
- Source(s): R-33 p. 12 (upward slope); R-39; R-41.
- Risks / what would change our mind: the outer score of that configuration; the AUC cost is real, about 0.001.
- Human reviewer (initials) and date:

### D-126 About 40% of full2's AUC gain buys back the cost of the constraints; the constraints stay
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (constraints kept; unconstrained twin is a diagnostic only, not outer scored)
- Decision: report both parts of the gain. full2 + GB-08's relaxation is left for a joint decision with phase A.
- Alternatives considered: unconstrained full2 (fails face validity, as GB-02 did, D-064); full2 + GB-08 relax.
- Business reason: separates new information from a looser effective constraint, which a validator would ask about.
- Evidence (inner, fold 2; fold 3): gain from full2 with constraints +0.0053; +0.0093. Without constraints on both +0.0033 [-0.0005, +0.0073]; +0.0054 [+0.0024, +0.0075]. Log loss -0.00022; -0.00062 vs -0.00005; -0.00031.
  Constraint cost on full2 +0.0032; +0.0048 AUC (v1: +0.0052; +0.0088). So constraints cost about 40% less on full2, not zero. GB-08 relax adds about the same on either set: v1 +0.0024, +0.0037; full2 +0.0023, +0.0028.
- Source(s): assumption (face validity, D-051); R-39.
- Risks / what would change our mind: the relaxed set frees interest_coverage but leaves interest_coverage_sect_pct constrained, an inconsistency to settle first.
- Human reviewer (initials) and date:

### D-127 GB-23 (full2 + GB-09's tuned settings): not adopted
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: REJECTED as the entry; tuning remains an option for the final-window refit
- Decision: keep GB-20's pre-declared grid.
- Alternatives considered: per-fold A-selected mix (GB-20 fold 1, GB-23 folds 2-3): not built, because it was never a pre-declared configuration.
- Business reason: simpler, and no reliable gain.
- Evidence: inner choice recorded before GB-23's outer run (`GB-23/inner_choice.csv`). A gain on full2: -3e-7, +5e-8, +1.0e-4 (GB-09 cfg "wins" folds 2-3, effectively a tie in 2 of 3). B check +2.1e-4, +2.5e-4, +1.9e-4. So the v1 tuning does not hurt on full2, but its settings differ widely by fold (learning rate 0.024 / 0.05 / 0.016; final trees 674 / 253 / 1,271).
  Outer, reported only: vs GB-20 pooled +0.0007 [+0.0002, +0.0012], 2009 -0.0004 with log loss +0.00015 [+0.00003, +0.00026], 2010 +0.0020 [+0.0011, +0.0026]. Fails "better in all 3 years". vs GB-01 it wins, like GB-20.
- Source(s): D-052; assumption.
- Risks / what would change our mind: fold-3 prefers the tuned cfg on A, B and 2010, on v1 too (GB-09 2010 +0.0022). It may be that low learning rate and stronger penalties pay off once data are plentiful. Testable in the final window, on inner slices.
- Human reviewer (initials) and date:

### D-128 More features did widen the same-year gap; the new-firm shortfall is 2008 only
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (reporting)
- Decision: state the wider gap in the deck, with the out-of-time evidence next to it.
- Alternatives considered: reporting only the OOT slope, which would hide the gap.
- Business reason: a validator will look for overfitting; PD spread drives pricing.
- Evidence: same-year in-sample minus OOT AUC, GB-01 vs GB-20 (out-of-fold encodings): 2008 fold 2 0.0262 vs 0.0349; 2008 fold 3 0.0173 vs 0.0259; 2009 fold 3 0.0146 vs 0.0217. So +0.007 to +0.009. Full-table encodings add about 0.001. Final trees 338/490/386 vs 444/722/737. Inner Platt slopes dropped (1.017/0.943/0.998 to 0.953/0.919/0.961), the map corrects this; OOT slopes 0.996 pooled, 2010 unchanged at 0.957.
  In-sample AUC rises more than OOT AUC, but OOT still rises. Reading: extra fit is partly noise, and does not cost out-of-time performance.
  New firms: pooled dAUC -0.0011 [-0.0061, +0.0040] vs seen +0.0069. By year new: 2008 -0.0100 [-0.0174, -0.0021] (log loss +0.00088 [+0.00039, +0.00134]), 2009 +0.0083 [+0.0000, +0.0167], 2010 +0.0079 [-0.0025, +0.0189]. Seen-minus-new gap: 2008 +0.0166 [+0.0085, +0.0244], 2009 +0.0002, 2010 -0.0015 (ns).
  So only fold 1 (one training year, 1,379 defaults) loses on new firms. Our hypothesis: too little data for 45 features and one-year encodings. Not tested further; this is one fold.
- Source(s): D-117; R-33 Sec. 4.7-4.8 (same-year gap reporting).
- Risks / what would change our mind: if the holdout is new-firm heavy and resembles 2008; final training has more years, like folds 2-3.
- Human reviewer (initials) and date:

### D-129 What didn't work, what was not done, and process disclosures
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: REJECTED (items 1-3); OPEN (items 4-5); disclosures for the referee (item 6)
1. Isotonic: B log loss worse than Platt in every fold (0.04700 vs 0.04410; 0.05585 vs 0.05522; 0.06414 vs 0.06380). Fold-1 B has 203 defaults, not eligible anyway. Platt kept.
2. full2 does not fix the level: 2010 obs/pred at common level 1.19 vs CAL-01 1.18. Level stays the calibration round's job.
3. Sector percentiles as a source of power, and "the tree uses TFR in interaction with leverage": neither holds (D-122, D-123).
4. NOT run on full2: native NaN vs shared imputation, the depth/grid search beyond GB-09's cfg, XGBoost. Round-1 tie (D-056) is assumed to carry over.
5. Prediction cost measured under CPU contention: 4.7-5.3 s per 100k rows at 4 threads vs 1.0 s for GB-01 (8 threads, D-059). The harness-engineer must re-measure; most of it is the feature transform.
6. Disclosures. Outer folds scored once each for GB-20 and GB-23; bootstraps only re-read saved predictions. Lockbox: all reads use the fs_year < 2011 filter. GB-24 (inner-only) and GB-23's inner choice were run after GB-20's outer result was known, and the interest_to_sales proposal comes from those runs, so they are not blind. 2010 labels appear in diagnostics (encoding persistence, TFR cells, SHAP tables), no configuration choice uses them. TFR and suppression readings were developed after looking at training data and 2010 cells. Fold 1 inner is a firm split within 2007, as D-061 notes for GB-01. The driver gained subcommands after GB-20/GB-21 ran; their code path is unchanged. `score_r1.py` functions are exec'd from source (its body would rescore everything), with an L1 column mirrored and then discarded. gbm_lib.py and phase A's files were not edited.
- Source(s): D-056, D-059, D-061, D-072.
- Human reviewer (initials) and date:

### D-130 GB-25: restoring interest_to_sales' + sign costs 0.0011 AUC pooled and 0.0021 on 2010; proposed as the GBM entry
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: proposed (referee and human sign-off OPEN); supersedes GB-20 as the proposed entry in D-120
- Decision: use GB-25 (GB-20 with interest_to_sales constrained +1, all else identical) as the GBM entry. Pre-specified before any outer score for it; nothing was changed after seeing the result. This is a paid face-validity trade, not a performance win: GB-20 is better on the pooled metrics.
- Alternatives considered: keep GB-20 (better metrics, but a reversal we cannot defend); drop the feature (inner -0.0008, -0.0023, D-125).
- Business reason: a decline or price must not rest on "more interest per euro of sales lowers PD". The sign also returns to its a priori direction, removing the post-hoc change in D-099 item 6.
- Evidence (`GB-25/gb25_*.csv`, 1,000 draws):
  Cost vs GB-20: pooled dAUC -0.0011 [-0.0017, -0.0006], d log loss +0.00013 [+0.00009, +0.00018]. 2008 -0.0007 [-0.0017, +0.0002]; 2009 -0.0005 [-0.0015, +0.0005]; 2010 -0.0021 [-0.0029, -0.0013], d log loss +0.00021 [+0.00014, +0.00028]. Common calibration pooled -0.0011 [-0.0017, -0.0006], +0.00012 [+0.00008, +0.00016]. The 2010 cost is about double the inner estimate (-0.0010).
  Vs GB-01: pooled +0.0049 [+0.0037, +0.0062], d log loss -0.00035 [-0.00045, -0.00026]; 2008 +0.0027 [+0.0001, +0.0054]; 2009 +0.0081 [+0.0065, +0.0101]; 2010 +0.0042 [+0.0021, +0.0059]. Meets the win rule vs GB-01. New firms 2008 stays negative (-0.0101 [-0.0173, -0.0027]), 2009 and 2010 positive (ns on AUC).
  AUC by year 0.8468 / 0.8441 / 0.8348, pooled 0.8424; log loss 0.05411 / 0.06563 / 0.07098, pooled 0.06380; slope 1.005 / 1.014 / 0.957, pooled 0.991.
  Monotone: 36 constrained, 0 violating (ICE 500 rows, 0); neg_equity, debt_split_missing and n_core_imputed have 0 splits.
  Same-year gap: 0.0292 / 0.0220 / 0.0172 vs GB-20 0.0349 / 0.0259 / 0.0217 and GB-01 0.0262 / 0.0173 / 0.0146. About 45-65% of the widening in D-128 goes away (final trees 461 / 576 / 548).
  Reversal (fold 3, same 20,000 2010 rows): interest_to_sales SHAP in decile 10 -0.149 to +0.059, decile 9 -0.030 to +0.060. Partial dependence now rises (range 0.34 to 0.14; splits 297 to 66). Mean SHAP plateaus across deciles 7-10 (+0.06 to +0.07) and dips slightly in the lowest bins; these are averages over firms, not violations.
  Partly moved, not fully: in decile 10 the feature's own SHAP rises 0.207, offset by ar_days -0.069, asset_turnover -0.029 and ateco_group -0.017; total log-odds rises 0.112, and mean raw PD goes from 1.54% to 1.67% against an observed 2.36%.
  interest_to_assets: stays monotone + and does not take up the slack. It weakens: mean |SHAP| 0.057 to 0.045 (rank 19 to 21), partial-dependence range 0.30 to 0.23, decile 10 SHAP +0.100 to +0.082. GB-01 had 0.069 and 0.34.
- Source(s): R-33 Table 3 p. 11, p. 12 (upward slope); R-39 (monotone_constraints); R-41 (TreeSHAP).
- Risks / what would change our mind: the humans may weigh the 2010 cost more than the shape. The re-chosen fold-2 cfg (min_data_in_leaf 2000 vs 500) follows the same recipe, so that fold's comparison also includes a cfg change. Part of the reversal now sits in correlated unconstrained asset_turnover. GB-25 is the third full2 configuration scored on 2008-10; the sign was pre-specified, but the proposal came after GB-20's outer result (D-129). D-119's lockbox must name the chosen entry before anyone opens 2011.
- Human reviewer (initials) and date:
