"""Fill the @@...@@ narrative blocks of r1b_diagnostics_template.md (text by the referee; every number is
copied from the r1b_*.csv files rendered in the same document). Run before r1b_build_md.py."""
from pathlib import Path

OUT = Path(__file__).resolve().parent
TPL = OUT / "r1b_diagnostics_template.md"

N = {}
N["SUMMARY"] = """\
| # | Finding | Key numbers (files) | What it means |
|---|---|---|---|
| 1 | **EBM verdict: CLEAN, 0 face-validity violations, not promoted.** GAM-01 loses to GB-01 on both metrics in every year and firm group; beats SC-03 and RC-02. | GAM-01 vs GB-01 pooled ΔAUC −0.0120 [−0.0139, −0.0102], ΔLL +0.00102 [+0.00088, +0.00115]; vs SC-03 +0.0089 [+0.0073, +0.0105], −0.00139 [−0.00155, −0.00124]; vs RC-02 +0.0061 [+0.0044, +0.0078], −0.00063 [−0.00078, −0.00048] (`r1b_gam_paired_bootstrap.csv`) | Bank: GB-01 stays the champion. Roger: the glass box costs about 0.012 AUC; GAM-01 is now the best fully transparent model and should replace RC-03 as the documented fallback. |
| 2 | **Power by segment.** GB-01 is the best or tied model in every cell; no segment where a transparent model is significantly better. Power is weakest for small firms, construction (F) and new firms. | GB-01 AR: size Q1 0.626 → Q5 0.735; C 0.745, F 0.559, L 0.643; seen 0.688, new 0.573 (`r1b_segment_power.csv`) | Bank: rankings are least reliable for small firms, builders and first-time borrowers; cutoffs there deserve a manual review buffer. Roger: the RiskCalc-style AR-by-segment tables (R-33 Tables 9-14) now exist for our models. |
| 3 | **Missing inputs (L1): GB-01 shows no extra under-prediction** on rows with missing debt split or missing core inputs; the EBM and the WoE scorecard do, on the 0.36% of rows with a missing core ratio. | Relative obs/pred (vs same-year average) GB-01: debt_split_missing 1.04 [0.98, 1.09], core NaN 1.01 [0.84, 1.22]; GAM-01+CAL core NaN 1.21 [1.00, 1.46], SC-03 1.30 [1.07, 1.56] (`r1b_missing_tests.csv`) | Bank: no extra "missing-data" add-on is needed for GB-01. Roger: Moody's warning about mean/median fills (R-34) is visible in our data, but only for the models that fill with an average. |
| 4 | **Re-split grade grid: 16 grades, none above 10.3% of firms** (edges set on 2007-2009 predictions). On 2010, observed default rates rise monotonically across all 16 grades. | max share 9.9% (edge sample), 10.3% (OOT 2008-09), 10.0% (2010) (`r1b_grade_summary.csv`) | Bank: a usable master scale; no grade lumps a fifth of the book. Roger: the grade design follows R-33 Sec. 4.9 and was not fitted to the test year. |
| 5 | **The 2010 under-prediction, quantified.** With the shipped level logic (CAL-01), 2,968 defaults were observed against 2,511 expected (obs/pred 1.18, 457 extra defaults). The shortfall sits in grades 6 and 8-14 (PD 0.5%-10%, 50% of firms, +502 defaults); grades 1-5 are on target and the top grade (PD ≥ 20%) is over-predicted. | `r1b_grade_table.csv`, `r1b_grade_summary.csv` | Bank: in a 2010-like year the PDs of the middle of the book were 20-30% too low (observed 1.26-1.41x predicted); the safest and the riskiest firms are priced about right. Roger: this is a level (cycle) miss, not a ranking failure. |
| 6 | **Overfitting evidence.** OOT calibration slopes 0.96-1.01 by year (no over-fitting of the PD spread, except new firms 0.88 for every model). GB-01's in-sample AUC is 0.015-0.026 above its OOT AUC on the same year; RC-02 0.004-0.006, GAM-01 0.006. 2008-2010 have been scored by about 90 prediction vectors. | `r1b_stability_summary.csv`, `r1b_same_year_overfit.csv`, `r1b_insample_vs_oot.csv`, `r1b_selection_ledger.json` | Bank: quote only OOT numbers; the in-sample fit of the GBM is optimistic. Roger: walk-forward OOT is our evidence, as in R-10/R-33; the test years are now heavily reused, so one sealed check is needed. |
| 7 | **Lockbox: fs_year 2011 is NOT untouched** (correction to D-087). GB-01 and RC-02 fold-4 models were already scored on 2011 (AUC columns in `reports/calibration/q5_2011_other_labels_fold4.csv`, `q6_2011_pseudo_test.csv`; D-076), and the level tau is anchored to the 2011 default rate. GAM-01/02 and SC-03 have not been scored on 2011. | file headers only (values not read) | Use 2011 once, pre-registered, for the GB-01 vs GAM-01 ranking decision only; 2011 cannot test the level. |
"""

N["PROCESS"] = """\
Process audit (code read: `experiments/gam/run_gam.py`, `gam_lib.py`, `explain_gam.py`; `src/features/build.py`):

- **In-window fits.** Feature parameters (`B.fit`) on inner-train rows for the inner model and on all
  training rows for the final model; the EBM's early-stopping/validation rows are passed explicitly via
  `bags` and are always training rows (folds 2-3: A = half of the last training year, by firm; fold 1:
  random firm split inside 2007); grid choice by A log loss; Platt slope on B (other half of the last
  training year); level anchored to the training default rate; missing-bin reset (D-082) uses training
  default counts only. The +CAL shift uses `reports/calibration/ct_targets.csv` rule e, which is built
  from training-window labels (D-070). No `def_date`, label, or future-year field is a feature. Test
  rows enter only `predict_pd`, which calls `B.transform` with frozen parameters (row-wise; verified above
  with label columns removed and rows shuffled).
- **Same splits.** Train/test row counts equal `walkforward.json` (asserted in the runner and recomputed here).
- **Process notes (not leakage).** (a) The D-082 missing-bin reset rule was added after the first full run
  had produced OOT metrics (`gam_lib.py` 11:39:30, `metrics_run1_before_missing_reset.md` 11:39:47 on
  2026-10-30). Unlike RC-03 (D-063) it is not presented as a performance result, its trigger was the fitted
  training shapes (19 training rows, 0 defaults → −5.1 log-odds), it is a harness-robustness rule, and its
  OOT effect is negligible (pooled AUC 0.82504 → 0.82545 per the challenger's run-1 file). (b) The final
  run reused run 1's in-window grid choice (`--fixed-from GAM-01`), and GAM-02 reused GAM-01's choice
  without its own grid: both choices are in-window. (c) The +CAL files inherit the selection optimism of
  rule e (chosen among 7 level rules after seeing 2008-2010, D-071), exactly like CAL-01. (d) Shared D-062
  caveat (feature set designed on 2007-2009). (e) Single seed.
"""

N["FACE"] = """\
Result: **0 violations in 20 sign-expected terms × 3 folds × 2 models** (the constraints hold; the
missing bin and the unknown bin are outside the constraint and are reported separately). GAM-02's 10 pair
terms per fold involve only features with no expected sign (asset_turnover, extraord_to_assets,
rev_nonpositive, ateco_group, legal_group, macro_area), so pairs cannot break a sign. This confirms D-084.
Items a credit officer will question (not counted as violations because the spec gives no sign):
`rev_nonpositive = 1` lowers the score (−0.19 in fold 3; it overlaps with asset_turnover ≈ 0, which is
−0.47); `legal_group OTHER` −0.87 and `macro_area OTHER` +0.65 are thin pooled levels that unseen codes
also map to; most missing bins were reset to 0 (average firm), which matters for section 3.
"""

N["VERDICT"] = """\
- **GAM-01: CLEAN** (leakage) with the process note above; contract PASS; reproduction exact; 0 face-validity
  violations. **GAM-02: CLEAN**, same notes.
- **Promotion vs GB-01: NO.** Rule 2 fails in the wrong direction: pooled ΔAUC −0.0120 [−0.0139, −0.0102],
  ΔLL +0.00102 [+0.00088, +0.00115]; worse in 2008, 2009, 2010, for seen and for new firms (every CI excludes
  zero against GAM-01). Same with the level shift (GAM-01+CAL vs CAL-01: −0.0119 [−0.0137, −0.0101],
  +0.00098 [+0.00085, +0.00111]). This is not a statistical tie, so transparency does not decide it.
- **vs SC-03 (the round-1 benchmark): GAM-01 wins** on both metrics, in every year and for new firms.
- **vs RC-02 (best transparent model so far): GAM-01 wins** (ΔLL CI excludes zero pooled, in each year and
  for seen firms; ΔAUC excludes zero pooled and in 2009/2010; 2008 ΔAUC +0.0020 [−0.0014, +0.0053] is a tie;
  new-firm ΔLL −0.00029 [−0.00071, +0.00014] is a tie) and has fewer face-validity violations (0 vs 2,
  D-064). **GAM-01 should replace RC-03 as the transparent fallback named in D-067.**
- **GAM-02 vs GAM-01:** ΔAUC +0.0008 [+0.0004, +0.0012], ΔLL −0.00008 [−0.00010, −0.00005]; 2008 and new-firm
  AUC ties. Statistically detectable, economically small, ten extra heat maps: transparency wins, GAM-01 kept
  (agrees with D-083).
"""

N["SEGMENTS"] = """\
Reading the segment tables:
- **GB-01 is the best or statistically tied model in every cell.** The GB-01 minus GAM-01+CAL AUC CI
  excludes zero in every year, every size quintile, both firm groups and sections C, E, F, G, J, L, M;
  it includes zero in A, D, H, I, N, Q, R, S and the pooled cell (30-225 defaults each), where single-model
  AUC CIs are 0.04-0.23 wide. In D (32 defaults) and S (30 defaults) GAM-01's point estimate is higher, both ties.
- **Power rises with size**: GB-01 AR 0.626 (Q1, smallest) to 0.735 (Q5). Every model shows this gradient.
- **Sectors**: highest power in manufacturing C (AR 0.745) and hotels/restaurants I (0.756); lowest in
  construction F (0.559), agriculture A (0.550), professional services M (0.579) and the thin sections D, S.
  Real estate L (100,645 rows, 997 defaults, the second-largest section) has AR 0.643, close to the
  average; L has the lowest default rate of the big sections (0.99%). R-33 advises against scoring
  single-project real-estate developers (Sec. 4.6 fn. 14 p. 22); we cannot identify them in the data.
- **New firms**: AR 0.573 vs 0.688 for seen firms (all models show a 0.057-0.061 AUC gap, `r1b_stability_summary.csv`).
- Per year, GB-01 AR 0.688 / 0.672 / 0.661: a slow decline, as for every model.
"""

N["SEGCAL"] = """\
Level by cell (CAL-01, the shipped level logic): pooled obs/pred 1.09 (2008 1.02, 2009 1.05, 2010 1.18).
Largest misses with enough defaults: real estate L 1.17 and the largest firms Q5 1.15 (under-predicted,
also for GB-01 at its own level: L 1.46, Q5 1.43, against 1.34 overall); E 1.36 (63 defaults); D 0.66
and Q 0.71 are over-predicted (32 and 34 defaults, CIs wide). S 1.79 rests on 30 defaults. The smallest
quintile Q1 is close (1.07). These are diagnostic: correcting them would be a group-level fit on test years
(D-078 rejected that).
"""

N["MISSING"] = """\
Answer to L1 for GB-01: **no, GB-01 does not under-predict these rows more than it under-predicts everyone.**
- `debt_split_missing = 1` (17.5% of test rows, 1,446 defaults): GB-01 obs/pred 1.40 vs 1.34 for all rows;
  relative 1.04 [0.98, 1.09], p = 0.16. With CAL-01: 1.13 vs 1.09. The flag is in the model and carries
  the extra risk.
- At least one core ratio NaN after the identity rebuild (1,800 rows, 0.36%, 114 defaults, default rate
  6.3%, four times the average): GB-01 relative 1.01 [0.84, 1.22]; CAL-01 1.05 [0.87, 1.26]. GB-01's native
  NaN handling learns that these rows are risky (mean PD 4.6%).
- Core raw input blank in the raw file (405 rows, 8 defaults): too few defaults to say anything (all CIs
  span 0.5-2.3).
- The L1 mechanism is visible in the models that fill with an average: GAM-01+CAL (missing bins reset to
  the average firm, D-082) relative 1.21 [1.00, 1.46], p = 0.049; SC-03 1.30 [1.07, 1.56], p = 0.009.
  RC-02 over-predicts them (0.75 [0.62, 0.90]): its T_i transforms price missingness explicitly (D-042).
- Recommendation: no `n_core_imputed` flag for GB-01 (no evidence it is needed). If GAM-01 is ever the
  submission model, replace the "reset to 0" rule for missing bins by a pooled "any core missing" bin
  (in-window), because average-firm scoring under-prices these rows by about 20%.
"""

N["GRADEMETHOD"] = """\
Edge-setting sample: the fold-3 GB-01 model (fit on 2007-2009) scored **in-sample on its 475,861 training
rows** (`experiments/calibration/cache/gb_train_3.parquet`), shifted to the CAL-01 level with the fold-3
prior-shift delta (0.2175 = logit(τ_e,2010) − logit(r_w,2007-09); recomputed from the files and from the
training rate: identical, `r1b_grade_edges.json`). 2010 PDs and labels were not used to choose edges.
Rule: keep floor 0.03%, cap 99% and the 11 existing edges; repeatedly split the grade with the largest share
above 12% at the median PD of its rows rounded to two significant digits. Result: 5 new edges (0.17%, 0.36%,
0.62%, 1.2%, 1.9%), 16 grades. R-33 Sec. 4.9 p. 24 designs EDF-implied rating scales so that no category holds
too many credits (< 12% of observations per category in their sample); the 12% is their number, used here as
our design target.
"""

N["GRADES"] = """\
2010 under-prediction, quantified (CAL-01, new grid):
- Total: 2,968 observed vs 2,511.4 expected defaults, obs/pred 1.18 (+457 defaults, 18%).
- Grades 1-5 (PD < 0.5%, 42.5% of firms, 5.9% of defaults): 174 observed vs 186.1 expected: on target.
- Under-predicted (observed above the acceptance band): grades 6 and 8-14 (PD 0.5%-10%), 50.0% of firms,
  +501.6 defaults, obs/pred 1.26-1.41 per grade (grade 7 1.17 is just inside its band).
- Grade 15 (10-20%) on target (1.01); grade 16 (≥ 20%) OVER-predicted: 451 observed vs 498.9 expected (0.90).
- At GB-01's own level (no shift) the under-prediction is 1.43 overall and grades 5-15 are flagged.
- Pattern: the miss is a level shift in the middle of the scale plus a slightly compressed top
  (2010 calibration slope 0.957). It is the cycle (default rate 1.28% → 1.62% → 1.70%) that no
  through-the-data level rule anticipated (D-071/D-073), not a ranking failure: observed default rates rise
  monotonically across all 16 grades.
"""

N["OVERFIT"] = """\
**Overfitting and validation evidence (summary table)**

| Evidence | What we have | Reading |
|---|---|---|
| OOT calibration slope (1 = right spread) | GB-01 pooled 0.976, by year 0.957-1.005; GAM-01 0.964 (0.952-0.974); RC-02 1.022 (0.975-1.099); SC-03 0.989 (0.970-1.030); new firms 0.83-0.89 for every model | No over-fitted spread for the population; new firms' PDs are too spread for all models (a population issue, D-047/D-079) |
| Per-year stability | GB-01 AUC 0.844 / 0.836 / 0.831 (range 0.0135, smallest of all models); GAM-01 range 0.017, RC-02 0.021 | Gentle decline with the cycle, same for all models; GB-01 better in every year |
| CI widths | Pooled AUC 95% CI width 0.009-0.010 (firm-clustered, 1,000 draws); per year about 0.014-0.018; paired ΔAUC GB-01 vs GAM-01 width 0.0036 | Model differences ≥ 0.006 AUC are resolvable; per-segment cells with < 100 defaults are not |
| In-sample vs OOT, same year | GB-01 +0.026 (2008, fold-2 model), +0.017 (2008, fold 3), +0.015 (2009, fold 3); RC-02 +0.004 to +0.006; GAM-01 +0.006 (last training year vs its OOT) | The GBM fits its training years harder (AR gap 2.9-5.2 points vs R-33's −0.5 points, Sec. 4.7-4.8); only OOT numbers may be quoted. The OOT slope ≈ 1 shows the extra fit does not distort the PDs |
| In-sample vs OOT, per fold | GB-01 +0.004 / +0.023 / +0.019; fold 1 RC-02 and GAM-01 are lower in-sample than OOT (2007 is a harder year to rank than 2008) | Confounded by year difficulty; use the same-year table |
| D-062 caveat | The feature set was designed on 2007-2009 (test years of folds 1-2); fold 3 (2010) is clean | Every promotion must also hold on 2010: GB-01 vs GAM-01 holds on 2010 (ΔAUC +0.0129 [+0.0098, +0.0161]) |
| Selection ledger | 30 OOT prediction files and 62 calibration configurations scored on 2008-2010 (≈ 90 looks at each test year); plus feature design (D-062) and RC inner validation on 2009 (D-063); round 2 adds more | The winning 2008-2010 numbers carry an unquantified selection optimism; GB-01's margins (0.012-0.021 AUC, CIs far from 0) are large relative to it, but its absolute AUC is probably slightly optimistic |
| fs_year 2011 | Not virgin: GB-01 and RC-02 fold-4 AUCs on 2011 already in `reports/calibration/q5_2011_other_labels_fold4.csv` and `q6_2011_pseudo_test.csv` (D-076 quotes them); τ is anchored to the fs2011 rate (D-073). GAM-01/02, SC-03 never scored on 2011 | 2011 can still settle the GB-01 vs GAM-01 ranking once; it cannot test the level |

**Recommendation: fs_year 2011 as a one-time lockbox for the final model choice** (refines D-087):
1. Freeze now, before opening 2011: the finalists (GB-01 and GAM-01, both with frozen code and seeds), the
   fold-4 fits on fs_year 2007-2010 (`label_status == ok`), the label (11-month y11 of D-077 as primary;
   complete s0/s3 labels as sensitivity), the metrics (AUC; log loss after one common prior shift per model so
   the level is not judged), and the decision rule above (paired firm bootstrap, 1,000 draws). Log this in the
   decision log with a timestamp.
2. One run, by the referee only, writing one file. The result is reported whatever it says. No re-tuning
   afterwards; if GAM-01 wins on 2011 it is a tie-break signal, not a reason to re-open the tournament.
3. Then refit the chosen model on 2007-2011 for the harness (D-077).
4. Do not use 2011 for any level claim (τ was set from it) and do not score any other model on 2011.
"""

N["DECISIONS"] = """\
```
### D-110 Referee verdict on the EBM challengers (GAM-01, GAM-02)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED
- Decision: GAM-01 and GAM-02 are CLEAN (no leakage). Contract PASS (506,159 rows, 0 missing/NaN/out-of-range);
  saved fold models reproduce the submitted own, raw and +CAL PDs exactly (max |diff| 0) on test rows with every
  label column removed and rows shuffled. Process notes: D-082 missing-bin reset was added after a full OOT run
  (robustness rule, training data only, OOT effect +0.0004 AUC, not used as evidence); GAM-02 reused GAM-01's
  in-window grid choice; +CAL inherits rule-e selection optimism (D-071); shared D-062 caveat.
- Alternatives considered: CONCERN by analogy with RC-03 (D-063) - rejected because the rule was triggered by
  training shapes, is not reported as a performance gain, and changes nothing measurable.
- Evidence: r1b_contract.csv, r1b_gam_repro.csv; code read of run_gam.py, gam_lib.py, explain_gam.py.
- Source(s): CLAUDE.md sec. 5 rule 5; data/splits/README.md.
- Risks / what would change our mind: a refit from current code not reproducing the saved models (not run:
  ~10 min per fold at 2 threads; saved-model reproduction done instead).
- Human reviewer (initials) and date:

### D-111 EBM face validity confirmed (D-084)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED
- Decision: 0 violations in 20 sign-expected terms x 3 folds x 2 models (step-function against-share 0); GAM-02
  pairs involve only unsigned features. Flag for officers: rev_nonpositive lowers the score (-0.19), OTHER levels
  of legal_group (-0.87) / macro_area (+0.65) are thin pools that unseen codes inherit.
- Evidence: r1b_gam_face_validity.csv (fold-3 GAM-01 shapes in section 1.5).
- Source(s): docs/FEATURE_SPEC.md expected-sign table.
- Human reviewer (initials) and date:

### D-112 GAM-01 not promoted over GB-01; GAM-01 becomes the transparent fallback (replaces RC-03 in D-067)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (referee); human sign-off requested for the fallback change
- Decision: GB-01 remains champion. GAM-01 replaces RC-03 as the documented transparent alternative.
- Evidence (1,000-draw firm bootstrap, r1b_gam_paired_bootstrap.csv): GAM-01 vs GB-01 dAUC -0.0120 [-0.0139,
  -0.0102], dLL +0.00102 [+0.00088, +0.00115], worse in every year and firm group; GAM-01+CAL vs CAL-01 -0.0119 /
  +0.00098. GAM-01 vs RC-02 dAUC +0.0061 [+0.0044, +0.0078], dLL -0.00063 [-0.00078, -0.00048] (2008 AUC and
  new-firm LL ties), 0 vs 2 face-validity violations. GAM-01 vs SC-03 +0.0089 / -0.00139. GAM-02 vs GAM-01
  +0.0008 [+0.0004, +0.0012] / -0.00008 [-0.00010, -0.00005]: small, GAM-01 kept (D-083).
- Alternatives considered: RC-03 as fallback (CONCERN, D-063, and dominated); GAM-02 (not worth ten heat maps).
- Business reason: if the team or Roger judges SHAP reason codes insufficient, the fallback with readable
  curves is also the stronger transparent model.
- Risks / what would change our mind: the 2011 lockbox reverses the GB-01 vs GAM-01 ranking.
- Human reviewer (initials) and date:

### D-113 Power by segment (lesson L4)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (reporting)
- Decision: report AR by year, size quintile (training-fold asst_tot edges), ATECO section (< 30 test defaults
  pooled; L separate) and seen/new, with firm-clustered CIs and default counts, for GB-01, GAM-01+CAL, RC-02, SC-03.
- Evidence (r1b_segment_power.csv): GB-01 best or tied in every cell; AR Q1 0.626 -> Q5 0.735; C 0.745, I 0.756,
  F 0.559, A 0.550, L 0.643 (997 defaults); seen 0.688, new 0.573; years 0.688 / 0.672 / 0.661. GB-01 minus
  GAM-01 CI excludes 0 in every year, size quintile, firm group and in sections C, E, F, G, J, L, M.
- Source(s): R-33 Tables 9-14 pp. 19-21 (AR by industry, size, year); R-33 Sec. 4.6 fn. 14 p. 22 (real estate).
- Risks: section cells with < 100 defaults have single-model AUC CIs 0.08-0.23 wide; do not rank models there.
- Human reviewer (initials) and date:

### D-114 Missing-input check (lesson L1): no extra under-prediction for GB-01; no n_core_imputed flag
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (no change to GB-01); OPEN for GAM-01 if it ever becomes the submission model
- Evidence (r1b_missing_tests.csv; relative obs/pred = group vs same-year average, Poisson CI): GB-01
  debt_split_missing 1.04 [0.98, 1.09] (88,593 rows, 1,446 defaults); core ratio NaN 1.01 [0.84, 1.22] (1,800 rows,
  114 defaults, DR 6.3%). GAM-01+CAL 1.21 [1.00, 1.46] (missing bins reset to average, D-082); SC-03 1.30
  [1.07, 1.56]; RC-02 0.75 [0.62, 0.90].
- Decision: keep GB-01 as is. For GAM-01, replace "reset sparse missing bins to 0" by a pooled in-window
  "any core missing" bin before any use.
- Source(s): R-34 (mean fills understate PDs), R-05 pp. 60-61, via docs/LIT_moodys_feature_cleaning_2026-10-30.md L1.
- Human reviewer (initials) and date:

### D-115 Re-split grade grid: 16 grades, no grade above ~10% of firms (supersedes the 11 bands of D-078 for reporting)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (proposal for the deck and harness documentation; human sign-off requested)
- Decision: edges 0.03, 0.10, 0.17, 0.25, 0.36, 0.50, 0.62, 0.75, 1.0, 1.2, 1.5, 1.9, 2.5, 5, 10, 20, 99 (%).
  Chosen on fold-3 GB-01 in-sample PDs on 2007-2009 training rows at the CAL-01 level (no 2010 data), by splitting
  any grade above 12% at its rounded median.
- Evidence (r1b_grade_summary.csv): max share 9.9% (edge sample), 10.3% (OOT 2008-09), 10.0% (2010); old grid had
  19.6% / 19.4% in grades 2-3. 2010 observed default rates are monotone across all 16 grades.
- Alternatives considered: equal-share quantile grades (edges not round, move with every refit); keeping 11 grades.
- Source(s): R-33 Sec. 4.9 p. 24 (< 12% of observations per category).
- Risks: the edge sample is in-sample (slightly more spread than OOT); shares on the holdout will differ.
- Human reviewer (initials) and date:

### D-116 The 2010 under-prediction, quantified on the new grid
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (finding; no group-level correction, per D-078)
- Evidence (r1b_grade_table.csv, CAL-01, 2010): 2,968 observed vs 2,511.4 expected (obs/pred 1.18, +457).
  Grades 1-5 (PD < 0.5%, 42.5% of firms): 174 vs 186.1, on target. Grades 6 and 8-14 (PD 0.5-10%, 50.0% of firms):
  above the acceptance band, +501.6 defaults, obs/pred 1.26-1.41. Grade 16 (>= 20%): 451 vs 498.9, over-predicted.
  GB-01 at its own level: obs/pred 1.43, grades 5-15 flagged.
- Reading: a cycle-driven level miss concentrated in the middle of the scale plus a slightly compressed top
  (2010 slope 0.957); ranking intact. Bands ignore default correlation (R-16 pp. 33-34): diagnostic only.
- Business reason: in a 2010-like year mid-grade PDs were 20-30% too low (observed 1.26-1.41x predicted), so pricing would not cover expected loss there.
- Human reviewer (initials) and date:

### D-117 Overfitting and validation evidence for the deck
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (reporting)
- Evidence: OOT calibration slopes 0.96-1.01 by year for GB-01 (new firms 0.88, all models 0.83-0.89); per-year
  AUC range 0.0135 (GB-01, smallest); pooled AUC CI width ~0.009; same-year in-sample minus OOT AUC: GB-01
  +0.015 to +0.026, RC-02 +0.004 to +0.006, GAM-01 +0.006 (r1b_same_year_overfit.csv, r1b_insample_vs_oot.csv,
  r1b_stability_summary.csv).
- Decision: the deck quotes only OOT numbers, shows the same-year gap honestly (GBM fits training years harder;
  R-33 reports -0.5 AR points for RiskCalc France, Sec. 4.7-4.8) and uses the OOT slope ~1 as the evidence that
  the PDs are not over-spread.
- Source(s): R-33 Sec. 4.7-4.8 pp. 22-23; R-10 Sec. 4.5 p. 22.
- Human reviewer (initials) and date:

### D-118 Selection ledger and correction to D-087 (fs_year 2011 is partially spent)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (correction of fact in D-087)
- Finding: 30 OOT prediction files and 62 calibration configurations have been scored on 2008-2010
  (r1b_selection_ledger.json). D-087 says 2011 "has so far been used only for a level statistic"; in fact GB-01
  and RC-02 fold-4 models were scored on fs2011 with AUC (reports/calibration/q5_2011_other_labels_fold4.csv,
  q6_2011_pseudo_test.csv; D-076 quotes 0.885 / 0.869 / 0.850) and tau is anchored to the fs2011 rate (D-073).
  GAM-01/02 and SC-03 have not been scored on 2011. (Referee read only the headers/model names of those files.)
- Decision: treat 2011 as valid for ONE ranking decision between GB-01 and GAM-01; never as a level test.
- Human reviewer (initials) and date:

### D-119 Pre-registered one-time 2011 lockbox protocol
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: OPEN (proposal; needs orchestrator + human sign-off before anyone opens 2011)
- Decision: freeze finalists (GB-01, GAM-01), code, seeds; fit fold 4 on fs 2007-2010 ok rows; label y11 (D-077)
  primary, s0/s3 sensitivity; metrics AUC and log loss after one common per-model prior shift; same decision
  rule, 1,000-draw firm bootstrap; one run by the referee, one file, result reported whatever it says; then
  refit the chosen model on 2007-2011 (D-077). No other model is scored on 2011.
- Alternatives considered: no lockbox (selection optimism unchecked); a random-firm holdout inside 2010 (not
  out-of-time, and 2010 is already used).
- Risks: one year, one cycle point; a reversal would be a tie-break signal, not proof.
- Source(s): assumption (standard validation practice); R-33 Sec. 4.8 p. 23 (walk-forward).
- Human reviewer (initials) and date:
```
"""

s = TPL.read_text()
for k, v in N.items():
    s = s.replace(f"@@{k}@@", v.rstrip())
assert "@@" not in s
(OUT / "r1b_diagnostics_filled_template.md").write_text(s)
print("ok")
