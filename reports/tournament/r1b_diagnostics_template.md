# Referee round 1b: validation diagnostics (referee, 2026-11-02)

Scope: (1) first independent referee check of the EBM challengers GAM-01 and GAM-02; (2) power and level
by segment (lesson L4); (3) missing-input calibration check (lesson L1); (4) a re-split grade grid and the
2010 grade table; (5) overfitting and validation evidence, incl. the test-year selection ledger and the
fs_year 2011 lockbox.

Produced by code, every number read from a file:
`reports/tournament/r1b_diagnostics.py` (contract, reproduction, face validity, metrics, 1,000-draw
firm-clustered bootstrap, segments, missing-input check, grade grid, in-sample vs OOT) →
`r1b_*.csv|json|npz`, `fig/r1b_*.png`; `r1b_addons.py` (same-year overfit gap, grade summary,
missing-input tests, stability summary); `r1b_build_md.py` renders this file from
`r1b_diagnostics_template.md`. Metrics: `src/common/metrics.py` only. Bootstrap: the fast
firm-multiplicity estimator of `score_r1.py`, re-verified identical to `metrics.paired_bootstrap`
(table A1). **Lockbox:** the panel was read with the parquet filter `fs_year < 2011`; no file
holding fs_year 2011/2012 rows was opened (the calibration cache `final_scored_2010_2012.parquet` was
not read; for the 2011 ledger only the column headers and model names of
`reports/calibration/q5_*/q6_*` were inspected, not their values).

Models: GB-01 (incumbent, own level), CAL-01 (GB-01 + rule-e prior shift, the shipped level logic),
SC-03 (WoE scorecard), RC-02 (RiskCalc-style probit; RC-02 not RC-03 per D-063), GAM-01 / GAM-02
(EBM, own calibration), GAM-01+CAL / GAM-02+CAL (own calibration + the same rule-e shift as CAL-01).
Like-for-like rule: own-level models are compared with own-level models (GAM vs GB-01, SC-03, RC-02),
shifted models with shifted models (GAM+CAL vs CAL-01). AUC within a test year is identical for a model
and its shifted version; pooled AUC differs slightly because the shift differs by fold.

## Decision rule (published with every scorecard)

A challenger replaces the incumbent only if:
1. its leakage / process verdict is CLEAN;
2. the 95% CI of Δlog-loss excludes zero in its favour, OR (the ΔAUC CI excludes zero in its favour
   AND log loss is not worse);
3. it has no more face-validity violations than the incumbent (or each violation is justified);
4. the per-year results show no year in which it is clearly worse (no hidden fragility).

Otherwise the simpler / more explainable model stays (statistical ties go to transparency).

## 0. Summary

@@SUMMARY@@

## 1. EBM (GAM-01, GAM-02): independent referee check

### 1.1 Contract

{{T:contract}}

### 1.2 Reproduction and process

Each saved fold model (`experiments/gam/GAM-0x/model_fold{k}`) was loaded and re-scored on its test rows
with **every label column removed** (`y`, `def_date`, `label_status`, `win_*`, `y_s*`, `status_*`,
`src_index`) and the rows **shuffled**; the prediction reproduces the three submitted files exactly
(own calibration, raw EBM, +CAL). `r_w` (the training default rate used for the level anchor and the
prior shift) and `n_train` are recomputed from the fold's training rows. Prediction time is with 2 threads.

{{T:repro}}

@@PROCESS@@

### 1.3 Metrics (shared code; AUC CI = firm-clustered bootstrap, 1,000 draws)

{{T:leader}}

Per test year and seen/new firms:

{{T:byyear}}

### 1.4 Paired firm-clustered bootstrap (1,000 draws, seed 0; positive ΔAUC and negative ΔLogLoss favour "new")

{{T:pairs}}

### 1.5 Face validity of the shape functions

Against-share = share of the step function's total movement (bins between the missing and the unknown bin)
that goes against the expected sign; 0 = monotone in the expected direction. All folds:

{{T:fv}}

Fold-3 GAM-01 shapes (log-odds contributions):

{{T:fv3}}

@@FACE@@

### 1.6 Verdict

@@VERDICT@@

## 2. Power and level by segment (lesson L4; R-33 Tables 9-14 pp. 19-21 report AR by industry, size and year)

AUC with firm-clustered 95% CI (1,000 draws); AR = 2·AUC − 1 (same information). Last column: paired
GB-01 minus GAM-01+CAL AUC difference with its CI. Cells pool the three test years unless the cell is a year.
GAM-01 here is the calibrated file `oot_predictions_calibrated.parquet` as requested.

Size quintile edges (total assets in EUR, quintiles of `asst_tot` on each fold's TRAINING rows):

{{T:size_edges}}

### 2.1 By test year

{{T:seg_year}}

### 2.2 By size quintile (training-fold edges)

{{T:seg_size}}

### 2.3 By ATECO section (raw `ateco_section`; sections with < 30 test defaults pooled; L real estate separate)

{{T:seg_ateco}}

### 2.4 Seen vs new firms

{{T:seg_firm}}

@@SEGMENTS@@

### 2.5 Observed vs mean predicted PD in every cell (GB-01 own level, CAL-01 shifted, GAM-01+CAL)

{{T:seg_cal}}

@@SEGCAL@@

Figure: `fig/r1b_segment_auc.png`.

## 3. Missing-input check (lesson L1)

Groups: `debt_split_missing = 1` (any of debt_bank_st, debt_bank_lt, AP_st missing in the raw row; GB-01
sees this flag); "core feature NaN after rebuild" = at least one core numeric feature is NaN after the
accounting-identity rebuild of `build.py` (exactly the rows a GLM view would median-fill); "core raw input
missing" = at least one raw field that feeds a core feature is blank in the raw data (before the rebuild).
`relative obs/pred` = the group's obs/pred divided by the obs/pred of all rows of the same year, i.e. how
much MORE the group is under-predicted than the average firm (1 = no extra under-prediction).

{{T:missing}}

Pooled over years, with the expectation adjusted for each year's overall level miss (Poisson test,
diagnostic; ignores default correlation):

{{T:missing_tests}}

@@MISSING@@

## 4. Grade grid with no grade over ~12% of firms

@@GRADEMETHOD@@

Edges:

{{T:grade_edges}}

Share of firms per grade: edge-setting sample (fold-3 model, 2007-2009 training rows), OOT 2008-2009 and 2010:

{{T:grade_shares}}

### 4.1 2010 grade table, new grid, CAL-01 (the level logic that would ship)

Flag = observed default count outside the 95% acceptance interval implied by the grade's PDs (normal
approximation to the Poisson-binomial; binomial(n, mean PD) when fewer than 20 expected defaults). The
binomial bands ignore default correlation and therefore understate the uncertainty (R-16 pp. 33-34): read
the flags as diagnostic.

{{T:grade_new_2010}}

### 4.2 Same grid, GB-01 at its own level (no prior shift)

{{T:grade_new_2010_gb}}

### 4.3 Old 11-grade grid, CAL-01, 2010 (reproduces `reports/calibration/grade_table_2010.csv`)

{{T:grade_old_2010}}

Summary:

{{T:grade_summary}}

@@GRADES@@

Figure: `fig/r1b_grade_2010.png`.

## 5. Overfitting and validation evidence

### 5.1 In-sample (training rows) vs out-of-time AUC per fold

{{T:insample}}

### 5.2 Same year, in-sample vs out-of-time (removes the year-difficulty confound)

Year t scored by a later fold's model that was trained on t (in-sample) vs by the fold for which t was the
test year (OOT). Same rows.

{{T:sameyear}}

### 5.3 Stability summary

{{T:stability}}

### 5.4 Selection ledger (how often 2008-2010 have been looked at)

{{T:ledger}}

@@OVERFIT@@

## Appendix A1: fast bootstrap vs `metrics.paired_bootstrap` (GAM-01 vs GB-01, 15,000-firm subsample, 40 draws, seed 0)

{{T:equiv}}

## Decision entries for the orchestrator to merge (DECISION_LOG template)

@@DECISIONS@@
