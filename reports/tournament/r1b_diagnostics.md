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

| # | Finding | Key numbers (files) | What it means |
|---|---|---|---|
| 1 | **EBM verdict: CLEAN, 0 face-validity violations, not promoted.** GAM-01 loses to GB-01 on both metrics in every year and firm group; beats SC-03 and RC-02. | GAM-01 vs GB-01 pooled ΔAUC −0.0120 [−0.0139, −0.0102], ΔLL +0.00102 [+0.00088, +0.00115]; vs SC-03 +0.0089 [+0.0073, +0.0105], −0.00139 [−0.00155, −0.00124]; vs RC-02 +0.0061 [+0.0044, +0.0078], −0.00063 [−0.00078, −0.00048] (`r1b_gam_paired_bootstrap.csv`) | Bank: GB-01 stays the champion. Roger: the glass box costs about 0.012 AUC; GAM-01 is now the best fully transparent model and should replace RC-03 as the documented fallback. |
| 2 | **Power by segment.** GB-01 is the best or tied model in every cell; no segment where a transparent model is significantly better. Power is weakest for small firms, construction (F) and new firms. | GB-01 AR: size Q1 0.626 → Q5 0.735; C 0.745, F 0.559, L 0.643; seen 0.688, new 0.573 (`r1b_segment_power.csv`) | Bank: rankings are least reliable for small firms, builders and first-time borrowers; cutoffs there deserve a manual review buffer. Roger: the RiskCalc-style AR-by-segment tables (R-33 Tables 9-14) now exist for our models. |
| 3 | **Missing inputs (L1): GB-01 shows no extra under-prediction** on rows with missing debt split or missing core inputs; the EBM and the WoE scorecard do, on the 0.36% of rows with a missing core ratio. | Relative obs/pred (vs same-year average) GB-01: debt_split_missing 1.04 [0.98, 1.09], core NaN 1.01 [0.84, 1.22]; GAM-01+CAL core NaN 1.21 [1.00, 1.46], SC-03 1.30 [1.07, 1.56] (`r1b_missing_tests.csv`) | Bank: no extra "missing-data" add-on is needed for GB-01. Roger: Moody's warning about mean/median fills (R-34) is visible in our data, but only for the models that fill with an average. |
| 4 | **Re-split grade grid: 16 grades, none above 10.3% of firms** (edges set on 2007-2009 predictions). On 2010, observed default rates rise monotonically across all 16 grades. | max share 9.9% (edge sample), 10.3% (OOT 2008-09), 10.0% (2010) (`r1b_grade_summary.csv`) | Bank: a usable master scale; no grade lumps a fifth of the book. Roger: the grade design follows R-33 Sec. 4.9 and was not fitted to the test year. |
| 5 | **The 2010 under-prediction, quantified.** With the shipped level logic (CAL-01), 2,968 defaults were observed against 2,511 expected (obs/pred 1.18, 457 extra defaults). The shortfall sits in grades 6 and 8-14 (PD 0.5%-10%, 50% of firms, +502 defaults); grades 1-5 are on target and the top grade (PD ≥ 20%) is over-predicted. | `r1b_grade_table.csv`, `r1b_grade_summary.csv` | Bank: in a 2010-like year the PDs of the middle of the book were 20-30% too low (observed 1.26-1.41x predicted); the safest and the riskiest firms are priced about right. Roger: this is a level (cycle) miss, not a ranking failure. |
| 6 | **Overfitting evidence.** OOT calibration slopes 0.96-1.01 by year (no over-fitting of the PD spread, except new firms 0.88 for every model). GB-01's in-sample AUC is 0.015-0.026 above its OOT AUC on the same year; RC-02 0.004-0.006, GAM-01 0.006. 2008-2010 have been scored by about 90 prediction vectors. | `r1b_stability_summary.csv`, `r1b_same_year_overfit.csv`, `r1b_insample_vs_oot.csv`, `r1b_selection_ledger.json` | Bank: quote only OOT numbers; the in-sample fit of the GBM is optimistic. Roger: walk-forward OOT is our evidence, as in R-10/R-33; the test years are now heavily reused, so one sealed check is needed. |
| 7 | **Lockbox: fs_year 2011 is NOT untouched** (correction to D-087). GB-01 and RC-02 fold-4 models were already scored on 2011 (AUC columns in `reports/calibration/q5_2011_other_labels_fold4.csv`, `q6_2011_pseudo_test.csv`; D-076), and the level tau is anchored to the 2011 default rate. GAM-01/02 and SC-03 have not been scored on 2011. | file headers only (values not read) | Use 2011 once, pre-registered, for the GB-01 vs GAM-01 ranking decision only; 2011 cannot test the level. |

## 1. EBM (GAM-01, GAM-02): independent referee check

### 1.1 Contract

| exp        | file                                                      |   n_rows |   missing_test_rows |   dup_row_fold |   nan_pd |   pd_outside_open_01 |   min_pd |   max_pd | PASS   |
|:-----------|:----------------------------------------------------------|---------:|--------------------:|---------------:|---------:|---------------------:|---------:|---------:|:-------|
| GB-01      | experiments/gbm/GB-01/oot_predictions.parquet             |   506159 |                   0 |              0 |        0 |                    0 |  0.00022 |   0.8467 | True   |
| CAL-01     | experiments/calibration/CAL-01/oot_predictions.parquet    |   506159 |                   0 |              0 |        0 |                    0 |  0.0003  |   0.8808 | True   |
| SC-03      | experiments/scorecard/SC-03/oot_predictions.parquet       |   506159 |                   0 |              0 |        0 |                    0 |  0.0002  |   0.5674 | True   |
| RC-02      | experiments/riskcalc/RC-02/oot_predictions.parquet        |   506159 |                   0 |              0 |        0 |                    0 |  0.00026 |   0.6309 | True   |
| GAM-01     | experiments/gam/GAM-01/oot_predictions.parquet            |   506159 |                   0 |              0 |        0 |                    0 |  0.0003  |   0.8113 | True   |
| GAM-01+CAL | experiments/gam/GAM-01/oot_predictions_calibrated.parquet |   506159 |                   0 |              0 |        0 |                    0 |  0.0003  |   0.8434 | True   |
| GAM-01raw  | experiments/gam/GAM-01/oot_predictions_raw.parquet        |   506159 |                   0 |              0 |        0 |                    0 |  0.00011 |   0.8398 | True   |
| GAM-02     | experiments/gam/GAM-02/oot_predictions.parquet            |   506159 |                   0 |              0 |        0 |                    0 |  0.0003  |   0.8255 | True   |
| GAM-02+CAL | experiments/gam/GAM-02/oot_predictions_calibrated.parquet |   506159 |                   0 |              0 |        0 |                    0 |  0.0003  |   0.8547 | True   |
| GAM-02raw  | experiments/gam/GAM-02/oot_predictions_raw.parquet        |   506159 |                   0 |              0 |        0 |                    0 |  2e-05   |   0.8529 | True   |

### 1.2 Reproduction and process

Each saved fold model (`experiments/gam/GAM-0x/model_fold{k}`) was loaded and re-scored on its test rows
with **every label column removed** (`y`, `def_date`, `label_status`, `win_*`, `y_s*`, `status_*`,
`src_index`) and the rows **shuffled**; the prediction reproduces the three submitted files exactly
(own calibration, raw EBM, +CAL). `r_w` (the training default rate used for the level anchor and the
prior shift) and `n_train` are recomputed from the fold's training rows. Prediction time is with 2 threads.

| exp    |   fold |   n_test |   max_abs_diff_own |   max_abs_diff_raw |   max_abs_diff_cal |   r_w_config |   r_w_recomputed |   n_train_config |   n_train_recomputed |   tau_rule_e |   delta | chosen_cfg                                  |   predict_secs_2threads |
|:-------|-------:|---------:|-------------------:|-------------------:|-------------------:|-------------:|-----------------:|-----------------:|---------------------:|-------------:|--------:|:--------------------------------------------|------------------------:|
| GAM-01 |      1 |   161261 |                  0 |                  0 |                  0 |     0.009562 |         0.009562 |           144214 |               144214 |      0.01165 |  0.1997 | {"max_leaves": 2, "min_samples_leaf": 500}  |                    1.66 |
| GAM-01 |      2 |   170386 |                  0 |                  0 |                  0 |     0.011271 |         0.011271 |           305475 |               305475 |      0.01502 |  0.2913 | {"max_leaves": 3, "min_samples_leaf": 1000} |                    1.35 |
| GAM-01 |      3 |   174512 |                  0 |                  0 |                  0 |     0.013019 |         0.013019 |           475861 |               475861 |      0.01613 |  0.2175 | {"max_leaves": 3, "min_samples_leaf": 200}  |                    1.34 |
| GAM-02 |      1 |   161261 |                  0 |                  0 |                  0 |     0.009562 |         0.009562 |           144214 |               144214 |      0.01165 |  0.1997 | {"max_leaves": 2, "min_samples_leaf": 500}  |                    1.77 |
| GAM-02 |      2 |   170386 |                  0 |                  0 |                  0 |     0.011271 |         0.011271 |           305475 |               305475 |      0.01502 |  0.2913 | {"max_leaves": 3, "min_samples_leaf": 1000} |                    1.7  |
| GAM-02 |      3 |   174512 |                  0 |                  0 |                  0 |     0.013019 |         0.013019 |           475861 |               475861 |      0.01613 |  0.2175 | {"max_leaves": 3, "min_samples_leaf": 200}  |                    1.43 |

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

### 1.3 Metrics (shared code; AUC CI = firm-clustered bootstrap, 1,000 draws)

| exp        |      n |   defaults |    auc | auc_95ci         |    ar |   log_loss |   brier | mean_pd   | observed_dr   |   ratio_obs_to_pred |   cal_intercept |   cal_slope |
|:-----------|-------:|-----------:|-------:|:-----------------|------:|-----------:|--------:|:----------|:--------------|--------------------:|----------------:|------------:|
| GB-01      | 506159 |       7784 | 0.8375 | [0.8328, 0.8415] | 0.675 |    0.06415 | 0.01368 | 1.1451%   | 1.5379%       |               1.343 |           0.335 |       0.976 |
| CAL-01     | 506159 |       7784 | 0.8374 | [0.8327, 0.8414] | 0.675 |    0.06352 | 0.01364 | 1.4135%   | 1.5379%       |               1.088 |           0.096 |       0.976 |
| SC-03      | 506159 |       7784 | 0.8166 | [0.8116, 0.8214] | 0.633 |    0.06656 | 0.01408 | 1.1047%   | 1.5379%       |               1.392 |           0.359 |       0.988 |
| RC-02      | 506159 |       7784 | 0.8194 | [0.8141, 0.8243] | 0.639 |    0.0658  | 0.01393 | 1.1288%   | 1.5379%       |               1.362 |           0.338 |       1.022 |
| GAM-01     | 506159 |       7784 | 0.8254 | [0.8202, 0.8300] | 0.651 |    0.06517 | 0.01379 | 1.1309%   | 1.5379%       |               1.36  |           0.345 |       0.964 |
| GAM-01+CAL | 506159 |       7784 | 0.8255 | [0.8204, 0.8301] | 0.651 |    0.06451 | 0.01374 | 1.3995%   | 1.5379%       |               1.099 |           0.106 |       0.963 |
| GAM-02     | 506159 |       7784 | 0.8262 | [0.8210, 0.8308] | 0.652 |    0.06509 | 0.01379 | 1.1294%   | 1.5379%       |               1.362 |           0.346 |       0.97  |
| GAM-02+CAL | 506159 |       7784 | 0.8264 | [0.8211, 0.8309] | 0.653 |    0.06443 | 0.01374 | 1.3979%   | 1.5379%       |               1.1   |           0.108 |       0.97  |

Per test year and seen/new firms:

| exp        | group   |      n |   defaults |    auc | auc_95ci         |   log_loss |   ratio_obs_to_pred |   cal_slope |
|:-----------|:--------|-------:|-----------:|-------:|:-----------------|-----------:|--------------------:|------------:|
| GB-01      | 2008    | 161261 |       2064 | 0.8441 | [0.8355, 0.8524] |    0.05423 |               1.212 |       0.969 |
| GB-01      | 2009    | 170386 |       2752 | 0.8359 | [0.8276, 0.8436] |    0.06617 |               1.362 |       1.005 |
| GB-01      | 2010    | 174512 |       2968 | 0.8307 | [0.8235, 0.8377] |    0.07136 |               1.432 |       0.957 |
| GB-01      | seen    | 442415 |       6862 | 0.8438 | [0.8389, 0.8484] |    0.06375 |               1.347 |       0.984 |
| GB-01      | new     |  63744 |        922 | 0.7865 | [0.7726, 0.8004] |    0.06695 |               1.317 |       0.883 |
| CAL-01     | 2008    | 161261 |       2064 | 0.8441 | [0.8355, 0.8524] |    0.05398 |               1.016 |       0.97  |
| CAL-01     | 2009    | 170386 |       2752 | 0.8359 | [0.8276, 0.8436] |    0.0654  |               1.054 |       1.006 |
| CAL-01     | 2010    | 174512 |       2968 | 0.8307 | [0.8235, 0.8377] |    0.07051 |               1.182 |       0.957 |
| CAL-01     | seen    | 442415 |       6862 | 0.8438 | [0.8389, 0.8484] |    0.0631  |               1.092 |       0.985 |
| CAL-01     | new     |  63744 |        922 | 0.786  | [0.7719, 0.7997] |    0.06646 |               1.061 |       0.892 |
| SC-03      | 2008    | 161261 |       2064 | 0.8278 | [0.8185, 0.8367] |    0.05631 |               1.237 |       1.03  |
| SC-03      | 2009    | 170386 |       2752 | 0.8117 | [0.8024, 0.8200] |    0.06913 |               1.422 |       0.975 |
| SC-03      | 2010    | 174512 |       2968 | 0.8115 | [0.8034, 0.8196] |    0.07353 |               1.494 |       0.97  |
| SC-03      | seen    | 442415 |       6862 | 0.823  | [0.8177, 0.8283] |    0.06626 |               1.4   |       1.004 |
| SC-03      | new     |  63744 |        922 | 0.7648 | [0.7502, 0.7794] |    0.06864 |               1.335 |       0.842 |
| RC-02      | 2008    | 161261 |       2064 | 0.8328 | [0.8236, 0.8415] |    0.05561 |               1.221 |       1.099 |
| RC-02      | 2009    | 170386 |       2752 | 0.8174 | [0.8082, 0.8256] |    0.06804 |               1.396 |       1.011 |
| RC-02      | 2010    | 174512 |       2968 | 0.8115 | [0.8033, 0.8198] |    0.07303 |               1.446 |       0.974 |
| RC-02      | seen    | 442415 |       6862 | 0.8261 | [0.8207, 0.8312] |    0.06545 |               1.371 |       1.035 |
| RC-02      | new     |  63744 |        922 | 0.7654 | [0.7506, 0.7801] |    0.06819 |               1.3   |       0.888 |
| GAM-01     | 2008    | 161261 |       2064 | 0.8348 | [0.8257, 0.8437] |    0.05507 |               1.231 |       0.974 |
| GAM-01     | 2009    | 170386 |       2752 | 0.8233 | [0.8140, 0.8318] |    0.06725 |               1.38  |       0.969 |
| GAM-01     | 2010    | 174512 |       2968 | 0.8177 | [0.8099, 0.8254] |    0.07247 |               1.445 |       0.952 |
| GAM-01     | seen    | 442415 |       6862 | 0.8318 | [0.8262, 0.8370] |    0.06478 |               1.368 |       0.978 |
| GAM-01     | new     |  63744 |        922 | 0.7744 | [0.7606, 0.7892] |    0.0679  |               1.299 |       0.833 |
| GAM-01+CAL | 2008    | 161261 |       2064 | 0.8348 | [0.8257, 0.8437] |    0.05479 |               1.029 |       0.974 |
| GAM-01+CAL | 2009    | 170386 |       2752 | 0.8233 | [0.8140, 0.8318] |    0.06644 |               1.066 |       0.969 |
| GAM-01+CAL | 2010    | 174512 |       2968 | 0.8177 | [0.8099, 0.8254] |    0.0716  |               1.189 |       0.951 |
| GAM-01+CAL | seen    | 442415 |       6862 | 0.832  | [0.8265, 0.8372] |    0.06408 |               1.106 |       0.978 |
| GAM-01+CAL | new     |  63744 |        922 | 0.7742 | [0.7604, 0.7888] |    0.06744 |               1.048 |       0.832 |
| GAM-02     | 2008    | 161261 |       2064 | 0.8353 | [0.8264, 0.8443] |    0.05503 |               1.236 |       0.998 |
| GAM-02     | 2009    | 170386 |       2752 | 0.8239 | [0.8146, 0.8326] |    0.06719 |               1.379 |       0.968 |
| GAM-02     | 2010    | 174512 |       2968 | 0.8192 | [0.8114, 0.8267] |    0.07235 |               1.447 |       0.95  |
| GAM-02     | seen    | 442415 |       6862 | 0.8327 | [0.8272, 0.8377] |    0.06471 |               1.371 |       0.983 |
| GAM-02     | new     |  63744 |        922 | 0.7747 | [0.7607, 0.7899] |    0.06779 |               1.297 |       0.843 |
| GAM-02+CAL | 2008    | 161261 |       2064 | 0.8353 | [0.8264, 0.8443] |    0.05474 |               1.032 |       0.998 |
| GAM-02+CAL | 2009    | 170386 |       2752 | 0.8239 | [0.8146, 0.8326] |    0.06638 |               1.066 |       0.968 |
| GAM-02+CAL | 2010    | 174512 |       2968 | 0.8192 | [0.8114, 0.8267] |    0.07148 |               1.19  |       0.949 |
| GAM-02+CAL | seen    | 442415 |       6862 | 0.8328 | [0.8274, 0.8380] |    0.06401 |               1.108 |       0.982 |
| GAM-02+CAL | new     |  63744 |        922 | 0.7745 | [0.7607, 0.7896] |    0.06734 |               1.046 |       0.841 |

### 1.4 Paired firm-clustered bootstrap (1,000 draws, seed 0; positive ΔAUC and negative ΔLogLoss favour "new")

| new        | old        | subset    |   defaults | dAUC [95% CI]              | dLogLoss [95% CI]             |   n_draws |
|:-----------|:-----------|:----------|-----------:|:---------------------------|:------------------------------|----------:|
| GAM-01     | GB-01      | ALL|ALL   |       7784 | -0.0120 [-0.0139, -0.0102] | +0.00102 [+0.00088, +0.00115] |      1000 |
| GAM-01     | GB-01      | year|2008 |       2064 | -0.0094 [-0.0127, -0.0060] | +0.00084 [+0.00062, +0.00107] |      1000 |
| GAM-01     | GB-01      | year|2009 |       2752 | -0.0127 [-0.0159, -0.0097] | +0.00109 [+0.00084, +0.00133] |      1000 |
| GAM-01     | GB-01      | year|2010 |       2968 | -0.0129 [-0.0161, -0.0098] | +0.00111 [+0.00087, +0.00134] |      1000 |
| GAM-01     | GB-01      | firm|seen |       6862 | -0.0120 [-0.0142, -0.0101] | +0.00102 [+0.00088, +0.00117] |      1000 |
| GAM-01     | GB-01      | firm|new  |        922 | -0.0121 [-0.0178, -0.0059] | +0.00095 [+0.00055, +0.00133] |      1000 |
| GAM-01     | SC-03      | ALL|ALL   |       7784 | +0.0089 [+0.0073, +0.0105] | -0.00139 [-0.00155, -0.00124] |      1000 |
| GAM-01     | SC-03      | year|2008 |       2064 | +0.0070 [+0.0038, +0.0104] | -0.00124 [-0.00149, -0.00098] |      1000 |
| GAM-01     | SC-03      | year|2009 |       2752 | +0.0115 [+0.0087, +0.0146] | -0.00188 [-0.00216, -0.00160] |      1000 |
| GAM-01     | SC-03      | year|2010 |       2968 | +0.0062 [+0.0032, +0.0090] | -0.00106 [-0.00132, -0.00079] |      1000 |
| GAM-01     | SC-03      | firm|seen |       6862 | +0.0088 [+0.0071, +0.0106] | -0.00148 [-0.00166, -0.00132] |      1000 |
| GAM-01     | SC-03      | firm|new  |        922 | +0.0096 [+0.0038, +0.0159] | -0.00074 [-0.00118, -0.00032] |      1000 |
| GAM-01     | RC-02      | ALL|ALL   |       7784 | +0.0060 [+0.0044, +0.0078] | -0.00063 [-0.00078, -0.00048] |      1000 |
| GAM-01     | RC-02      | year|2008 |       2064 | +0.0020 [-0.0014, +0.0053] | -0.00054 [-0.00080, -0.00027] |      1000 |
| GAM-01     | RC-02      | year|2009 |       2752 | +0.0058 [+0.0029, +0.0091] | -0.00078 [-0.00106, -0.00053] |      1000 |
| GAM-01     | RC-02      | year|2010 |       2968 | +0.0062 [+0.0034, +0.0091] | -0.00056 [-0.00083, -0.00031] |      1000 |
| GAM-01     | RC-02      | firm|seen |       6862 | +0.0057 [+0.0039, +0.0076] | -0.00068 [-0.00083, -0.00052] |      1000 |
| GAM-01     | RC-02      | firm|new  |        922 | +0.0090 [+0.0029, +0.0151] | -0.00029 [-0.00071, +0.00014] |      1000 |
| GAM-02     | GB-01      | ALL|ALL   |       7784 | -0.0112 [-0.0130, -0.0094] | +0.00094 [+0.00081, +0.00107] |      1000 |
| GAM-02     | GB-01      | year|2008 |       2064 | -0.0088 [-0.0120, -0.0055] | +0.00080 [+0.00058, +0.00102] |      1000 |
| GAM-02     | GB-01      | year|2009 |       2752 | -0.0120 [-0.0152, -0.0091] | +0.00102 [+0.00079, +0.00127] |      1000 |
| GAM-02     | GB-01      | year|2010 |       2968 | -0.0114 [-0.0145, -0.0084] | +0.00099 [+0.00077, +0.00123] |      1000 |
| GAM-02     | GB-01      | firm|seen |       6862 | -0.0112 [-0.0132, -0.0093] | +0.00095 [+0.00081, +0.00110] |      1000 |
| GAM-02     | GB-01      | firm|new  |        922 | -0.0118 [-0.0175, -0.0055] | +0.00085 [+0.00045, +0.00123] |      1000 |
| GAM-02     | SC-03      | ALL|ALL   |       7784 | +0.0097 [+0.0080, +0.0114] | -0.00147 [-0.00163, -0.00131] |      1000 |
| GAM-02     | SC-03      | year|2008 |       2064 | +0.0076 [+0.0041, +0.0111] | -0.00128 [-0.00154, -0.00103] |      1000 |
| GAM-02     | SC-03      | year|2009 |       2752 | +0.0122 [+0.0094, +0.0154] | -0.00194 [-0.00222, -0.00166] |      1000 |
| GAM-02     | SC-03      | year|2010 |       2968 | +0.0077 [+0.0045, +0.0108] | -0.00117 [-0.00146, -0.00089] |      1000 |
| GAM-02     | SC-03      | firm|seen |       6862 | +0.0096 [+0.0078, +0.0115] | -0.00156 [-0.00174, -0.00138] |      1000 |
| GAM-02     | SC-03      | firm|new  |        922 | +0.0099 [+0.0039, +0.0162] | -0.00085 [-0.00130, -0.00041] |      1000 |
| GAM-02     | GAM-01     | ALL|ALL   |       7784 | +0.0008 [+0.0004, +0.0012] | -0.00008 [-0.00010, -0.00005] |      1000 |
| GAM-02     | GAM-01     | year|2008 |       2064 | +0.0005 [-0.0002, +0.0012] | -0.00005 [-0.00009, -0.00001] |      1000 |
| GAM-02     | GAM-01     | year|2009 |       2752 | +0.0007 [+0.0003, +0.0011] | -0.00006 [-0.00009, -0.00003] |      1000 |
| GAM-02     | GAM-01     | year|2010 |       2968 | +0.0015 [+0.0007, +0.0024] | -0.00012 [-0.00018, -0.00005] |      1000 |
| GAM-02     | GAM-01     | firm|seen |       6862 | +0.0008 [+0.0005, +0.0012] | -0.00007 [-0.00010, -0.00004] |      1000 |
| GAM-02     | GAM-01     | firm|new  |        922 | +0.0003 [-0.0010, +0.0016] | -0.00010 [-0.00018, -0.00003] |      1000 |
| GAM-01+CAL | CAL-01     | ALL|ALL   |       7784 | -0.0119 [-0.0137, -0.0101] | +0.00098 [+0.00085, +0.00111] |      1000 |
| GAM-01+CAL | CAL-01     | year|2008 |       2064 | -0.0094 [-0.0127, -0.0060] | +0.00081 [+0.00058, +0.00103] |      1000 |
| GAM-01+CAL | CAL-01     | year|2009 |       2752 | -0.0127 [-0.0159, -0.0097] | +0.00104 [+0.00079, +0.00128] |      1000 |
| GAM-01+CAL | CAL-01     | year|2010 |       2968 | -0.0129 [-0.0161, -0.0098] | +0.00109 [+0.00085, +0.00132] |      1000 |
| GAM-01+CAL | CAL-01     | firm|seen |       6862 | -0.0118 [-0.0140, -0.0099] | +0.00098 [+0.00084, +0.00113] |      1000 |
| GAM-01+CAL | CAL-01     | firm|new  |        922 | -0.0118 [-0.0175, -0.0057] | +0.00098 [+0.00058, +0.00136] |      1000 |
| GAM-02+CAL | CAL-01     | ALL|ALL   |       7784 | -0.0110 [-0.0128, -0.0092] | +0.00090 [+0.00077, +0.00104] |      1000 |
| GAM-02+CAL | CAL-01     | year|2008 |       2064 | -0.0088 [-0.0121, -0.0055] | +0.00076 [+0.00054, +0.00098] |      1000 |
| GAM-02+CAL | CAL-01     | year|2009 |       2752 | -0.0120 [-0.0152, -0.0091] | +0.00098 [+0.00074, +0.00122] |      1000 |
| GAM-02+CAL | CAL-01     | year|2010 |       2968 | -0.0114 [-0.0145, -0.0084] | +0.00097 [+0.00075, +0.00120] |      1000 |
| GAM-02+CAL | CAL-01     | firm|seen |       6862 | -0.0110 [-0.0130, -0.0091] | +0.00091 [+0.00077, +0.00105] |      1000 |
| GAM-02+CAL | CAL-01     | firm|new  |        922 | -0.0114 [-0.0172, -0.0051] | +0.00088 [+0.00048, +0.00127] |      1000 |
| GAM-02+CAL | GAM-01+CAL | ALL|ALL   |       7784 | +0.0008 [+0.0005, +0.0012] | -0.00008 [-0.00011, -0.00005] |      1000 |
| GAM-02+CAL | GAM-01+CAL | year|2008 |       2064 | +0.0005 [-0.0002, +0.0012] | -0.00005 [-0.00009, -0.00001] |      1000 |
| GAM-02+CAL | GAM-01+CAL | year|2009 |       2752 | +0.0007 [+0.0003, +0.0011] | -0.00006 [-0.00009, -0.00003] |      1000 |
| GAM-02+CAL | GAM-01+CAL | year|2010 |       2968 | +0.0015 [+0.0007, +0.0024] | -0.00012 [-0.00019, -0.00005] |      1000 |
| GAM-02+CAL | GAM-01+CAL | firm|seen |       6862 | +0.0009 [+0.0005, +0.0013] | -0.00008 [-0.00011, -0.00005] |      1000 |
| GAM-02+CAL | GAM-01+CAL | firm|new  |        922 | +0.0004 [-0.0009, +0.0017] | -0.00010 [-0.00018, -0.00002] |      1000 |

### 1.5 Face validity of the shape functions

Against-share = share of the step function's total movement (bins between the missing and the unknown bin)
that goes against the expected sign; 0 = monotone in the expected direction. All folds:

| term               |   expected |   GAM-01 f1 |   GAM-01 f2 |   GAM-01 f3 |   GAM-02 f1 |   GAM-02 f2 |   GAM-02 f3 |
|:-------------------|-----------:|------------:|------------:|------------:|------------:|------------:|------------:|
| ap_days            |          1 |           0 |           0 |           0 |           0 |           0 |           0 |
| ar_days            |          1 |           0 |           0 |           0 |           0 |           0 |           0 |
| bank_debt_ratio    |          1 |           0 |           0 |           0 |           0 |           0 |           0 |
| cash_ratio         |         -1 |           0 |           0 |           0 |           0 |           0 |           0 |
| cfo_to_debt        |         -1 |           0 |           0 |           0 |           0 |           0 |           0 |
| current_ratio      |         -1 |           0 |           0 |           0 |           0 |           0 |           0 |
| debt_split_missing |          1 |           0 |           0 |           0 |           0 |           0 |           0 |
| ebitda_margin      |         -1 |           0 |           0 |           0 |           0 |           0 |           0 |
| equity_ratio       |         -1 |           0 |           0 |           0 |           0 |           0 |           0 |
| intangible_ratio   |          1 |           0 |           0 |           0 |           0 |           0 |           0 |
| interest_coverage  |         -1 |           0 |           0 |           0 |           0 |           0 |           0 |
| interest_to_assets |          1 |           0 |           0 |           0 |           0 |           0 |           0 |
| log_assets         |         -1 |           0 |           0 |           0 |           0 |           0 |           0 |
| neg_equity         |          1 |           0 |           0 |           0 |           0 |           0 |           0 |
| roa_net            |         -1 |           0 |           0 |           0 |           0 |           0 |           0 |
| roa_oper           |         -1 |           0 |           0 |           0 |           0 |           0 |           0 |
| st_debt_ratio      |          1 |           0 |           0 |           0 |           0 |           0 |           0 |
| st_debt_share      |          1 |           0 |           0 |           0 |           0 |           0 |           0 |
| tangible_ratio     |         -1 |           0 |           0 |           0 |           0 |           0 |           0 |
| wc_ratio           |         -1 |           0 |           0 |           0 |           0 |           0 |           0 |

Fold-3 GAM-01 shapes (log-odds contributions):

| term               |   expected_sign |   range_logodds |   lowest_bin_score |   highest_bin_score |   missing_bin_score |   against_share |
|:-------------------|----------------:|----------------:|-------------------:|--------------------:|--------------------:|----------------:|
| log_assets         |              -1 |           0.376 |              0.134 |              -0.242 |               0     |               0 |
| equity_ratio       |              -1 |           0.632 |              0.451 |              -0.181 |               0     |               0 |
| st_debt_ratio      |               1 |           0.741 |             -0.273 |               0.468 |               0     |               0 |
| bank_debt_ratio    |               1 |           0.146 |             -0.034 |               0.112 |              -0.025 |               0 |
| st_debt_share      |               1 |           0.293 |             -0.234 |               0.059 |               0     |               0 |
| wc_ratio           |              -1 |           0.184 |              0.136 |              -0.047 |               0     |               0 |
| current_ratio      |              -1 |           0.191 |              0.092 |              -0.099 |               0     |               0 |
| cash_ratio         |              -1 |           0.207 |              0.088 |              -0.119 |               0     |               0 |
| roa_net            |              -1 |           0.703 |              0.577 |              -0.126 |               0     |               0 |
| roa_oper           |              -1 |           0.58  |              0.48  |              -0.099 |               0     |               0 |
| ebitda_margin      |              -1 |           0.494 |              0.296 |              -0.198 |               0.29  |               0 |
| interest_coverage  |              -1 |           0.349 |              0.25  |              -0.099 |               0     |               0 |
| cfo_to_debt        |              -1 |           1.049 |              0.875 |              -0.174 |               0     |               0 |
| interest_to_assets |               1 |           0.959 |             -0.146 |               0.813 |               0     |               0 |
| asset_turnover     |               0 |           0.825 |             -0.465 |               0.239 |               0     |               0 |
| ar_days            |               1 |           0.489 |             -0.204 |               0.284 |               0.484 |               0 |
| ap_days            |               1 |           0.488 |             -0.168 |               0.32  |               0.013 |               0 |
| tangible_ratio     |              -1 |           1.05  |              0.436 |              -0.613 |               0     |               0 |
| intangible_ratio   |               1 |           0.109 |             -0.021 |               0.087 |               0     |               0 |
| extraord_to_assets |               0 |           0.757 |              0.557 |               0.605 |               0     |               0 |
| neg_equity         |               1 |           0.852 |             -0.032 |               0.82  |               0     |               0 |
| debt_split_missing |               1 |           0.165 |             -0.026 |               0.139 |               0     |               0 |
| rev_nonpositive    |               0 |           0.193 |              0.006 |              -0.187 |               0     |               0 |
| ateco_group        |               0 |           0.771 |             -0.371 |              -0.423 |               0     |               0 |
| legal_group        |               0 |           0.924 |             -0.477 |              -0.87  |               0     |               0 |
| macro_area         |               0 |           0.691 |             -0.002 |               0.655 |               0     |               0 |

Result: **0 violations in 20 sign-expected terms × 3 folds × 2 models** (the constraints hold; the
missing bin and the unknown bin are outside the constraint and are reported separately). GAM-02's 10 pair
terms per fold involve only features with no expected sign (asset_turnover, extraord_to_assets,
rev_nonpositive, ateco_group, legal_group, macro_area), so pairs cannot break a sign. This confirms D-084.
Items a credit officer will question (not counted as violations because the spec gives no sign):
`rev_nonpositive = 1` lowers the score (−0.19 in fold 3; it overlaps with asset_turnover ≈ 0, which is
−0.47); `legal_group OTHER` −0.87 and `macro_area OTHER` +0.65 are thin pooled levels that unseen codes
also map to; most missing bins were reset to 0 (average firm), which matters for section 3.

### 1.6 Verdict

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

## 2. Power and level by segment (lesson L4; R-33 Tables 9-14 pp. 19-21 report AR by industry, size and year)

AUC with firm-clustered 95% CI (1,000 draws); AR = 2·AUC − 1 (same information). Last column: paired
GB-01 minus GAM-01+CAL AUC difference with its CI. Cells pool the three test years unless the cell is a year.
GAM-01 here is the calibrated file `oot_predictions_calibrated.parquet` as requested.

Size quintile edges (total assets in EUR, quintiles of `asst_tot` on each fold's TRAINING rows):

|   fold | q1_edge_eur   | q2_edge_eur   | q3_edge_eur   | q4_edge_eur   |
|-------:|:--------------|:--------------|:--------------|:--------------|
|      1 | 1,988,987     | 2,768,098     | 4,250,368     | 8,310,610     |
|      2 | 2,001,663     | 2,810,112     | 4,348,225     | 8,555,797     |
|      3 | 2,002,632     | 2,814,278     | 4,352,539     | 8,560,716     |

### 2.1 By test year

|   cell |      n |   defaults | GAM-01+CAL           | GB-01                | RC-02                | SC-03                |   AR GAM-01+CAL |   AR GB-01 |   AR RC-02 |   AR SC-03 | GB-01 minus GAM-01+CAL   |
|-------:|-------:|-----------:|:---------------------|:---------------------|:---------------------|:---------------------|----------------:|-----------:|-----------:|-----------:|:-------------------------|
|   2008 | 161261 |       2064 | 0.835 [0.826, 0.844] | 0.844 [0.835, 0.852] | 0.833 [0.824, 0.841] | 0.828 [0.818, 0.837] |           0.67  |      0.688 |      0.666 |      0.656 | +0.009 [+0.006, +0.013]  |
|   2009 | 170386 |       2752 | 0.823 [0.814, 0.832] | 0.836 [0.828, 0.844] | 0.817 [0.808, 0.826] | 0.812 [0.802, 0.820] |           0.647 |      0.672 |      0.635 |      0.623 | +0.013 [+0.010, +0.016]  |
|   2010 | 174512 |       2968 | 0.818 [0.810, 0.825] | 0.831 [0.823, 0.838] | 0.812 [0.803, 0.820] | 0.812 [0.803, 0.820] |           0.635 |      0.661 |      0.623 |      0.623 | +0.013 [+0.010, +0.016]  |

### 2.2 By size quintile (training-fold edges)

| cell   |      n |   defaults | GAM-01+CAL           | GB-01                | RC-02                | SC-03                |   AR GAM-01+CAL |   AR GB-01 |   AR RC-02 |   AR SC-03 | GB-01 minus GAM-01+CAL   |
|:-------|-------:|-----------:|:---------------------|:---------------------|:---------------------|:---------------------|----------------:|-----------:|-----------:|-----------:|:-------------------------|
| Q1     |  99606 |       1789 | 0.796 [0.784, 0.807] | 0.813 [0.803, 0.823] | 0.790 [0.778, 0.801] | 0.787 [0.776, 0.798] |           0.591 |      0.626 |      0.579 |      0.574 | +0.017 [+0.013, +0.021]  |
| Q2     |  99563 |       1636 | 0.809 [0.797, 0.820] | 0.825 [0.815, 0.835] | 0.801 [0.789, 0.812] | 0.802 [0.790, 0.813] |           0.618 |      0.65  |      0.602 |      0.603 | +0.016 [+0.012, +0.021]  |
| Q3     | 100946 |       1497 | 0.814 [0.802, 0.825] | 0.825 [0.813, 0.834] | 0.808 [0.796, 0.818] | 0.806 [0.794, 0.817] |           0.629 |      0.649 |      0.616 |      0.612 | +0.010 [+0.006, +0.014]  |
| Q4     | 102567 |       1462 | 0.849 [0.840, 0.859] | 0.856 [0.847, 0.866] | 0.844 [0.834, 0.854] | 0.839 [0.830, 0.850] |           0.698 |      0.713 |      0.687 |      0.679 | +0.007 [+0.004, +0.011]  |
| Q5     | 103477 |       1400 | 0.862 [0.851, 0.872] | 0.867 [0.857, 0.877] | 0.856 [0.845, 0.867] | 0.851 [0.839, 0.861] |           0.724 |      0.735 |      0.712 |      0.701 | +0.006 [+0.002, +0.009]  |

### 2.3 By ATECO section (raw `ateco_section`; sections with < 30 test defaults pooled; L real estate separate)

| cell            |      n |   defaults | GAM-01+CAL           | GB-01                | RC-02                | SC-03                |   AR GAM-01+CAL |   AR GB-01 |   AR RC-02 |   AR SC-03 | GB-01 minus GAM-01+CAL   |
|:----------------|-------:|-----------:|:---------------------|:---------------------|:---------------------|:---------------------|----------------:|-----------:|-----------:|-----------:|:-------------------------|
| A               |   8090 |         66 | 0.759 [0.694, 0.825] | 0.775 [0.714, 0.839] | 0.763 [0.700, 0.829] | 0.752 [0.690, 0.816] |           0.519 |      0.55  |      0.526 |      0.504 | +0.016 [-0.004, +0.033]  |
| C               | 123921 |       2257 | 0.867 [0.860, 0.875] | 0.872 [0.866, 0.879] | 0.862 [0.854, 0.870] | 0.860 [0.852, 0.867] |           0.735 |      0.745 |      0.724 |      0.719 | +0.005 [+0.003, +0.007]  |
| D               |   4349 |         32 | 0.733 [0.627, 0.841] | 0.723 [0.609, 0.835] | 0.722 [0.602, 0.840] | 0.720 [0.606, 0.833] |           0.465 |      0.445 |      0.443 |      0.44  | -0.010 [-0.033, +0.014]  |
| E               |   4581 |         63 | 0.837 [0.789, 0.885] | 0.861 [0.819, 0.900] | 0.850 [0.802, 0.894] | 0.841 [0.791, 0.886] |           0.674 |      0.722 |      0.7   |      0.681 | +0.024 [+0.006, +0.043]  |
| F               |  90170 |       1429 | 0.763 [0.750, 0.776] | 0.779 [0.768, 0.791] | 0.747 [0.734, 0.761] | 0.742 [0.728, 0.756] |           0.526 |      0.559 |      0.494 |      0.485 | +0.016 [+0.011, +0.021]  |
| G               |  89604 |       1603 | 0.842 [0.831, 0.852] | 0.847 [0.837, 0.856] | 0.839 [0.829, 0.849] | 0.837 [0.826, 0.847] |           0.683 |      0.693 |      0.678 |      0.673 | +0.005 [+0.002, +0.008]  |
| H               |  13613 |        225 | 0.848 [0.821, 0.873] | 0.855 [0.832, 0.876] | 0.839 [0.812, 0.865] | 0.837 [0.810, 0.862] |           0.697 |      0.71  |      0.679 |      0.674 | +0.007 [-0.003, +0.017]  |
| I               |  14989 |        131 | 0.869 [0.841, 0.897] | 0.878 [0.851, 0.903] | 0.854 [0.822, 0.886] | 0.854 [0.824, 0.884] |           0.738 |      0.756 |      0.709 |      0.708 | +0.009 [-0.001, +0.021]  |
| J               |  11156 |        215 | 0.805 [0.773, 0.834] | 0.819 [0.791, 0.847] | 0.806 [0.777, 0.837] | 0.806 [0.775, 0.836] |           0.609 |      0.639 |      0.612 |      0.613 | +0.015 [+0.004, +0.028]  |
| L               | 100645 |        997 | 0.808 [0.796, 0.821] | 0.822 [0.811, 0.834] | 0.798 [0.785, 0.811] | 0.797 [0.785, 0.810] |           0.617 |      0.643 |      0.597 |      0.595 | +0.013 [+0.008, +0.018]  |
| M               |  22265 |        373 | 0.775 [0.751, 0.799] | 0.790 [0.766, 0.812] | 0.776 [0.750, 0.801] | 0.773 [0.748, 0.797] |           0.55  |      0.579 |      0.552 |      0.546 | +0.014 [+0.004, +0.024]  |
| N               |   9701 |        216 | 0.790 [0.755, 0.823] | 0.797 [0.765, 0.828] | 0.794 [0.762, 0.825] | 0.791 [0.759, 0.823] |           0.579 |      0.594 |      0.588 |      0.581 | +0.007 [-0.002, +0.017]  |
| Q               |   4302 |         34 | 0.833 [0.750, 0.905] | 0.847 [0.781, 0.914] | 0.829 [0.756, 0.894] | 0.827 [0.750, 0.891] |           0.666 |      0.695 |      0.658 |      0.655 | +0.014 [-0.012, +0.038]  |
| R               |   3830 |         73 | 0.781 [0.721, 0.833] | 0.798 [0.735, 0.849] | 0.778 [0.715, 0.829] | 0.783 [0.725, 0.833] |           0.562 |      0.595 |      0.555 |      0.567 | +0.017 [-0.003, +0.037]  |
| S               |   1799 |         30 | 0.790 [0.732, 0.850] | 0.776 [0.705, 0.842] | 0.780 [0.707, 0.845] | 0.778 [0.708, 0.837] |           0.581 |      0.553 |      0.559 |      0.555 | -0.014 [-0.043, +0.015]  |
| pooled(B,O,P,U) |   3144 |         40 | 0.828 [0.748, 0.896] | 0.834 [0.756, 0.900] | 0.830 [0.752, 0.894] | 0.831 [0.752, 0.895] |           0.656 |      0.667 |      0.66  |      0.661 | +0.006 [-0.010, +0.025]  |

### 2.4 Seen vs new firms

| cell   |      n |   defaults | GAM-01+CAL           | GB-01                | RC-02                | SC-03                |   AR GAM-01+CAL |   AR GB-01 |   AR RC-02 |   AR SC-03 | GB-01 minus GAM-01+CAL   |
|:-------|-------:|-----------:|:---------------------|:---------------------|:---------------------|:---------------------|----------------:|-----------:|-----------:|-----------:|:-------------------------|
| new    |  63744 |        922 | 0.774 [0.760, 0.789] | 0.787 [0.773, 0.800] | 0.765 [0.751, 0.780] | 0.765 [0.750, 0.779] |           0.548 |      0.573 |      0.531 |      0.53  | +0.012 [+0.006, +0.018]  |
| seen   | 442415 |       6862 | 0.832 [0.826, 0.837] | 0.844 [0.839, 0.848] | 0.826 [0.821, 0.831] | 0.823 [0.818, 0.828] |           0.664 |      0.688 |      0.652 |      0.646 | +0.012 [+0.010, +0.014]  |

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

### 2.5 Observed vs mean predicted PD in every cell (GB-01 own level, CAL-01 shifted, GAM-01+CAL)

| dimension   | cell            |      n |   defaults | obs DR [CI]             | CAL-01   | GAM-01+CAL   | GB-01   |   obs/pred CAL-01 |   obs/pred GAM-01+CAL |   obs/pred GB-01 |
|:------------|:----------------|-------:|-----------:|:------------------------|:---------|:-------------|:--------|------------------:|----------------------:|-----------------:|
| ateco       | A               |   8090 |         66 | 0.816% [0.632%, 1.037%] | 0.835%   | 0.841%       | 0.666%  |              0.98 |                  0.97 |             1.22 |
| ateco       | C               | 123921 |       2257 | 1.821% [1.748%, 1.897%] | 1.798%   | 1.735%       | 1.472%  |              1.01 |                  1.05 |             1.24 |
| ateco       | D               |   4349 |         32 | 0.736% [0.504%, 1.037%] | 1.107%   | 1.009%       | 0.885%  |              0.66 |                  0.73 |             0.83 |
| ateco       | E               |   4581 |         63 | 1.375% [1.058%, 1.756%] | 1.011%   | 0.994%       | 0.819%  |              1.36 |                  1.38 |             1.68 |
| ateco       | F               |  90170 |       1429 | 1.585% [1.504%, 1.668%] | 1.411%   | 1.439%       | 1.135%  |              1.12 |                  1.1  |             1.4  |
| ateco       | G               |  89604 |       1603 | 1.789% [1.703%, 1.878%] | 1.609%   | 1.632%       | 1.303%  |              1.11 |                  1.1  |             1.37 |
| ateco       | H               |  13613 |        225 | 1.653% [1.445%, 1.881%] | 1.407%   | 1.401%       | 1.136%  |              1.17 |                  1.18 |             1.46 |
| ateco       | I               |  14989 |        131 | 0.874% [0.731%, 1.036%] | 0.807%   | 0.825%       | 0.652%  |              1.08 |                  1.06 |             1.34 |
| ateco       | J               |  11156 |        215 | 1.927% [1.680%, 2.200%] | 1.851%   | 1.746%       | 1.501%  |              1.04 |                  1.1  |             1.28 |
| ateco       | L               | 100645 |        997 | 0.991% [0.930%, 1.054%] | 0.847%   | 0.846%       | 0.678%  |              1.17 |                  1.17 |             1.46 |
| ateco       | M               |  22265 |        373 | 1.675% [1.511%, 1.853%] | 1.476%   | 1.426%       | 1.189%  |              1.13 |                  1.18 |             1.41 |
| ateco       | N               |   9701 |        216 | 2.227% [1.942%, 2.540%] | 1.876%   | 1.790%       | 1.520%  |              1.19 |                  1.24 |             1.46 |
| ateco       | Q               |   4302 |         34 | 0.790% [0.548%, 1.103%] | 1.119%   | 1.078%       | 0.907%  |              0.71 |                  0.73 |             0.87 |
| ateco       | R               |   3830 |         73 | 1.906% [1.497%, 2.391%] | 1.781%   | 1.843%       | 1.449%  |              1.07 |                  1.03 |             1.32 |
| ateco       | S               |   1799 |         30 | 1.668% [1.128%, 2.372%] | 0.931%   | 0.865%       | 0.745%  |              1.79 |                  1.93 |             2.24 |
| ateco       | pooled(B,O,P,U) |   3144 |         40 | 1.272% [0.910%, 1.728%] | 1.103%   | 1.014%       | 0.892%  |              1.15 |                  1.26 |             1.43 |
| firm        | new             |  63744 |        922 | 1.446% [1.355%, 1.542%] | 1.363%   | 1.381%       | 1.099%  |              1.06 |                  1.05 |             1.32 |
| firm        | seen            | 442415 |       6862 | 1.551% [1.515%, 1.588%] | 1.421%   | 1.402%       | 1.152%  |              1.09 |                  1.11 |             1.35 |
| size        | Q1              |  99606 |       1789 | 1.796% [1.715%, 1.880%] | 1.680%   | 1.647%       | 1.364%  |              1.07 |                  1.09 |             1.32 |
| size        | Q2              |  99563 |       1636 | 1.643% [1.565%, 1.724%] | 1.512%   | 1.497%       | 1.226%  |              1.09 |                  1.1  |             1.34 |
| size        | Q3              | 100946 |       1497 | 1.483% [1.409%, 1.559%] | 1.412%   | 1.406%       | 1.142%  |              1.05 |                  1.05 |             1.3  |
| size        | Q4              | 102567 |       1462 | 1.425% [1.354%, 1.500%] | 1.305%   | 1.303%       | 1.055%  |              1.09 |                  1.09 |             1.35 |
| size        | Q5              | 103477 |       1400 | 1.353% [1.283%, 1.425%] | 1.173%   | 1.157%       | 0.949%  |              1.15 |                  1.17 |             1.43 |
| year        | 2008            | 161261 |       2064 | 1.280% [1.226%, 1.336%] | 1.260%   | 1.243%       | 1.056%  |              1.02 |                  1.03 |             1.21 |
| year        | 2009            | 170386 |       2752 | 1.615% [1.556%, 1.676%] | 1.533%   | 1.515%       | 1.185%  |              1.05 |                  1.07 |             1.36 |
| year        | 2010            | 174512 |       2968 | 1.701% [1.641%, 1.763%] | 1.439%   | 1.431%       | 1.188%  |              1.18 |                  1.19 |             1.43 |

Level by cell (CAL-01, the shipped level logic): pooled obs/pred 1.09 (2008 1.02, 2009 1.05, 2010 1.18).
Largest misses with enough defaults: real estate L 1.17 and the largest firms Q5 1.15 (under-predicted,
also for GB-01 at its own level: L 1.46, Q5 1.43, against 1.34 overall); E 1.36 (63 defaults); D 0.66
and Q 0.71 are over-predicted (32 and 34 defaults, CIs wide). S 1.79 rests on 30 defaults. The smallest
quintile Q1 is close (1.07). These are diagnostic: correcting them would be a group-level fit on test years
(D-078 rejected that).

Figure: `fig/r1b_segment_auc.png`.

## 3. Missing-input check (lesson L1)

Groups: `debt_split_missing = 1` (any of debt_bank_st, debt_bank_lt, AP_st missing in the raw row; GB-01
sees this flag); "core feature NaN after rebuild" = at least one core numeric feature is NaN after the
accounting-identity rebuild of `build.py` (exactly the rows a GLM view would median-fill); "core raw input
missing" = at least one raw field that feeds a core feature is blank in the raw data (before the rebuild).
`relative obs/pred` = the group's obs/pred divided by the obs/pred of all rows of the same year, i.e. how
much MORE the group is under-predicted than the average firm (1 = no extra under-prediction).

| group                                                      | test_year   | model   |      n |   defaults | observed_dr   | dr_ci_low   | dr_ci_high   | mean_pd   |   obs_over_pred |   obs_over_pred_all_rows_same_year |   relative_obs_over_pred |
|:-----------------------------------------------------------|:------------|:--------|-------:|-----------:|:--------------|:------------|:-------------|:----------|----------------:|-----------------------------------:|-------------------------:|
| debt_split_missing=1                                       | ALL         | GB-01   |  88593 |       1446 | 1.632%        | 1.550%      | 1.718%       | 1.168%    |            1.4  |                               1.34 |                     1.04 |
| debt_split_missing=1                                       | ALL         | CAL-01  |  88593 |       1446 | 1.632%        | 1.550%      | 1.718%       | 1.445%    |            1.13 |                               1.09 |                     1.04 |
| debt_split_missing=1                                       | 2008        | GB-01   |  28297 |        375 | 1.325%        | 1.195%      | 1.465%       | 1.005%    |            1.32 |                               1.21 |                     1.09 |
| debt_split_missing=1                                       | 2008        | CAL-01  |  28297 |        375 | 1.325%        | 1.195%      | 1.465%       | 1.202%    |            1.1  |                               1.02 |                     1.08 |
| debt_split_missing=1                                       | 2009        | GB-01   |  31133 |        523 | 1.680%        | 1.540%      | 1.829%       | 1.207%    |            1.39 |                               1.36 |                     1.02 |
| debt_split_missing=1                                       | 2009        | CAL-01  |  31133 |        523 | 1.680%        | 1.540%      | 1.829%       | 1.562%    |            1.08 |                               1.05 |                     1.02 |
| debt_split_missing=1                                       | 2010        | GB-01   |  29163 |        548 | 1.879%        | 1.726%      | 2.042%       | 1.285%    |            1.46 |                               1.43 |                     1.02 |
| debt_split_missing=1                                       | 2010        | CAL-01  |  29163 |        548 | 1.879%        | 1.726%      | 2.042%       | 1.555%    |            1.21 |                               1.18 |                     1.02 |
| debt_split_missing=0                                       | ALL         | GB-01   | 417566 |       6338 | 1.518%        | 1.481%      | 1.555%       | 1.140%    |            1.33 |                               1.34 |                     0.99 |
| debt_split_missing=0                                       | ALL         | CAL-01  | 417566 |       6338 | 1.518%        | 1.481%      | 1.555%       | 1.407%    |            1.08 |                               1.09 |                     0.99 |
| debt_split_missing=0                                       | 2008        | GB-01   | 132964 |       1689 | 1.270%        | 1.211%      | 1.332%       | 1.067%    |            1.19 |                               1.21 |                     0.98 |
| debt_split_missing=0                                       | 2008        | CAL-01  | 132964 |       1689 | 1.270%        | 1.211%      | 1.332%       | 1.272%    |            1    |                               1.02 |                     0.98 |
| debt_split_missing=0                                       | 2009        | GB-01   | 139253 |       2229 | 1.601%        | 1.535%      | 1.668%       | 1.181%    |            1.36 |                               1.36 |                     1    |
| debt_split_missing=0                                       | 2009        | CAL-01  | 139253 |       2229 | 1.601%        | 1.535%      | 1.668%       | 1.526%    |            1.05 |                               1.05 |                     1    |
| debt_split_missing=0                                       | 2010        | GB-01   | 145349 |       2420 | 1.665%        | 1.600%      | 1.732%       | 1.168%    |            1.43 |                               1.43 |                     1    |
| debt_split_missing=0                                       | 2010        | CAL-01  | 145349 |       2420 | 1.665%        | 1.600%      | 1.732%       | 1.416%    |            1.18 |                               1.18 |                     1    |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | ALL         | GB-01   |   1800 |        114 | 6.333%        | 5.252%      | 7.559%       | 4.550%    |            1.39 |                               1.34 |                     1.04 |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | ALL         | CAL-01  |   1800 |        114 | 6.333%        | 5.252%      | 7.559%       | 5.408%    |            1.17 |                               1.09 |                     1.08 |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | 2008        | GB-01   |    517 |         25 | 4.836%        | 3.153%      | 7.056%       | 2.731%    |            1.77 |                               1.21 |                     1.46 |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | 2008        | CAL-01  |    517 |         25 | 4.836%        | 3.153%      | 7.056%       | 3.199%    |            1.51 |                               1.02 |                     1.49 |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | 2009        | GB-01   |    614 |         43 | 7.003%        | 5.114%      | 9.318%       | 4.008%    |            1.75 |                               1.36 |                     1.28 |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | 2009        | CAL-01  |    614 |         43 | 7.003%        | 5.114%      | 9.318%       | 5.018%    |            1.4  |                               1.05 |                     1.32 |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | 2010        | GB-01   |    669 |         46 | 6.876%        | 5.078%      | 9.065%       | 6.453%    |            1.07 |                               1.43 |                     0.74 |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | 2010        | CAL-01  |    669 |         46 | 6.876%        | 5.078%      | 9.065%       | 7.472%    |            0.92 |                               1.18 |                     0.78 |
| no_core_feature_NaN                                        | ALL         | GB-01   | 504359 |       7670 | 1.521%        | 1.487%      | 1.555%       | 1.133%    |            1.34 |                               1.34 |                     1    |
| no_core_feature_NaN                                        | ALL         | CAL-01  | 504359 |       7670 | 1.521%        | 1.487%      | 1.555%       | 1.399%    |            1.09 |                               1.09 |                     1    |
| no_core_feature_NaN                                        | 2008        | GB-01   | 160744 |       2039 | 1.268%        | 1.214%      | 1.324%       | 1.051%    |            1.21 |                               1.21 |                     1    |
| no_core_feature_NaN                                        | 2008        | CAL-01  | 160744 |       2039 | 1.268%        | 1.214%      | 1.324%       | 1.253%    |            1.01 |                               1.02 |                     1    |
| no_core_feature_NaN                                        | 2009        | GB-01   | 169772 |       2709 | 1.596%        | 1.537%      | 1.656%       | 1.175%    |            1.36 |                               1.36 |                     1    |
| no_core_feature_NaN                                        | 2009        | CAL-01  | 169772 |       2709 | 1.596%        | 1.537%      | 1.656%       | 1.520%    |            1.05 |                               1.05 |                     1    |
| no_core_feature_NaN                                        | 2010        | GB-01   | 173843 |       2922 | 1.681%        | 1.621%      | 1.742%       | 1.167%    |            1.44 |                               1.43 |                     1.01 |
| no_core_feature_NaN                                        | 2010        | CAL-01  | 173843 |       2922 | 1.681%        | 1.621%      | 1.742%       | 1.416%    |            1.19 |                               1.18 |                     1    |
| core_raw_input_missing_in_raw_data                         | ALL         | GB-01   |    405 |          8 | 1.975%        | 0.857%      | 3.855%       | 1.232%    |            1.6  |                               1.34 |                     1.19 |
| core_raw_input_missing_in_raw_data                         | ALL         | CAL-01  |    405 |          8 | 1.975%        | 0.857%      | 3.855%       | 1.567%    |            1.26 |                               1.09 |                     1.16 |
| core_raw_input_missing_in_raw_data                         | 2008        | GB-01   |    161 |          3 | 1.863%        | 0.386%      | 5.349%       | 0.492%    |            3.79 |                               1.21 |                     3.13 |
| core_raw_input_missing_in_raw_data                         | 2008        | CAL-01  |    161 |          3 | 1.863%        | 0.386%      | 5.349%       | 0.599%    |            3.11 |                               1.02 |                     3.06 |
| core_raw_input_missing_in_raw_data                         | 2009        | GB-01   |    163 |          3 | 1.840%        | 0.381%      | 5.284%       | 1.432%    |            1.29 |                               1.36 |                     0.94 |
| core_raw_input_missing_in_raw_data                         | 2009        | CAL-01  |    163 |          3 | 1.840%        | 0.381%      | 5.284%       | 1.894%    |            0.97 |                               1.05 |                     0.92 |
| core_raw_input_missing_in_raw_data                         | 2010        | GB-01   |     81 |          2 | 2.469%        | 0.300%      | 8.636%       | 2.302%    |            1.07 |                               1.43 |                     0.75 |
| core_raw_input_missing_in_raw_data                         | 2010        | CAL-01  |     81 |          2 | 2.469%        | 0.300%      | 8.636%       | 2.832%    |            0.87 |                               1.18 |                     0.74 |
| no_core_raw_input_missing                                  | ALL         | GB-01   | 505754 |       7776 | 1.538%        | 1.504%      | 1.572%       | 1.145%    |            1.34 |                               1.34 |                     1    |
| no_core_raw_input_missing                                  | ALL         | CAL-01  | 505754 |       7776 | 1.538%        | 1.504%      | 1.572%       | 1.413%    |            1.09 |                               1.09 |                     1    |
| no_core_raw_input_missing                                  | 2008        | GB-01   | 161100 |       2061 | 1.279%        | 1.225%      | 1.335%       | 1.057%    |            1.21 |                               1.21 |                     1    |
| no_core_raw_input_missing                                  | 2008        | CAL-01  | 161100 |       2061 | 1.279%        | 1.225%      | 1.335%       | 1.260%    |            1.02 |                               1.02 |                     1    |
| no_core_raw_input_missing                                  | 2009        | GB-01   | 170223 |       2749 | 1.615%        | 1.556%      | 1.676%       | 1.185%    |            1.36 |                               1.36 |                     1    |
| no_core_raw_input_missing                                  | 2009        | CAL-01  | 170223 |       2749 | 1.615%        | 1.556%      | 1.676%       | 1.532%    |            1.05 |                               1.05 |                     1    |
| no_core_raw_input_missing                                  | 2010        | GB-01   | 174431 |       2966 | 1.700%        | 1.640%      | 1.762%       | 1.187%    |            1.43 |                               1.43 |                     1    |
| no_core_raw_input_missing                                  | 2010        | CAL-01  | 174431 |       2966 | 1.700%        | 1.640%      | 1.762%       | 1.438%    |            1.18 |                               1.18 |                     1    |

Pooled over years, with the expectation adjusted for each year's overall level miss (Poisson test,
diagnostic; ignores default correlation):

| group                                                      | model      |      n |   defaults |   expected_at_model_level |   obs_over_pred |   expected_after_year_level_adjustment |   relative_obs_over_pred |   relative_ci_low |   relative_ci_high |   poisson_p_two_sided |
|:-----------------------------------------------------------|:-----------|-------:|-----------:|--------------------------:|----------------:|---------------------------------------:|-------------------------:|------------------:|-------------------:|----------------------:|
| core_feature_NaN_after_rebuild (median-filled in GLM view) | CAL-01     |   1800 |        114 |                      97.3 |            1.17 |                                  108.4 |                     1.05 |              0.87 |               1.26 |                 0.612 |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | GAM-01+CAL |   1800 |        114 |                      83.5 |            1.37 |                                   93.9 |                     1.21 |              1    |               1.46 |                 0.049 |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | GB-01      |   1800 |        114 |                      81.9 |            1.39 |                                  112.5 |                     1.01 |              0.84 |               1.22 |                 0.91  |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | RC-02      |   1800 |        114 |                     109.9 |            1.04 |                                  152.4 |                     0.75 |              0.62 |               0.9  |                 0.001 |
| core_feature_NaN_after_rebuild (median-filled in GLM view) | SC-03      |   1800 |        114 |                      62   |            1.84 |                                   88   |                     1.3  |              1.07 |               1.56 |                 0.009 |
| core_raw_input_missing_in_raw_data                         | CAL-01     |    405 |          8 |                       6.3 |            1.26 |                                    6.9 |                     1.15 |              0.5  |               2.27 |                 0.786 |
| core_raw_input_missing_in_raw_data                         | GAM-01+CAL |    405 |          8 |                       3.9 |            2.05 |                                    4.2 |                     1.88 |              0.81 |               3.71 |                 0.134 |
| core_raw_input_missing_in_raw_data                         | GB-01      |    405 |          8 |                       5   |            1.6  |                                    6.8 |                     1.18 |              0.51 |               2.32 |                 0.745 |
| core_raw_input_missing_in_raw_data                         | RC-02      |    405 |          8 |                       4.7 |            1.7  |                                    6.4 |                     1.25 |              0.54 |               2.45 |                 0.633 |
| core_raw_input_missing_in_raw_data                         | SC-03      |    405 |          8 |                       3.4 |            2.34 |                                    4.7 |                     1.69 |              0.73 |               3.33 |                 0.213 |
| debt_split_missing=0                                       | CAL-01     | 417566 |       6338 |                    5874.2 |            1.08 |                                 6389.7 |                     0.99 |              0.97 |               1.02 |                 0.522 |
| debt_split_missing=0                                       | GAM-01+CAL | 417566 |       6338 |                    5821.5 |            1.09 |                                 6396.4 |                     0.99 |              0.97 |               1.02 |                 0.47  |
| debt_split_missing=0                                       | GB-01      | 417566 |       6338 |                    4760.9 |            1.33 |                                 6390.7 |                     0.99 |              0.97 |               1.02 |                 0.515 |
| debt_split_missing=0                                       | RC-02      | 417566 |       6338 |                    4726.9 |            1.34 |                                 6440.3 |                     0.98 |              0.96 |               1.01 |                 0.204 |
| debt_split_missing=0                                       | SC-03      | 417566 |       6338 |                    4624.9 |            1.37 |                                 6438.9 |                     0.98 |              0.96 |               1.01 |                 0.21  |
| debt_split_missing=1                                       | CAL-01     |  88593 |       1446 |                    1280.2 |            1.13 |                                 1394.3 |                     1.04 |              0.98 |               1.09 |                 0.171 |
| debt_split_missing=1                                       | GAM-01+CAL |  88593 |       1446 |                    1262.3 |            1.15 |                                 1387.6 |                     1.04 |              0.99 |               1.1  |                 0.122 |
| debt_split_missing=1                                       | GB-01      |  88593 |       1446 |                    1035   |            1.4  |                                 1393.3 |                     1.04 |              0.98 |               1.09 |                 0.163 |
| debt_split_missing=1                                       | RC-02      |  88593 |       1446 |                     986.6 |            1.47 |                                 1343.7 |                     1.08 |              1.02 |               1.13 |                 0.006 |
| debt_split_missing=1                                       | SC-03      |  88593 |       1446 |                     966.8 |            1.5  |                                 1345.1 |                     1.08 |              1.02 |               1.13 |                 0.007 |
| no_core_feature_NaN                                        | CAL-01     | 504359 |       7670 |                    7057   |            1.09 |                                 7675.6 |                     1    |              0.98 |               1.02 |                 0.955 |
| no_core_feature_NaN                                        | GAM-01+CAL | 504359 |       7670 |                    7000.2 |            1.1  |                                 7690.1 |                     1    |              0.98 |               1.02 |                 0.825 |
| no_core_feature_NaN                                        | GB-01      | 504359 |       7670 |                    5714   |            1.34 |                                 7671.5 |                     1    |              0.98 |               1.02 |                 0.992 |
| no_core_feature_NaN                                        | RC-02      | 504359 |       7670 |                    5603.5 |            1.37 |                                 7631.6 |                     1.01 |              0.98 |               1.03 |                 0.663 |
| no_core_feature_NaN                                        | SC-03      | 504359 |       7670 |                    5529.8 |            1.39 |                                 7696   |                     1    |              0.97 |               1.02 |                 0.773 |
| no_core_raw_input_missing                                  | CAL-01     | 505754 |       7776 |                    7148   |            1.09 |                                 7777.1 |                     1    |              0.98 |               1.02 |                 0.996 |
| no_core_raw_input_missing                                  | GAM-01+CAL | 505754 |       7776 |                    7079.8 |            1.1  |                                 7779.8 |                     1    |              0.98 |               1.02 |                 0.972 |
| no_core_raw_input_missing                                  | GB-01      | 505754 |       7776 |                    5790.9 |            1.34 |                                 7777.2 |                     1    |              0.98 |               1.02 |                 0.995 |
| no_core_raw_input_missing                                  | RC-02      | 505754 |       7776 |                    5708.7 |            1.36 |                                 7777.6 |                     1    |              0.98 |               1.02 |                 0.992 |
| no_core_raw_input_missing                                  | SC-03      | 505754 |       7776 |                    5588.3 |            1.39 |                                 7779.3 |                     1    |              0.98 |               1.02 |                 0.976 |

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

## 4. Grade grid with no grade over ~12% of firms

Edge-setting sample: the fold-3 GB-01 model (fit on 2007-2009) scored **in-sample on its 475,861 training
rows** (`experiments/calibration/cache/gb_train_3.parquet`), shifted to the CAL-01 level with the fold-3
prior-shift delta (0.2175 = logit(τ_e,2010) − logit(r_w,2007-09); recomputed from the files and from the
training rate: identical, `r1b_grade_edges.json`). 2010 PDs and labels were not used to choose edges.
Rule: keep floor 0.03%, cap 99% and the 11 existing edges; repeatedly split the grade with the largest share
above 12% at the median PD of its rows rounded to two significant digits. Result: 5 new edges (0.17%, 0.36%,
0.62%, 1.2%, 1.9%), 16 grades. R-33 Sec. 4.9 p. 24 designs EDF-implied rating scales so that no category holds
too many credits (< 12% of observations per category in their sample); the 12% is their number, used here as
our design target.

Edges:

old: 0.03%, 0.1%, 0.25%, 0.5%, 0.75%, 1%, 1.5%, 2.5%, 5%, 10%, 20%, 99%

new: 0.03%, 0.1%, 0.17%, 0.25%, 0.36%, 0.5%, 0.62%, 0.75%, 1%, 1.2%, 1.5%, 1.9%, 2.5%, 5%, 10%, 20%, 99%

Share of firms per grade: edge-setting sample (fold-3 model, 2007-2009 training rows), OOT 2008-2009 and 2010:

|   grade | pd_band        | new_grid_2010_CAL-01   | new_grid_2010_GB-01_own_level   | new_grid_OOT_2008-2009_CAL-01   | new_grid_training_years_2007-2009_fold3_insample   |
|--------:|:---------------|:-----------------------|:--------------------------------|:--------------------------------|:---------------------------------------------------|
|       1 | [0.03%, 0.1%)  | 3.5%                   | 6.7%                            | 5.6%                            | 2.8%                                               |
|       2 | [0.1%, 0.17%)  | 9.8%                   | 11.9%                           | 10.3%                           | 8.8%                                               |
|       3 | [0.17%, 0.25%) | 9.8%                   | 10.4%                           | 9.7%                            | 9.2%                                               |
|       4 | [0.25%, 0.36%) | 10.0%                  | 10.3%                           | 10.0%                           | 9.8%                                               |
|       5 | [0.36%, 0.5%)  | 9.3%                   | 9.5%                            | 10.1%                           | 9.3%                                               |
|       6 | [0.5%, 0.62%)  | 6.3%                   | 6.7%                            | 6.8%                            | 6.2%                                               |
|       7 | [0.62%, 0.75%) | 5.9%                   | 6.6%                            | 6.2%                            | 6.0%                                               |
|       8 | [0.75%, 1%)    | 10.0%                  | 9.9%                            | 9.3%                            | 9.9%                                               |
|       9 | [1%, 1.2%)     | 6.3%                   | 5.5%                            | 5.6%                            | 6.4%                                               |
|      10 | [1.2%, 1.5%)   | 6.8%                   | 5.9%                            | 6.1%                            | 7.2%                                               |
|      11 | [1.5%, 1.9%)   | 6.1%                   | 4.9%                            | 5.4%                            | 6.6%                                               |
|      12 | [1.9%, 2.5%)   | 5.5%                   | 4.1%                            | 4.8%                            | 6.2%                                               |
|      13 | [2.5%, 5%)     | 6.9%                   | 4.8%                            | 6.1%                            | 7.5%                                               |
|      14 | [5%, 10%)      | 2.2%                   | 1.6%                            | 2.1%                            | 2.3%                                               |
|      15 | [10%, 20%)     | 0.9%                   | 0.7%                            | 1.0%                            | 1.0%                                               |
|      16 | [20%, 99%)     | 0.7%                   | 0.6%                            | 0.8%                            | 0.8%                                               |

### 4.1 2010 grade table, new grid, CAL-01 (the level logic that would ship)

Flag = observed default count outside the 95% acceptance interval implied by the grade's PDs (normal
approximation to the Poisson-binomial; binomial(n, mean PD) when fewer than 20 expected defaults). The
binomial bands ignore default correlation and therefore understate the uncertainty (R-16 pp. 33-34): read
the flags as diagnostic.

|   grade | pd_band        |     n | share_of_firms   |   defaults | share_of_defaults   | mean_pd   | observed_dr   | dr_ci_low   | dr_ci_high   |   expected_defaults |   accept_lo_defaults |   accept_hi_defaults |   excess_defaults |   obs_over_pred | flag            |
|--------:|:---------------|------:|:-----------------|-----------:|:--------------------|:----------|:--------------|:------------|:-------------|--------------------:|---------------------:|---------------------:|------------------:|----------------:|:----------------|
|       1 | [0.03%, 0.1%)  |  6158 | 3.5%             |          4 | 0.1%                | 0.082%    | 0.065%        | 0.018%      | 0.166%       |                 5.1 |                    1 |                   10 |              -1.1 |            0.79 |                 |
|       2 | [0.1%, 0.17%)  | 17162 | 9.8%             |         14 | 0.5%                | 0.136%    | 0.082%        | 0.045%      | 0.137%       |                23.3 |                   14 |                   33 |              -9.3 |            0.6  |                 |
|       3 | [0.17%, 0.25%) | 17032 | 9.8%             |         37 | 1.2%                | 0.208%    | 0.217%        | 0.153%      | 0.299%       |                35.5 |                   24 |                   47 |               1.5 |            1.04 |                 |
|       4 | [0.25%, 0.36%) | 17513 | 10.0%            |         46 | 1.5%                | 0.302%    | 0.263%        | 0.192%      | 0.350%       |                52.9 |                   39 |                   67 |              -6.9 |            0.87 |                 |
|       5 | [0.36%, 0.5%)  | 16292 | 9.3%             |         73 | 2.5%                | 0.426%    | 0.448%        | 0.351%      | 0.563%       |                69.4 |                   53 |                   86 |               3.6 |            1.05 |                 |
|       6 | [0.5%, 0.62%)  | 10985 | 6.3%             |         77 | 2.6%                | 0.558%    | 0.701%        | 0.554%      | 0.875%       |                61.3 |                   46 |                   77 |              15.7 |            1.26 | UNDER-predicted |
|       7 | [0.62%, 0.75%) | 10326 | 5.9%             |         83 | 2.8%                | 0.684%    | 0.804%        | 0.641%      | 0.995%       |                70.6 |                   54 |                   87 |              12.4 |            1.17 |                 |
|       8 | [0.75%, 1%)    | 17390 | 10.0%            |        194 | 6.5%                | 0.871%    | 1.116%        | 0.965%      | 1.283%       |               151.4 |                  127 |                  175 |              42.6 |            1.28 | UNDER-predicted |
|       9 | [1%, 1.2%)     | 10911 | 6.3%             |        155 | 5.2%                | 1.097%    | 1.421%        | 1.207%      | 1.661%       |               119.7 |                   98 |                  141 |              35.3 |            1.3  | UNDER-predicted |
|      10 | [1.2%, 1.5%)   | 11803 | 6.8%             |        210 | 7.1%                | 1.339%    | 1.779%        | 1.548%      | 2.034%       |               158   |                  134 |                  183 |              52   |            1.33 | UNDER-predicted |
|      11 | [1.5%, 1.9%)   | 10694 | 6.1%             |        226 | 7.6%                | 1.682%    | 2.113%        | 1.849%      | 2.404%       |               179.9 |                  154 |                  206 |              46.1 |            1.26 | UNDER-predicted |
|      12 | [1.9%, 2.5%)   |  9675 | 5.5%             |        276 | 9.3%                | 2.172%    | 2.853%        | 2.530%      | 3.204%       |               210.1 |                  182 |                  238 |              65.9 |            1.31 | UNDER-predicted |
|      13 | [2.5%, 5%)     | 11961 | 6.9%             |        571 | 19.2%               | 3.376%    | 4.774%        | 4.399%      | 5.171%       |               403.8 |                  365 |                  442 |             167.2 |            1.41 | UNDER-predicted |
|      14 | [5%, 10%)      |  3764 | 2.2%             |        333 | 11.2%               | 6.806%    | 8.847%        | 7.959%      | 9.800%       |               256.2 |                  226 |                  286 |              76.8 |            1.3  | UNDER-predicted |
|      15 | [10%, 20%)     |  1552 | 0.9%             |        218 | 7.3%                | 13.870%   | 14.046%       | 12.355%     | 15.875%      |               215.3 |                  189 |                  242 |               2.7 |            1.01 |                 |
|      16 | [20%, 99%)     |  1294 | 0.7%             |        451 | 15.2%               | 38.555%   | 34.853%       | 32.255%     | 37.520%      |               498.9 |                  467 |                  531 |             -47.9 |            0.9  | OVER-predicted  |

### 4.2 Same grid, GB-01 at its own level (no prior shift)

|   grade | pd_band        |     n | share_of_firms   |   defaults | share_of_defaults   | mean_pd   | observed_dr   | dr_ci_low   | dr_ci_high   |   expected_defaults |   accept_lo_defaults |   accept_hi_defaults |   excess_defaults |   obs_over_pred | flag            |
|--------:|:---------------|------:|:-----------------|-----------:|:--------------------|:----------|:--------------|:------------|:-------------|--------------------:|---------------------:|---------------------:|------------------:|----------------:|:----------------|
|       1 | [0.03%, 0.1%)  | 11759 | 6.7%             |          9 | 0.3%                | 0.078%    | 0.077%        | 0.035%      | 0.145%       |                 9.2 |                    4 |                   16 |              -0.2 |            0.98 |                 |
|       2 | [0.1%, 0.17%)  | 20790 | 11.9%            |         28 | 0.9%                | 0.134%    | 0.135%        | 0.090%      | 0.195%       |                27.8 |                   17 |                   38 |               0.2 |            1.01 |                 |
|       3 | [0.17%, 0.25%) | 18140 | 10.4%            |         44 | 1.5%                | 0.208%    | 0.243%        | 0.176%      | 0.325%       |                37.7 |                   26 |                   50 |               6.3 |            1.17 |                 |
|       4 | [0.25%, 0.36%) | 17924 | 10.3%            |         62 | 2.1%                | 0.302%    | 0.346%        | 0.265%      | 0.443%       |                54.1 |                   40 |                   69 |               7.9 |            1.15 |                 |
|       5 | [0.36%, 0.5%)  | 16579 | 9.5%             |        109 | 3.7%                | 0.427%    | 0.657%        | 0.540%      | 0.793%       |                70.8 |                   54 |                   87 |              38.2 |            1.54 | UNDER-predicted |
|       6 | [0.5%, 0.62%)  | 11707 | 6.7%             |         95 | 3.2%                | 0.559%    | 0.811%        | 0.657%      | 0.991%       |                65.4 |                   50 |                   81 |              29.6 |            1.45 | UNDER-predicted |
|       7 | [0.62%, 0.75%) | 11502 | 6.6%             |        126 | 4.2%                | 0.684%    | 1.095%        | 0.913%      | 1.303%       |                78.6 |                   61 |                   96 |              47.4 |            1.6  | UNDER-predicted |
|       8 | [0.75%, 1%)    | 17238 | 9.9%             |        243 | 8.2%                | 0.867%    | 1.410%        | 1.239%      | 1.597%       |               149.5 |                  126 |                  173 |              93.5 |            1.63 | UNDER-predicted |
|       9 | [1%, 1.2%)     |  9525 | 5.5%             |        167 | 5.6%                | 1.094%    | 1.753%        | 1.499%      | 2.037%       |               104.2 |                   84 |                  124 |              62.8 |            1.6  | UNDER-predicted |
|      10 | [1.2%, 1.5%)   | 10224 | 5.9%             |        211 | 7.1%                | 1.338%    | 2.064%        | 1.797%      | 2.358%       |               136.8 |                  114 |                  160 |              74.2 |            1.54 | UNDER-predicted |
|      11 | [1.5%, 1.9%)   |  8610 | 4.9%             |        241 | 8.1%                | 1.684%    | 2.799%        | 2.461%      | 3.170%       |               145   |                  122 |                  168 |              96   |            1.66 | UNDER-predicted |
|      12 | [1.9%, 2.5%)   |  7127 | 4.1%             |        264 | 8.9%                | 2.163%    | 3.704%        | 3.278%      | 4.169%       |               154.2 |                  130 |                  178 |             109.8 |            1.71 | UNDER-predicted |
|      13 | [2.5%, 5%)     |  8336 | 4.8%             |        479 | 16.1%               | 3.369%    | 5.746%        | 5.256%      | 6.267%       |               280.9 |                  249 |                  313 |             198.1 |            1.71 | UNDER-predicted |
|      14 | [5%, 10%)      |  2758 | 1.6%             |        285 | 9.6%                | 6.855%    | 10.334%       | 9.222%      | 11.530%      |               189.1 |                  163 |                  215 |              95.9 |            1.51 | UNDER-predicted |
|      15 | [10%, 20%)     |  1238 | 0.7%             |        208 | 7.0%                | 13.888%   | 16.801%       | 14.759%     | 19.003%      |               171.9 |                  148 |                  196 |              36.1 |            1.21 | UNDER-predicted |
|      16 | [20%, 99%)     |  1055 | 0.6%             |        397 | 13.4%               | 37.669%   | 37.630%       | 34.698%     | 40.632%      |               397.4 |                  368 |                  427 |              -0.4 |            1    |                 |

### 4.3 Old 11-grade grid, CAL-01, 2010 (reproduces `reports/calibration/grade_table_2010.csv`)

|   grade | pd_band       |     n | share_of_firms   |   defaults | share_of_defaults   | mean_pd   | observed_dr   | dr_ci_low   | dr_ci_high   |   expected_defaults |   accept_lo_defaults |   accept_hi_defaults |   excess_defaults |   obs_over_pred | flag            |
|--------:|:--------------|------:|:-----------------|-----------:|:--------------------|:----------|:--------------|:------------|:-------------|--------------------:|---------------------:|---------------------:|------------------:|----------------:|:----------------|
|       1 | [0.03%, 0.1%) |  6158 | 3.5%             |          4 | 0.1%                | 0.082%    | 0.065%        | 0.018%      | 0.166%       |                 5.1 |                    1 |                   10 |              -1.1 |            0.79 |                 |
|       2 | [0.1%, 0.25%) | 34194 | 19.6%            |         51 | 1.7%                | 0.172%    | 0.149%        | 0.111%      | 0.196%       |                58.8 |                   44 |                   74 |              -7.8 |            0.87 |                 |
|       3 | [0.25%, 0.5%) | 33805 | 19.4%            |        119 | 4.0%                | 0.362%    | 0.352%        | 0.292%      | 0.421%       |               122.3 |                  101 |                  144 |              -3.3 |            0.97 |                 |
|       4 | [0.5%, 0.75%) | 21311 | 12.2%            |        160 | 5.4%                | 0.619%    | 0.751%        | 0.639%      | 0.876%       |               132   |                  110 |                  154 |              28   |            1.21 | UNDER-predicted |
|       5 | [0.75%, 1%)   | 17390 | 10.0%            |        194 | 6.5%                | 0.871%    | 1.116%        | 0.965%      | 1.283%       |               151.4 |                  127 |                  175 |              42.6 |            1.28 | UNDER-predicted |
|       6 | [1%, 1.5%)    | 22714 | 13.0%            |        365 | 12.3%               | 1.223%    | 1.607%        | 1.447%      | 1.779%       |               277.7 |                  245 |                  310 |              87.3 |            1.31 | UNDER-predicted |
|       7 | [1.5%, 2.5%)  | 20369 | 11.7%            |        502 | 16.9%               | 1.914%    | 2.465%        | 2.256%      | 2.687%       |               390   |                  352 |                  428 |             112   |            1.29 | UNDER-predicted |
|       8 | [2.5%, 5%)    | 11961 | 6.9%             |        571 | 19.2%               | 3.376%    | 4.774%        | 4.399%      | 5.171%       |               403.8 |                  365 |                  442 |             167.2 |            1.41 | UNDER-predicted |
|       9 | [5%, 10%)     |  3764 | 2.2%             |        333 | 11.2%               | 6.806%    | 8.847%        | 7.959%      | 9.800%       |               256.2 |                  226 |                  286 |              76.8 |            1.3  | UNDER-predicted |
|      10 | [10%, 20%)    |  1552 | 0.9%             |        218 | 7.3%                | 13.870%   | 14.046%       | 12.355%     | 15.875%      |               215.3 |                  189 |                  242 |               2.7 |            1.01 |                 |
|      11 | [20%, 99%)    |  1294 | 0.7%             |        451 | 15.2%               | 38.555%   | 34.853%       | 32.255%     | 37.520%      |               498.9 |                  467 |                  531 |             -47.9 |            0.9  | OVER-predicted  |

Summary:

| table                                                                  |   n_grades | max_share   | observed_defaults   | expected_defaults   | obs_over_pred_total   | grades_under                |   grades_over | excess_defaults_in_under_grades   | share_firms_in_under_grades   |   observed_dr_monotone |
|:-----------------------------------------------------------------------|-----------:|:------------|:--------------------|:--------------------|:----------------------|:----------------------------|--------------:|:----------------------------------|:------------------------------|-----------------------:|
| old_11_grades_2010_CAL-01 (reproduces grade_table_2010.csv)            |         11 | 19.6%       | 2968                | 2511.4              | 1.182                 | 4,5,6,7,8,9                 |            11 | 513.9                             | 55.9%                         |                      1 |
| new_grid_training_years_2007-2009_fold3_insample (edge-setting sample) |         16 | 9.9%        |                     |                     |                       | nan                         |           nan |                                   |                               |                    nan |
| new_grid_OOT_2008-2009_CAL-01 (share check)                            |         16 | 10.3%       |                     |                     |                       | nan                         |           nan |                                   |                               |                    nan |
| new_grid_2010_CAL-01                                                   |         16 | 10.0%       | 2968                | 2511.4              | 1.182                 | 6,8,9,10,11,12,13,14        |            16 | 501.6                             | 50.0%                         |                      1 |
| new_grid_2010_GB-01_own_level                                          |         16 | 11.9%       | 2968                | 2072.5              | 1.432                 | 5,6,7,8,9,10,11,12,13,14,15 |           nan | 881.6                             | 60.1%                         |                      1 |

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

Figure: `fig/r1b_grade_2010.png`.

## 5. Overfitting and validation evidence

### 5.1 In-sample (training rows) vs out-of-time AUC per fold

| model   |   fold | train_years        |   test_year |   n_train_scored |   auc_in_sample_all_train | auc_in_sample_last_train_year   |   auc_oot |   gap_in_minus_oot |   ar_gap_points |   cal_slope_oot |
|:--------|-------:|:-------------------|------------:|-----------------:|--------------------------:|:--------------------------------|----------:|-------------------:|----------------:|----------------:|
| GB-01   |      1 | [2007]             |        2008 |           144214 |                    0.8476 | 0.8476                          |    0.8441 |             0.0035 |             0.7 |           0.969 |
| RC-02   |      1 | [2007]             |        2008 |           144214 |                    0.809  |                                 |    0.8328 |            -0.0238 |            -4.8 |           1.099 |
| GAM-01  |      1 | [2007]             |        2008 |           144214 |                    0.8097 | 0.8097                          |    0.8348 |            -0.0251 |            -5   |           0.974 |
| GB-01   |      2 | [2007, 2008]       |        2009 |           305475 |                    0.859  | 0.8704                          |    0.8359 |             0.0231 |             4.6 |           1.005 |
| RC-02   |      2 | [2007, 2008]       |        2009 |           305475 |                    0.8248 |                                 |    0.8174 |             0.0073 |             1.5 |           1.011 |
| GAM-01  |      2 | [2007, 2008]       |        2009 |           305475 |                    0.827  | 0.8411                          |    0.8233 |             0.0037 |             0.7 |           0.969 |
| GB-01   |      3 | [2007, 2008, 2009] |        2010 |           475861 |                    0.8498 | 0.8505                          |    0.8307 |             0.0192 |             3.8 |           0.957 |
| RC-02   |      3 | [2007, 2008, 2009] |        2010 |           475861 |                    0.8223 |                                 |    0.8115 |             0.0108 |             2.2 |           0.974 |
| GAM-01  |      3 | [2007, 2008, 2009] |        2010 |           475861 |                    0.8283 | 0.8294                          |    0.8177 |             0.0105 |             2.1 |           0.952 |

### 5.2 Same year, in-sample vs out-of-time (removes the year-difficulty confound)

Year t scored by a later fold's model that was trained on t (in-sample) vs by the fold for which t was the
test year (OOT). Same rows.

| model   |   year |   in_sample_fold |   n_in |   n_oot |   auc_in_sample |   auc_oot |   gap_auc |   gap_ar_points |
|:--------|-------:|-----------------:|-------:|--------:|----------------:|----------:|----------:|----------------:|
| GB-01   |   2008 |                2 | 161261 |  161261 |          0.8704 |    0.8441 |    0.0262 |             5.2 |
| RC-02   |   2008 |                2 | 161261 |  161261 |          0.8387 |    0.8328 |    0.006  |             1.2 |
| GB-01   |   2008 |                3 | 161261 |  161261 |          0.8614 |    0.8441 |    0.0173 |             3.5 |
| RC-02   |   2008 |                3 | 161261 |  161261 |          0.8368 |    0.8328 |    0.0041 |             0.8 |
| GB-01   |   2009 |                3 | 170386 |  170386 |          0.8505 |    0.8359 |    0.0146 |             2.9 |
| RC-02   |   2009 |                3 | 170386 |  170386 |          0.8223 |    0.8174 |    0.0049 |             1   |

### 5.3 Stability summary

| model      |   auc_pooled |   auc_ci_width_pooled |   auc_2008 |   auc_2009 |   auc_2010 |   auc_range_years |   cal_slope_pooled |   cal_slope_min_year |   cal_slope_max_year |   cal_slope_new_firms |   obs_over_pred_2008 |   obs_over_pred_2010 |   seen_minus_new_auc |
|:-----------|-------------:|----------------------:|-----------:|-----------:|-----------:|------------------:|-------------------:|---------------------:|---------------------:|----------------------:|---------------------:|---------------------:|---------------------:|
| GB-01      |       0.8375 |                0.0087 |     0.8441 |     0.8359 |     0.8307 |            0.0135 |              0.976 |                0.957 |                1.005 |                 0.883 |                1.212 |                1.432 |               0.0573 |
| CAL-01     |       0.8374 |                0.0087 |     0.8441 |     0.8359 |     0.8307 |            0.0135 |              0.976 |                0.957 |                1.006 |                 0.892 |                1.016 |                1.182 |               0.0578 |
| SC-03      |       0.8166 |                0.0098 |     0.8278 |     0.8117 |     0.8115 |            0.0163 |              0.988 |                0.97  |                1.03  |                 0.842 |                1.237 |                1.494 |               0.0582 |
| RC-02      |       0.8194 |                0.0102 |     0.8328 |     0.8174 |     0.8115 |            0.0212 |              1.022 |                0.974 |                1.099 |                 0.888 |                1.221 |                1.446 |               0.0608 |
| GAM-01     |       0.8254 |                0.0098 |     0.8348 |     0.8233 |     0.8177 |            0.0171 |              0.964 |                0.952 |                0.974 |                 0.833 |                1.231 |                1.445 |               0.0574 |
| GAM-01+CAL |       0.8255 |                0.0097 |     0.8348 |     0.8233 |     0.8177 |            0.0171 |              0.963 |                0.951 |                0.974 |                 0.832 |                1.029 |                1.189 |               0.0578 |
| GAM-02     |       0.8262 |                0.0098 |     0.8353 |     0.8239 |     0.8192 |            0.0161 |              0.97  |                0.95  |                0.998 |                 0.843 |                1.236 |                1.447 |               0.058  |

### 5.4 Selection ledger (how often 2008-2010 have been looked at)

|   oot_prediction_files_covering_2008_2010 |   calibration_configs_scored_on_2008_2010 |
|------------------------------------------:|------------------------------------------:|
|                                        30 |                                        62 |

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

## Appendix A1: fast bootstrap vs `metrics.paired_bootstrap` (GAM-01 vs GB-01, 15,000-firm subsample, 40 draws, seed 0)

| metric     |   shared_lo |     fast_lo |   shared_hi |     fast_hi |   max_abs_diff |
|:-----------|------------:|------------:|------------:|------------:|---------------:|
| d_auc      | -0.0175286  | -0.0175286  | -0.00590859 | -0.00590859 |    1.04083e-16 |
| d_log_loss |  0.00049005 |  0.00049005 |  0.00132987 |  0.00132987 |    2.67798e-17 |

## Decision entries for the orchestrator to merge (DECISION_LOG template)

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
