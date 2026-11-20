# Frozen walk-forward splits (data-auditor, 2026-10-30)

Written once by `src/data/build_panel.py`; files are read-only (chmod 444). Nobody edits them. A
change requires a logged decision (DECISION_LOG) and a new version file, never an overwrite.

## Files
- `walkforward.json`: the fold definitions, label definition and counts.
- `walkforward_test_rows.parquet`: one row per (fold, test row): `row_id, fold, test_year, firm_seen_in_train`.
- Panel: `data/interim/panel.parquet` (1,023,552 rows, all fs_years, no imputation).

## Label used by the folds (D-002, D-009; assumptions pending Roger/TA)
- `y` = 1 if the firm's `def_date` falls in [stmt_date + 6 months, stmt_date + 18 months); 0 otherwise.
- `label_status`: `ok` (usable), `already_defaulted` (default before the window starts: not a
  performing applicant), `censored` (window ends after 2013-05-31, the last observed default date),
  `after_default` (statement dated on/after default; 0 rows in training).
- D-009 primary treatment: 2013 default dates used as given; data_end = 2013-05-31.

## Design (D-005)
Expanding window, out-of-time. For test year t+1, train on every row with fs_year <= t and
`label_status == "ok"`; test on rows with fs_year == t+1 and `label_status == "ok"`.
Only test years whose label window is fully observed are used: 2008, 2009, 2010.
fs_year 2011 (window ends 2013-06-30, one month past data_end) and 2012 are excluded as test years
and are NOT usable training rows under the primary label.

| fold | train fs_years | test fs_year | n_train | train defaults | n_test | test defaults | test rate | seen-firm rows | new-firm rows |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2007 | 2008 | 144,214 | 1,379 | 161,261 | 2,064 | 1.28% | 131,426 | 29,835 |
| 2 | 2007-2008 | 2009 | 305,475 | 3,443 | 170,386 | 2,752 | 1.62% | 153,214 | 17,172 |
| 3 | 2007-2009 | 2010 | 475,861 | 6,195 | 174,512 | 2,968 | 1.70% | 157,775 | 16,737 |

(Copied from `walkforward.json` / `reports/data/walkforward_folds.csv`; those files are authoritative.)

`sensitivity_folds_not_primary` in the JSON lists an extra test year 2011 that is valid ONLY for the
start-0 (`y_s0`) and start-3 (`y_s3`) labels. Use it for the label-window sensitivity study, not
for model selection.

## Seen vs new firms
`firm_seen_in_train` = the firm `id` has at least one panel row (any label status) with fs_year in
the fold's training years. The holdout contains both kinds of firm (Appendix B), so every
challenger reports metrics for both groups. A firm's other rows are NEVER used as features unless
Roger/TA approve (EMAIL_DRAFT question 4).

## Rules for users
1. Fit everything (imputation, scaling, bins, transforms, calibration, selection, tuning) on the
   training rows of the fold only. Inner tuning splits must be time-ordered inside the training years.
2. Columns `def_date`, `y*`, `status*`, `win_start`, `win_end`, `label_status` are label-only:
   never features. `src_index` is a file index, never a feature.
3. Output contract: `experiments/<agent>/oot_predictions.parquet` with `row_id, fold, pd`.
4. `row_id = id * 10000 + fs_year` (int64), reproducible from the raw file.

---

## v2 (2026-11-08, D-151): s5 label window — use for all new work

v1 above (start-6 label, `walkforward.json`, `walkforward_test_rows.parquet`, `data/interim/panel.parquet`)
is kept unchanged as the record of rounds 1-2. New work uses:

- `2026-11-08_walkforward_v2.json`: folds, lockbox, final_fit, label definition, counts, sha256 of the files below.
- `2026-11-08_walkforward_v2_test_rows.parquet`: `row_id, fold, test_year, firm_seen_in_train, role`
  (`role` = `model_selection` for folds 1-3, `lockbox` for fold 4). **Filter `role == "model_selection"`
  for any tuning or model comparison.**
- Panel: `data/interim/2026-11-08_panel_s5.parquet` (same 1,023,552 rows and 64 v1 columns, plus `y_s5`, `status_s5`,
  `win_start_s5`, `win_end_s5`, `y_s5_d2012`, `status_s5_d2012`). The primary `y` / `label_status` /
  `win_start` / `win_end` columns hold the **s5** values; the v1 label is still in `y_s6` / `status_s6`.
- Built by `src/data/2026-11-08_build_labels_s5.py` (deterministic; refuses to overwrite these files).
  Report: `reports/data/2026-11-08_labels_s5.md`; fold table `reports/data/2026-11-08_walkforward_v2_folds.csv`.

Label s5: y = 1 if def_date in [1 Jun t+1, 1 Jun t+2) for fs_year t; `already_defaulted` if def_date < 1 Jun t+1
(drop); `censored` if the window ends after data_end 2013-05-31 (D-009 primary), whether or not a default is seen.
Built from `stmt_date`/`def_date` directly, not with `src/common/labels.make_labels` (D-011/D-068).
D-009 sensitivity (2013 dates blanked, data_end 2012-12-31): `y_s5_d2012` / `status_s5_d2012`.

| fold | role | train fs_years | test fs_year | n_train | train defaults | n_test | test defaults | test rate | seen-firm rows | new-firm rows |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | model selection | 2007 | 2008 | 144,251 | 1,260 | 161,383 | 1,945 | 1.21% | 131,538 | 29,845 |
| 2 | model selection | 2007-2008 | 2009 | 305,634 | 3,205 | 170,549 | 2,687 | 1.58% | 153,363 | 17,186 |
| 3 | model selection | 2007-2009 | 2010 | 476,183 | 5,892 | 174,674 | 2,881 | 1.65% | 157,924 | 16,750 |
| 4 | **one-time lockbox per D-087/D-119 — not for model selection** | 2007-2010 | 2011 | 650,857 | 8,773 | 184,159 | 2,563 | 1.39% | 167,667 | 16,492 |
| final_fit | refit of the chosen model (no test rows) | 2007-2011 | — | 835,016 | 11,336 | — | — | — | — | — |

(Copied from the v2 JSON / folds CSV; those files are authoritative.) fs2012 is never usable (all rows censored
or already_defaulted). Under the D-009 sensitivity, fs2011 has 0 usable rows (the lockbox and the 2011 part of
final_fit disappear); folds 1-3 are identical under both treatments.

---

## v4 (2026-11-20, D-156): temporal_year walk-forward + FY2011 lockbox — use for the final model

Supersedes the v3 `company_random` proposal (D-155; v3 files live in the agent-kit repo,
`data/splits/2026-11-18_walkforward_v3_company_random.json`). Label s5 (D-151), status `ok` rows only.

| Round | Train | Validation (early stopping only) | Test | Role |
|---|---|---|---|---|
| 1 | FY2007 | FY2008 | FY2009 | model selection |
| 2 | FY2007-08 | FY2009 | FY2010 | model selection |
| 3 | FY2007-09 | FY2010 | FY2011 | **LOCKBOX** (file in `LOCKBOX_do_not_open/`; score once, after model choice) |
| final | FY2007-11, all 835,016 labelled rows | none | - | harness model |

- Split definition: `2026-11-20_walkforward_v4_temporal_year.json` (read-only): counts, default rates (lockbox labels
  withheld), seen/new firm row counts per validation and test set, the test-year firm ids seen in training, code and
  schema sha256, and the sha256 of every feature file and preprocessor artifact.
- Feature files: `data/processed/2026-11-20_v4_temporal_year/round_0{1,2,3}_{train,valid,test}.parquet`
  (round 3 test in `LOCKBOX_do_not_open/`), `final_fit.parquet`, `artifacts/*_preprocessor.joblib`.
  Columns: `id, fs_year, label`, then 81 features (schema 1.3.0, same names and order in every file).
- Built by `src/data/2026-11-20_build_v4_temporal_year.py` with the team's `preprocess.py` in `firm_year` mode:
  `fit_preprocessor` sees only the round's training years; validation, test and lockbox use
  `transform_with_preprocessor` with the frozen artifact. Checks: `reports/data/2026-11-20_v4_checks.md`.
- Rules: the validation year is for early stopping only. Do not fit Platt/isotonic on it (D-150); the level comes from
  the prior shift (D-072). Never tune on round 3. Rounds 1-2 test years (FY2009, FY2010) are the only selection years.
