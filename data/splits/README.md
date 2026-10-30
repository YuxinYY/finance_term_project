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
