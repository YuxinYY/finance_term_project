# Harness test report: TA test-run submission (2026-11-16)

Bundle tested: Chloe's `~/Downloads/turquoise.zip` (round-03 LightGBM, 80 features, 424 trees,
trained FY2007-09, Platt-calibrated on FY2010). Fixed bundle: `harness/turquoise.zip`
(sha256 `0120b96454c318f3e7ea1866410301f0c828cee74390f05bb52294b18194c04e`), source in
`harness/2026-11-16_turquoise_submission/`. Stress script: `reports/harness/2026-11-16_stress_test.py`.

## Environment (mirrors the TA instructions)
Fresh unzip into an empty folder; fresh venv; `pip install -r requirements.txt` only; command run from a
different working directory: `python3 harness.py --input_csv X --output_csv Y`.
Python 3.13.7 and 3.11.14 (arm64 macOS), numpy 2.1.3, pandas 3.0.1, scipy 1.16.3, lightgbm 4.6.0.
Outputs byte-identical across the two Python versions.

## Defects found in the original bundle and fixed
| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | Method G imputation ("previous year was 0 -> 0") reads the same firm's other rows in the test file. The assignment allows "no test data besides the current record, for any purpose" (Term Project Instructions, Appendix A, p. 5). | 84 of 1,518 firm-years (firms where G fires) changed PD when other rows were removed; max abs change 0.0124, max rel change 85.5%. G fires on 291 of 1,023,552 raw rows. | `fill_missing(..., use_firm_history=False)` at inference; affected values fall back to Method B (frozen sector medians). After fix: 0 of 1,518 rows change. |
| 2 | Missing input column -> `ValueError`, no output (assignment: run "without throwing errors or halting"). | drop_roe / drop_id / drop_fs_year / drop_cf_operations / drop_HQ_city all exited 1. | Missing columns treated as all-NaN and imputed; warning on stderr. |
| 3 | Checksum mismatch on any bundled file (incl. README) -> hard error. | Code review (`inference.load_parameters`). | Warning only. |
| 4 | No last-resort guarantee of output. | Code review. | Any pipeline exception -> frozen fallback PD (0.012373) for all rows, error on stderr. |
README updated (Method G, missing columns, computational complexity section); manifest checksums refreshed.

## Stress tests on the fixed bundle: all PASS
Shuffled rows (same PD per row), reversed columns, single row, row-independence (random 300 and
targeted 1,518), messy values (blank, "NA", "n/a", "-", "inf", unseen city/sector/legal form, 0 and
negative assets, missing id/year, fully blank rows, 1e30), all-NaN rows, 11 dropped-column cases, extra
columns, header-only file (empty output), pandas index column, unseen ids + future year, duplicate rows,
thousand separators. Output always: n rows, no header, finite, strictly in (0,1).

## Full-file run and performance of the packaged raw-input pipeline
Full `train-2025.csv`, 1,023,552 rows: 22.9 s, single thread; all PDs finite in (0,1); 0 fallback rows.
Labels: v2 s5 (D-151), `data/interim/2026-11-08_panel_s5.parquet`.

| FY | n | default rate | mean PD | PD/DR | AUC | AR | role for this model |
|---|---|---|---|---|---|---|---|
| 2007 | 144,251 | 0.0087 | 0.0150 | 1.72 | 0.867 | 0.734 | training |
| 2008 | 161,383 | 0.0121 | 0.0168 | 1.40 | 0.889 | 0.779 | training |
| 2009 | 170,549 | 0.0158 | 0.0169 | 1.08 | 0.885 | 0.770 | training |
| 2010 | 174,674 | 0.0165 | 0.0150 | 0.91 | 0.844 | 0.688 | Platt calibration only |
| 2011 | 184,159 | 0.0139 | 0.0153 | 1.10 | 0.854 | 0.708 | out-of-time (lockbox year, D-119) |

Note: FY2011 is the D-119 lockbox. It was used here once to verify a model already chosen, not for
selection (D-152).

## Feature reconciliation vs Chloe's 11.15 feature files (round 1)
Harness-built features vs `features_test.csv` (FY2009) and `features_train.csv` (FY2007), joined on
(id, fs_year): on firm-years with complete raw inputs (129,160 and 120,569 rows) all 79 non-encoding features
match exactly (0 mismatches). Differences elsewhere are expected: imputation medians are round-specific
(round 1 vs round 3), and `ateco_sector_te` is round-specific (out-of-fold on training rows).
Chloe's pipeline drops 601 FY2009 and 277 FY2007 raw rows; the harness scores every row.
