# Harness test report, 2026-11-21 bundle (Team Turquoise)

Bundle: `harness/2026-11-21_turquoise_submission/`, zip `harness/2026-11-21_Turquoise_harness.zip` (files at the zip root,
same layout as the 2026-11-16 zip). Built by `experiments/final/2026-11-21_final_fit/2026-11-21_03_build_bundle.py`.

## Model in the bundle
- LightGBM, **unconstrained** (KN decision 2026-11-21, D-159), D-158 winner setting (num_leaves 31, min_child_samples 3000,
  feature_fraction 0.5, lambda_l2 10, learning_rate 0.05), fixed 210 trees, 75 VIF-selected features.
- Preprocessing (schema 1.3.0: no Method G, roe recomputed, 3 smallest industry groups use the overall median) fitted once on all
  835,016 labelled FY2007-2011 rows, frozen in `artifacts/preprocessing.json`.
- Level: prior shift (D-072) to tau = 0.014640906 (5-month label, D-150/D-151), floor 0.03%, cap 99% (D-075);
  anchor a = 0.0011066, delta = 0.0766119 (= the calibration study's production delta). Fallback PD = tau.
- In-sample sanity (`final_insample_summary.json`): mean raw PD 1.3563% vs training rate 1.3576%; mean final PD 1.4510%; PD range
  0.03%-88.1%. Out-of-sample evidence for this recipe is the Phase-1 walk-forward (D-158), not these numbers.

## Environment (clean venv = Chloe's requirements.txt, byte-identical to the 2026-11-16 bundle)
Python 3.13.7, numpy 2.1.3, pandas 3.0.1, scipy 1.16.3, lightgbm 4.6.0 (model trained under lightgbm 4.7.0 / pandas 2.3.3; loads and
scores identically). Python 3.14 cannot install these pins (no numpy 2.1.3 wheel); the README keeps "tested with Python 3.13".

## Results
1. Code parity with Chloe's development data (`parity_round03.csv`): the vendored teammate code (project-update @ 7949207) refit on
   FY2007-09 and applied to FY2010 reproduces all 75 features of her 11.20 round-03 files on every validation row; 74/75 on training
   rows (ateco_sector_te on training rows is out-of-fold and depends on fold assignment; its frozen mapping matches).
2. Full file through the CLI: 1,023,552 rows in, 1,023,552 PDs out, 0 fallbacks, 46.5 s wall time.
3. Harness vs training pipeline (`parity_harness_vs_training.json`): max |difference| 2.8e-16 over all 1,023,552 rows.
4. Stress suite (`reports/harness/2026-11-21_stress_test.py`, 25 checks): 25 PASS, 0 FAIL.

```
PASS  shuffled: rows in=5000 out=5000 range=[0.0002999999999999, 0.8256963138418361]
      same PD per row after shuffle: True
PASS  cols_reversed: rows in=5000 out=5000 range=[0.0002999999999999, 0.8256963138418361]
PASS  single_0: rows in=1 out=1 range=[0.0035346384756485, 0.0035346384756485]
PASS  row-independence: 0 of 300 rows change PD when other rows are removed
PASS  messy_values: rows in=500 out=500 range=[0.0003673140178002, 0.6249098319568033]
PASS  all_nan: rows in=100 out=100 range=[0.0077005042909053, 0.0077005042909053]
PASS  drop_def_date: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  drop_stmt_date: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  drop_Unnamed: 0: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  drop_roa: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  drop_roe: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  drop_days_rec: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  drop_eqty_corp_family_tot: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  drop_id: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  drop_fs_year: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  drop_cf_operations: rows in=200 out=200 range=[0.0005356693981722, 0.3781567953812638]
PASS  drop_HQ_city: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  extra_cols: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  header_only: rows in=0 out=0 range=[-, -]
PASS  with_index_column: exit 0 Wrote 200 PDs to /private/tmp/claude-501/-Users-kietnguyen-Downloads-pd-agent-kit/95463a29-60e3-428c-b43c-50deac56a531/scratchpad/harness_test/out/with_index.csv.
PASS  unseen_ids_future_year: rows in=200 out=200 range=[0.0004218614754407, 0.5403594548839135]
PASS  duplicate_rows: rows in=300 out=300 range=[0.0004218614754407, 0.0496697199907952]
PASS  thousand_sep: rows in=50 out=50 range=[0.0009637343224351, 0.0589066822665963]
PASS  python predict(df): shape (5000, 1), equals CLI output: True
PASS  same-firm multi-year rows: max change 0.00e+00 when scored alone
```

## Additional checks (same day, before sending)
5. Python 3.11.14 venv with the same pins: installs and runs; output identical to Python 3.13.7.
6. Zip unzipped into a folder whose path contains spaces, run from a different working directory with the single-dash
   aliases (`-input_csv`, `-output_csv`): runs, no warnings.
7. Chloe's first-round TA test input `data/raw/harness_test_data_2011.csv` (teammate/project-update @ 4e0ba3a, 184,973 rows):
   new bundle 184,973 PDs, 0 fallbacks, all in (0,1), mean 1.525%, range 0.03%-86.0%. The 2026-11-16 zip on the same file in
   the same venv: 184,973 PDs, mean 1.635%; Spearman(new, old) = 0.962.
Not tested: Windows or Linux (only macOS available). Runtime code is plain Python (pathlib, pandas) like the 11-16 bundle;
the LightGBM text model is platform-independent.

## Variant bundles (D-161), same tests in the same clean venv (Python 3.13.7, pinned requirements)
| variant | constraints | trees | full file (1,023,552 rows) | harness vs training max diff | stress tests |
|---|---|---|---|---|---|
| mono20 | 20 | 240 | 1,023,552 PDs, 0 fallbacks, 48.2 s | 2.2e-16 | 25 PASS / 0 FAIL |
| lit13 | 13 | 200 | 1,023,552 PDs, 0 fallbacks, 28.1 s | 2.2e-16 | 25 PASS / 0 FAIL |
| mono19 | 19 | 310 | 1,023,552 PDs, 0 fallbacks, 37.1 s | 3.3e-16 | 25 PASS / 0 FAIL |
Zips: `harness/2026-11-21_variants/2026-11-21_Turquoise_harness_<variant>.zip` (files at the zip root, same layout).
