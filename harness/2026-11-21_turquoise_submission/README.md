# Team Turquoise: PD harness (test bundle, 2026-11-21)

Install Python 3.11+ and the pinned dependencies, then run the same command as the 2026-11-16 test bundle:

```bash
python3 -m pip install -r requirements.txt
python3 harness.py --input_csv /path/to/input.csv --output_csv /path/to/predictions.csv
```

Or, as one Python function call:

```python
from harness import predict
pds = predict(df)        # df: pandas DataFrame of raw rows (or a CSV path); returns an n x 1 numpy array
```

Both double-dash arguments and the aliases `-input_csv` / `-output_csv` are accepted. No training data,
network access or project installation is needed. Bundled resources are found relative to the code, so the
harness runs from any working directory. LightGBM needs an OpenMP runtime (`libomp` on macOS).

## What changed since the 2026-11-16 test bundle

| | 2026-11-16 bundle | this bundle |
|---|---|---|
| Model | round-03 model, trained FY2007-2009, 80 features, 424 trees | **final model, trained on all labelled FY2007-2011 rows (835,016)**, 75 features, 210 trees |
| Missing-value rules | Method G switched off at inference only (the model had been trained with it) | **Method G removed in training and inference**; `roe` recomputed as profit / equity for every row; the three smallest industry groups use the overall median (feature schema 1.3.0) |
| Imputation / encoding parameters | fitted on FY2007-2009 | **fitted once on all FY2007-2011 training rows and frozen** in `artifacts/preprocessing.json` |
| Level calibration | Platt map fitted on FY2010 | **prior shift** to a stated central tendency tau = 1.4641%, then PD floor 0.03% and cap 99% |
| Fallback PD | training default rate 1.2373% | tau = 1.4641% |

Interface, output format and robustness behaviour are unchanged.

## Files

- `harness.py`: command-line entry point and the one-call Python function `predict`.
- `inference.py`: loads the frozen parameters, applies preprocessing, predicts, calibrates, and writes PDs.
- `anomaly_rules.py`, `missing_rules.py`, `feature_rules.py`: the team's anomaly, missing-value and feature
  rules (schema 1.3.0), as normal Python modules.
- `bundle_paths.py`: resolves lookup files inside this folder.
- `config/feature_schema.json`: frozen feature contract.
- `data/interim/ateco_sector_english.csv`: ATECO code to A-U industry lookup.
- `artifacts/lightgbm.txt`: the frozen LightGBM trees.
- `artifacts/preprocessing.json`: frozen training medians, industry target encoding, category lists, the
  ordered 75 features, calibration constants, fallback PD and model settings.
- `artifacts/manifest.json`: provenance and checksums.
- `requirements.txt`: pinned runtime libraries.

No model, median, encoding or calibration constant is fitted on the supplied CSV.

## Input and pipeline

Input is an unlabeled raw firm-year CSV with the columns of the training file (the schema's `raw_required`
and `impute_columns` lists); individual values may be missing, and a whole missing column is treated as
all-missing (warning on stderr). `roa` may be absent. Labels, default dates, statement dates, index columns and
extra columns are ignored. Blanks, `NA` and other non-numeric spellings in numeric columns become missing;
infinities become missing. Unseen industries fall back to the overall frozen median and encoding; unseen
legal forms and provinces get all-zero dummies.

Pipeline per row: numeric checks, negative balance-sheet flag, the team's formula and industry-median
imputations (frozen medians), financial ratios (a zero denominator becomes one), frozen industry target
encoding, legal-form and region dummies, the 75 model features, the frozen LightGBM, then the frozen level
calibration:

    PD = clip( expit( a + logit(p_raw) + delta ), 0.0003, 0.99 )
    a = 0.0011066 (in-sample anchor), delta = logit(tau) - logit(training default rate) = 0.0766119

Every row is processed using only that row and the frozen parameter files. Nothing reads another record of the
same firm, and no statistic is computed from the testing input, so a row's PD does not change if other rows
are added, removed or reordered.

## Output and fallback PDs

Exactly one PD per input row, in the original order, one column, no header or index. Diagnostics go to stderr.
A row whose final features are still missing or infinite after imputation gets the frozen fallback PD
tau = **0.014640906163076605** instead of a model PD. If any unexpected error occurs inside the pipeline, the
harness writes the fallback PD for every row and reports the error on stderr, so an output with one PD per
input row is always produced. A CSV with zero data rows gives an empty output.

## Computational complexity

For n rows, p = 75 features and T = 210 trees with at most 31 leaves each: preprocessing is O(n p) with
constant-time lookups into frozen tables, plus one O(n log n) sort by firm and year inherited from the team's
feature code (it reorders rows only; no value of one row is used for another, and the original order is
restored). Scoring is O(n T L) for tree depth L <= 30. Memory is O(n p). Measured: the full 1,023,552-row
training file takes about 47 seconds end to end on a laptop (Apple M1, Python 3.13, the pinned versions),
i.e. roughly linear in n.

Tested 2026-11-21 in a fresh virtual environment with exactly `requirements.txt` (Python 3.13.7, numpy 2.1.3,
pandas 3.0.1, scipy 1.16.3, lightgbm 4.6.0): one PD per row for the full training file, PDs identical to the
training pipeline (max difference 3e-16), and 25 stress tests passed (shuffled/reversed input, row
independence, messy values, missing and extra columns, header-only file, unseen firms and categories,
duplicate rows).
