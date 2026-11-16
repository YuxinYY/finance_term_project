# Turquoise inference submission

Install Python 3.11+ (tested with 3.13.12), then install the pinned dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 harness.py --input_csv /path/to/input.csv --output_csv /path/to/predictions.csv
```

Both double-dash arguments and the aliases `-input_csv` / `-output_csv` are
accepted. No further arguments, training data, network calls, or project
installation are needed. Paths to bundled resources are relative to the code,
so invocation works from a different working directory. LightGBM requires a
working OpenMP runtime; on macOS the usual system dependency is `libomp`.

## Files

- `harness.py`: two-argument command-line entry point.
- `inference.py`: loads frozen parameters, applies preprocessing, predicts, and
  writes calibrated probabilities.
- `anomaly_rules.py`: exported anomaly-check code from `01_flag_anomalies.ipynb`.
- `missing_rules.py`: exported imputation code from `02_missing_values.ipynb`.
- `feature_rules.py`: frozen feature construction and province-to-region rules
  extracted from the project's `preprocess.py`.
- `bundle_paths.py`: resolves lookup files within this folder.
- `config/feature_schema.json`: frozen feature contract.
- `data/interim/ateco_sector_english.csv`: detailed-code to A–U industry lookup.
- `artifacts/lightgbm.txt`: frozen LightGBM trees from
  `temporal_firm_year_v2/round_03/lightgbm/model.joblib`.
- `artifacts/preprocessing.json`: matching round-03 training medians, target
  encoding, category lists, ordered features, Platt coefficients, and fallback PD.
- `artifacts/manifest.json`: source provenance, versions, and bundle checksums.
- `requirements.txt`: pinned runtime libraries.

There is no training step in this submission. The model is the round-03 model,
trained on FY2007–2009 and calibrated using FY2010. Its 80 inputs and 424 trees
are preserved. No model is fitted on the supplied testing CSV.

The saved round-03 test feature CSV is not reproducible from the currently saved
round-03 preprocessor: its target encodings vary within the same coarse industry
(and even within the same detailed sector). A frozen industry mapping would give
one value per industry. This harness consistently uses the saved training
preprocessor. Its frozen trees and calibration reproduce the saved predictions
when supplied those original feature values, but the reported v2 test metrics
are not validation of this corrected raw-input inference path. Audit and
regenerate validation/test features before using those metrics as evidence for
the packaged pipeline. No artifact or evaluation CSV was changed to build this
submission.

## Input and pipeline

Input is an unlabeled raw firm-year CSV. It needs the columns in the bundled
schema's `raw_required` and `impute_columns` lists, including `roe`; individual
values may be missing. `roa` may be absent and will be derived. Labels, default
dates, statement dates, index exports, and extra columns are ignored. Input row
order is preserved in the output. Repeated company IDs across years are allowed;
no unique-ID or unique-firm-year assumption is required by this harness.

The pipeline runs numeric/missing checks, flags reported negative balance-sheet
values, applies the notebook's formula and A–U group median imputations, builds
financial features with the existing zero-denominator rule (zero becomes one),
applies frozen industry target encoding and categorical dummies, checks all 80
features, and applies the frozen model followed by its frozen Platt calibrator
to rows with valid features. Rows with remaining NaN or infinity receive the
frozen training default rate instead. Numeric input infinities are
treated as missing before imputation. Negative reported finite values are kept.
The notebook's previous-observed-zero rule uses earlier records in the supplied
input; no external company history is loaded. Medians and target encodings are
never recalculated from the testing input.

## Output and fallback PDs

The output has exactly one PD for every input row, in the original order, in a
single column with no header or index. Diagnostics are printed to stderr.

When a row still contains a missing or infinite final model feature after
imputation, the model is not called for that row. Its output is the frozen
round-03 training default rate: **0.012373394262289918** (about **1.23734%**).
This value is stored in `artifacts/preprocessing.json` as `fallback_pd` and is
independent of the testing data. It is used directly as a probability, without
Platt calibration. Rows with valid features retain their normal calibrated
model predictions. The fallback decision does not require LightGBM's native
missing-value handling.

Even if every row needs a fallback, every row receives a finite PD. A CSV with
zero data rows produces an empty output. `inference.transform` returns only the
rows suitable for model scoring, with original positional indices;
`inference.predict_frame` inserts their model PDs into a full-length output
initialized to the fallback PD. Missing required columns or malformed inputs
still raise an explanatory error before predictions are written; the fallback
handles invalid feature values, not a broken input schema or missing artifacts.
