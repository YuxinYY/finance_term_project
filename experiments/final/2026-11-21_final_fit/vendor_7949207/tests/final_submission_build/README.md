# Final submission source

Runtime uses ordinary Python modules; it never executes notebook cells.

| File | Role |
| --- | --- |
| `harness.py` | Two-argument CSV entry point. |
| `inference.py` | Raw checks, preprocessing, frozen encoding, predictions and fallback PDs. |
| `anomaly_rules.py` | Current exported rules from `01_flag_anomalies.ipynb`. |
| `missing_rules.py` | Current exported rules from `02_missing_values.ipynb`. |
| `feature_rules.py` | Current schema-driven financial features from `preprocess.py`. |
| `bundle_paths.py` | Resource paths relative to this folder. |
| `sync_rules.py` | Developer utility to refresh modules and lookup/schema files. |
| `build_submission.py` | Developer utility to export existing compatible fitted artifacts and ZIP runtime files. |
| `verify_rules.py` | Developer checks for notebook parity, ROE, group fallback and record independence. |
| `rules_provenance.json` | Hashes of the source notebooks, training script, schema and lookup. |

`anomaly_rules.py` is the intended spelling (not `anomaly_rulels.py`). Generated
modules should be refreshed with the utility rather than edited independently.
The feature notebooks contain older exploratory formulas; `feature_rules.py`
uses the current training implementation and `config/feature_schema.json` so
feature names, formulas and flags match the actual model training pipeline.

## Current rule changes

The synchronized missing-value rules use A–U industry medians, omitting the three
groups with the fewest unique training firms so they use overall medians. They
recompute ROE from filled profit and equity for every row, treating zero equity
as one; add `roe_negative` for negative profit and negative equity; and remove
the former previous-observed-zero rule. No imputation depends on other holdout
records. Group exclusion and medians are determined during training only.

`src/finance_project/data_processing/preprocess.py` continues loading the latest
notebook cells tagged `export` for `sector_medians` and `fill_missing`. It therefore
already uses these updated methods, and `roe_negative` is retained through the
current feature schema. This submission uses synchronized copies of those same
functions with normal imports.

## Refresh and build

From the repository root:

```bash
python tests/final_submission_build/sync_rules.py
/opt/miniconda3/envs/daily/bin/python tests/final_submission_build/verify_rules.py
PYTHONPATH=src /opt/miniconda3/envs/daily/bin/python \
  tests/final_submission_build/build_submission.py \
  --preprocessor path/to/current_preprocessor.joblib \
  --model path/to/current_model_bundle.joblib
```

The build script does not train a model. It checks the artifact schema and
feature order, then creates `artifacts/` and `turquoise.zip` inside this folder.
The model bundle must use the existing training runner's LightGBM/Platt format.
Only runtime files are included in the ZIP; developer utilities are excluded.

**The saved v2 round-03 artifacts use schema 1.2.1, whereas the current schema is
1.3.0. They must not be combined with these updated rules.** No stale artifacts
are copied here. Refit preprocessing and the model under the new methods before
building a deployable ZIP. The older `tests/harness_submission/` and
`tests/turquoise.zip` remain their original frozen v2 snapshots.

## Inference after building

```bash
python3 -m pip install -r requirements.txt
python3 harness.py --input_csv input.csv --output_csv predictions.csv
```

The input is an unlabeled raw firm-year CSV with the columns required by the
bundled schema. `roa` may be absent; ROE is recomputed. Targets, dates and extra
columns are ignored. Parameters and data lookups are loaded from this folder;
no network or training data is used at runtime. Python 3.11+ is supported;
LightGBM requires a working OpenMP runtime (usually `libomp` on macOS).

Output contains exactly one PD per input row, in its original order, without a
header or index. Rows with invalid final features receive the frozen training
default-rate fallback; other rows receive calibrated model PDs. Diagnostics go
to stderr. Runtime is O(n log n + n T D) for n records, T trees and tree depth D,
including the inherited firm-year sort; memory is O(n F) for F model features.
