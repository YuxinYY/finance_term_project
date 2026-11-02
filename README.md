# Finance ML Term Project

This repository uses a lightweight, reproducible ML workflow. Source data is
never overwritten; each stage writes a clearly named artifact that feeds the
next stage.

## Layout

```text
data/raw/                    immutable source files (including the demo subset)
data/interim/                temporary transformation outputs
data/processed/              model-ready datasets (generated, ignored)
notebooks/                   exploratory work; 01_playground.ipynb is preserved
src/finance_project/
  data_processing/           clean data and write EDA/data-quality metrics
  model_development/         train and persist a baseline model
  model_evaluation/          score held-out data and write model metrics
  reporting/                 inject generated metrics into the Markdown report
models/                      serialized models (generated, ignored)
reports/report.md            committed, human-authored report with generated blocks
reports/metrics/             machine-readable JSON metrics (generated, ignored)
reports/figures/             generated figures (ignored)
docs/                        assignment and domain reference material
```

## Quick start

Create an environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the walk-forward data preparation from the repository root after creating
the 12-month `label` column. Each output row remains a distinct `(id, fs_year)`
prediction sample; the later preprocessing step aggregates its prior financial
statements without looking ahead.

```bash
PYTHONPATH=src python -m finance_project.data_processing.make_dataset \
  --input data/processed/labeled_data.csv \
  --output-dir data/processed/walk_forward
```

For every round, fit preprocessing only on `train.csv`, then use the saved
artifact to transform validation and test. This is required to keep sector
medians, target encoding, and categorical levels from leaking across splits.

```bash
PYTHONPATH=src python -m finance_project.data_processing.preprocess fit \
  --input data/processed/walk_forward/round_01/train.csv \
  --history data/processed/walk_forward/round_01/history.csv \
  --output data/processed/walk_forward/round_01/features_train.csv \
  --artifacts models/preprocessing/round_01.joblib

PYTHONPATH=src python -m finance_project.data_processing.preprocess transform \
  --input data/processed/walk_forward/round_01/validation.csv \
  --history data/processed/walk_forward/round_01/history.csv \
  --output data/processed/walk_forward/round_01/features_validation.csv \
  --artifacts models/preprocessing/round_01.joblib

PYTHONPATH=src python -m finance_project.data_processing.preprocess transform \
  --input data/processed/walk_forward/round_01/test.csv \
  --history data/processed/walk_forward/round_01/history.csv \
  --output data/processed/walk_forward/round_01/features_test.csv \
  --artifacts models/preprocessing/round_01.joblib
```

Repeat the three preprocessing commands for rounds 2--4. The transformed files
retain `id`, `fs_year`, `pred_start_dt` when available, and `label`, followed
by model features; do not use the identifier/date columns as model inputs.

## Working conventions

- Keep raw files in `data/raw/`; do not modify them in scripts or notebooks.
- Use notebooks for exploration, then promote reusable logic into `src/`.
- Commit code, configuration, the report template, and the demo dataset; do
  not commit large raw files, generated models, or generated metrics.
- Add a new experiment configuration before comparing multiple model families.
