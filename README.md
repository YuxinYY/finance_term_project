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

Run the pipeline from the repository root. The first command uses the tracked
1,000-row demo dataset and removes the CSV index column (for example,
`Unnamed: 0`) safely.

```bash
PYTHONPATH=src python -m finance_project.data_processing.make_dataset \
  --input data/raw/train_demo_subset.csv

# Replace profit with the approved project target column.
PYTHONPATH=src python -m finance_project.model_development.train \
  --data data/processed/train_demo_subset_clean.csv --target profit

PYTHONPATH=src python -m finance_project.model_evaluation.evaluate \
  --data data/processed/train_demo_subset_clean.csv --target profit

PYTHONPATH=src python -m finance_project.reporting.update_report
```

The training and evaluation scripts intentionally require `--target`. Choose
the target only after the project team agrees on the prediction question. The
report updater reads JSON artifacts, so report numbers are never manually
copied from a notebook.

## Working conventions

- Keep raw files in `data/raw/`; do not modify them in scripts or notebooks.
- Use notebooks for exploration, then promote reusable logic into `src/`.
- Commit code, configuration, the report template, and the demo dataset; do
  not commit large raw files, generated models, or generated metrics.
- Add a new experiment configuration before comparing multiple model families.
