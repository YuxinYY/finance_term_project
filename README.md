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
PYTHONPATH=src python -m finance_project.data_processing.make_walk_forward_splits \
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

Minority-class bootstrapping is optional and applies only to the transformed
training output. Add these arguments to the `preprocess fit` command to target
one default for every five non-defaults; omit them to retain the original class
distribution for comparison:

```bash
--bootstrap-minority --minority-to-majority-ratio 0.2
```

## Walk-forward model development

The first model-development runner compares a training-only WoE scorecard with
regularised logistic regression and monotone LightGBM.  It reads the already
transformed feature files, uses the validation set for logistic-model selection
or LightGBM early stopping, fits a Platt calibrator on validation predictions,
and evaluates the test split once.  It does not bootstrap the minority class.

Run both models for all four rounds from the repository root:

```bash
PYTHONPATH=src /opt/miniconda3/envs/daily/bin/python \
  -m finance_project.model_development.run_walk_forward_models
```

For a quicker single-round run, or to run one model family:

```bash
PYTHONPATH=src /opt/miniconda3/envs/daily/bin/python \
  -m finance_project.model_development.run_walk_forward_models \
  --rounds 1 --models lightgbm --run-name lightgbm_round_01
```

Every generated artifact for the default run is written below
`model_artifacts/baseline_v1/`, grouped by round and model family.  Supply a
new `--run-name` to preserve earlier experiments, or `--output-dir` to place
all artifacts elsewhere.  Each model folder contains its serialised model,
metrics JSON, and validation/test prediction CSVs; the JSON contains ROC-AUC,
PR-AUC, raw and calibrated PD-quality metrics, and a calibrated test-PD decile
table.  ROC and precision--recall curves for calibrated test PDs are written to
`reports/figures/<run-name>/round_XX/<model-family>/`.  Pass `--no-figures` to
skip those PNG outputs, or `--figure-dir` to use another figure root.  See
`src/finance_project/model_development/MODEL_DEVELOPMENT_PLAN.md` for the
evaluation contract and the later FURIA, GAM, and TabPFN phases.

### PD calibration

The runner retains the model's unadjusted probability as `pd_raw`, then fits a
Platt calibrator on that round's validation predictions and labels.  It learns
the two parameters in the following mapping:

\[
\operatorname{pd\_calibrated} = \sigma\!\left(a \cdot
\operatorname{logit}(\operatorname{pd\_raw}) + b\right),
\]

where \(\sigma\) is the logistic sigmoid.  The same frozen mapping is then
applied to the test-set `pd_raw` values; test labels are never used to fit the
calibrator.  Raw probabilities are clipped to \([10^{-6}, 1 - 10^{-6}]\)
before the logit is calculated.  This produces calibrated PDs, not binary
default decisions; the runner does not use a 0.5 (or any other) classification
threshold.

## Working conventions

- Keep raw files in `data/raw/`; do not modify them in scripts or notebooks.
- Use notebooks for exploration, then promote reusable logic into `src/`.
- Commit code, configuration, the report template, and the demo dataset; do
  not commit large raw files, generated models, or generated metrics.
- Add a new experiment configuration before comparing multiple model families.
