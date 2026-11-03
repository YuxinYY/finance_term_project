# Model-development plan

## Goal

Compare transparent and flexible models for the `label` default outcome without
letting information from a future walk-forward test period affect feature
preparation, tuning, calibration, or model selection.  Each row of the
transformed data is one `(id, fs_year)` prediction sample; `id`, `fs_year`,
`pred_start_dt`, and `label` are metadata, not predictors.

The four model tracks are:

1. a WoE scorecard with logistic regression;
2. FURIA and monotone LightGBM;
3. a generalized additive model (GAM); and
4. TabPFN, only if its scale and dependency checks pass.

## Common experimental contract

### Data and folds

- Use the existing four walk-forward rounds.  For every round, fit model
  components on `features_train.csv`, tune/select them on
  `features_validation.csv`, and make one final evaluation on
  `features_test.csv`.
- Keep the test set completely untouched until its final score.  In
  particular, bins, WoE values, scalers, imputers, rules, target encoding,
  hyperparameters, monotonic constraints, and calibrators must not use test
  rows or test labels.
- Training and validation company IDs are already disjoint.  A company may
  occur in training history and later in the test year; that is deliberate
  walk-forward evaluation, not leakage.
- Reuse the saved preprocessing artifact fitted for that round.  Do not refit
  anomaly rules, sector medians, target encoding, or category levels inside a
  model using validation or test data.

### Class imbalance

- Run each applicable model in two training modes: the original class
  distribution and optional minority bootstrap at a 1:5 default:non-default
  ratio (`--bootstrap-minority --minority-to-majority-ratio 0.2`).
- Bootstrap only transformed *training* rows, after all target-derived
  transformations have been fitted.  Never bootstrap validation or test rows.
- For LightGBM, also compare class weighting (`scale_pos_weight`) against
  bootstrap, rather than combining them by default.  Select the approach on
  validation log loss/AUC and report the choice.

### Metrics and promotion rule

- Primary discrimination metric: ROC-AUC.  Also report PR-AUC because defaults
  are rare.
- Probability-quality metrics: log loss, Brier score, calibration-in-the-large,
  calibration slope, and observed-versus-predicted default rates by PD decile.
- Report metrics for every round and pooled out-of-time test predictions.  Use
  a firm-clustered bootstrap confidence interval for pooled comparisons because
  firms can appear in multiple years.
- A candidate is promoted only when it improves validation performance without
  materially worsening calibration or violating its stated interpretability
  requirement.  The final test set is not used to choose a model.

### Calibration and artifacts

- Fit a probability calibrator using training/validation data only after the
  base estimator is selected.  Compare no calibration, Platt/sigmoid, and
  isotonic calibration; prefer the simplest option that improves validation
  log loss without degrading ranking.
- Save, per round: input feature list, feature exclusions, preprocessing
  artifact reference, model parameters, random seed, bootstrap/weighting mode,
  calibrator, validation metrics, test metrics, fitted model, and predictions
  containing `id`, `fs_year`, `label`, and `pd`.

## 1. WoE + logistic-regression scorecard

**Purpose:** first serious benchmark: transparent, auditable, and naturally
suited to financial-ratio risk bands.

1. Start with a plain regularized logistic regression on the model-ready
   features as a debugging baseline.  Median-impute remaining numeric values
   using training values only, add missing indicators where needed, and
   standardize only for this plain-logit baseline.
2. For the scorecard, derive bins and smoothed Weight of Evidence values from
   the training data only.  Give missing values a separate bin.  Merge sparse
   bins and apply pseudo-count smoothing so zero defaults/non-defaults cannot
   create infinite WoE.
3. For financially directional ratios, test monotonic binning consistent with
   economic expectations (for example, more liquidity should not increase
   risk).  Record exceptions rather than silently forcing a questionable
   relationship.
4. Transform validation and test data using frozen training bin edges and WoE
   values.  Fit L1/L2-regularized logistic regression on the WoE features;
   select regularization on validation data.
5. Produce a scorecard table: bin definitions, default/survivor counts, WoE,
   logistic coefficient, point contribution, and example borrower reasons.

**Decision output:** plain logit versus WoE scorecard; a compact, stable
scorecard is retained as the core explainable benchmark.

## 2. FURIA and monotone LightGBM

### FURIA

**Purpose:** an interpretable fuzzy-rule challenger to the scorecard.

1. First run a feasibility spike: select a reproducible FURIA implementation,
   confirm that it supports numeric inputs, missing-value behavior, class
   scores/probabilities, deterministic seeds, and exported rule text.  The
   likely route is the maintained Weka FURIA implementation, invoked through a
   small reproducible adapter.
2. Train only on round training data; tune rule-complexity controls on the
   validation set.  Compare original and bootstrapped training data.
3. Export the complete rule set and report rule count, average antecedents per
   rule, coverage, and three representative borrower explanations.
4. Treat FURIA's output as an uncalibrated risk score unless its probability
   semantics are verified.  Calibrate it using the common training/validation
   protocol before comparing PD-quality metrics.

**Promotion criterion:** FURIA must be competitive with the WoE scorecard and
remain compact enough that the full rule set can be reviewed.  A large,
overlapping rule list is not considered globally interpretable merely because
individual rules are readable.

### Monotone LightGBM

**Purpose:** the principal nonlinear performance challenger.

1. Use the full transformed feature set.  Preserve missing values as `NaN`
   where LightGBM can handle them natively; do not introduce batch-level
   imputation.
2. Build an explicit feature-to-direction constraint map from the financial
   rationale.  Features with no defensible monotonic direction (for example,
   some activity or categorical-encoding features) receive no constraint.
3. Tune conservative depth/leaves, minimum observations per leaf, learning
   rate, feature subsampling, and regularization on validation only.  Compare
   native imbalance handling, class weighting, and optional bootstrap as
   separate training choices.
4. Inspect monotonicity, calibration, SHAP summaries by *feature family*, and
   local explanation examples.  Check that changes in constrained ratios move
   PD in the stated direction.

**Decision output:** the best calibrated, economically defensible LightGBM
configuration, plus a comparison against the explainable benchmarks.

## 3. Explainable Boosting Machine (EBM)

**Purpose:** determine whether smooth, one-feature risk curves recover much of
the LightGBM performance while staying globally explainable.

1. Use `interpret.glassbox.ExplainableBoostingClassifier`, which fits binned,
   boosted univariate functions and exports each learned shape.  The EBM's
   internal validation split is drawn only from the training partition for
   early stopping; outer validation remains for calibration.
2. Fit the same core ratio set first; then evaluate a fuller feature set only
   if validation data supports it.  Limit complexity/number of bins and use
   regularization to avoid wiggly curves.
3. Enforce monotonic constraints for ratios with clear expected directions.
   Keep non-monotonic effects only when they are stable across rounds and
   documented.
4. Start with additive main effects.  Consider a small, pre-specified set of
   pairwise interactions only if they improve validation performance enough to
   justify the added explanation burden.
5. Export every learned feature curve, its missing-value treatment, and local
   additive contributions for example firms.

**Decision output:** quantify the price of one-readable-curve-per-feature
interpretability relative to LightGBM.

## 4. TabPFN (deferred)

**Status:** deferred.  Do not add TabPFN or its dependencies in the current
development phase.

TabPFN is optional because the walk-forward training sets are large and highly
imbalanced, while its computational/data-size limits and behavior on financial
default panels must be verified.  It is also not a globally interpretable
model.

Reconsider only after the WoE, LightGBM, FURIA, and EBM comparisons are stable.

## Implementation order

1. Build shared data loading, metadata exclusion, metrics, calibration,
   prediction-writing, and per-round experiment configuration.
2. Implement the plain logistic baseline and WoE scorecard.
3. Implement monotone LightGBM, then FURIA feasibility/export support.
4. Implement the EBM benchmark.
5. Freeze the selected configuration, refit it on the permitted training data
   for each round, and write a model-comparison report.
