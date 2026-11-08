# Validation-strategy recommendation

## Decision

Add an optional `rolling_year` validation strategy to
`make_walk_forward_splits.py`, while retaining the existing `company_random`
strategy as a new-company sensitivity analysis.

```text
--validation-strategy company_random
--validation-strategy rolling_year
```

Use `rolling_year` as the primary hyperparameter-tuning design because the
production task is to score future fiscal-year statements.  Use
`company_random` when the team wants an explicit test of generalisation to
companies absent from the training partition.

## Current method: `company_random`

The existing split process selects approximately 20% of company IDs from the
historical training window.  Every row for those companies is validation data;
the remaining company IDs form training data.

- Train and validation company IDs are disjoint.
- It can tune hyperparameters, provide LightGBM early stopping, and fit Platt
  calibration.
- It is a useful conservative check for firms not seen in model fitting.
- It is not fully time-ordered: a model can train on a later fiscal year from
  one company while validating on an earlier fiscal year from another company.

## Proposed method: `rolling_year`

For each outer test year \(t\), use all years before \(t\) as the outer
training window.  Create expanding inner folds by year inside that window.

Example: outer test year FY2010.

```text
Outer training: FY2007–FY2009
Outer test:     FY2010

Inner fold 1: train FY2007       → validate FY2008
Inner fold 2: train FY2007–FY2008 → validate FY2009
```

For every candidate hyperparameter set, compute the validation metric in each
inner fold and use the mean across folds to select the configuration.  Then
refit the selected model on the complete outer training window and score the
outer test year exactly once.

This is temporally realistic: every inner validation year is later than its
corresponding training years.  The same company may appear in an earlier inner
training year and a later inner validation year.  That is appropriate when the
deployment use case includes existing borrowers with prior statements.

## Calibration rule

Do **not** fit a PD calibrator on in-sample predictions from the final model
refit.  Those predictions are overly optimistic because the model was trained
on the same labels.

Instead:

1. Train each inner-fold model and save its validation-year raw PDs.
2. Pool those out-of-fold raw PDs and their observed labels.
3. Fit the Platt calibrator on that pooled out-of-fold data.
4. Refit the selected base model on all outer-training rows.
5. Apply the frozen calibrator to the outer-test raw PDs.

For LightGBM, each inner validation year supplies early stopping.  After
tuning, use the median best iteration from the inner folds as the fixed number
of trees for the full outer-training refit.

## Expected split layout

```text
round_03/
  train.csv                 # complete outer training window: FY2007–FY2009
  test.csv                  # outer test year: FY2010
  history.csv
  inner_folds/
    fold_2008/
      train.csv             # FY2007
      validation.csv        # FY2008
      history.csv           # FY2007–FY2008
    fold_2009/
      train.csv             # FY2007–FY2008
      validation.csv        # FY2009
      history.csv           # FY2007–FY2009
```

Each inner fold must run the existing preprocessing `fit` command on its own
inner training CSV, then use that fold's saved preprocessing artifact to
transform its validation CSV.  This keeps imputation statistics, target
encoding, categories, and feature engineering leakage-safe.

## Round-1 limitation and recommendation

The first outer round has FY2007 training and FY2008 test data only.  It has no
earlier/later year pair within its training window, so it cannot create a
rolling-year inner validation fold.

Recommendation:

- Run round 1 as an untuned baseline with fixed, predeclared model settings.
- Use rolling-year hyperparameter tuning and out-of-fold calibration in rounds
  2–4.
- Retain `company_random` as an optional alternative for all rounds and as a
  new-company robustness check.
