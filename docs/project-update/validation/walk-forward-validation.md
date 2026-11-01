# Walk-forward validation

Anchored window, fixed features. Described for round t.

## 1. Split

- Training window: FY2007 to FY(t−1). Test dataset: FY(t).
- Within the window: training dataset FY2007 to FY(t−2); validation dataset FY(t−1).
- If default labels need an observation period, add a gap between the datasets.
- Labels are usable for FY2007–FY2011. The earliest test year is FY2010 (training dataset FY2007–FY2008, so at least one fold) and the latest is FY2011: two rounds.

## 2. Tune hyperparameters (training dataset only)

- Define a candidate grid for the chosen model (e.g. regularization strength for a linear model, tree depth and learning rate for boosted trees).
- Use expanding-window folds split by year: train FY2007 → validate FY2008; train FY2007–08 → validate FY2009; and so on. Folds stay within FY2007 to FY(t−2), so FY(t−1) is reserved for calibration.
- Run every hyperparameter combination on every fold: fit imputation, scaling and the model inside the fold, then compute log-loss on the validation year. This avoids leakage.
- Pick the combination with the highest mean AUC across folds. Ranking power comes first; the PD level is fixed by the calibration in step 4, so log-loss is only reported for reference.
## 3. Refit

Refit imputation, scaling and the model on the whole training dataset (FY2007 to FY(t−2)) with the selected hyperparameters.

## 4. Calibrate (on the validation dataset FY(t−1))

1. Score the validation dataset with the refit model, sort by score, and divide into k buckets of N/k companies each (the first bucket has the worst scores).
2. Count the defaults in each bucket to get its actual default rate.
3. Fit a curve mapping quantiles to default rates, and record the model output associated with each quantile.
4. Apply the Bayesian prior-shift formula to convert from the validation default rate to the population default rate:

```
pi* = (T·pi − pi·S·T) / (pi·T − pi·S + S − S·T)
```

- S: overall default rate of the validation dataset FY(t−1)
- T: true population default rate (state its source)
- pi: mapped PD from step 3
- pi*: calibrated PD

If S and T are close, this step has almost no effect.

## 5. Test (FY(t), evaluated once)

- Score the test dataset with the model from step 3, then apply the mapping and transformation from step 4 to get the calibrated PD.
- Report ROC/AUC and the calibration plot, separately for existing and new companies. Define "new" explicitly (e.g. companies not present in the training data).

## 6. Roll forward

Add FY(t) to the training window and repeat steps 1 to 5.

## 7. Stability

Report metrics per test year, plus the selected hyperparameters per round.

## 8. Final model

Same procedure as each round: tune and refit on data before the latest year, and keep the latest year for calibration. Do not train on all the data, or there is no calibration set left.

## To discuss: Bootstrap

**Bootstrap confidence intervals (left out for now; whether to add is to be discussed).** Bias is judged only from the calibration plot and the point estimate of (mean calibrated PD − observed default rate), so there is no way to tell whether a gap is noise. Read it with care when the test year has few defaults. If added later: resample the test dataset n times and compute confidence intervals for the AUC and for (mean calibrated PD − observed default rate) separately; bias is judged by whether the latter interval contains 0.
