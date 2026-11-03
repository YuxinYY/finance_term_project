# Walk-forward validation

Anchored window, fixed features. Described for round t.

## 1. Split

- Training dataset: FY2007 to FY(t−1). Test dataset: FY(t). There is no separate calibration year; calibration uses the training dataset itself (step 4).
- If default labels need an observation period, add a gap between the training and test datasets.
- Labels are usable for FY2007–FY2011. The earliest test year is FY2009 (training dataset FY2007–FY2008, so at least one fold) and the latest is FY2011: three rounds.

## 2. Tune hyperparameters (training dataset only)

- Define a candidate grid for the chosen model (e.g. regularization strength for a linear model, tree depth and learning rate for boosted trees).
- Use expanding-window folds split by year inside the training dataset: train FY2007 → validate FY2008; train FY2007–08 → validate FY2009; and so on, up to train FY2007 to FY(t−2) → validate FY(t−1).
- Run every hyperparameter combination on every fold: fit imputation, scaling and the model inside the fold, then compute AUC on the validation year. This avoids leakage.
- Pick the combination with the highest mean AUC across folds. Ranking power comes first; the PD level is fixed by the calibration in step 4, so log-loss is at most reported for reference.
- The folds only select hyperparameters (and the early-stopping point, if used). They play no part in calibration or testing.

## 3. Refit

Refit imputation, scaling and the model on the whole training dataset (FY2007 to FY(t−1)) with the selected hyperparameters.

## 4. Calibrate (on the training dataset)

1. Score the training dataset with the refit model, sort by score, and divide into k buckets of N/k companies each (the first bucket has the worst scores). Choose k so that each bucket holds enough defaults.
2. Count the defaults in each bucket to get its actual default rate.
3. Fit a monotone curve (e.g. isotonic regression) mapping quantiles to default rates, and record the model output associated with each quantile.
4. Apply the Bayesian prior-shift formula to convert from the training default rate to the population default rate:

```
pi* = (T·pi − pi·S·T) / (pi·T − pi·S + S − S·T)
```

- S: overall default rate of the training dataset FY2007 to FY(t−1), pooled over years
- T: true population default rate (state its source)
- pi: mapped PD from step 3
- pi*: calibrated PD

If S and T are close, this step has almost no effect.

The model has seen this data, so its scores here look better than on new data and the curve will be somewhat optimistic; the more flexible the model, the larger the effect. The calibration plot on the test year (step 5) shows how large it is. Calibration is a monotone transform, so it does not change the ranking or the AUC.

## 5. Test (FY(t), evaluated once)

- Score the test dataset with the model from step 3, then apply the mapping and transformation from step 4 to get the calibrated PD.
- Report ROC/AUC and the calibration plot, separately for existing and new companies. Define "new" explicitly (e.g. companies not present in the training data).

## 6. Roll forward

Add FY(t) to the training dataset and repeat steps 1 to 5.

## 7. Stability

Report metrics per test year, plus the selected hyperparameters per round. Also check the calibration plot year by year, since the curve pools several years with different default rates.

## 8. Final model

Same procedure as each round: tune on the folds, refit on all years up to the latest, and calibrate on that same training dataset.

## Basis for the design

Calibrating on the data the model was estimated on follows Falkenstein et al. (2000) and Kocagil et al. (2002, Appendix B). Stein (2007) and Stein et al. (2003) say only that the calibration data is a historical default database assumed representative of the population, and the RiskCalc 3.1 whitepaper does not say how development, validation and calibration data were divided. The walk-forward folds and the choice of AUC for tuning are our own design, not prescribed by these papers.

## To discuss: Bootstrap

**Bootstrap confidence intervals (left out for now; whether to add is to be discussed).** Bias is judged only from the calibration plot and the point estimate of (mean calibrated PD − observed default rate), so there is no way to tell whether a gap is noise. Read it with care when the test year has few defaults. If added later, it would be an extra part of step 5, run once per round on the test year (no change to steps 1 to 4):

1. Keep the model (step 3) and the calibration mapping (step 4) fixed. Only the test dataset is resampled, so the interval reflects noise in the test sample, not variance from refitting the model. Measuring the latter would mean resampling the training dataset and repeating steps 2 to 4 each time, which is far more expensive and is left out.
2. Score the test dataset once and keep each company's score, calibrated PD and default label.
3. Repeat n times (e.g. n = 1,000): draw N companies with replacement from the N in the test dataset (one row per company within a test year), and compute on the resample
   - the AUC;
   - the bias = mean calibrated PD − observed default rate.
4. Take the 2.5th and 97.5th percentiles of each as the 95% interval. Bias is judged by whether its interval contains 0; the AUC interval shows how much the ranking power can be trusted.
5. Report existing and new companies separately: resample within each group, because the new-company group may have few defaults and a wide interval.

Notes:

- Use a plain bootstrap, not one stratified by default status. Stratifying fixes the number of defaults in every resample, which hides the noise in the observed default rate, and that noise is exactly what the bias interval needs to show.
- If several test years are pooled, resample by company `id` rather than by row, to keep a company's years together.
- To compare two models, score both on the same resamples and take the interval of the AUC difference.
- Cost is small because nothing is refit; only the metrics are recomputed on the resamples.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_test(y, pd_cal, n=1000, seed=0):
    # y: 0/1 default labels, pd_cal: calibrated PD, both for the test year
    rng = np.random.default_rng(seed)
    aucs, biases = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        aucs.append(roc_auc_score(y[i], pd_cal[i]))
        biases.append(pd_cal[i].mean() - y[i].mean())
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(biases, [2.5, 97.5])
```

The AUC can use the calibrated PD because calibration is monotone and does not change the ranking.
