# Temporary review report: `temporal_firm_year_v1`

## Scope

This preliminary experiment uses the **firm-year** feature mode and the
three-round **temporal-year validation** design.  Each observation retains its
direct fiscal-year financial features; it is not a decayed aggregation of a
company's historical statements.  The rounds are: FY2007 train / FY2008
validation / FY2009 test, then the train window expands by one year in each
round.

The comparison includes the unbootstrapped WoE + logistic-regression scorecard,
monotone LightGBM, and an additive Explainable Boosting Machine (EBM/GAM).
Validation data select the WoE/logistic regularisation or provide LightGBM
early stopping; EBM uses an internal training-only validation split for early
stopping.  The outer validation set fits the Platt calibrator for every model.
Test labels are not used in fitting or calibration.

## Test-set results

`Calibration gap` is mean calibrated PD minus the observed test default rate.
Negative values mean that the model underpredicts the default level in that
test period.

| Round | Model | Observed DR | ROC-AUC | PR-AUC | Mean calibrated PD | Calibration gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | WoE/logistic | 1.58% | 0.8270 | 0.1963 | 1.19% | -0.39 pp |
| 1 | LightGBM | 1.58% | **0.8431** | **0.2306** | 1.19% | -0.39 pp |
| 1 | GAM (EBM) | 1.58% | 0.8018 | 0.1759 | 1.30% | -0.27 pp |
| 2 | WoE/logistic | 1.65% | 0.8215 | 0.1724 | 1.27% | -0.38 pp |
| 2 | LightGBM | 1.65% | **0.8370** | **0.1920** | 1.33% | -0.32 pp |
| 2 | GAM (EBM) | 1.65% | 0.8042 | 0.1695 | 1.30% | -0.35 pp |
| 3 | WoE/logistic | 1.39% | 0.8380 | 0.1669 | 1.68% | +0.29 pp |
| 3 | LightGBM | 1.39% | **0.8534** | **0.1983** | 1.67% | +0.28 pp |
| 3 | GAM (EBM) | 1.39% | 0.8240 | 0.1660 | 1.63% | +0.23 pp |

**Calibration note.** Each model is fit on its training years only.  The next
year is the outer validation set: it is used for WoE/logistic selection or
LightGBM early stopping, while EBM early stopping uses an internal split of
the training years.  The outer validation raw PDs and labels fit the Platt
mapping, \(\sigma(a\operatorname{logit}(pd_{raw}) + b)\), for every model.  That
frozen mapping is applied once to the following test year.  `Calibration gap`
is mean calibrated test PD minus the observed test default rate.

## Production refit: `production_lightgbm_v1`

Following the walk-forward selection of LightGBM, the production bundle refits
the firm-year preprocessing and LightGBM model on **all mature labeled data**:
FY2007–2011.  This is a deployment artifact, not a fourth evaluation round;
it therefore has no new held-out test metric.

| Item | Production-refit result |
| --- | --- |
| Training rows | 835,016 |
| Training fiscal years | 2007–2011 |
| Training default rate | 1.36% |
| Model features | 171 |
| LightGBM trees | 314 fixed trees |
| Calibration method | Platt scaling |
| Calibration observations | 529,382 pooled out-of-time rows from rounds 1–3 |
| Calibration default rate | 1.54% |

The 314-tree count is the median of the temporal-run LightGBM early-stopping
iterations (287, 399, and 314).  The preprocessor, model, calibrator, and
training manifest are stored in `model_artifacts/production_lightgbm_v1/`.

| Pooled out-of-time calibration metric | Raw PD | Platt-calibrated PD |
| --- | ---: | ---: |
| ROC-AUC | 0.8412 | 0.8412 |
| PR-AUC | 0.2011 | 0.2011 |
| Mean PD | 1.10% | 1.54% |
| Calibration gap | -0.43 pp | +0.00 pp |
| Calibration slope | 0.9682 | 1.0001 |
| Brier score | 0.01355 | 0.01350 |

These pooled predictions are used to estimate the production Platt mapping;
they should not be reported as an additional independent test result.  The
three temporal test rounds above remain the relevant evidence for model
selection and expected out-of-time performance.

## Initial interpretation

- **LightGBM ranks defaults better in every temporal test year.** Its ROC-AUC
  and PR-AUC exceed both benchmarks in all three rounds.
- **The additive GAM/EBM is a useful interpretability benchmark, but trails
  the other models here.** Its ROC-AUC is 0.8018–0.8240 and its PR-AUC is
  0.1660–0.1759; it is closest to WoE/logistic on round-3 PR-AUC.
- **Ranking is reasonably stable.** Test ROC-AUC is 0.8215–0.8534 and PR-AUC
  is 0.1660–0.2306 across the three models and three out-of-time years.
- **Calibration shifts with the default-rate regime.** All three models
  underpredict in rounds 1–2 and overpredict in round 3, even though Platt
  calibration is nearly exact on the preceding validation year.  This is
  temporal distribution shift, not calibration fitted on the test set.

## PR figures

The curves below use calibrated test PDs.  PR curves are shown because they
are especially informative for this rare-default outcome; the corresponding
ROC curves are stored beside them under `reports/figures/temporal_firm_year_v1/`
and `reports/figures/temporal_firm_year_ebm_v1/`.

| Round | Model | PR curve |
| --- | --- | --- |
| 1 | WoE/logistic | ![Round 1 WoE/logistic PR curve](../reports/figures/temporal_firm_year_v1/round_01/woe_logistic/pr_curve.png) |
| 1 | LightGBM | ![Round 1 LightGBM PR curve](../reports/figures/temporal_firm_year_v1/round_01/lightgbm/pr_curve.png) |
| 1 | GAM (EBM) | ![Round 1 GAM PR curve](../reports/figures/temporal_firm_year_ebm_v1/round_01/ebm/pr_curve.png) |
| 2 | WoE/logistic | ![Round 2 WoE/logistic PR curve](../reports/figures/temporal_firm_year_v1/round_02/woe_logistic/pr_curve.png) |
| 2 | LightGBM | ![Round 2 LightGBM PR curve](../reports/figures/temporal_firm_year_v1/round_02/lightgbm/pr_curve.png) |
| 2 | GAM (EBM) | ![Round 2 GAM PR curve](../reports/figures/temporal_firm_year_ebm_v1/round_02/ebm/pr_curve.png) |
| 3 | WoE/logistic | ![Round 3 WoE/logistic PR curve](../reports/figures/temporal_firm_year_v1/round_03/woe_logistic/pr_curve.png) |
| 3 | LightGBM | ![Round 3 LightGBM PR curve](../reports/figures/temporal_firm_year_v1/round_03/lightgbm/pr_curve.png) |
| 3 | GAM (EBM) | ![Round 3 GAM PR curve](../reports/figures/temporal_firm_year_ebm_v1/round_03/ebm/pr_curve.png) |
