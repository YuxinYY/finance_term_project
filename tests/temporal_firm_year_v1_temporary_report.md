# Temporary review report: `temporal_firm_year_v1`

## Scope

This preliminary experiment uses the **firm-year** feature mode and the
three-round **temporal-year validation** design.  Each observation retains its
direct fiscal-year financial features; it is not a decayed aggregation of a
company's historical statements.  The rounds are: FY2007 train / FY2008
validation / FY2009 test, then the train window expands by one year in each
round.

The comparison includes the unbootstrapped WoE + logistic-regression scorecard
and monotone LightGBM.  Validation data select the WoE/logistic regularisation
or provide LightGBM early stopping, and also fit the Platt calibrator.  Test
labels are not used in either fitting or calibration.

## Test-set results

`Calibration gap` is mean calibrated PD minus the observed test default rate.
Negative values mean that the model underpredicts the default level in that
test period.

| Round | Model | Observed DR | ROC-AUC | PR-AUC | Mean calibrated PD | Calibration gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | WoE/logistic | 1.58% | 0.8270 | 0.1963 | 1.19% | -0.39 pp |
| 1 | LightGBM | 1.58% | **0.8431** | **0.2306** | 1.19% | -0.39 pp |
| 2 | WoE/logistic | 1.65% | 0.8215 | 0.1724 | 1.27% | -0.38 pp |
| 2 | LightGBM | 1.65% | **0.8370** | **0.1920** | 1.33% | -0.32 pp |
| 3 | WoE/logistic | 1.39% | 0.8380 | 0.1669 | 1.68% | +0.29 pp |
| 3 | LightGBM | 1.39% | **0.8534** | **0.1983** | 1.67% | +0.28 pp |

**Calibration note.** Each model is fit on its training years only.  The next
year is the outer validation set: it is used for WoE/logistic selection or
LightGBM early stopping, then its raw PDs and labels fit the Platt mapping,
\(\sigma(a\operatorname{logit}(pd_{raw}) + b)\).  That frozen mapping is applied
once to the following test year.  `Calibration gap` is mean calibrated test PD
minus the observed test default rate.

## Initial interpretation

- **LightGBM ranks defaults better in every temporal test year.** Its ROC-AUC
  and PR-AUC exceed the WoE/logistic benchmark in all three rounds.
- **Ranking is reasonably stable.** Test ROC-AUC is 0.8215–0.8534 and PR-AUC
  is 0.1669–0.2306 across the two models and three out-of-time years.
- **Calibration shifts with the default-rate regime.** Both models
  underpredict in rounds 1–2 and overpredict in round 3, even though Platt
  calibration is nearly exact on the preceding validation year.  This is
  temporal distribution shift, not calibration fitted on the test set.

## PR figures

The curves below use calibrated test PDs.  PR curves are shown because they
are especially informative for this rare-default outcome; the corresponding
ROC curves are stored beside them under `reports/figures/temporal_firm_year_v1/`.

| Round | Model | PR curve |
| --- | --- | --- |
| 1 | WoE/logistic | ![Round 1 WoE/logistic PR curve](../reports/figures/temporal_firm_year_v1/round_01/woe_logistic/pr_curve.png) |
| 1 | LightGBM | ![Round 1 LightGBM PR curve](../reports/figures/temporal_firm_year_v1/round_01/lightgbm/pr_curve.png) |
| 2 | WoE/logistic | ![Round 2 WoE/logistic PR curve](../reports/figures/temporal_firm_year_v1/round_02/woe_logistic/pr_curve.png) |
| 2 | LightGBM | ![Round 2 LightGBM PR curve](../reports/figures/temporal_firm_year_v1/round_02/lightgbm/pr_curve.png) |
| 3 | WoE/logistic | ![Round 3 WoE/logistic PR curve](../reports/figures/temporal_firm_year_v1/round_03/woe_logistic/pr_curve.png) |
| 3 | LightGBM | ![Round 3 LightGBM PR curve](../reports/figures/temporal_firm_year_v1/round_03/lightgbm/pr_curve.png) |
