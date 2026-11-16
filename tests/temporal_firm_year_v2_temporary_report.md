# Temporary review report: `temporal_firm_year_v2`

## Scope

This experiment uses **firm-year** financial features and the same three-round
**temporal-year validation** design as the
[v1 review](temporal_firm_year_v1_temporary_report.md). Each observation retains
its fiscal-year features rather than a decayed average of historical statements.
The completed v2 run contains **monotone LightGBM only**; no new WoE/logistic or
EBM results are available under this run name.

| Round | Training years | Validation year | Test year | Training rows | Validation rows | Test rows | Test defaults |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 2007 | 2008 | 2009 | 144,251 | 161,383 | 170,549 | 2,687 |
| 2 | 2007–2008 | 2009 | 2010 | 305,634 | 170,549 | 174,674 | 2,881 |
| 3 | 2007–2009 | 2010 | 2011 | 476,183 | 174,674 | 184,159 | 2,563 |

Counts above come from the saved split and feature CSVs, with validation/test
counts confirmed against the run's metrics. Raw and transformed row counts and
default counts match in all nine splits: there is no observed row loss at the
final preprocessing step for these datasets. Earlier data exclusions are not
measured by this comparison.

## Features and training settings

Each round uses **80 model features**, compared with **171** in v1. The saved
feature lists include the revised financial ratios, raw income-statement
amounts, ROE, negative-P&L and zero-value flags, region dummies (`city_*`), legal
form dummies, and `ateco_sector_te`. Region dummies replace the earlier
province-code dummies.

The current preprocessing uses coarse **A–U ATECO industry groups** for median
imputation and target encoding. Raw financial inputs are imputed before feature
construction; the final feature check removes rows containing missing or
infinite model features. These preprocessing descriptions reflect the current
code; the model metrics retain the actual fitted feature lists and constraints,
but do not independently archive every preprocessing rule.

| Setting | Value |
| --- | --- |
| Maximum trees | 5,000 |
| Learning rate | 0.03 |
| Leaves | 15 |
| Minimum child samples | 200 |
| Subsample / column sample | 0.8 / 0.8 |
| L2 regularisation | 1.0 |
| Early-stopping patience | 150 rounds |
| Random seed | 42 |
| Selected trees, rounds 1 / 2 / 3 | 88 / 357 / 424 |
| Probability calibration | Platt scaling on the outer validation set |

Features with a nonzero entry in the saved monotonic-constraint map are
constrained; other features are unconstrained. For example, the newly included
`debt_to_eqty`, `operating_cash_flow_margin`, `cash_interest_coverage`, and
`operating_cash_flow_to_debt` have constraint 0 in this run.

## Test-set results

All results below use **Platt-calibrated test PDs**. Calibration gap is mean PD
minus observed default rate, in percentage points (pp).

| Round | Model | Observed DR | ROC-AUC | PR-AUC | Mean calibrated PD | Calibration gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | LightGBM | 1.58% | 0.8401 | 0.2420 | 1.22% | -0.35 pp |
| 2 | LightGBM | 1.65% | 0.8368 | 0.1904 | 1.43% | -0.22 pp |
| 3 | LightGBM | 1.39% | 0.8529 | 0.1970 | 1.65% | +0.26 pp |

The model is fitted on training years, validation selects the early-stopping
iteration and fits the Platt mapping, and the frozen model and mapping score
the following test year. Test labels are used only for evaluation. Validation
calibration statistics are therefore fitted statistics, not independent
evidence of calibration quality.

| Round | Raw test log loss | Calibrated test log loss | Raw test Brier score | Calibrated test Brier score | Calibrated test slope |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.06547 | 0.06341 | 0.01402 | 0.01350 | 1.0308 |
| 2 | 0.06861 | 0.06738 | 0.01474 | 0.01462 | 0.9492 |
| 3 | 0.05742 | 0.05757 | 0.01229 | 0.01233 | 1.0014 |

Lower log loss and Brier score are better. Platt calibration improves both
metrics in rounds 1–2, but slightly worsens them in round 3. ROC-AUC and PR-AUC
are unchanged by the fitted monotone Platt transformations in all three rounds.

## Comparison with v1 LightGBM

The saved v1 and v2 metrics have the same test row and default counts in each
round. This is a descriptive comparison of complete pipeline versions: several
features and preprocessing choices changed together, so the differences cannot
be attributed to one change. No uncertainty or significance test is included.
PR-AUC here is the runner's average-precision metric.

| Round | v1 ROC-AUC | v2 ROC-AUC | Change | v1 PR-AUC | v2 PR-AUC | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8431 | 0.8401 | -0.0030 | 0.2306 | 0.2420 | +0.0114 |
| 2 | 0.8370 | 0.8368 | -0.0002 | 0.1920 | 0.1904 | -0.0016 |
| 3 | 0.8534 | 0.8529 | -0.0006 | 0.1983 | 0.1970 | -0.0014 |

| Round | v1 calibration gap | v2 calibration gap | v1 calibrated Brier | v2 calibrated Brier |
| --- | ---: | ---: | ---: | ---: |
| 1 | -0.39 pp | -0.35 pp | 0.01361 | 0.01350 |
| 2 | -0.32 pp | -0.22 pp | 0.01462 | 0.01462 |
| 3 | +0.28 pp | +0.26 pp | 0.01235 | 0.01233 |

## Initial interpretation

- **Ranking remains close to v1 with fewer features.** v2 ROC-AUC is slightly
  lower in every round; PR-AUC improves in round 1 and falls slightly in rounds
  2–3. These results do not establish an overall improvement over v1.
- **Mean-PD calibration gaps are smaller in magnitude in all three rounds.**
  The pattern remains underprediction in rounds 1–2 and overprediction in
  round 3. Test slopes are near one, but that does not remove the level gaps.
- **Calibration must transfer across years.** Round 3 shows that a mapping
  fitted to the preceding year's higher default rate can raise PDs too much
  for the next year. Its calibrated log loss and Brier score are slightly
  worse than the raw model's.
- **The 80-feature specification is a credible candidate for further review.**
  The run supports comparing the revised pipeline with v1; it does not yet
  provide a new three-model comparison or a production-refit evaluation.

## PR figures

The curves use calibrated test PDs. ROC curves are stored beside each PR curve.

| Round | Model | PR curve |
| --- | --- | --- |
| 1 | LightGBM | ![Round 1 LightGBM PR curve](../reports/figures/temporal_firm_year_v2/round_01/lightgbm/pr_curve.png) |
| 2 | LightGBM | ![Round 2 LightGBM PR curve](../reports/figures/temporal_firm_year_v2/round_02/lightgbm/pr_curve.png) |
| 3 | LightGBM | ![Round 3 LightGBM PR curve](../reports/figures/temporal_firm_year_v2/round_03/lightgbm/pr_curve.png) |

## Sources and production status

- [Run summary](../model_artifacts/temporal_firm_year_v2/summary.json)
- Per-round metrics: [round 1](../model_artifacts/temporal_firm_year_v2/round_01/lightgbm/metrics.json), [round 2](../model_artifacts/temporal_firm_year_v2/round_02/lightgbm/metrics.json), [round 3](../model_artifacts/temporal_firm_year_v2/round_03/lightgbm/metrics.json).
- Split counts: `data/processed/walk_forward_temporal/round_XX/{train,validation,test}.csv` and the corresponding `features_*.csv` files.
- [v1 report](temporal_firm_year_v1_temporary_report.md) and the saved v1 LightGBM `metrics.json` files supply the comparison values.

This report covers the three v2 evaluation rounds. It does not include a new
production refit; the v1 report's `production_lightgbm_v1` results belong to the
earlier pipeline and should not be presented as v2 production results.
