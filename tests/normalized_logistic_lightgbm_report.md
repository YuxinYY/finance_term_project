# Normalized logistic regression vs. LightGBM

## Scope and takeaway

These results compare the logistic and LightGBM models in `temporal_no_goodwill`, using the same 74 selected model inputs after manually removing `goodwill_ratio`. Each round has its own training-fitted normalization. The reported test scores below use validation-fitted Platt calibration. Round 1 tests fiscal year 2009, round 2 tests 2010, and round 3 tests 2011.

**LightGBM ranks defaults more effectively in all three test years.** Its ROC AUC is higher by 0.049–0.057 and its PR AUC is higher by 0.086–0.087. Its feature explanations also show that it relies on several interpretable balance-sheet ratios and can represent changes in their effects across their range. These curves are descriptive evidence of the fitted model, not proof that flexibility alone caused the performance gain.

## Out-of-time results

| Round / test FY | Test records | Defaults | Model | ROC AUC | PR AUC | Mean PD | Observed default rate | Log loss | Brier score |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 / 2009 | 170,549 | 2,687 | Logistic | 0.7942 | 0.1565 | 1.255% | 1.576% | 0.07037 | 0.01461 |
| 1 / 2009 | 170,549 | 2,687 | LightGBM | 0.8429 | 0.2422 | 1.192% | 1.576% | 0.06320 | 0.01350 |
| 2 / 2010 | 174,674 | 2,881 | Logistic | 0.7793 | 0.1053 | 1.434% | 1.649% | 0.07478 | 0.01571 |
| 2 / 2010 | 174,674 | 2,881 | LightGBM | 0.8365 | 0.1924 | 1.382% | 1.649% | 0.06743 | 0.01462 |
| 3 / 2011 | 184,159 | 2,563 | Logistic | 0.8016 | 0.1082 | 1.657% | 1.392% | 0.06446 | 0.01322 |
| 3 / 2011 | 184,159 | 2,563 | LightGBM | 0.8536 | 0.1952 | 1.681% | 1.392% | 0.05764 | 0.01235 |

LightGBM also has lower log loss and Brier score in every round. PR AUC is especially informative given default rates of only 1.39–1.65%. Calibration is not uniformly better: both models underpredict the average default rate in rounds 1–2 and overpredict in round 3; logistic's mean PD is closer to the observed rate in each round. Lower Brier score and log loss do not imply better calibration by every measure.

## Feature-level evidence

Permutation importance measures how much a test-sample metric falls when a feature is shuffled. Each round uses a stratified sample of 3,000 test rows and three permutations, with raw-model probabilities. The numbers below are drops in ROC AUC, not percentage-point changes in predicted PD.

| Feature | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|
| Equity ratio | 0.0066 | 0.0327 | 0.0203 |
| Accounts receivable ratio | 0.0121 | 0.0080 | 0.0170 |
| Tangible asset ratio | 0.0134 | 0.0215 | 0.0095 |
| Current asset ratio | 0.0200 | 0.0060 | — |

Equity, receivables and tangible asset ratios appear among the top five LightGBM features by ROC-AUC permutation importance in every round. The current asset ratio is shown for rounds 1 and 2; the dash does not mean zero importance. Some PR-AUC permutation drops are negative, including tangible asset ratio and current asset ratio in round 2; importance depends on the metric and sample. Correlated features can share importance, and three permutations on 3,000 rows can be noisy.

The ALE curves add shape to those rankings. In round 3, the equity-ratio curve moves from a positive centered contribution at low values toward a flatter negative contribution at higher values. The tangible-asset curve also generally declines before flattening, with small reversals. This illustrates LightGBM's ability to represent changing slopes and thresholds. However, the equity ratio is explicitly constrained to have a non-increasing effect, and receivables is constrained to have a non-decreasing effect: those directions partly reflect modeling assumptions. The extreme left equity-ratio edge spans a wide bin, so its large endpoint difference should not be read as a precisely located threshold.

For logistic regression, the round 3 median-baseline marginal effect for tangible asset ratio varies from **−0.00319** at its training 5th percentile to **−0.00110** at the 95th percentile, in probability change per one standardized input unit. The equity-ratio effect varies from **−0.000220** to **−0.000205**. Logistic probability effects can vary across values even though its log-odds are linear. These local slopes are not numerically comparable to centered ALE levels: they use different reference populations and summarize different quantities. The LightGBM ALE uses the test sample and raw predictions, while the logistic table uses training quantiles and mean/median reference profiles.

To make a direct shape comparison, calculate ALE for **both models** on the same rows and probability scale. In particular, ALE values are centered contributions, not absolute PDs; standardized x-axis values are relative to each round's training scaler. Neither ALE nor permutation importance establishes causality.

## Presentation-ready message

> Across three future test years, LightGBM improved ROC AUC by 4.9–5.7 points and PR AUC by 8.6–8.7 points over logistic regression. Its explanations highlight equity, receivables, and tangible assets as influential, while showing that the model can represent varying response shapes across feature ranges. Average PD calibration still requires attention.

## Saved outputs and limitations

- Per-round model metrics and predictions: `model_artifacts/temporal_no_goodwill/round_XX/<model>/`.
- Logistic explanations: `marginal_effects.csv`; LightGBM explanations: `ale.csv` and `permutation_importance.csv` in the corresponding model folders.
- Example plot: [Round 3 top-feature ALE curves](../reports/figures/temporal_no_goodwill/round_03/lightgbm/ale_top_features.png).
- Logistic marginal effects are raw-model point estimates without confidence intervals. The runner uses validation data for both model selection/early stopping and calibration. The feature explanations use test data post hoc; avoid using them repeatedly for tuning if the same test years are to remain a final unbiased assessment.
- Earlier checks identified a reproducibility concern with the older v2 feature files and artifacts. This report describes the newly saved normalized run outputs; it does not independently establish end-to-end reproduction from raw data or validate the submission harness.
