# Normalized logistic regression vs. LightGBM

## Scope and takeaway

These results compare the saved `temporal_normalized_logistic` and `temporal_normalized_lightgbm` walk-forward runs using the same 75 selected model inputs. Each round has its own training-fitted normalization. The reported test scores below use validation-fitted Platt calibration. Round 1 tests fiscal year 2009, round 2 tests 2010, and round 3 tests 2011.

**LightGBM ranks defaults more effectively in all three test years.** Its ROC AUC is higher by 0.048–0.058 and its PR AUC is higher by 0.085–0.087. Its feature explanations also show that it relies on several interpretable balance-sheet ratios and can represent changes in their effects across their range. These curves are descriptive evidence of the fitted model, not proof that flexibility alone caused the performance gain.

## Out-of-time results

| Round / test FY | Test records | Defaults | Model | ROC AUC | PR AUC | Mean PD | Observed default rate | Log loss | Brier score |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 / 2009 | 170,549 | 2,687 | Logistic | 0.7955 | 0.1567 | 1.254% | 1.576% | 0.07030 | 0.01461 |
| 1 / 2009 | 170,549 | 2,687 | LightGBM | 0.8432 | 0.2423 | 1.190% | 1.576% | 0.06312 | 0.01351 |
| 2 / 2010 | 174,674 | 2,881 | Logistic | 0.7790 | 0.1070 | 1.433% | 1.649% | 0.07468 | 0.01569 |
| 2 / 2010 | 174,674 | 2,881 | LightGBM | 0.8367 | 0.1921 | 1.378% | 1.649% | 0.06739 | 0.01461 |
| 3 / 2011 | 184,159 | 2,563 | Logistic | 0.8009 | 0.1077 | 1.659% | 1.392% | 0.06455 | 0.01323 |
| 3 / 2011 | 184,159 | 2,563 | LightGBM | 0.8530 | 0.1943 | 1.679% | 1.392% | 0.05778 | 0.01237 |

LightGBM also has lower log loss and Brier score in every round. PR AUC is especially informative given default rates of only 1.39–1.65%. Calibration is not uniformly better: both models underpredict the average default rate in rounds 1–2 and overpredict in round 3; logistic's mean PD is closer to the observed rate in each round. Lower Brier score and log loss do not imply better calibration by every measure.

## Feature-level evidence

Permutation importance measures how much a test-sample metric falls when a feature is shuffled. Each round uses a stratified sample of 3,000 test rows and three permutations, with raw-model probabilities. The numbers below are drops in ROC AUC, not percentage-point changes in predicted PD.

| Feature | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|
| Equity ratio | 0.0055 | 0.0272 | 0.0166 |
| Accounts receivable ratio | 0.0230 | 0.0115 | 0.0117 |
| Tangible asset ratio | 0.0153 | 0.0239 | 0.0074 |
| Current asset ratio | 0.0133 | 0.0086 | — |

All three highlighted ratios appear among the top four LightGBM features by ROC-AUC permutation importance in every round. The current asset ratio is shown where it appears in the top four; the dash does not mean zero importance. Some PR-AUC permutation drops are negative, including the tangible asset ratio in round 2 and receivables ratio in rounds 2–3; importance depends on the metric and sample. Correlated features can share importance, and three permutations on 3,000 rows can be noisy.

The ALE curves add shape to those rankings. In round 3, the equity-ratio curve moves from a positive centered contribution at low values toward a flatter negative contribution at higher values. The tangible-asset curve also generally declines before flattening, with small reversals. This illustrates LightGBM's ability to represent changing slopes and thresholds. However, the equity ratio is explicitly constrained to have a non-increasing effect, and receivables is constrained to have a non-decreasing effect: those directions partly reflect modeling assumptions. The extreme left equity-ratio edge spans a wide bin, so its large endpoint difference should not be read as a precisely located threshold.

For logistic regression, the round 3 median-baseline marginal effect for tangible asset ratio varies from **−0.00324** at its training 5th percentile to **−0.00109** at the 95th percentile, in probability change per one standardized input unit. The equity-ratio effect varies from **−0.000227** to **−0.000211**. Logistic probability effects can vary across values even though its log-odds are linear. These local slopes are not numerically comparable to centered ALE levels: they use different reference populations and summarize different quantities. The LightGBM ALE uses the test sample and raw predictions, while the logistic table uses training quantiles and mean/median reference profiles.

To make a direct shape comparison, calculate ALE for **both models** on the same rows and probability scale. In particular, ALE values are centered contributions, not absolute PDs; standardized x-axis values are relative to each round's training scaler. Neither ALE nor permutation importance establishes causality.

## Presentation-ready message

> Across three future test years, LightGBM improved ROC AUC by 4.8–5.8 points and PR AUC by 8.5–8.7 points over logistic regression. Its explanations highlight equity, receivables, and tangible assets as influential, while showing that the model can represent varying response shapes across feature ranges. Average PD calibration still requires attention.

## Saved outputs and limitations

- Per-round model metrics and predictions: `model_artifacts/temporal_normalized_{logistic,lightgbm}/round_XX/<model>/`.
- Logistic explanations: `marginal_effects.csv`; LightGBM explanations: `ale.csv` and `permutation_importance.csv` in the corresponding model folders.
- Example plot: [Round 3 top-feature ALE curves](../reports/figures/temporal_normalized_lightgbm/round_03/lightgbm/ale_top_features.png).
- Logistic marginal effects are raw-model point estimates without confidence intervals. The runner uses validation data for both model selection/early stopping and calibration. The feature explanations use test data post hoc; avoid using them repeatedly for tuning if the same test years are to remain a final unbiased assessment.
- Earlier checks identified a reproducibility concern with the older v2 feature files and artifacts. This report describes the newly saved normalized run outputs; it does not independently establish end-to-end reproduction from raw data or validate the submission harness.
