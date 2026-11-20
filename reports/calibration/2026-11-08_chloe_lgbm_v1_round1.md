# Calibration reference study: Chloe's LightGBM, round 1 (2026-11-08)

Calibration-specialist. Code and every CSV are in `experiments/calibration/2026-11-08_chloe_lgbm_v1/`. File names
below drop the `2026-11-08_` prefix. This is a reference study: nothing here changes the harness.

## What was done

- **Model.** Chloe's monotone LightGBM was refit with her settings (production manifest `lightgbm_config`, her
  `MONOTONE_DIRECTIONS` copied verbatim, 27 of 171 features constrained, matching the manifest). It was trained on
  FY2007 and early-stopped on FY2008, as in her round 1. Best iteration 287. FY2009 test AUC on her label was
  **0.84309, against her reported 0.8431** (difference -0.000006, `fit_and_map_params.json`). We did NOT use the
  production model to evaluate anything, because it was trained on these years.
- **Files checked** (`file_checks.csv`): train is FY2007 only (144,251 rows, 0.873%), validation FY2008 only
  (161,383, 1.205%), test FY2009 only (170,549, 1.576%). No duplicate (id, fs_year) pairs. Her label, rebuilt from our
  panel as "default between 1 Jun fs_year+1 and 31 May fs_year+2", matches her file on 100% of rows in all three files.
- **Nothing was fitted on FY2009.** FY2009 labels are used only for scoring.

## The variants (FY2009), and which rows set each parameter

| Variant | Shape (slope) from | Level from |
|---|---|---|
| raw | none | the booster itself (≈ FY2007 rate) |
| (b) her Platt | FY2008 (also used for early stopping) | FY2008 Platt intercept, i.e. roughly the FY2008 rate |
| (c) ours, D-053 | FY2008 Platt slope (0.996) | intercept re-anchored so the mean PD on FY2007 training rows = FY2007 rate (0.873%) |
| (d) ours + rule e, D-071/D-072 | as (c) | prior shift to rule e = defaults in months 7-12 of the FY2008 cohort (Dec 2009-May 2010), annualised: **1.516%**. These are known on 1 Jun 2010, the FY2009 decision date in her design |
| isotonic (reference) | isotonic step map fitted on FY2008 (her code's option) | FY2008, or rule e for "iso + e" |

Sensitivities in the CSVs: rule b (FY2008 rate 1.205%), a D-073-style random-walk hedge (1.350%; its volatility
comes from a single year-on-year change, so it is weak), and a "clean slice" refit (FY2008 split by firm: early
stopping on one half, Platt slope on the other; slope 0.971).

## What to expect: the table (FY2009, every variant clipped to [0.03%, 99%])

Source: `metrics_fy2009.csv`; the log-loss differences and CIs come from `bootstrap_vs_b_platt.csv` (1,000 firm-clustered
paired draws, seed 0, `src/common/metrics.paired_bootstrap`). Δ log loss is in units of 1e-4, negative = better than (b).

| Label | Variant | AUC | Log loss | Δ LL vs (b), 1e-4 [95% CI] | Brier | Mean PD | Observed | Obs/pred | Slope | Intercept |
|---|---|---|---|---|---|---|---|---|---|---|
| Hers | raw | 0.8431 | 0.06459 | +11.42 [+10.19, +12.72] | 0.01373 | 0.970% | 1.576% | 1.62 | 1.008 | +0.554 |
| Hers | (b) her Platt (FY2008) | 0.8431 | 0.06345 | 0 (ref.) | 0.01361 | 1.186% | 1.576% | 1.33 | 1.012 | +0.326 |
| Hers | (c) ours: slope + FY2007 anchor | 0.8431 | 0.06459 | +11.46 [+10.23, +12.75] | 0.01374 | 0.969% | 1.576% | 1.63 | 1.012 | +0.554 |
| Hers | (d) ours + rule e | 0.8431 | 0.06278 | -6.63 [-8.47, -4.88] | 0.01352 | 1.581% | 1.576% | 1.00 | 1.012 | -0.004 |
| Hers | isotonic (FY2008) | 0.8427 | 0.06341 | -0.31 [-1.15, +0.57] | 0.01361 | 1.188% | 1.576% | 1.33 | 1.005 | +0.325 |
| Hers | isotonic + rule e | 0.8427 | 0.06275 | -6.91 [-8.91, -4.99] | 0.01352 | 1.577% | 1.576% | 1.00 | 1.005 | -0.001 |
| Ours (s6) | raw | 0.8343 | 0.06771 | +12.78 [+11.56, +13.93] | 0.01430 | 0.953% | 1.615% | 1.69 | 0.977 | +0.599 |
| Ours (s6) | (b) her Platt (FY2008) | 0.8343 | 0.06643 | 0 (ref.) | 0.01418 | 1.167% | 1.615% | 1.38 | 0.981 | +0.370 |
| Ours (s6) | (c) ours: slope + FY2007 anchor | 0.8343 | 0.06771 | +12.76 [+11.56, +13.95] | 0.01430 | 0.953% | 1.615% | 1.70 | 0.981 | +0.599 |
| Ours (s6) | (d) ours + rule e | 0.8343 | 0.06557 | -8.61 [-10.31, -6.88] | 0.01411 | 1.559% | 1.615% | 1.04 | 0.981 | +0.041 |
| Ours (s6) | isotonic (FY2008) | 0.8339 | 0.06639 | -0.39 [-1.32, +0.51] | 0.01419 | 1.169% | 1.615% | 1.38 | 0.973 | +0.370 |
| Ours (s6) | isotonic + rule e | 0.8339 | 0.06553 | -8.96 [-10.86, -6.93] | 0.01411 | 1.554% | 1.615% | 1.04 | 0.973 | +0.044 |

Her label: 170,549 rows, 2,687 defaults (1.576%). Our label (start 6, D-002): 170,386 rows, 2,752 defaults (1.615%).

## Reading it as a credit officer

1. **Ranking is untouched.** Every Platt or shift variant has exactly the raw AUC (0.8431 on her label, 0.8343 on ours).
   They are monotone maps, so they cannot reorder firms. Isotonic loses 0.0004 AUC because it puts thousands of
   firms into tied steps (116 steps).
2. **The problem is the level, not the shape.** The calibration slope is about 1.01 (her label) and 0.98 (ours)
   for every variant. The raw model priced FY2009 at 0.97% against an observed 1.58%: 1.62x too low. Her Platt only
   lifts the level to the FY2008 rate, so it is still 1.33x too low. Its PDs sit below the observed band in 7 of
   10 deciles, the riskiest decile included (6.9% predicted vs 8.9% observed). For pricing, these loans would be
   under-priced in a rising-default year (compare D-146).
3. **Our recipe without the level step (c) is no better than raw.** The Platt slope is about 1 and LightGBM's
   in-sample mean already equals the training rate, so (c) ≈ raw. All of our gain comes from the explicit level choice (d).
4. **Rule e hit the FY2009 level almost exactly** on her label (mean PD 1.581% vs observed 1.576%; obs/pred 1.00) and
   was 1.04 on ours. In 9 of 10 deciles the mean PD falls inside the observed rate's 95% binomial band (7 of 10 on our
   label). Her Platt manages 3 of 10 on both labels (`deciles_fy2009.csv`).
5. **Our label scores lower on every variant** (log loss about 0.0028 higher) because the window and the rows differ.
   The ordering of the variants is the same on both labels.

## What worked / what didn't

- Worked: the explicit level step (prior shift to rule e). It gives the best or equal-best log loss and Brier on both
  labels, with the slope left alone.
- Didn't help: re-estimating the shape. Platt slope 0.996 (0.971 on the clean slice), so it changes nothing.
- Isotonic + e ties (d) on log loss: -6.91 vs -6.63e-4 on her label, with heavily overlapping CIs. It loses 0.0004
  AUC. That loss is not statistically significant (CI [-0.0008, +0.0001]), but it is a cost with no matching gain.
  Its deciles are jagged (4 of 10 inside the band), and without the floor it would give a PD of exactly 0 to 5,324
  firms. All of its gain comes from the rule-e level, as for (d). We keep D-072's rejection.
- The clean-slice refit (early stopping and slope on separate halves) gives a slightly different model (AUC 0.8427).
  Its log loss is within 0.6e-4 of (d). Using FY2008 for both early stopping and the slope is therefore not a material leak.

## Caution for rounds 2 and 3: rule e is not safe at a turning point

Label-only preview on her window (`rule_preview_her_window.csv`, no model): rule e targets 1.516 / 1.82 / 1.99% for
test years 2009 / 2010 / 2011, against observed 1.576 / 1.649 / 1.392%. Level-only extra log loss (1e-4): rule e 0.12 / 0.88 / 10.19; rule b
5.25 / 0.17 / 2.16; pooled 11.59 / 6.29 / 0.07. Rule e wins only 2009. In 2011 it over-predicts by 43%. This matches D-071 (2011 turn) and is why the holdout level uses the D-073 hedge,
not rule e.

## Production parameters (her model, her label window)

`params_for_production.json`, status REFERENCE ONLY. On her window every cohort FY2007-2011 is complete (FY2011
ends 31 May 2013 = data end). Rates 0.873 / 1.205 / 1.576 / 1.649 / 1.392% (`rates_by_cohort_her_window.csv`). The pooled
rate recomputed from our panel is 1.3576%, matching her manifest's training rate exactly. The D-073 logic gives
anchor FY2011 = 1.392%, sigma 0.23 / yr (logit scale), **tau (h = 2) = 1.464%**, **delta = +0.077** vs 1.3576%. Slope 0.996
(round-1 estimate). The anchor intercept must be solved on the production model's own training rows (procedure in the
file), floor 0.03%, cap 99%. Other candidates are in `production_level_candidates.csv`.

**Flag on her production calibrator.** Per her manifest, the production Platt map was fitted on pooled round-1 to
round-3 out-of-time raw PDs: mean raw 1.105% vs observed 1.536%, so the Platt map lifts PDs by ≈1.39x. It is then
applied to a model refit on FY2007-2011, whose raw level already reflects those years. That double-counts the level
(D-053, alternative b). We did not run it, so the size of the effect is unmeasured.

## Flags (not fixed)

- **Harness rule conflict.** Her features include `annualized_sales_growth`, `annualized_change_in_roa` and
  `change_in_ar_turnover`. They need the firm's prior-year statement. Under CLAUDE.md §2.2 and D-020, the harness may
  only use the current record. These features need a frozen training-data lookup (email Q4, unanswered) or removal.
- **Label join.** All 170,549 test rows are in our panel. 163 of them (all her defaults, dated 4-29 Jun 2010) are
  "already defaulted" under our start-6 window (ours starts 30 Jun), so they drop out of our-label scoring. FY2007: 37
  rows; FY2008: 122. None of our usable rows are missing from her files (`label_join.csv`).
- **Deviations.** LightGBM 4.7.0 here; her version is unknown (her pyproject says >= 4.0). `subsample` 0.8 is inactive
  without `subsample_freq`, both in her code and here; reproduced as is. Our label is scored on the 170,386 rows
  common to both labels.
- **Limits.** One test year. Rule e was chosen after seeing 2008-2010 on our label (D-071), so its FY2009 hit is
  partly selection, not proof.
