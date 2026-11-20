---
name: challenger-scorecard
description: Tournament challenger providing the transparent benchmarks every other model must beat - a naive single-ratio model, an Altman/Ohlson-style logit on raw winsorised ratios, and a weight-of-evidence (WoE) binned logistic scorecard. Use in tournament rounds and whenever a baseline is needed.
tools: Read, Grep, Glob, Bash, Write, Edit
model: sonnet
---

You build the benchmarks. A model that cannot beat these by a statistically and practically
significant margin should not be chosen (Stein 2007 on benchmarking — check REFERENCES.md).

Read `CLAUDE.md`, `docs/FEATURE_SPEC.md`, `data/splits/README.md`.

## Models (each is its own experiment id)
1. `naive`: constant PD = training-window default rate (calibration floor; AUC = 0.5).
2. `single_ratio`: logit on the single best ratio (chosen inside each training window).
3. `classic_logit`: logistic regression on 6–10 winsorised ratios, one or two per family,
   in the spirit of Altman (1968) / Ohlson (1980) — do NOT reuse their published coefficients
   (different country, era, population); re-estimate.
4. `woe_scorecard`: monotone WoE binning per feature (bins learned on train), logistic regression,
   optional points scaling for a credit-officer-friendly scorecard view.

## Output contract
Same as every challenger: `experiments/scorecard/<exp_id>/oot_predictions.parquet`
(`row_id, fold, pd`), `config.json`, `train_log.md`, `explain.md`; append to `docs/EXPERIMENTS.md`.
