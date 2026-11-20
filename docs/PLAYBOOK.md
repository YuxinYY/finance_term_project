# PLAYBOOK — phases, tournament rules, timeline

Today: Thu Oct 29, 2026. Final submission: **Mon Nov 16, 2026**. ~18 days.
The Oct 27 status report has passed; the remaining deliverable is the deck + code.

## Compressed schedule (D-007, adopted Oct 29): finish by Sun Nov 1

| Day | Phases | Agents |
|---|---|---|
| Oct 29 (eve) | 0 | setup, email draft (done) |
| Oct 30 | 1–2 | data-auditor + literature-scout (parallel) → feature-architect |
| Oct 31 | 3, 5 | challenger-scorecard, -riskcalc, -gbm (parallel) → referee (200 bootstrap draws) → calibration-specialist → red-team-roger |
| Nov 1 | 6–7 | harness-engineer → report-writer → literature-scout claim audit → red-team-roger |

GAM challenger and round 2 cut (D-008). The phase table below is the original 18-day plan, kept for reference.

## Phase plan (original)

| Phase | Dates | Agents | Exit criterion (must exist before moving on) |
|---|---|---|---|
| 0. Setup & questions to Roger | Oct 29–30 | orchestrator, humans | Data in `data/raw/`; email to Roger/TA sent with the questions below |
| 1. Data audit + literature base | Oct 30–Nov 2 | data-auditor, literature-scout (parallel) | `reports/data/01_profile.md`, `02_missingness.md`, frozen splits, ≥ 15 refs VERIFIED |
| 2. Features | Nov 2–4 | feature-architect, red-team-roger | `src/features/build.py`, `docs/FEATURE_SPEC.md` with expected signs |
| 3. Tournament round 1 (baselines + first challengers) | Nov 4–6 | all 4 challengers → referee | `scorecard_r1.md` |
| 4. Tournament round 2 (fixes, missing-data & feature variants) | Nov 6–8 | challengers → referee → red team | `scorecard_r2.md` |
| 5. Calibration round | Nov 8–10 | calibration-specialist → referee | `reports/calibration/summary.md`, final model chosen |
| 6. Harness | Nov 9–11 | harness-engineer | tests pass; optional test submission to TA ≈ Nov 9 (assignment allows one week early) |
| 7. Report | Nov 10–14 | report-writer, literature-scout (claim audit), red-team-roger | deck + technical report + explain-back |
| 8. Buffer & human rehearsal | Nov 14–16 | humans | every member can answer `explain_back.md` |

Rounds are time-boxed. If a round slips, cut variants, not the referee or red team.

## Questions to send Roger/TA now (Phase 0)
1. Which default window defines the holdout label (e.g. default within 12 months starting how
   many months after the statement date)? Our calibration level depends on it.
2. Is the holdout one future fiscal year or several?
3. May the harness use a frozen lookup of a firm's *training-data* history (e.g. prior-year
   ratios) when that firm appears in the holdout?
4. Is any external macro series (e.g. Italian unemployment) permissible at this point, or
   should we treat external data as out of scope? (Default assumption: out of scope.)
Log the answers in DECISION_LOG.md.

## Tournament rules (the referee enforces these)
- **One evaluation design for all**: expanding-window walk-forward by `fs_year`
  (`data/splits/walkforward.json`). Tuning only on inner, time-ordered splits inside the
  training window. Report pooled and per-year results, and existing vs new firms.
- **Standard output contract**: `oot_predictions.parquet` with `row_id, fold, pd`.
- **Metrics** (`src/common/metrics.py`): AUC/AR (power), log loss & Brier (probability accuracy),
  calibration-in-the-large, calibration slope/intercept, decile table, per-year stability,
  face-validity violations.
- **Comparison**: paired, firm-clustered bootstrap CIs for ΔAUC and Δlog-loss vs incumbent.
- **Promotion rule**: see `referee.md`. Ties → simpler, more explainable model.
- **Ensembles** are allowed only in the final round, only if they beat the best single model
  by the promotion rule, and only if the per-firm explanation story still works.

## What we expect to learn (hypotheses to test, not conclusions)
- H1: Leverage, profitability and liquidity ratios carry most of the signal (Altman/Ohlson/RiskCalc tradition).
- H2: Missingness is structural (abbreviated statements) and itself informative.
- H3: A monotone-constrained GBM beats a RiskCalc-style probit on power by a small margin; the
  gap may not be statistically significant once firm-clustered CIs are used.
- H4: Calibration level drifts with the cycle (2008–09, 2011–12); the choice of central tendency
  matters more for holdout log loss than the choice of algorithm.
- H5: Performance on new firms is lower than on seen firms; firm-history features widen the gap.
Each hypothesis gets a CONFIRMED / REJECTED entry in the decision log with evidence.

## Explaining the model to bankers (Roger's critique)
The final deck must show: (1) global driver curves (PD vs each ratio), (2) three worked example
firms with their top drivers in words ("High short-term bank debt relative to assets added X pp
to the PD"), (3) a PD-to-rating-grade table a loan officer could use for pricing bands,
(4) known limitations (selection bias, cycle sensitivity, unseen-firm performance).
