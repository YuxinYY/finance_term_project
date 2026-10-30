---
description: Run one tournament round - challengers submit, referee scores, red team reviews, decision log updated
argument-hint: [round number] [optional focus, e.g. "missing data" or "calibration"]
---

Run tournament round $ARGUMENTS.

1. Read `docs/PLAYBOOK.md` (tournament rules), `docs/DECISION_LOG.md`, and the latest
   `reports/tournament/scorecard_*.md`. State the current incumbent and what this round tests.
2. Launch the challengers that have something new to test this round, **in parallel**, each
   with: the round focus, the incumbent's numbers, and the output contract. Typical set:
   `challenger-riskcalc`, `challenger-gbm`, `challenger-gam`, `challenger-scorecard`.
3. When all have written `oot_predictions.parquet`, launch `referee` with the list of new
   experiment ids. Wait for the scorecard.
4. If calibration was in scope, launch `calibration-specialist` on the top two models.
5. Launch `red-team-roger` on the new scorecard and the decision-log entries from this round.
6. Update `docs/DECISION_LOG.md` (ADOPTED / REJECTED with evidence) and `docs/AI_USE_LOG.md`.
7. Report to the user in ≤ 15 lines: incumbent before/after, the key numbers with CIs, what was
   rejected and why, open questions for the humans, and the proposed next round.
