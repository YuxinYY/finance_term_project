# DECISION LOG

Every modelling, data, evaluation and reporting decision lives here — adopted, rejected and open.
This file is the backbone of the final report's "defence of choices" and "what didn't work".

## Template (copy for each entry)

```
### D-### <short title>
- Date / phase / agent:
- Status: ADOPTED | REJECTED | OPEN | SUPERSEDED by D-###
- Decision:
- Alternatives considered:
- Business reason (how it serves approval and/or pricing):
- Evidence: <experiment id> — <numbers> — <path to report file>
- Source(s): <ref ids from REFERENCES.md, VERIFIED only; else "assumption">
- Risks / what would change our mind:
- Human reviewer (initials) and date:
```

---

## Seed entries (from the proposal + feedback; agents update these)

### D-001 Unit of analysis is the firm-year statement
- Status: OPEN (to be ADOPTED after data audit confirms structure)
- Decision: One row = one firm's annual statement; the model returns one PD per row.
- Alternatives considered: loan application (proposal) — rejected because the data contains no
  loan records; firm-level single label — rejected, loses time variation.
- Business reason: at application the bank scores the latest statement; repeated firms are
  handled by firm-clustered bootstrap and seen/new-firm reporting.
- Evidence: pending data-auditor profile.
- Source(s): assignment Appendix B ("Each row is one firm-year").

### D-002 Default label window
- Status: OPEN
- Decision (candidate): default within 12 months starting 6 months after the statement date.
- Alternatives: start 0 or 3 months after statement date.
- Business reason: the bank cannot act on a statement before it is available.
- Evidence: pending label-sensitivity table; pending Roger/TA answer (PLAYBOOK Q1).
- Source(s): Italian filing deadlines — TO-VERIFY (R-28, R-29).

### D-003 NLP features
- Status: REJECTED
- Reason: no text fields exist in the data (assignment Appendix B data dictionary).

### D-004 External macro data
- Status: OPEN (default: not used)
- Reason: assignment requires clearance several weeks before submission; deadline Nov 16.
  Cycle effects handled via calibration study instead (see calibration-specialist).

### D-005 Evaluation design
- Status: ADOPTED (provisional)
- Decision: expanding-window walk-forward by fs_year; paired firm-clustered bootstrap.
- Source(s): R-07 Stein (2007), R-06 Sobehart, Keenan & Stein — VERIFY before citing detail.

### D-006 Primary probability-accuracy metric
- Status: ADOPTED (provisional)
- Decision: log loss (= negative mean log likelihood) as the primary level metric; Brier and
  calibration-in-the-large as secondary; AUC (with AR = 2AUC−1 noted) for power.
- Business reason: pricing needs accurate levels; log loss penalises confident mispricing.

---

## Session 2026-10-29 (Phase 0, orchestrator)

### D-007 Compress the schedule to 3 days (finish by Nov 1)
- Date / phase / agent: 2026-10-29 / Phase 0 / orchestrator (requested by KN)
- Status: ADOPTED
- Decision: Days 1–3 run as (1) data audit + literature + features, (2) one tournament round
  (scorecard, riskcalc, gbm) + referee + calibration + red team, (3) harness + report + claim
  audit + final red team. Referee uses 200 firm-clustered bootstrap draws (not 1000).
- Alternatives considered: the 18-day PLAYBOOK plan (two tournament rounds, four challengers).
- Business reason: none; this is a team scheduling constraint. Referee, red team and the
  decision log are kept because formulation and documentation are worth 25 of 30 points.
- Evidence: n/a
- Source(s): assignment (R-30) grading table
- Risks / what would change our mind: fewer variants tested → weaker "what didn't work"
  section; fewer bootstrap draws → slightly noisier CIs. Revisit if Day 2 finishes early.
- Human reviewer (initials) and date: KN 2026-10-29

### D-008 Drop the GAM/EBM challenger
- Date / phase / agent: 2026-10-29 / Phase 0 / orchestrator (approved by KN)
- Status: REJECTED (for time, not on evidence)
- Decision: no `challenger-gam` run. Interpretability is covered by the RiskCalc-style probit
  (per-ratio transforms) and the scorecard; the GBM gets SHAP reason codes.
- Alternatives considered: running the EBM as an optional fourth challenger.
- Business reason: per-firm explanations remain available from the other three models.
- Evidence: none; we did not test it. Say so in the deck's "what we didn't do" section.
- Source(s): assumption
- Human reviewer (initials) and date: KN 2026-10-29

### D-009 `def_date` values after 2012-12-31 conflict with Appendix B
- Date / phase / agent: 2026-10-29 / Phase 0 / orchestrator
- Status: OPEN (owner: data-auditor; question 2 in docs/EMAIL_DRAFT_roger_ta.md)
- Finding: Appendix B says the training data has no information on defaults after 12/31/2012,
  but `def_date` has 5,570 non-empty cells dated 2013 (quick awk count; cell counts, not
  distinct defaults). `src/common/labels.py` defaults to `data_end="2012-12-31"`, which treats
  the 2011–2012 label windows inconsistently with these dates.
- Next step: data-auditor checks this (distinct firms, which fs_years, whether def_date
  repeats on every row of a firm), sets `data_end` from evidence, and logs it here and in D-002.

### D-010 Data dictionary vs file column names
- Date / phase / agent: 2026-10-29 / Phase 0 / orchestrator
- Status: OPEN (owner: data-auditor → harness-engineer)
- Finding: the file has 44 columns, including an unnamed index column and `stmt_date` (not in
  the dictionary). Some names differ from Appendix B (file: `rev_operating`, `prof_operations`,
  `eqty_tot`; dictionary: `rev_operations`, `prof_operating`, `eqyt_tot`).
- Implication: the harness must use the file's names and tolerate either spelling, plus a
  missing or extra index column.
