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
