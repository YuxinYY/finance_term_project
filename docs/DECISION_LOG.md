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
- Status: ADOPTED (data-auditor, 2026-10-30; evidence below)
- Decision: One row = one firm's annual statement; the model returns one PD per row.
- Alternatives considered: loan application (proposal) — rejected because the data contains no
  loan records; firm-level single label — rejected, loses time variation.
- Business reason: at application the bank scores the latest statement; repeated firms are
  handled by firm-clustered bootstrap and seen/new-firm reporting.
- Evidence: 1,023,552 rows, 237,711 firms, 0 duplicate (id, fs_year) pairs; no loan, application
  or pricing fields exist, so the proposal's loan-application unit contradicts the data
  (`reports/data/01_profile.md` §1). 12.6% of firms appear once, 42.6% in all 6 years (§2).
- Source(s): assignment Appendix B ("Each row is one firm-year").

### D-002 Default label window
- Date / phase / agent: 2026-10-30 / Phase 1 / data-auditor
- Status: ADOPTED (provisional assumption pending Roger/TA; see the OPEN item D-002a)
- Decision: y = 1 if def_date in [stmt_date + 6m, stmt_date + 18m). `stmt_date` exists and is
  31 Dec of fs_year on every row (no assumption needed). Exclusions: already_defaulted (default
  before window start), after_default (0 rows exist), censored (window ends after data_end
  2013-05-31; a row is censored even if a default is observed inside its incomplete window).
  Censored rows are DROPPED (not kept as 0).
- Alternatives considered: start 0 (2007-2010 default rate 1.07%, 4 folds incl. test 2011);
  start 3 (4 folds, but the 2011 fold depends on the 2013 dates, D-009). Keep-as-0 for censored
  rows (biases rates down: 2012 rows at start 0 would show 0.35%); make_labels' default of y=1
  for observed defaults in incomplete windows (100% rate in that year) — rejected.
- Business reason: accounts are approved within 120 days of year-end (180 if the statute allows)
  and filed within 30 days (R-28, R-29), so a decision using the FY t statement is made from
  about mid-year t+1; the bank needs the 12-month PD from then.
- Evidence: `reports/data/00_labels.md` §2, §5; `labels_sensitivity_summary.csv`. Start 6:
  650,373 usable rows, 9,163 defaults, 1.41% (2007-2010); start 0: 837,913 rows, 9,328 defaults,
  1.11% (2007-2011). Same years 2007-2010: 1.41% vs 1.07% (ratio 1.32). 4,237 rows have a
  default 0-6 months after fiscal year-end (excluded as already_defaulted at start 6).
- Source(s): R-28, R-29 (VERIFIED current text; 2007-2012 wording SECONDARY).
- Risks / what would change our mind: Roger/TA say the holdout uses start 0 → switch primary to
  `y_s0` (already in the panel) and use the S0-2011 fold; calibration level moves ~1/3.
- Human reviewer (initials) and date:

### D-002a OPEN: ask Roger/TA which window the holdout label uses
- Status: OPEN (EMAIL_DRAFT question 1)
- Why: the calibration level depends on it (1.41% vs 1.07% on the same years, D-002). The panel
  carries `y_s0`, `y_s3`, `y_s6` so any answer can be adopted without rebuilding data.

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
- **Update 2026-10-30 (data-auditor)**: Status → ADOPTED (provisional; Roger/TA not yet answered).
  `def_date` is firm-level (0 firms with >1 date or partially filled). 16,863 defaulting firms
  (44,725 rows carry the date); 1,416 firms have 2013 dates (5,570 rows, confirming the awk
  count), all between 2013-01 and 2013-05-31, at 229-377 firms/month vs 261-372 in Jan-May 2012:
  consistent with default tracking through May 2013, not stray errors. No firm has a statement
  for its default year; 0 rows dated after default.
  Decision: PRIMARY = dates as given, data_end = 2013-05-31 (last observed date). SENSITIVITY =
  2013 dates blanked, data_end 2012-12-31 (`y_s6_d2012` etc. in the panel).
  Effect: under the primary start-6 window the two treatments give identical labels for fs_year
  2007-2011; they differ only for start 3 (fs_year 2011, 2012) and start 6 (fs_year 2012, which
  is censored either way). No walk-forward fold depends on D-009.
  Evidence: `reports/data/00_labels.md` §1-2; `labels_def_by_year.csv`.

### D-010 Data dictionary vs file column names
- Date / phase / agent: 2026-10-29 / Phase 0 / orchestrator
- Status: OPEN (owner: data-auditor → harness-engineer)
- Finding: the file has 44 columns, including an unnamed index column and `stmt_date` (not in
  the dictionary). Some names differ from Appendix B (file: `rev_operating`, `prof_operations`,
  `eqty_tot`; dictionary: `rev_operations`, `prof_operating`, `eqyt_tot`).
- Implication: the harness must use the file's names and tolerate either spelling, plus a
  missing or extra index column.
- **Update 2026-10-30 (data-auditor)**: Status → ADOPTED. The dictionary lists 41 names; the file
  has those 41 (3 spelled differently) + `def_date` + `stmt_date` + `Unnamed: 0` = 44.
  `src/data/load_raw.py` maps both spellings and drops/keeps the index as `src_index`. Field
  meanings differ from the dictionary in places: `goodwill` behaves like value added (median
  EUR 420,706; ebitda <= goodwill on 99.99% of rows); `roa` = 100 x prof_operations / asst_tot
  (operating ROA, exact on 100%); `inc_financing` looks like the NET financial result (negative
  median); `eqty_corp_family_tot` is 100% empty. Evidence: `reports/data/01_profile.md` §1, §5.

---

## Session 2026-10-30 (Phase 1, data-auditor)

### D-011 Proposed change to `src/common/labels.py` (censoring of observed defaults)
- Date / phase / agent: 2026-10-30 / Phase 1 / data-auditor
- Status: OPEN (referee to decide; data-auditor did not edit shared code)
- Finding: `make_labels` marks a row `censored` only if NO default is observed in its window, so
  rows in an incomplete window that do show a default are `ok` with y=1. In fs_year 2012 (start 0)
  that is 650 rows, all defaults; in fs_year 2011 (start 6) 2,357 rows. Any fit on `status=="ok"`
  would then include years with a 100% default rate.
- Proposed change: `censored = win_end > data_end + 1 day` regardless of outcome, plus an
  `after_default` status for statements dated on/after def_date. Implemented on top of make_labels
  in `src/data/build_panel.py::label_one`; the panel's `label_status` already follows it.
- Evidence: `reports/data/00_labels.md` §2 column `censored_default_observed`.

### D-012 HQ_city codes are treated as ISTAT province codes → region / macro-area
- Date / phase / agent: 2026-10-30 / Phase 1 / data-auditor
- Status: ADOPTED (hypothesis with strong data support; code table [UNVERIFIED] for literature-scout)
- Decision: map HQ_city (codes 1-110) to 20 regions and 5 macro-areas (North-West, North-East,
  Centre, South, Islands) with `src/data/codes.py`; city code stays categorical.
- Evidence: exactly 110 codes 1-110; most frequent 15, 58, 17, 1, 63 = Milano, Roma, Brescia,
  Torino, Napoli under ISTAT coding (`01_profile.md` §3c). Region default rates 0.69%-1.89%;
  macro-area 1.30%-1.62% (Centre highest, not the South). 45 of 110 codes have < 30 defaults.
- Alternatives: raw HQ_city one-hot (too thin); no geography.
- Risks: if the codes are not ISTAT, the region labels are wrong but the grouping is still a
  fixed partition of codes; model use should rely on fold-fitted encodings, not the names.

### D-013 ATECO section mapping (Roger M7)
- Date / phase / agent: 2026-10-30 / Phase 1 / data-auditor
- Status: ADOPTED (mapping); grouping rule left to feature-architect
- Decision: `ateco_sector` is the 2-digit ATECO 2007 division; mapped to sections A-U by
  NACE Rev.2 division ranges (`src/data/codes.py`, [UNVERIFIED] reference for the ranges).
  `docs/2603_-CODES_ATECO_(003).pdf` is a 2020 list of non-suspended activities, not a full
  classification; it only confirms the code format.
- Evidence: 83 divisions, none in 64-66; sections O, P, U have < 30 defaults → pool
  (`01_profile.md` §3a, `profile_by_ateco_section.csv`). Section default rates: L 0.92%, C 1.67%,
  G 1.62%, F 1.45%, N 1.99%.

### D-014 Missing-value handling in the panel (no imputation)
- Date / phase / agent: 2026-10-30 / Phase 1 / data-auditor
- Status: ADOPTED
- Decision: all NA tokens (broad list in `src/data/load_raw.py`) and non-finite numbers → NaN.
  Candidate sentinels (-999, -9999, ...) are NOT masked: they occur only as plausible small euro
  amounts (e.g. 148 rows of inc_extraord = -999 or similar). Zeros are kept as values.
- Evidence: training uses only the empty string as a missing code (`01_profile.md` §4).
- Rejected: masking -999-type values (would delete real data).

### D-015 Frozen walk-forward splits
- Date / phase / agent: 2026-10-30 / Phase 1 / data-auditor
- Status: ADOPTED (files read-only)
- Decision: `data/splits/walkforward.json`: train fs_year <= t, test t+1, primary label, test
  years 2008, 2009, 2010 (2011, 2012 windows incomplete). row_id = id*10000 + fs_year.
  Fold test sizes 161,261 / 170,386 / 174,512; defaults 2,064 / 2,752 / 2,968; new-firm rows
  29,835 / 17,172 / 16,737. Sensitivity fold S0-2011 (start-0 label only).
- Risks: fold 1 trains on one year (2007); only 3 folds. If D-002 changes to start 0, use the
  S0-2011 fold too (write a v2 file, never overwrite).

### D-016 Missingness findings for feature-architect (H2)
- Date / phase / agent: 2026-10-30 / Phase 1 / data-auditor
- Status: H2 PARTLY CONFIRMED (evidence for the feature-architect's choice)
- Findings (`reports/data/02_missingness.md`): the bank/fin/AP debt split is missing as a block
  (all 3 ST fields together on 98.2% of rows where any is missing); the ST block is missing on
  11.8% of rows, more for small firms (13.6% at revenue < 8M vs 1.2% at >= 10M), persistent within
  firm (57.2% vs 5.9%), and doubles in 2008-2010 (16.3% vs 7.3%: vintage effect). Weakly
  informative: default rate 1.57% missing vs 1.38% present. `margin_fin` is missing on all 40,748
  negative-equity rows and `roe` when |ROE| >= 150% or equity <= 0: their missingness is a
  negative-equity flag (default rate 10.3% / 8.0% vs 1.06% / 0.93%), and both are fully
  reconstructable from eqty_tot, profit and fixed assets. `days_rec` 72% missing and not equal to
  365 x AR / revenue → recompute. Abbreviated statements (R-27, art. 2435-bis) are a plausible but
  unproven cause.
