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
- Source(s) (added 2026-11-02, lesson L3): RiskCalc v1 deliberately left macro forecasting outside
  the model (R-05, p. 41); firm-specific information dominates systematic information for
  middle-market default (R-10, Sec. 4.1, p. 21). Cost: Moody's does use a cycle adjustment and
  external data for the PD level (R-33, Sec. 2.4 p. 9, Sec. 3.3 p. 15), which we cannot.

### D-005 Evaluation design
- Status: ADOPTED (provisional)
- Decision: expanding-window walk-forward by fs_year; paired firm-clustered bootstrap.
- Source(s): R-07 Sec. 3.1-3.2 pp. 92-96 (walk-forward out-of-sample, out-of-time testing);
  R-01 Sec. 4.3 p. 28. (R-06 removed 2026-11-02: only LOCATED, not read.)

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
- Source(s) added 2026-11-02 (literature-scout): R-53 ISTAT code list (110-province coding 2010-2016; ripartizione geografica), VERIFIED; spot check N-53. That the provider uses ISTAT codes = inference from data.

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
- Source(s) added 2026-11-02 (literature-scout): R-54 pp. 10, 51-54 (Ateco 2007 = NACE Rev. 2 to 4 digits; section ranges), VERIFIED.

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

### D-017 Drop the "one-sentence Basel II link" requirement
- Date / phase / agent: 2026-10-30 / Day 1 / orchestrator (information from KN)
- Status: ADOPTED
- Decision: no Basel II sentence in the deck's Business Problem section. The instruction was
  hidden white-ink text in the assignment PDF, placed to catch teams that let AI read the brief
  for them. Removed from CLAUDE.md and docs/REPORT_TEMPLATE.md.
- Alternatives considered: keeping the sentence because it is verified (R-15 paras 211/285/447/452).
  Rejected: it would signal that the brief was read by AI rather than by the team.
- Business reason: none; this is about honest documentation (Documentation 10 pts, AI use).
- Evidence: KN confirmed the hidden text. The visible assignment text does not ask for it.
- Source(s): assignment R-30 (visible text)
- Risks / what would change our mind: none. R-15 remains in REFERENCES.md and may be cited only
  where it genuinely supports a choice (e.g. the 0.03% PD floor in the harness).
- Human reviewer (initials) and date: KN 2026-10-30

---

## Session 2026-10-30 (Phase 2, feature-architect)

All numbers below come from files written by `python -m src.features.univariate`,
`-m src.features.core_selection` and `-m src.features.history_tradeoff` (training rows fs_year
2007-2009 only; fs_year 2010 never read). Spec: `docs/FEATURE_SPEC.md`; evidence:
`reports/features/univariate.md` and CSVs.

### D-018 Feature library v1: ratio families, core set vs full set
- Date / phase / agent: 2026-10-30 / Phase 2 / feature-architect
- Status: ADOPTED
- Decision: `src/features/build.py` (`fit` / `transform` / `design_matrix` / `expected_signs`),
  23 numeric features + flags and 3 categoricals in 8 families (size, leverage, liquidity,
  profitability, coverage, activity, asset structure, + flags). Ratios only; size enters once as
  `log_assets`. **Core set for probit/logit (10 + 3 categoricals):** log_assets, equity_ratio,
  st_debt_ratio, cash_ratio, roa_net, interest_coverage, interest_to_assets, ar_days,
  tangible_ratio, neg_equity + ateco_group, legal_group, macro_area. **Full set (26)** for GBM.
  The feature set is fixed for every fold; only parameters are refitted per fold (Roger M12).
- Alternatives considered: 10 core variants on an inner split (fit 2007-08, score 2009):
  A (coverage only) AUC 0.7968 / LL 0.06965 with log_assets flipped; B (cfo_to_debt) 0.7950 with
  cfo_to_debt flipped (+0.231); E (coverage + interest_to_assets, ADOPTED) 0.8006 / 0.06940, no
  flips; J (E + bank debt + flag) 0.8010 / 0.06937 but bank_debt_ratio flipped → not adopted.
- Business reason: one or two interpretable ratios per family give a credit officer a reason
  code per family (leverage, profitability, liquidity, debt service) and serve both ranking
  (approve/decline) and pricing (level, via the challengers' calibration step).
- Evidence: `reports/features/core_selection.csv`; univariate AUCs `univariate_stats.csv`
  (e.g. cfo_to_debt 0.248, roa_net 0.256, equity_ratio 0.267, interest_coverage 0.276,
  st_debt_ratio 0.706; AUC < 0.5 = PD falls as the ratio rises).
- Source(s): R-01 Sec. 3.1 p. 13 (families incl. growth; one per family; too many ratios
  overfit), Sec. 3.1 p. 14 (industry indicators in the probit); R-05 p. 9 (ratios + size into a
  probit).
- Risks / what would change our mind: a challenger's within-fold selection repeatedly
  dropping a core feature; a core sign flip in the tournament models.
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): tangible_ratio / asset-structure family. Data: within-industry AUC 0.56-0.73 (SC1 T16). Mechanism: R-48 Sec. 1 p. 1 (tangible assets raise creditor recovery in default and support borrowing; US public firms; their proxy is not our balance-sheet ratio). The lower-PD link for Italian private firms is OUR EMPIRICAL FINDING; no study located (REFERENCES Contradiction 11).

### D-019 Ratio pathologies: safe division rules, winsorisation, signed log
- Date / phase / agent: 2026-10-30 / Phase 2 / feature-architect
- Status: ADOPTED
- Decision: (1) most ratios divide by total assets (≥ EUR 1.5MM by construction); (2) bounded
  alternatives instead of fragile ratios: equity/assets (not debt/equity), profit/assets (not
  ROE); (3) where a denominator can be zero, the economically correct limit is assigned and
  documented per feature: no interest expense → coverage = +inf if EBITDA > 0, 0 if EBITDA ≤ 0;
  no debt → cfo_to_debt = +inf if cash flow > 0 else 0; zero revenue with receivables →
  ar_days = +inf; debt_st = 0 → current ratio +inf; negative denominators that are data errors
  (debt < 0, revenue < 0, COGS ≤ 0) → NaN; (4) winsorise at the training-window p1/p99 (params),
  ±inf land on the bound; (5) ratios with unbounded denominators (current_ratio,
  ebitda_margin, interest_coverage, ar_days, ap_days) are put on a signed-log scale after
  winsorising (monotone: AUC and tree splits unchanged, linear models not dominated by tails;
  e.g. the interest_coverage p1/p99 bounds are about −364 and 5,782).
- Alternatives considered: dropping rows with bad denominators (not allowed: a PD for every
  row); NaN for every zero denominator (loses the meaning "no interest to pay").
- Evidence: bounds in `reports/features/params_fold3_example.json`; unit tests
  `src/features/test_build.py::test_infinite_and_zero_denominators`, `test_messy_row`.
- Source(s): assumption (standard practice); R-01 Sec. 3.1 fn. 8 (transforms reduce the impact
  of noisy ratios) supports bounding outliers in general. Added 2026-11-02 (L3): Moody's
  RiskCalc v1 truncated raw ratios at the 2nd/98th percentiles in estimation and testing
  (R-05, p. 55, fn. 42); our p1/p99 is our own choice, not Moody's.
- Human reviewer (initials) and date:

### D-020 Growth / lagged (firm-history) features
- Date / phase / agent: 2026-10-30 / Phase 2 / feature-architect
- Status: REJECTED for v1 (re-open only if Roger/TA answer email Q4 "yes" AND Day-2 time allows)
- Decision: no growth or lag features. The current record has no prior-year fields, and the
  harness may not use a firm's other holdout rows (CLAUDE.md §2.2). The only permissible route
  is a frozen lookup of training-data statements (email Q4, unanswered).
- Evidence (inner split inside the training window, `reports/features/history_tradeoff.csv`):
  core logit vs core + 4 lag deltas (equity_ratio, roa_net, st_debt_ratio, log_assets) from a
  frozen prior-year lookup, fit on 2008 (2007 lookup), scored on 2009 (2008 lookup): AUC all
  rows 0.7992 → 0.8012, log loss 0.06896 → 0.06831; seen firms (148,279 rows) AUC 0.8075 →
  0.8096; new firms (22,107 rows) 0.7390 → 0.7379. Gain ≈ +0.002 AUC on seen firms only,
  inconsistent treatment of new firms, and a dependence on an unanswered rule question.
- Side finding for H5: even without history features, new-firm AUC (0.739) is well below
  seen-firm AUC (0.807) on this inner split.
- Source(s): R-01 Sec. 3.1 p. 13 lists growth as a family (two-sided sign); we cannot build it
  under the harness rules without approval. Added 2026-11-02 (L3): support — levels beat trends,
  and the latest year beats a 3-year average (R-05, pp. 38-40); cost — RiskCalc 4.0 France still
  gives growth inputs (sales growth, change in ROA, U-shaped) 13.0% of the weight
  (R-33, Table 4 p. 14, Fig. 5 p. 13).
- Risks: leaves a small amount of power unused for seen firms.

### D-021 Missing-data treatment per field (Roger G2)
- Date / phase / agent: 2026-10-30 / Phase 2 / feature-architect
- Status: ADOPTED
- Decision: (a) **identity reconstruction first** (`_reconstruct`, row-wise): debt_st =
  asst_current − wc_net; eqty_tot = margin_fin + fixed assets; prof_operations = rev − COGS or
  roa·assets/100; profit / ebitda / cf_operations from the cf identity; asst_tot ≤ 0 or missing →
  sum of asset components. (b) **Debt-split block** (bank/fin/AP, missing as a block): flag
  `debt_split_missing` + identity-bounded imputation, component = train sector-median share of
  the total × the firm's own debt_st / debt_lt (component ≤ total). (c) **NaN left for trees**
  in `transform(impute=False)`; `impute=True` fills residual NaN with the train sector median.
  `roe`, `margin_fin` (blank exactly when equity ≤ 0), `days_rec` (72% missing, inconsistent) and
  `eqty_corp_family_tot` (100% empty) are not used; information carried by equity_ratio,
  neg_equity, roa_net, ar_days instead.
- Alternatives considered: plain sector-median imputation of the ratio; dropping block-missing rows.
- Evidence (`reports/features/missing_treatment.csv`, `univariate.md` §3, §7): block missing on
  15.67% of 2007-09 training rows, default rate 1.39% vs 1.29% present (weak). AUC on the
  block-missing rows themselves: bank_debt_ratio 0.553 (share × total) vs 0.462 (sector median);
  ap_days 0.612 vs 0.538. All rows: bank_debt_ratio 0.574 vs 0.568; ap_days 0.595 vs 0.584. The
  share-based rebuild carries real information from the firm's own totals; the median does not.
  neg_equity flag: 3.78% of rows, default rate 9.74% vs 0.97%.
- Source(s): D-016 (data-auditor evidence); R-27 (abbreviated statements, plausible cause; unproven).
- Risks: holdout missing rate for the block may differ (vintage effect, D-016); the GBM
  challenger should compare native NaN vs imputed.

### D-022 Where the data contradict the expected sign
- Date / phase / agent: 2026-10-30 / Phase 2 / feature-architect
- Status: ADOPTED (handling) / OPEN for log_assets (watch in tournament)
- Findings (train 2007-09, `univariate_stats.csv`, `core_logit_signs.csv`, `deciles.csv`):
  1. `asset_turnover`: a priori − (idle assets), data inverted-U (decile default rates 1.48%,
     0.72%, … 1.71%, … 1.22%; AUC 0.504) → expected sign set to **none** (no monotone constraint),
     not in core. Reason: high turnover also marks thin-margin trading firms.
  2. `log_assets`: univariate − as expected (D1 1.57% → D10 1.08%, AUC 0.477), but its
     conditional coefficient in the core logit is ≈ 0 and its sign varies (+0.013 standardised on
     2007-09; − on the 2007-08 inner fit, set E). Kept with sign −: size is a RiskCalc family
     and the constraint costs nothing if the true conditional effect is ≈ 0. Challengers report
     it if their fitted effect is positive.
  3. `cfo_to_debt` (best univariate, AUC 0.248) flips to + next to roa_net (Spearman 0.80) →
     full set only, not core.
  4. `debt_split_missing`: + but tiny (AUC 0.505) → full set / variant, not core.
  5. `extraord_to_assets`: two-sided (U-shape: D1 2.80%, D10 2.01%) → sign none, as expected.
- Source(s): R-01 Sec. 3.1 p. 13 (expected signs; growth two-sided).

### D-023 Categorical encoding (Roger M7) and code-table spot check
- Date / phase / agent: 2026-10-30 / Phase 2 / feature-architect
- Status: ADOPTED
- Decision: `ateco_group` = ATECO 2007 section (codes.py) with sections having < 30 training
  defaults pooled into OTHER, learned per fold in `fit`; `legal_group` = legal_struct with the
  same rule; `macro_area` = HQ_city → region → 5 NUTS-1 areas; missing / unseen / unparsable
  codes → OTHER. One-hot in `design_matrix` with the largest training level as reference
  (C, SRL, North-West in fold 3). Never numeric codes.
- Evidence (fold-3 training window, `categorical_rates.csv`): pooled sections B (20 defaults),
  D (23), O (1), P (8), Q (26), S (22), U (7); legal forms SAA (1), SRS (0); HQ_city missing (6).
  Pooling is fold-dependent: fold 1 (2007 only) pools more sections.
- Spot check of `src/data/codes.py`: all 110 province codes and the 21 NACE/ATECO section
  division ranges match the standard ISTAT 2009-2016 province list (incl. 93 Pordenone,
  94 Isernia, 95 Oristano, 96 Biella, 97 Lecco, 98 Lodi, 99 Rimini, 100 Prato, 101 Crotone,
  102 Vibo Valentia, 103 VCO, 104-107 Sardinian provinces, 108 Monza-Brianza, 109 Fermo,
  110 BAT) and the NACE Rev. 2 section ranges. **No errors found.** This check is from
  the agent's own knowledge, not a document read in this session, so the tables remain
  [UNVERIFIED] until literature-scout checks an ISTAT / Eurostat source.
- Alternatives: region (20 levels; several < 30 defaults); raw division one-hot (36 of 83 < 30 defaults).
- Risks: OTHER mixes low-risk (D, Q) and high-risk (U) sections; small share of rows.
- Source(s) added 2026-11-02 (literature-scout): Provinces: R-53 (VERIFIED, spot check N-53). ATECO sections: R-54 pp. 10, 51-54 (VERIFIED).

### D-024 Rejected features, and the `taxes` field is pre-tax profit (D-010 addendum)
- Date / phase / agent: 2026-10-30 / Phase 2 / feature-architect
- Status: REJECTED (features); finding ADOPTED as a data-derived hypothesis
- Finding: `taxes` = prof_operations + inc_financing + inc_extraord + prof_financing on 99.7%
  of 2007-09 rows, i.e. **profit before tax**, not the tax charge; Spearman with profit 0.95.
  It is consistent with D-010: `inc_financing` enters EBT as the net financial result.
- Rejected (evidence `reports/features/rejected_candidates.csv`): tax_to_assets (AUC 0.259,
  Spearman 0.96 with roa_net: duplicate); recomputed ROE (undefined for negative equity;
  0.89 with roa_net); file `roe` (truncated, missingness = negative equity); file `days_rec`
  (AUC 0.508, 72% missing); margin_fin/assets; inc_financing/assets (Spearman −0.92 with
  interest_to_assets; meaning is a hypothesis); goodwill(=value added)/assets (cannot be
  explained under its file name); log_revenue (second size measure); liab_lt_emp/assets
  (no expected sign, AUC 0.510); debt_lt/assets (ambiguous sign, AUC 0.467).
- Raw euro amounts (Roger M9): exp_financing enters only as interest_coverage and
  interest_to_assets; cf_operations only as cfo_to_debt; inc_financing dropped.
- D-003 (NLP) re-confirmed REJECTED: no text fields in the data. D-004: no macro features; the
  cycle is left to the calibration-specialist.

### D-025 Collinearity rule (Roger M8)
- Date / phase / agent: 2026-10-30 / Phase 2 / feature-architect
- Status: ADOPTED
- Decision: core set must have every pairwise |Spearman| < 0.70 and every VIF < 5 (VIF = diag of
  the inverse correlation matrix of winsorised, imputed features); a violating feature is dropped
  from the core (kept in the full set). In the full GBM set high VIF is tolerated (trees do not
  invert the design) but redundancy is reported and SHAP explanations are reported by family.
- Evidence (`vif.csv`, `univariate.md` §6): core max |rho| 0.67, max VIF 1.98. Full set: st_debt_ratio
  12.5, roa_net 6.6, st_debt_share 6.5, roa_oper 6.3, wc_ratio 5.7, equity_ratio 5.2. Pairs ≥ 0.70:
  wc_ratio–current_ratio 0.91, roa_net–cfo_to_debt 0.80, interest_coverage–cfo_to_debt 0.76,
  roa_net–roa_oper 0.76. Actions: tax_to_assets dropped (Spearman 0.96 with roa_net, `rejected_candidates.csv`);
  cfo_to_debt, roa_oper, current_ratio kept out of core.
- Source(s): R-01 Sec. 4.2 p. 26 (VIFs used to limit collinearity). Added 2026-11-02 (L3):
  RiskCalc 4.0 France reports Spearman correlations on transformed inputs up to 0.66 and VIFs
  <= 3.04, and treats VIF 4-10 as the warning zone (R-33, Sec. 4.3, Tables 7-8, fn. 13, p. 18);
  our |rho| < 0.70 and VIF < 5 rules sit inside that practice. Panel dependence widens standard
  errors but left RiskCalc's conclusions unchanged (R-01, Sec. 3.4.2, p. 22), hence our
  firm-clustered bootstrap.

---

## Tournament round 1: challenger-scorecard (merged by orchestrator from experiments/scorecard/decisions.md)

### D-030 Benchmark family and pre-declared settings (SC-01..SC-04)
- Date / phase / agent: 2026-10-31 / Round 1 / challenger-scorecard
- Status: ADOPTED
- Decision: four benchmarks on the frozen walk-forward folds, each refit per fold on `fs_year <= t`, `label_status == ok`, with `B.fit/B.transform(impute=True)`: SC-01 single best core numeric ratio (chosen by in-window |AUC-0.5|; it was roa_net in all 3 folds) -> logit; SC-02 unpenalised logit on the 10 core features + 3 categoricals; SC-03 monotone-WoE scorecard; SC-04 Z'-style logit on wc_ratio, equity_ratio, roa_oper, asset_turnover.
- Tuning: none needed. GLMs are unpenalised (no hyper-parameter). SC-03 settings were declared before looking at results and are identical in all folds, including fold 1 (single training year): 20 initial quantile bins, pool-adjacent-violators to monotone in the expected-sign direction, minimum final bin 2% of training rows. No inner validation split was used; no fold uses a default instead of a tuned value because nothing was tuned.
- Evidence: `experiments/scorecard/SC-0*/metrics.md`. Pooled AUC 0.746 / 0.803 / 0.817 / 0.770; log loss 0.0713 / 0.0672 / 0.0666 / 0.0699.
- Source(s): Stein (benchmarking logic, course); Altman lineage only via R-32; Ohlson (R-12) cited only as existence of logit lineage (content unread).
- Source(s) added 2026-11-02 (literature-scout): R-52 Sec. 2 (WoE definition), Sec. 1/3.1 (small-bin problem) (VERIFIED preprint); R-44 (PAV); R-07, R-17 Conclusions MS p. 25. Bin settings (20 bins, 2% minimum) = pre-declared design (assumption). Siddiqi R-50 not read. ('Stein (course)' is not a reference ID.)

### D-031 SC-03 (WoE scorecard) is the benchmark to beat; SC-02 is the transparent logit reference
- Status: ADOPTED (round-1 provisional; the referee decides)
- Evidence: pooled AUC 0.8166 vs 0.8027 (SC-02); AUC higher in all three folds (+0.014, +0.011, +0.017); log loss 0.0666 vs 0.0672. Brier 0.0141 vs 0.0140 (overall probability accuracy, essentially tied). Calibration-in-the-large is poor for all (see D-033), so level comparisons between SC-02 and SC-03 are mostly about shape, not level. Paired bootstrap is the referee's job.
- Interpretation (hypothesis, not tested): the AUC gain of SC-03 over SC-02 comes from non-linear shapes in the ratios (e.g. roa_net default rate 9.4% in the lowest bin, falling to 0.43% at the top; the logit is linear in a winsorised ratio). A GAM/GBM challenger should beat SC-03 if shape is the only story.

### D-032 Sign check (Roger face-validity)
- Status: ADOPTED (reported)
- Result: SC-03 betas (on risk-oriented WoE) positive in all 13 features in all 3 folds; monotone bins enforce the numeric direction by construction. SC-01, SC-04: no violations. SC-02: one violation, `log_assets` has a positive coefficient in fold 1 (+0.049) and fold 3 (+0.012, SE 0.017, not significant) against expected sign -; negative in fold 2. Cause: size is correlated with the leverage/profit ratios and carries almost no marginal signal once they are in (univariate AUC 0.4765). Not "fixed" by dropping or sign-constraining; reported. The SC-03 `log_assets` WoE is monotone decreasing by construction with beta 0.43.
- Source(s): FEATURE_SPEC expected-sign table.

### D-033 Level calibration: all benchmarks under-predict (rising default rate, no cycle information)
- Status: OPEN (hand-off to calibration-specialist)
- Evidence: pooled mean PD vs observed default rate: SC-01 1.21% vs 1.54%; SC-02 1.08%; SC-03 1.10%; SC-04 1.17%. Calibration intercepts +0.25 to +0.45. Train-window default rates (1.0%, 1.1%, 1.3%) trail the test-year rates (1.28%, 1.62%, 1.70%) in every fold. Calibration slopes are near 1 (0.95 to 1.10), so ordering and spread are fine and the miss is a level shift. The logit models also lose a little level because the GLM fits the training years' mean exactly, which by construction was lower than the next year's.
- Not fixed here: any level adjustment is the calibration-specialist's decision (central tendency / cycle). The fix is not tuned on test years.

### D-034 SC-04 Altman-lineage benchmark is a proxy, not Z'
- Status: ADOPTED with caveat
- Decision: Z'-style ratios are rebuilt from the feature library: working capital/TA (wc_ratio), equity/assets (stands in for both retained earnings/TA and book equity/liabilities; retained earnings is not in the data), operating ROA (EBIT/TA proxy), sales/TA (asset_turnover); re-estimated as a logit, no published coefficients used. It is lineage only (R-32: original Z needs market equity; Z' replaces it with book equity); we do not claim it reproduces Altman's model. Pooled AUC 0.770, below SC-02 (0.803) with 4 vs 13 inputs.

### D-035 REJECTED / not done
- Points scaling is illustrative (PDO 20, 600 points at 50:1 good:bad odds): conventional choices, not a bank policy. Reported, not tuned.
- Isotonic/other post-hoc recalibration of the benchmarks: not done (calibration-specialist's scope).
- Penalised (L1/L2) logit: not used; with 475k rows and 13 features, no overfit visible (calibration slope 0.95 to 1.03).

---

## Tournament round 1: challenger-riskcalc (merged by orchestrator from experiments/riskcalc/decisions.md)

# challenger-riskcalc decisions (for the orchestrator to merge into docs/DECISION_LOG.md)

All numbers are from files written by code in `experiments/riskcalc/` (paths given). Outer metrics
are diagnostic and come from `src/common/metrics.py`; the referee's scorecard is authoritative.

### D-040 RiskCalc-style architecture and its in-window estimation recipe
- Date / phase / agent: 2026-10-31 / Phase 3 (tournament round 1) / challenger-riskcalc
- Status: ADOPTED (as the RiskCalc challenger; promotion is the referee's call)
- Decision: PD = F(Phi(sum_i beta_i * Phi^-1(T_i(x_i)) + sum_j gamma_j I_j)), with I_j = ATECO
  section-group indicators. Everything is refitted in each fold's training window:
  (1) T_i = the ratio's default rate by training quantile bin (K = clip(train_defaults // 30, 10, 50):
  45 bins in fold 1, 50 in folds 2–3), shrunk toward the base rate with a Beta prior of 2
  pseudo-defaults, made monotone in the FEATURE_SPEC direction by weighted pool-adjacent-violators,
  interpolated on the probit scale between bin centres and flat beyond the outer centres; a missing
  ratio gets its own shrunk empirical default rate. (2) statsmodels Probit. (3) F = 100 quantile bins
  of the probit index → shrunk rates → PAV (increasing) → one knot per PAV block, linear
  interpolation on the probit scale, slope-1 extrapolation beyond the outer knots. Features come from
  `B.transform(impute=False)` (NaNs kept, so missingness is priced inside T_i, not imputed).
- Alternatives considered: imputation + linear probit on raw ratios (that is the scorecard
  challenger's territory); isotonic step function for F (ties ranks in the tails; rejected in
  favour of interpolation); 1-D logistic recalibration as F (inner check "F = identity", i.e. the
  raw probit, gave AUC 0.8169 / log loss 0.0681 vs 0.8170 / 0.0680 with F: F changes almost nothing
  in-window; kept because it is part of the published architecture and gives calibration-specialist a
  fitted hook).
- Business reason: each T_i is a driver curve a credit officer can read ("default rate vs this
  ratio"); flat tails stop one extreme or manipulated ratio from dominating the PD (R-01 fn. 8);
  monotone curves make the reason codes face-valid.
- Evidence: RC-01 / RC-02 / RC-03 below; inner checks in `experiments/riskcalc/inner_validation.md`.
- Source(s): R-01 Sec. 3.1 p. 14 (functional form, industry indicators, "closely related to a GAM"),
  fn. 8 (transforms reduce the impact of noisy ratios), p. 13 (ratio families). The specific bin
  rule, shrinkage prior, PAV and the shape of F are **our choices (assumption)**: the paper's T_i
  and F estimation details are in R-02, which we have not read. Do not say "as in RiskCalc" for them.
- Risks / what would change our mind: if the referee finds a monotone-GBM with clearly better power
  and comparable explanation, the RiskCalc model becomes the benchmark rather than the champion.
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-44 Sec. 4.1/4.5 (PAV = optimal monotone fit); R-05 pp. 55-57. Bins, prior and F remain our design; do not say 'as in RiskCalc' for these details.

### D-041 In-window sign rule drops `neg_equity` and `bank_debt_ratio`
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-riskcalc
- Status: ADOPTED (RC-03)
- Decision: after the probit is fitted on a fold's training rows, any T_i with coefficient <= 0 is
  dropped (most negative z first) and the probit refitted. The rule runs inside every fold (no test
  data used). It dropped the same two terms in all three folds (`RC-03/train_log.md`):
  `T_bank_debt_ratio` (z = −1.24, −3.17, −2.56 in folds 1–3) and `T_neg_equity` (z = −0.81, −1.43,
  −1.78). **Selected set is identical across folds** (10 ratios: log_assets, equity_ratio,
  st_debt_ratio, cash_ratio, roa_net, interest_coverage, interest_to_assets, ar_days,
  tangible_ratio, ap_days) — answers Roger's "does each year's model have different variables?": no.
- Why the signs flip (explanation, not a data error): (a) `neg_equity` is a deterministic function
  of `equity_ratio` < 0; the nonparametric T_equity_ratio already prices that region (its bottom bin
  has a 16.0% default rate in fold 3, `explain.md` §1), so the flag is redundant and picks up a small
  negative offset. The flag was designed for linear models (FEATURE_SPEC: "adds a jump that
  winsorised equity_ratio cannot show"), which the RiskCalc transform does not need. (b)
  `bank_debt_ratio`: univariately + (AUC 0.579), but conditional on short-term liabilities, equity
  and interest/assets, more bank debt means less of the same leverage is trade/other debt; ap_days
  (trade payables) keeps a strong + coefficient. Treated as collinearity inside the leverage family,
  not as a finding that bank debt is safe.
- Alternatives considered: keep them with negative signs (RC-02; face-validity violation in the
  reason codes); force positivity by constrained estimation (more complex, same effect).
- Evidence: RC-02 vs RC-03 pooled AUC 0.8194 vs 0.8192, log loss 0.0658 vs 0.0658 (the rule costs
  nothing measurable); `RC-02/train_log.md`, `RC-03/train_log.md`.
- Source(s): R-01 Sec. 3.1 p. 13 (expected signs per family; at least one ratio per family);
  dropping on sign is our rule (assumption).
- Risks: bank debt is the lender-facing leverage measure a credit officer may ask about; answer: it is
  represented by the leverage family (equity_ratio, st_debt_ratio) and the interest burden.
- Human reviewer (initials) and date:

### D-042 Missing values priced inside T_i; H2 evidence from the debt-split block
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-riskcalc
- Status: ADOPTED (method); H2 → PARTLY CONFIRMED for this model
- Decision: a missing ratio gets its own shrunk empirical default rate inside T_i (no imputation, no
  separate indicator). RC-02 tested H2 by adding the two block ratios (`bank_debt_ratio`, `ap_days`,
  missing on 15.7% of fold-3 training rows).
- Evidence (fold-3 params, `RC-03/explain.md` §1): T_ap_days(missing) = 1.39% vs base rate 1.30%
  (68,730 rows): the block's missingness is only weakly informative, matching D-016. In contrast the
  rare missing cases of `ar_days` (1,109 rows; revenue undefined/non-positive) carry T = 5.94%, and
  `interest_coverage` missing (321 rows) 1.90%: missingness *is* informative where it marks an
  abnormal statement. Adding ap_days lifted pooled AUC 0.8175 → 0.8194 and per-year AUC in every
  year (2008 0.8311→0.8328, 2009 0.8144→0.8174, 2010 0.8102→0.8115); log loss 0.0660 → 0.0658.
  That gain comes from the ratio's values, not from the missing bin.
- Alternatives: median imputation + `debt_split_missing` flag (the GLM path; scorecard challenger).
- Source(s): assumption (method); D-016 (data-auditor) for the block pattern.
- Risks: paired bootstrap needed before claiming the +0.002 AUC is real (referee).
- Human reviewer (initials) and date:

### D-043 Probit vs logit link
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-riskcalc
- Status: ADOPTED (probit), logit REJECTED as a separate experiment (near-identical)
- Evidence: inner validation (train 2007–08 → validate 2009, inside fold-3 window,
  `inner_validation.md`): probit AUC 0.8170 / log loss 0.0680; logit 0.8153 / 0.0679; PD rank
  correlation (Spearman) 0.9973. Small AUC edge for probit is expected because the T_i enter on the
  probit scale.
- Source(s): R-01 Sec. 3.1 p. 14 (probit link).
- Human reviewer (initials) and date:

### D-044 Bins, smoothing, monotonicity: pre-declared defaults kept
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-riskcalc
- Status: ADOPTED (defaults); alternatives REJECTED
- Decision: K = clip(train_defaults // 30, 10, 50), prior 2 pseudo-defaults, PAV on. Pre-declared
  before any outer result; fold 1 (one training year, 1,379 defaults) uses the same rule (45 bins), no
  tuning possible there.
- Evidence (inner validation 2007–08 → 2009, `inner_validation.md`): base AUC 0.8170 / LL 0.0680;
  K <= 20: 0.8165 / 0.0683; K <= 100 (15 defaults/bin): 0.8160 / 0.0680; PAV off: 0.8146 / 0.0681;
  prior 10: 0.8171 / 0.0684 (cal. slope 1.15). No alternative is better on both metrics.
- Source(s): assumption.
- Human reviewer (initials) and date:

### D-045 Industry, legal form and macro-area indicators
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-riskcalc
- Status: ADOPTED (ateco_group only, as in R-01's form); legal_group + macro_area OPEN (not run
  out-of-time)
- Evidence (inner validation, `inner_validation.md`): base (industry) AUC 0.8170 / LL 0.0680;
  + legal_group + macro_area 0.8183 / 0.0680; no industry indicators 0.8184 / 0.0681. All
  differences ≤ 0.0015 AUC on one validation year (2009), within noise. Fold-3 industry coefficients
  are all negative vs manufacturing (ATECO C, the reference), e.g. real estate (L) −0.287 (SE 0.019),
  construction (F) −0.215 (SE 0.017): conditional on ratios, manufacturing is the riskiest large
  sector in 2007–09 training data. [Section names follow D-013's mapping.]
- Source(s): R-01 Sec. 3.1 p. 14 (industry indicators in the probit).
- Risks: in 2009 validation industry indicators did not add power; they may matter more for level
  (calibration) than for ranking. Re-test with legal/macro if round 2 happens.
- Human reviewer (initials) and date:

### D-046 Final transform F does not fix the level; under-prediction left to calibration-specialist
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-riskcalc
- Status: OPEN (owner: calibration-specialist)
- Evidence: RC-03 pooled mean PD 1.13% vs observed 1.54% (obs/pred 1.36); per year obs/pred 1.22
  (2008), 1.39 (2009), 1.44 (2010); calibration intercept +0.33, slope 1.02 (`RC-03/metrics.md`). The
  slope ≈ 1 says the *shape* of the PD scale is right; the *level* follows the training-window base
  rate (0.96%, 1.13%, 1.30%) while test years run at 1.28%, 1.62%, 1.70%. The driver curves shift up
  fold by fold (`RC-03/transforms.png`). This is H4 territory: a central-tendency adjustment of the
  final transform, not a change of model.
- Source(s): R-01 Sec. 4.1 p. 25 (power and calibration are separate validation dimensions).
- Human reviewer (initials) and date:

### D-047 New firms are ranked worse than seen firms (H5, first evidence)
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-riskcalc
- Status: OPEN (H5 evidence; referee to confirm with CIs)
- Evidence: RC-03 pooled AUC seen firms 0.8259 (442,415 rows, 6,862 defaults) vs new firms 0.7651
  (63,744 rows, 922 defaults); calibration slope 1.04 vs 0.89 (`RC-03/metrics.md`). The model uses
  no firm history, so the gap is a property of the population (new firms in the data), not of
  history features.
- Source(s): assumption.
- Human reviewer (initials) and date:

---

## Tournament round 1: challenger-gbm (merged by orchestrator from experiments/gbm/decisions.md)

# challenger-gbm decisions (D-050..D-059), for the orchestrator to merge into docs/DECISION_LOG.md

All numbers come from `experiments/gbm/<exp>/metrics.md`, `train_log.md`, `config.json`,
`experiments/gbm/comparison.md` and `experiments/gbm/GB-01/explain.md` (code: `gbm_lib.py`,
`run_experiment.py`, `compare.py`, `explain.py` in `experiments/gbm/`). Pooled = test years
2008-2010, 506,159 rows, 7,784 defaults.

### D-050 GBM challenger = monotone-constrained LightGBM, full feature set, native NaN (GB-01)
- Date / phase / agent: 2026-10-31 / Phase 3 (tournament round 1) / challenger-gbm
- Status: ADOPTED (as the GBM entry to the tournament; promotion is the referee's call)
- Decision: LightGBM binary log-loss objective; monotone constraints from `B.expected_signs()` on every
  feature with sign +1/-1 (asset_turnover, extraord_to_assets, rev_nonpositive and the 3 categoricals
  unconstrained); categoricals as LightGBM categoricals (never numeric codes); NaN left to LightGBM's
  native missing-value handling; no class re-weighting or resampling.
- Alternatives considered: unconstrained twin (GB-02, D-051), core features only (GB-03, D-055), shared
  imputation (GB-04, D-056), 'advanced' monotone method (GB-05, D-057), partial constraints (GB-06,
  D-051). XGBoost was not tested (team decision: not installed). So the past-cohort tip that
  "calibrated XGBoost did well" is tested only in its LightGBM form.
- Business reason: one model that ranks well (approval) and whose drivers all move PD in the direction
  a credit officer expects (defensible declines, reason codes).
- Evidence: GB-01 pooled AUC 0.8375 (AR 0.675), log loss 0.06416, Brier 0.01368; per year AUC
  0.8441 / 0.8359 / 0.8307 (2008/2009/2010); seen firms AUC 0.8438, new firms 0.7865. All 20
  sign-expected features have zero partial-dependence movement against the expected sign (explain.md §1).
- Source(s): monotone constraints = FEATURE_SPEC expected signs (D-018, D-022); R-19 (abstract) for the
  need to check boosted-tree probabilities post hoc.
- Risks / what would change our mind: referee bootstrap shows the RiskCalc/scorecard challengers equal on
  AUC (then the simpler model wins under the tie rule); the new-firm AUC gap (0.787 vs 0.844) also
  appears in the other challengers or is specific to GBM.
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-38 Sec. 2.1 p. 2 (GBDT algorithm, VERIFIED); R-39 LightGBM Parameters docs (monotone_constraints, use_missing, categorical; VERIFIED documentation); R-19 abstract (boosted-tree probabilities can be distorted); expected signs = FEATURE_SPEC.

### D-051 Monotone constraints kept despite a significant cost in power
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED (constraints kept). Unconstrained model REJECTED for submission ("what didn't work" /
  trade-off row)
- Decision: keep the constraints. The unconstrained twin GB-02 is better on both metrics, but it gets
  there partly by using ratios against their economic direction.
- Alternatives considered: GB-02 (no constraints); GB-06 (constraints dropped only on the three
  decile-break features current_ratio, tangible_ratio, st_debt_share flagged by feature-architect).
- Business reason: a model that says "more working capital raises PD" or "more bank debt lowers PD"
  cannot be defended to a credit officer or a validator, and its reason codes would be wrong-signed.
- Evidence (comparison.md, firm-clustered paired bootstrap, 200 draws, d = variant - GB-01):
  GB-02 dAUC +0.0065 [+0.0052, +0.0079], d log loss -0.00083 [-0.00096, -0.00072] (-1.3%), both
  significant; per year AUC 0.8485 / 0.8419 / 0.8402. GB-02 fold-3 partial dependence CONTRADICTS the
  expected sign for wc_ratio (84.7% of movement against), bank_debt_ratio (55.5%), interest_coverage
  (57.4%), and partly violates nearly every other ratio (GB-01/explain.md §4). GB-06 recovers only
  dAUC +0.0005 [+0.0000, +0.0010], d log loss -0.00011, so the gap is not caused by the three decile-
  break features. It is spread over collinear ratios used conditionally against sign.
- Source(s): assumption (face validity is a course grading criterion and a credit-officer requirement).
- Risks / what would change our mind: if the referee's face-validity test also accepts limited
  non-monotonicity, a partially constrained model could be revisited in round 2. The gap is about
  0.0065 AUC, about the same size as the extra power the full feature set adds over the core set (D-055).
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-01 Sec. 3.1 p. 13 (expected signs, VERIFIED); R-39 (constraint option). Face validity as the reason for the constraint = our design choice.

### D-052 Inner validation scheme, tuning grid and number of trees
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED
- Decision: folds 2-3: inner-train = training years except the last; the last training year is
  split by firm (crc32 hash of id) into A (early stopping + grid choice by A log loss) and B
  (calibration). Fold 1 (one training year): random firm split 70/15/15 with the pre-declared DEFAULT
  config (num_leaves 8, depth 3, min_data_in_leaf 1000), no grid. Pre-declared grid (folds 2-3):
  num_leaves/max_depth in {4/2, 8/3} x min_data_in_leaf in {500, 2000}; fixed lr 0.03,
  feature/bagging fraction 0.8, lambda_l2 10, early stopping 150 rounds (max 4000). Final booster
  refit on all training rows with round(1.2 x best_iter) trees (heuristic: more rows support
  slightly more trees). Feature params (`B.fit`) refit on inner-train for the inner model and on the
  full window for the final model.
- Alternatives considered: Optuna search (rejected: time box, and the grid differences are tiny);
  no refit (final model = inner model; rejected because it would drop the most recent training year).
- Business reason: tuning never sees the test year; the latest year is used for validation, which
  mimics deployment.
- Evidence: chosen per fold (config.json): fold 1 DEFAULT, 282 -> 338 trees; fold 2 8 leaves/min 500,
  408 -> 490; fold 3 8 leaves/min 2000, 322 -> 386. The grid spread in A log loss is <= 0.0003
  (train_log.md), so the choice matters little.
- Source(s): assumption (standard practice). The 1.2 factor is a heuristic, not tuned.
- Risks / what would change our mind: the 1.2 factor is untested; the harness refit on 2007-2010
  should use the same rule (encoded in `fit_pd_model`).
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-07 Sec. 3.2 p. 95 (VERIFIED). The 1.2 x trees factor = heuristic (assumption).

### D-053 Own calibration: Platt slope from held-out slice B, level anchored to the training default rate
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED (own calibration); final level-setting is DEFERRED to calibration-specialist
- Decision: Platt map logit(PD) = a + b*logit(p_raw). Slope b is fitted on slice B with the inner model
  (B was used neither for fitting nor for early stopping). The intercept a is re-anchored so that the
  mean calibrated PD over the full training window equals the training default rate. Isotonic was
  compared but not adopted.
- Alternatives considered: (a) raw booster probabilities; (b) Platt as fitted on B (its intercept
  mixes in the year-on-year rise in the default rate, then would be applied to a refit model that
  already contains that year, so it double-counts); (c) isotonic (eligible only if B has >= 500
  defaults: fold 1 B has 203 defaults, so not eligible there).
- Business reason: pricing needs the right level. A shape-only fix from held-out data, plus an
  explicit, documented level choice (training-window average), keeps the level decision with one
  owner (calibration-specialist) for all challengers.
- Evidence: 2-fold firm-cross-fitted log loss on B (raw / Platt / isotonic): fold 1 0.04398 / 0.04401
  / 0.04553; fold 2 0.05579 / 0.05549 / 0.05573; fold 3 0.06518 / 0.06447 / 0.06476. Platt slope b on B
  = 1.017 / 0.943 / 0.998. So the LightGBM log-loss booster (no re-weighting) is close to slope 1. In
  this data, the Platt gain on B is almost all intercept (level drift between years), not the sigmoid
  distortion R-19 describes for boosting. OOT effect of our map vs raw: pooled AUC unchanged, log loss
  0.064155 vs raw within 1e-4 (metrics.md). OOT level: mean PD 1.145% vs observed 1.538% (obs/pred
  1.34; 1.21 / 1.36 / 1.43 by year), calibration slope 0.976. The level gap grows as default rates rise,
  so it is a central-tendency problem, not a shape problem.
- Source(s): R-19 (abstract only: boosted-tree probabilities can be distorted and need post-hoc
  calibration; we tested this and found little distortion here). Platt (R-20) is TO-VERIFY, so
  "Platt scaling" is cited as a method name only [UNVERIFIED].
- Risks / what would change our mind: the calibration-specialist's chosen central tendency. Raw OOT
  predictions (`oot_predictions_raw.parquet`) and inner-B predictions (`inner_B_predictions.parquet`:
  row_id, y, pd_raw_inner, fold) are provided so the common method can be fitted without touching
  test years.
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): Platt scaling as restated in R-43 Sec. 1 eqs. 1-2 (original R-20 SECONDARY); isotonic/PAV properties R-44 Sec. 4.1, 4.5 (R-21 LOCATED, name only); R-19 abstract.

### D-054 No class re-weighting or resampling
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED
- Decision: train on the natural ~1-1.3% default rate with strong min_data_in_leaf (500-2000) and
  shallow trees; no scale_pos_weight, no up/down-sampling.
- Alternatives considered: scale_pos_weight plus re-calibration (rejected: it destroys the probability
  level and needs another calibration stage, with no expected gain in AUC).
- Business reason: the PD level is used for pricing.
- Evidence: calibration slope 0.976 pooled without any re-weighting (metrics.md).
- Source(s): assumption.
- Risks / what would change our mind: none expected.
- Human reviewer (initials) and date:

### D-055 Full feature set vs core set for the GBM
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED full set (core-only GB-03 REJECTED)
- Decision: the GBM uses all 26 features from FEATURE_SPEC.
- Alternatives considered: the 13 core features only (GB-03).
- Business reason: more power for approval ranking at no cost in face validity (all constrained).
- Evidence: GB-03 vs GB-01: dAUC -0.0077 [-0.0087, -0.0066], d log loss +0.00077 [+0.00067, +0.00085];
  GB-03 per year AUC 0.8381 / 0.8274 / 0.8226. It is also a like-for-like reference for the probit /
  logit challengers that use the core set: GB-03 pooled AUC 0.8298.
- Source(s): assumption.
- Risks / what would change our mind: if the referee rates the GBM vs probit gap as mostly a feature-set
  effect, the probit challengers could add non-core features in round 2.
- Human reviewer (initials) and date:

### D-056 Native NaN handling vs shared imputation (build.py impute=True)
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED native NaN (imputation GB-04 REJECTED as unnecessary, not as harmful)
- Decision: keep NaN for LightGBM; the `debt_split_missing` flag stays in the feature set.
- Alternatives considered: GB-04 = same per-fold configs as GB-01 with sector-median / block-share
  imputation.
- Business reason: simpler harness (no imputation step for this model); identical results.
- Evidence: GB-04 vs GB-01 dAUC +0.0001 [-0.0003, +0.0006], d log loss -0.000007 [-0.00004, +0.00003]:
  no difference.
- Source(s): assumption.
- Risks / what would change our mind: none. Either is acceptable for the harness.
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-39 use_missing (native NaN handling, VERIFIED documentation); evidence GB-04 vs GB-01.

### D-057 LightGBM monotone method 'intermediate' (vs 'advanced')
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED 'intermediate' ('advanced' GB-05 REJECTED: no material gain)
- Evidence: GB-05 vs GB-01 dAUC +0.0002 [-0.0000, +0.0004], d log loss -0.00002 [-0.00004, +0.00000].
- Business reason: same face validity, simpler and faster; the difference is below any business relevance.
- Source(s): assumption.
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-39 monotone_constraints_method; R-40 abstract p. 1 (un-refereed arXiv report, Adult dataset only).

### D-058 Explainability: TreeSHAP reason codes and dependence checks
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED
- Decision: per-firm reason codes = top-3 positive TreeSHAP contributions (LightGBM `pred_contrib`, log-odds
  scale) in plain words with natural-scale values and percentiles; global = mean |SHAP| + partial
  dependence vs expected sign; report driver FAMILIES because correlated ratios split attribution.
- Evidence: GB-01/explain.md. Top drivers (mean |SHAP|): tangible_ratio 0.41, equity_ratio 0.21, roa_net
  0.21, st_debt_ratio 0.18, ap_days 0.15, ar_days 0.14, ateco_group 0.12. Every sign-expected feature
  agrees. asset_turnover (unconstrained by design) rises with PD over its range in the fold-3 model
  (-5.36 -> -4.87 log-odds).
- Source(s): R-23 (Shapley-value properties, Sec. 3). The TreeSHAP algorithm is NOT in R-23, so it is
  cited as [UNVERIFIED] until literature-scout verifies a source.
- Risks: tangible_ratio being the #1 driver should be sanity-checked by a human (sector mix: real
  estate / asset-heavy firms; FEATURE_SPEC notes decile breaks). ar_days hits the p99 cap (~60,000
  days) for near-zero-revenue firms, so the reason code must be worded "almost no revenue".
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-41 Sec. 2.5, Methods 10.1 (Tree SHAP exact, O(TLD^2)), Sec. 2.7 (summary/dependence plots), Methods 5 (gain importance inconsistent) (VERIFIED, preprint numbering); R-23 Sec. 3 Thm 1 p. 4; R-33 Sec. 5.2-5.4 pp. 27-29 (style).

### D-059 Reusable estimation for the harness; prediction cost
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED
- Decision: `experiments/gbm/gbm_lib.py`: `fit_pd_model(train_df, "GB-01")` fits feature params +
  inner model + calibrator + final booster on any training frame (needs columns id, fs_year, y, raw
  fields); `save_model` / `load_model` write and read booster.txt, feature_params.json,
  calibrator.json, meta.json; `predict_pd(art, df)` is frozen (no re-estimation, row-wise).
- Evidence: round trip of the saved fold-3 model reproduces the fold-3 OOT PDs exactly (max |diff| 0.0,
  comparison.md). Cost: 100,000 rows in 1.0 s (0.8 s of it is the shared feature transform), 8
  threads, laptop. Complexity O(n x (F + T x D)): 386 trees of depth <= 3 in fold 3.
- Source(s): n/a.
- Risks: the harness must call `predict_pd` with the same LightGBM major version (4.7.0 used).
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-38 Sec. 2.1 (tree ensemble); prediction cost = our derivation.

---

## Tournament round 1: referee (2026-10-31)

All numbers below are read from `reports/tournament/r1_*.csv` written by
`reports/tournament/score_r1.py` (metrics from `src/common/metrics.py`); the full scorecard is
`reports/tournament/scorecard_r1.md`. Bootstrap: paired, firm-clustered, 200 draws (D-007), seed 0.

### D-060 Referee scoring method for round 1
- Date / phase / agent: 2026-10-31 / Day 2, round 1 / referee
- Status: ADOPTED
- Decision: score all 13 experiments pooled, per test year (2008/09/10) and seen/new firm with
  `metrics.score`; paired firm-clustered bootstrap (200 draws, same draws for every model) for dAUC
  and d log loss vs the incumbent SC-03 and for the key pairs. The bootstrap uses firm-multiplicity
  weights (weighted Mann-Whitney AUC with a single pre-sort; weighted mean log loss), which is
  algebraically the same estimator as `metrics.paired_bootstrap` with the same RNG call sequence.
  Two level-adjusted log losses are reported so that models are not compared only on their shared
  level error: L1 (leakage-free) = one common logit shift per fold, delta = logit(DR last training
  year) - logit(DR training window), from training labels only (fold 1: 0; fold 2: +0.129; fold 3:
  +0.219); L2 (oracle, diagnostic, uses test labels) = each model's own per-year
  calibration-in-the-large intercept. Neither is a level recommendation.
- Alternatives considered: the shared `paired_bootstrap` (O(firms x rows) index build, too slow on
  506k rows x 13 models); an in-window per-model level anchor (needs in-window predictions that only
  the GBM saved).
- Evidence: equivalence on a 15,000-firm subsample (GB-01 vs SC-03, 40 draws, seed 0): CI bounds
  identical to the shared function, max |diff| 1.0e-16 (`r1_bootstrap_equivalence.csv`). L1 and L2 do
  not change the ranking of any face-valid model, and no d log-loss CI changes sign (scorecard T1, T4, T6).
- Source(s): D-005, D-006; R-16 p. 32 (Brier = overall probability accuracy).
- Risks / what would change our mind: with 200 draws the 2.5%/97.5% percentiles are themselves noisy
  (each rests on about 5 draws); none of the promotion-relevant CIs lies near zero, but borderline
  pairs (e.g. GB-05 vs GB-01, RC-01 vs SC-03 AUC) should be re-run with 1,000 draws before being quoted.
- Human reviewer (initials) and date:

### D-061 Leakage / process verdicts, round 1
- Date / phase / agent: 2026-10-31 / round 1 / referee
- Status: ADOPTED
- Decision: CLEAN: SC-01..SC-04, RC-01, RC-02, GB-01..GB-06. CONCERN: RC-03 (D-063). FAIL: none.
  Shared CONCERN for all 13: D-062.
- Evidence: code read (runners, `scorecard_lib.py`, `riskcalc.py`, `gbm_lib.py`, `src/features/build.py`):
  every fit (B.fit bounds/medians/category pooling, WoE bins, T_i, probit, F, GBM inner split, early
  stopping, grid, Platt slope, level anchor) uses only `fs_year <= t, label_status == ok` rows. The
  referee refitted fold 3 of SC-03, RC-03 and GB-01 from current code on training rows only and
  predicted on a test frame with all label columns removed: submitted PDs reproduced exactly (max
  |diff| 0, `r1_repro_check.csv`). Contract check passes for all 13 (`r1_contract.csv`). GB notes:
  fold 1 uses a random 70/15/15 firm split inside 2007 and a pre-declared default config (deviation
  from "time-ordered inner splits", unavoidable with one training year; no test data); the 1.2 x
  early-stopping tree count is an untested but pre-fixed heuristic; calibration slope from in-window
  slice B, level anchored to the training rate. SC: nothing tuned. RiskCalc inner validation
  (2007-08 -> 2009) was run after the outer results and changed no setting.
- Source(s): CLAUDE.md section 5 rule 5; data/splits/README.md rules.
- Human reviewer (initials) and date:

### D-062 Shared caveat: the feature set was designed on fs_year 2007-2009 (test years of folds 1-2)
- Date / phase / agent: 2026-10-31 / round 1 / referee
- Status: ADOPTED (as a reporting rule); fix OPEN for round 2
- Finding: `docs/FEATURE_SPEC.md` (core set, which ratios exist, `sign = none` for asset_turnover and
  extraord_to_assets, rejected candidates) was chosen from 2007-2009 training rows, incl. an inner
  split scored on 2009 (`src/features/core_selection.py`, `univariate.py`). 2008 and 2009 are the
  outer test years of folds 1 and 2, so feature selection saw their labels. Common to every model, so
  paired comparisons are affected much less than absolute fold-1/2 levels; fold 3 (2010) is clean.
- Decision: (1) quote 2010 (fold 3) next to pooled numbers in the deck; (2) a promotion must also hold
  on 2010 alone. GB-01 vs SC-03 on 2010: dAUC +0.0191 [+0.0156, +0.0234], d log loss -0.00217
  [-0.00253, -0.00181]: it does. (3) Say plainly in the deck that folds 1-2 are mildly optimistic.
- Alternatives considered: re-run feature selection inside each fold (correct but changes the feature
  set per fold, Roger M12; not feasible on Day 2).
- Human reviewer (initials) and date:

### D-063 RC-03's sign-drop rule is post hoc; how to report RC-01/02/03
- Date / phase / agent: 2026-10-31 / round 1 / referee
- Status: ADOPTED
- Decision: RC-03 verdict CONCERN. The rule is executed in-window, but it was added after RC-01/RC-02
  outer coefficients and metrics were seen (`run.py` modified 23:59:03, after RC-02 ran 23:58). It is
  motivated by face validity, not by a performance search, and its measured effect vs RC-02 is nil
  (dAUC -0.0002 [-0.0004, +0.0001], d log loss -0.00003 [-0.00005, -0.00001]), so the optimism is small.
  Report RC-03 as "RC-02 + a post-hoc sign rule"; do not present RC-03 vs RC-01/RC-02 differences as
  evidence for the rule; it needs the real holdout to be confirmed. RC-01/RC-02 stay CLEAN: their settings
  are code defaults used unchanged, and the inner validation (which validates on 2009 = fold 2's test
  year) post-dates the outer runs and changed nothing. That inner validation may be quoted only as a
  fold-3 / final-model check (as `inner_validation.md` already says), never as tuning evidence.
- Evidence: `r1_paired_bootstrap.csv` (RC-03 vs RC-02); file times in experiments/riskcalc.
- Human reviewer (initials) and date:

### D-064 Face-validity verdicts, round 1
- Date / phase / agent: 2026-10-31 / round 1 / referee
- Status: ADOPTED
- Decision (violating features): SC-01 0, SC-02 1 (log_assets + in folds 1, 3; n.s.), SC-03 0, SC-04 0,
  RC-01 1 (T_neg_equity < 0 in all folds), RC-02 2 (neg_equity, bank_debt_ratio, all folds), RC-03 0,
  GB-01/03/04/05 0, GB-02 2 (wc_ratio 83% of partial-dependence movement against the sign,
  interest_coverage 57%; bank_debt_ratio borderline 45%), GB-06 1 (current_ratio 87% against).
  Method: coefficient signs per fold (GLM/probit); WoE/T_i monotonicity (111 checks, all pass);
  GBM partial dependence of the saved fold-3 model on 20,000 fold-3 training rows (folds 1-2 of the
  constrained GBMs are covered by construction only).
- tangible_ratio as GB-01's #1 driver: plausible. The expected direction holds within every ATECO group
  with >= 30 defaults (within-group AUC 0.56-0.73; overall 0.616; `r1_tangible_ratio_check.csv`), so it is
  not merely a real-estate/sector proxy; its SHAP share is inflated by attribution among correlated
  leverage/asset-structure ratios. Explain by family (D-058).
- Evidence: `r1_face_validity_long.csv`, `r1_face_validity_gbm_pd.csv`, `r1_monotone_shape_checks.json`.
- Source(s): docs/FEATURE_SPEC.md expected-sign table.
- Human reviewer (initials) and date:

### D-065 Promotion: GB-01 replaces SC-03 as incumbent
- Date / phase / agent: 2026-10-31 / round 1 / referee
- Status: ADOPTED (referee); human sign-off requested (explainability trade-off)
- Decision: GB-01 (monotone LightGBM, 26 features, native NaN) is the incumbent after round 1.
- Evidence (`r1_paired_bootstrap.csv`): vs SC-03 pooled dAUC +0.0209 [+0.0188, +0.0231], d log loss
  -0.00241 [-0.00260, -0.00223]; better in 2008, 2009 and 2010 and for seen and new firms (all CIs
  exclude 0); 0 face-validity violations (SC-03: 0); process CLEAN. vs RC-03 (best transparent model):
  dAUC +0.0183 [+0.0163, +0.0203], d log loss -0.00162 [-0.00177, -0.00147], better in every year, so
  this is not a tie that goes to transparency. Pooled GB-01: AUC 0.8375 [0.8321, 0.8408] (AR 0.675),
  log loss 0.06415, Brier 0.01368, mean PD 1.15% vs observed 1.54%, calibration slope 0.976.
  Part of the gain is the feature set and part is the shape/interactions: GB-03 (GBM on the core set)
  vs RC-01 (core-set probit): dAUC +0.0123 [+0.0106, +0.0141].
- Alternatives considered: RC-03 (transparent, D-067); GB-02 (better but fails face validity).
- Risks / what would change our mind: the humans judge per-firm SHAP reason codes insufficient for a
  credit officer → RC-03 becomes the submission model at a cost of ~0.018 AUC; the holdout behaves
  differently from 2008-2010.
- Human reviewer (initials) and date:

### D-066 Not promoted in round 1 (what didn't work / ties)
- Date / phase / agent: 2026-10-31 / round 1 / referee
- Status: REJECTED (for promotion)
- GB-02 (unconstrained) and GB-06 (partial constraints): fail face validity (D-064) despite better
  metrics (GB-02 vs GB-01 dAUC +0.0065 [+0.0052, +0.0080]); GB-02 is reported as the cost of face
  validity. GB-04 (imputation) and GB-05 ('advanced' monotone): statistical ties with GB-01 (dAUC
  +0.0001 [-0.0003, +0.0006] and +0.0002 [-0.0000, +0.0004]) → simpler GB-01 kept. GB-03: dominated by
  GB-01 (dAUC -0.0077 [-0.0087, -0.0066]). RC-01: log loss better than SC-03 but dAUC tie
  (+0.0009 [-0.0001, +0.0020]) and a wrong-signed T_neg_equity. RC-02: two wrong-signed transforms.
  RC-03: CONCERN and dominated by GB-01. SC-01, SC-02, SC-04: worse than SC-03 on both metrics.
- Human reviewer (initials) and date:

### D-067 Hand-off to calibration-specialist: GB-01 and RC-03
- Date / phase / agent: 2026-10-31 / round 1 / referee
- Status: ADOPTED
- Decision: calibrate GB-01 (champion) and RC-03 (transparent fallback). Facts for the specialist:
  every model under-predicts; obs/pred pooled 1.34 (GB-01) and 1.36 (RC-03), by year 1.21/1.36/1.43
  (GB-01) and 1.22/1.39/1.44 (RC-03); calibration slopes ≈ 1 pooled (0.976, 1.024), so the miss is a
  level shift that grows with the default-rate trend; new-firm slopes 0.88 / 0.89 (PDs too spread for
  new firms). L2 (oracle) shows the best achievable gain from a perfect per-year level: GB-01 log loss
  0.06415 → 0.06343; the leakage-free L1 shift reaches 0.06370. The central-tendency choice is the
  specialist's; it must be fitted in-window.
- Human reviewer (initials) and date:

### D-068 D-011: shared make_labels should censor incomplete windows regardless of outcome (proposed)
- Date / phase / agent: 2026-10-31 / round 1 / referee
- Status: OPEN (proposed by referee; needs human approval to edit src/common/labels.py)
- Finding: the panel `y`/`label_status` used for scoring is identical to `make_labels(start 6, horizon
  12, data_end 2013-05-31)` for fs_year 2007-2010 (0 status mismatches, 0 y mismatches, same default
  counts 1,379 / 2,064 / 2,752 / 2,968; `r1_d011_label_check.csv`). Round-1 results are unaffected.
  The bug bites in fs_year 2011: make_labels would mark 2,357 incomplete-window rows `ok`, all with
  y = 1 (100% default rate); the panel correctly censors them.
- Proposal: in `make_labels`, set `censored = (~already) & (win_end > data_end + 1 day)` regardless of
  `in_window`, add the `after_default` status, and change the default `data_end` to the D-009 value;
  add a unit test with one censored default. Until then, nobody may call make_labels directly to
  build training rows for the harness refit (use the panel).
- Human reviewer (initials) and date:

### D-069 Proposed (not made) change to src/common/metrics.paired_bootstrap
- Date / phase / agent: 2026-10-31 / round 1 / referee
- Status: OPEN (proposal; src/common not edited)
- Proposal: replace the per-firm index list with multiplicity weights (`np.bincount` of the resampled
  firm indices) and weighted AUC / log loss. Same RNG sequence, identical results (verified: max |diff|
  1e-16 on a 15,000-firm subsample, `r1_bootstrap_equivalence.csv`), and fast enough for 1,000 draws on
  506k rows. Reference implementation: `AUCEngine` + `bootstrap_draws` in `reports/tournament/score_r1.py`.
- Human reviewer (initials) and date:

---

## Calibration round (calibration-specialist, 2026-10-31)

Every number below is printed by `experiments/calibration/s1..s8_*.py` and stored in
`reports/calibration/*.csv|json` (figures in `reports/calibration/fig/`). Models: GB-01 and RC-02 (RC-02
used instead of RC-03 per D-063). "e-4" = 0.0001 of log loss. CIs = firm-clustered paired bootstrap, 200
draws, seed 0, same estimator as `reports/tournament/score_r1.py`. The GB-01 refit used for in-window
predictions reproduces the submitted OOT PDs exactly (max |diff| 0, `experiments/calibration/cache/gb_oot_check.csv`).
`reports/calibration/summary.md` was NOT written (the sub-agent harness blocks report files); the
orchestrator should render it from these entries and the CSVs.

### D-070 What a bank knows when it sets the level (leakage rule for every level parameter)
- Date / phase / agent: 2026-10-31 / calibration round / calibration-specialist
- Status: ADOPTED
- Decision: under start 6, fs_year t's label (30 Jun t+1 to 30 Jun t+2) is complete on 30 Jun t+2, when
  decisions on fs_year t+1 statements begin. So for test year t+1 the bank has seen defaults up to 30 Jun
  t+2 and nothing of the test window: a "partially observed next-year rate" is NOT available. The
  freshest legal signal is the default arrival speed in the last six months of the last training cohort
  (Jan-Jun t+2) = rule e. All targets, shifts, Platt/isotonic maps use training rows or GB-01's in-window
  slice B only.
- Alternatives considered: using the test cohort's 0-6-month defaults (they are excluded rows of the test
  population and would use other holdout rows: rejected, constraint 2).
- Evidence: `reports/calibration/ct_targets.csv`, `monthly_hazard.csv`.
- Source(s): D-002 (R-28, R-29 for filing timing); assumption for decision timing.
- Human reviewer (initials) and date:

### D-071 Central-tendency rules in the walk-forward: recency wins, long-run average loses, trend is fragile
- Date / phase / agent: 2026-10-31 / calibration round / calibration-specialist
- Status: ADOPTED (finding)
- Decision: in 2008-2010, level rules that follow the latest observed level beat the training-window
  average; trend extrapolation ties them in the rise but fails at the 2011 turn.
- Evidence (`methods_by_year.csv`, GB-01, prior shift M1, Delta log loss vs submitted, e-4, pooled):
  e last-6-months -6.31 [-7.09, -5.60] (per year -2.49 / -7.66 / -8.51); c2 half-trend -6.28 [-7.17, -5.44];
  c trend -6.16 [-7.43, -4.95]; b last year -4.50 [-5.02, -4.03] (= referee L1); f mean-last-2 -1.57;
  a pooled 0; d TTC proxy +0.37 [+0.34, +0.41]. RC-02 same order (e -6.65 [-7.50, -5.93]). Targets vs
  observed (2008/09/10): e 1.17/1.50/1.61%, b 0.96/1.28/1.62%, a 0.96/1.13/1.30%, observed 1.28/1.62/1.70%.
  obs/pred GB-01: submitted 1.21/1.36/1.43 -> rule e 1.02/1.05/1.18; pooled log loss 0.06415 -> 0.06352
  (oracle per-year level 0.06343: e recovers 88% of the attainable gain). 2011 pseudo-test (fold-4 models
  fit on 2007-2010; fs2011 12-month rate estimated at 1.43% from 11 months, `q6_2011_pseudo_test.csv`):
  level-only extra log loss (e-4) a 0.33, d 0.48, f 0.48, e 0.74, b 0.80, c2 3.85, c 9.42.
- Alternatives considered: a, b, c, c2, d, e, f (all in `ct_targets.csv`); a true TTC level is not
  estimable (no post-turn benign years in 2007-2010).
- Source(s): R-16 Sec. II p. 11 (pricing -> PIT, underwriting -> TTC).
- Risks / what would change our mind: e was selected among 7 rules after seeing 2008-2010 (optimistic);
  only one turning point (2011, estimated) tests the rules out of selection.
- Human reviewer (initials) and date:

### D-072 Method to move the level: prior shift (logit intercept shift); EM, isotonic, raw Platt rejected
- Date / phase / agent: 2026-10-31 / calibration round / calibration-specialist
- Status: ADOPTED (M1); REJECTED (M2, M3raw for the harness, M4 isotonic, Saerens EM)
- Decision: PD' = expit(logit(PD) + delta), delta = logit(tau) - logit(r_w): exactly the R-22 eq. (4)
  prior correction, i.e. the "simple prior adjustment" of R-07 pp. 88-89. One frozen number; no data
  needed at prediction time.
- Evidence (GB-01, rule e, pooled; `methods_by_year.csv`, `auc_invariance_check.csv`): M1 0.06352,
  -6.31e-4 [-7.09, -5.60], slope 0.98, within-year |dAUC| = 0. M2 (shift matched to the model's mean PD on
  the last training year) -6.16e-4: slightly worse. M3 (Platt slope from B + level) identical to M1 for
  GB-01. M3raw (Platt as fitted on B; level = the B year's drift) -6.21e-4: good in a rise but it is an
  implicit trend rule (D-071). M4 isotonic + level -4.82e-4, slope 0.92, within-year AUC -0.0008
  (ties): the miss is level, not shape. Saerens EM (prior estimated from the unlabeled test batch):
  2.18% / 1.79% / 0.45% vs observed 1.28 / 1.62 / 1.70% (`saerens_em_diagnostic.csv`), and it would need
  the holdout batch (constraint 2).
- Source(s): R-22 Sec. 2.2 eq. (4) pp. 5-6 (invariance assumption doubtful 2008-12; used as a level tool,
  not as a claim that the assumption holds); R-07 pp. 88-91 (monotone recalibration cannot add power);
  Platt/isotonic named as methods only [UNVERIFIED, R-20/R-21].
- Human reviewer (initials) and date:

### D-073 Holdout central tendency: tau = 1.49% (start-6), PIT anchor plus a horizon hedge
- Date / phase / agent: 2026-10-31 / calibration round / calibration-specialist
- Status: ADOPTED by calibration-specialist; human sign-off requested (the hedge is a judgment)
- Decision: tau = 1.493% = expected rate two years after the fs2011 anchor (1.431%) under a driftless
  logit random walk with sigma 0.21/yr (estimated from 2007-2011 cohort changes). Harness: delta =
  logit(0.01493) - logit(r_w) = +0.059 (fit on 2007-2010, r_w 1.409%) or +0.079 (fit incl. fs2011 with
  11-month label, r_w 1.381%); `experiments/calibration/harness_level_params.json`.
- Alternatives considered (`final_level_candidates.csv`): pooled 2007-2010 1.41%; last complete year
  (fs2010) 1.70%; fs2011 estimate 1.43%; latest half-year (Dec 2012-May 2013, seasonally scaled) 1.38%;
  equal-weight 2007-2011 1.40%; trend 2.21% (2007-2010) / 1.90% (2007-2011); f 1.57%; expected-loss
  optimum over fs2008-2011 rates 1.51%; minimax regret 1.49%; random walk h = 1/3: 1.46% / 1.53%.
- Business reason: pricing needs a PIT level (R-16 p. 11; D-071 shows recency wins). At the end of the data
  PIT (1.38-1.43%) and long-run (1.40-1.41%) agree near 1.4%; the holdout period is unknown (email Q3), and
  log loss under uncertainty is minimised by the EXPECTED rate, which with convexity and the asymmetry of
  D-074 sits a little above the anchor. Cost of the hedge if the truth is 1.43%: 0.17e-4; saving if the
  truth is 1.70%: 1.1e-4 vs shipping 1.43% (`scenario_regret.csv`, grid row tau = 1.50%).
- Evidence: expected holdout mean PD after the shift 1.40% / 1.41% / 1.61% on fs2010 / fs2011 / fs2012-like
  populations (GB-01 fold-4 model; the model's features raise the fs2012 mean by ~15%,
  `holdout_level_range.csv`); realised-rate p10-p90 at h = 2: 0.98%-2.08% (`recommended_level_random_walk.csv`).
  The decline in 2011 is not an end-of-file reporting artefact: monthly hazards fall from Jul 2012 while
  Feb-Apr 2013 match the prior year (`monthly_hazard.csv`); complete start-0/start-3 2011 rates also fall
  (1.39->1.27%, 1.58->1.31%).
- Source(s): R-16 p. 11; R-22 eq. (4); random-walk persistence and the volatility estimate are assumptions.
  Nothing is claimed about RiskCalc Italy's cycle adjustment (REFERENCES "must not claim" #12); R-01 Sec. 3.1
  p. 12 only: FSO mode updates with new statements, the full version adds market information.
- Risks / what would change our mind: holdout in a stressed year above 2010 (2.13% would cost 11.8e-4);
  Roger/TA disclose the holdout period (then set h accordingly); cleared macro data (D-004).
- Human reviewer (initials) and date:

### D-074 Asymmetry and scenario costs of the level
- Date / phase / agent: 2026-10-31 / calibration round / calibration-specialist
- Status: ADOPTED (finding)
- Evidence (`asymmetry_gb01.csv`; GB-01 from the oracle level, extra log loss e-4, 2008/09/10): 30% too
  low 8.17 / 10.35 / 10.93; 30% too high 5.57 / 7.02 / 7.40 (too low costs ~1.5x); log-symmetric 1/1.3:
  4.57 / 5.79 / 6.11. Scenarios for shipping ~1.5% (`scenario_regret.csv`, nearest grid row tau = 1.50%): truth 1.28% 1.7e-4, 1.43%
  0.2e-4, 1.62% 0.4e-4, 1.70% 1.3e-4, 2.13% 11.8e-4. Shipping 1.70% in a 2008-like year: 5.8e-4; shipping
  1.41% in a 2010-like year: 2.9e-4.
- Business reason (assumption, no rate/fee data): under-predicted PDs price loans below expected loss and
  invite adverse selection against better-priced competitors (R-08 Sec. 6 makes this argument for model
  power); over-prediction loses volume, not money on booked loans.
- H4 verdict: NOT confirmed in the walk-forward. Best leakage-free level gain 6.3e-4 vs algorithm gains
  16e-4 (GB-01 vs RC-02) and 24e-4 (vs SC-03). A 30% level miss (8-11e-4) is half the GB-01/RC-02 gap, so
  the level is second-order only if it is set carefully.
- Human reviewer (initials) and date:

### D-075 PD floor 0.03%, cap 99%; a 50% cap rejected
- Date / phase / agent: 2026-10-31 / calibration round / calibration-specialist
- Status: ADOPTED (floor 0.03%, cap 99%); REJECTED (cap 50%)
- Evidence (`floor_cap_effect.csv`, walk-forward GB-01 after M1-e): 31 PDs < 0.03%, 0 defaults among them,
  log loss unchanged. 855 PDs > 50% with 553 defaults (65% observed); a 50% cap raises log loss
  0.063524 -> 0.063621. Max PD 0.88, so a 99% cap never binds (numerical safety).
- Source(s): R-15 para. 285 p. 67 (corporate PD floor 0.03%), used by analogy only; we do not claim the bank
  is IRB (REFERENCES "must not claim" #3).
- Human reviewer (initials) and date:

### D-076 Label-window sensitivity (D-002a still open): ship the start-6 level, switchable
- Date / phase / agent: 2026-10-31 / calibration round / calibration-specialist
- Status: ADOPTED (default s6); OPEN pending Roger/TA
- Evidence (`q5_label_window_levels.csv`, same rule as D-073): tau s0 1.37%, s3 1.39%, s6 1.49% (ratios
  0.91 / 0.93 / 1). Cost matrix (`q5_label_window_cost_matrix.csv`, e-4, shipped vs true): s6 shipped,
  truth s0 0.57, truth s3 0.38; s0 shipped, truth s6 0.59. On 2007-2010 the s0 level is ~1/3 lower
  (D-002), but the 2011 rates are close across windows, so at the end of the data the gap is ~9%.
  Ranking is window-robust: fold-4 GB-01 AUC on 2011 = 0.885 (s0), 0.869 (s3), 0.850 (s6, 11-month)
  (`q5_2011_other_labels_fold4.csv`).
- Decision: harness reads `tau[s6]`; if Roger/TA confirm another window, switch to `tau[s0|s3]` (no refit
  of the level logic needed; the model's ranking stays).
- Human reviewer (initials) and date:

### D-077 Include fs2011 in the final fit with an 11-month label built from the panel
- Date / phase / agent: 2026-10-31 / calibration round / calibration-specialist
- Status: ADOPTED (recommendation to harness-engineer; human sign-off requested)
- Decision: y11 = 1 if win_start <= def_date <= 2013-05-31, else 0, for the 183,953 fs2011 rows alive at
  window start (2,357 defaults; 1.28% in 11 months). Build from the panel's win_start/def_date, NOT
  `src/common/labels.make_labels` (D-011/D-068 bug: it marks only the 2,357 defaults "ok" with y = 1).
  Recompute r_w and delta on the fitted rows (D-073).
- Evidence (fold-3 analogue, `q6_partial_cohort_test.csv`, test 2010 at the same level, point estimates,
  no CI): GB-01 AUC / log loss full 0.8307 / 0.07051, newest cohort cut to 11 months 0.8301 / 0.07063,
  newest cohort dropped 0.8263 / 0.07086; RC-02 0.8115 / 0.07216, 0.8114 / 0.07221, 0.8089 / 0.07242.
  Including a truncated newest cohort beats dropping it by 2.1-2.3e-4 and 0.003-0.004 AUC.
- Alternatives considered: exclude 2011 (current; loses the newest behaviour); keep-as-0 for the missing
  month is what the 11-month label does (~9% of the cohort's 12-month defaults fall in month 12,
  `cohort_share_11_of_12.csv`), and the level is set separately by tau.
- Human reviewer (initials) and date:

### D-078 Credit-officer grade table and residual group miscalibration
- Date / phase / agent: 2026-10-31 / calibration round / calibration-specialist
- Status: ADOPTED (reporting); group-level corrections REJECTED (they would be fitted on test years)
- Evidence: `grade_table_2010.csv` (CAL-01 on 2010; 11 pre-declared bands from 0.03% to 99%): observed
  default rate rises monotonically across all 11 grades; grades 1-3 hold 42.5% of firms and 5.9% of
  defaults; grades 9-11 hold 3.8% of firms and 33.8% of defaults; grades 4-9 run 20-40% above their
  grade PD in 2010 because rule e under-shot 2010 by 18%. Group calibration (`subgroup_calibration.csv`,
  pooled): new firms obs/pred 1.06, slope 0.89 (too spread: a shape issue, D-047); size quintile slopes
  0.88 (small) to 1.10 (large); big ATECO sections obs/pred 1.01-1.17; macro-areas 1.04-1.17.
- Source(s): R-16 pp. 33-34 (binomial bands understate error with correlated defaults: bands are diagnostic).
- Human reviewer (initials) and date:

### D-079 Hand-off from the calibration round
- Date / phase / agent: 2026-10-31 / calibration round / calibration-specialist
- Status: ADOPTED
- Decision: harness = GB-01 final fit (2007-2011, D-077) -> its own calibrator (anchored to r_w) -> M1
  shift with delta from `harness_level_params.json` -> clip [0.0003, 0.99]. Re-calibrated contract files
  for the referee: `experiments/calibration/CAL-01` (GB-01 + M1 + e), `CAL-02` (RC-02 + M1 + e),
  `CAL-03` (GB-01 + M1 + b), all floor 0.03% / cap 99%.
- Open for humans: (1) sign off the hedge (1.49% vs the pure PIT anchor 1.43%); (2) D-002a label window;
  (3) email Q3 holdout period (sets the horizon h); (4) new-firm over-dispersion (slope 0.89) needs a shape
  fix in a later round, not a level fix.
- Human reviewer (initials) and date:

---

## Glass-box challenger round (challenger-gam, 2026-10-30)

All numbers come from `experiments/gam/GAM-0x/{metrics.md,metrics_calibrated.md,config.json,train_log.md}`, `experiments/gam/paired_bootstrap.csv` (referee's fast firm-weight bootstrap, 200 draws, seed 0, via `experiments/gam/compare_gam.py`) and `experiments/gam/GAM-01/explain.md`. Pooled = test years 2008-2010. "+CAL" = own calibration followed by the D-072 prior shift to rule e (D-071), same taus as CAL-01.

### D-080 EBM / GAM challenger reinstated (supersedes D-008)
- Date / phase / agent: 2026-10-30 / tournament round 2 / challenger-gam (team request)
- Status: ADOPTED (supersedes D-008, which was REJECTED for time, not on evidence)
- Decision: run `ExplainableBoostingClassifier` (interpret 0.7.8) as a glass-box challenger to GB-01. Pre-declared experiments: GAM-01 (main effects), GAM-02 (+10 pairs). GAM-03 (pygam) NOT run (time box; coordinator allowed skipping).
- Business reason: every driver is a readable curve; RiskCalc v3.1 is "closely related to a GAM" (R-01 p. 14 fn. 11).
- Source(s): R-01. EBM / GA2M method papers R-24 (Lou et al. 2012), R-25 (Nori et al. 2019) are NOT verified: any method claim from them is [UNVERIFIED].
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-24 p. 150 eq. 1, p. 151 Tables 1-2; R-25 Sec. 3; R-01 p. 14 fn. 11 (VERIFIED).

### D-081 GAM-01: EBM main effects, monotone constraints from the expected-sign table
- Date / phase / agent: 2026-10-30 / round 2 / challenger-gam
- Status: ADOPTED as the glass-box entry; promotion is the referee's call
- Decision: 26 features (same as GB-01), categoricals as categorical, NaN native (own missing bin), `monotone_constraints = B.expected_signs()` (0 for asset_turnover, extraord_to_assets, rev_nonpositive, categoricals), learning rate 0.03, 128 bins, no class weighting. Tuning on the inner time split (as GB-01, D-052): folds 2-3 inner-train = earlier years, validation A = half of the last year (by firm), B = other half for the Platt slope; pre-declared grid max_leaves {2,3} x min_samples_leaf {200,1000}; fold 1 (one year) pre-declared default (2 leaves, 500). Chosen: fold 1 default; fold 2 and fold 3 3 leaves (min_samples_leaf had no effect: identical A log loss). Final EBM refit on all training rows, 8 outer bags; per bag a different firm-level 15% validation slice for early stopping, all from training rows (passed through `bags`). Own calibration as D-053 (Platt slope from B 1.02 / 0.99 / 0.96, intercept anchored to the training rate).
- Evidence: pooled AUC 0.8254 (AR 0.651), per year 0.8348 / 0.8233 / 0.8177, log loss 0.06517, Brier 0.01379, mean PD 1.13% vs observed 1.54% (obs/pred 1.36), slope 0.964. New firms AUC 0.774 vs seen 0.832. +CAL: log loss 0.06451, mean PD 1.40% (obs/pred 1.10), slope 0.963; per-year AUC unchanged. Gap to GB-01 (200-draw bootstrap): dAUC -0.0120 [-0.0140, -0.0104], d log loss +0.00102 [+0.00088, +0.00116]; +CAL vs CAL-01 (0.06352): dAUC -0.0119 [-0.0139, -0.0103], d log loss +0.00098 [+0.00085, +0.00113]. So the glass box gives up about 0.012 AUC (the sign constraints explain only part of it, GB-02 unconstrained being +0.0065 AUC above GB-01; the rest is the additive form, i.e. no interactions). The gap is about 2/3 of the GB-01 vs RC-02 gap (D-074: 0.0016 log loss).
- Business reason: ranking quality is the price of full readability; credit officers can read every curve.
- Source(s): R-01 p. 14 fn. 11; assumption elsewhere.
- Risks: the interaction-free form cannot capture sector-conditional effects (see D-083). Single seed.
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-25 Sec. 3 (EBM); monotone constraints and the D-082 reset rule = our design.

### D-082 Missing-bin reset rule (found by inspecting the shapes)
- Date / phase / agent: 2026-10-30 / round 2 / challenger-gam
- Status: ADOPTED
- Finding (data artefact, not economics): the EBM gives each feature's "missing" bin its own score. In the first GAM-01 run asset_turnover missing (19 training rows, 0 defaults in fold 3) got -5.10 log-odds, cash_ratio missing (15 rows, 0 defaults) -0.90: a messy-input row would get a PD about 150x too low, and the harness must score such rows. Other missing bins with rich data are meaningful (ar_days missing: 1,109 rows, 6.6% default rate, score +0.48).
- Decision: after the final fit, reset to 0 (= average firm) every missing-bin score (main and pair terms) whose training rows hold fewer than 30 defaults (same threshold as the categorical pooling rule in FEATURE_SPEC), then anchor the level. Implemented in `gam_lib.reset_sparse_missing`. Test rows are unaffected in practice (asset_turnover and cash_ratio have 0 missing test rows): pooled AUC 0.82504 -> 0.82545 (run 1 vs final, rest is re-anchoring), log loss unchanged to 4 digits. Run 1 outputs kept in `GAM-01/metrics_run1_before_missing_reset.md`, `train_log_grid_search_run1.md`. The final GAM-01 run reused the grid choice of run 1 (`--fixed-from`), to save 20 minutes.
- Source(s): assumption.
- Human reviewer (initials) and date:

### D-083 GAM-02: ten pairwise terms give a statistically detectable but economically small gain; not adopted over GAM-01
- Date / phase / agent: 2026-10-30 / round 2 / challenger-gam
- Status: GAM-02 REJECTED for submission as the glass-box entry (kept as evidence); GAM-01 preferred
- Decision: 10 FAST-selected pairs inside the training window (interaction bins 32), same config as GAM-01 per fold. Inner validation gain (A log loss, GAM-02 vs GAM-01): fold 1 0.04917 vs 0.04929, fold 2 0.05318 vs 0.05319, fold 3 0.06773 vs 0.06775.
- Evidence: OOT pooled AUC 0.8262 (+0.0008 [+0.0004, +0.0012]), log loss 0.06510 (-0.00008 [-0.00011, -0.00005]); +CAL 0.06443; 2010 AUC 0.8193. Still -0.0112 AUC vs GB-01 (and -0.0111 vs CAL-01). Gain is about 7% of the gap to GB-01. The 10 selected pairs (fold 3) all involve unconstrained features only: asset_turnover x ateco_group / extraord_to_assets / legal_group / macro_area, ateco x macro_area, extraord x ateco / macro / legal, legal x macro; largest mean |contribution| 0.046 (asset_turnover x ateco). No pair involves a feature with an expected sign, so the sign constraints cannot be violated by interactions (`explain_interaction_face_validity.csv` is empty for this reason). Cost: 10 more heat maps to read, fit time x1.5-2.
- Business reason: the 0.0008 AUC does not buy back the readability; the sector-conditional asset-turnover effect is the only economically meaningful pair (turnover norms differ by industry).
- Source(s): assumption; GA2M idea from R-24 [UNVERIFIED].
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-42 p. 623 eq. 2 (GA2M, FAST) (VERIFIED).

### D-084 Face validity of the EBM shape functions
- Date / phase / agent: 2026-10-30 / round 2 / challenger-gam
- Status: ADOPTED (finding)
- Evidence (`GAM-01/explain.md`, `explain_shapes.csv`, fold-3 model; folds 1-2 overlaid in `shape_functions.png`): all 20 sign-expected features are monotone in the expected direction (constraints; against_share 0). Unconstrained: asset_turnover rises from very low turnover (-0.47, mostly near-zero revenue) to a peak at about 0.5-1 (+0.2), dips, and rises again: consistent with high-turnover thin-margin traders and with low-revenue holding/real-estate firms being safer after controlling for the rest; extraord_to_assets is V-shaped around 0 (no extraordinary items = safest, any large item of either sign raises PD, +0.3 to +0.4); rev_nonpositive = 1 LOWERS the score (-0.19), against intuition: it is a flag mostly overlapping with asset_turnover = 0 (-0.47) and industry, so the two partly offset (read together; reason codes for such firms say "favourable no revenue", which is not credible to an officer: wording must be fixed before any use); macro_area OTHER +0.65 and legal_group OTHER -0.87 are pooled thin levels (unseen levels also map to OTHER in `B.transform`, so a new region or legal form gets these scores: flag for the harness); log_assets has almost no effect (importance 0.016), negative-equity jump +0.8, tangible_ratio the largest driver. Shape stability across folds is good for the large effects; st_debt_share, intangible_ratio and the staircase tails differ between training windows. No data artefact other than D-082. No true non-monotonicity of the "very high liquidity = distress" type appears: current_ratio, cash_ratio, wc_ratio are flat after the knee.
- Human reviewer (initials) and date:

### D-085 Reusable estimation and cost
- Date / phase / agent: 2026-10-30 / round 2 / challenger-gam
- Status: ADOPTED
- Decision: `experiments/gam/gam_lib.py`: `fit_pd_model(train_df, "GAM-01")`, `save_model` / `load_model` (ebm.pkl, feature_params.json, calibrator.json, meta.json), `predict_pd(art, df, delta=0.0)` frozen, row-wise, clipped to [0.0003, 0.99]; `delta` = D-072 prior-shift constant (fold-wise rule-e values are in config.json; the harness would use the `harness_level_params.json` delta). Models for all three folds are saved under `GAM-0x/model_fold{k}/`.
- Evidence: fit time per fold (feature params + inner model(s) + final fit; GAM-01 wall incl. prediction, final fit alone): fold 1 106 s (55 s), fold 2 198 s (114 s; rerun without grid), fold 3 305 s (174 s); the original grid run took 112 / 403 s / about 11 min. GAM-02 final fit 70 / 143 / 264 s. Prediction: 0.8-0.9 s per 100,000 rows (0.8 s of it is the shared feature transform), 8 threads, laptop; about 67 ms for one row (fixed pandas overhead). Complexity O(n x (F + T)) table lookups. Dependency: `interpret` 0.7.8 (pickle); an export of the term tables to JSON would remove it if GAM-01 wins.
- Human reviewer (initials) and date:
- Source(s) added 2026-11-02 (literature-scout): R-25 Sec. 3 (prediction = lookups + additions).

---

## Orchestrator, 2026-11-02

### D-086 Ensemble of GB-01 and the EBM (ENS-01): rejected
- Date / phase / agent: 2026-11-02 / round 2 / orchestrator
- Status: REJECTED
- Decision: do not average GB-01 with GAM-01 (or RC-02) on the log-odds scale.
- Evidence (pooled 2008-10 OOT predictions, 506,159 rows, 7,784 defaults; firm-clustered bootstrap,
  100 draws; `experiments/ensemble/ens01.py`, output `experiments/ensemble/ens01_output.txt`):
  pooled AUC GB-01 0.8375; 50/50 GB-01+GAM-01 0.8343 (dAUC -0.0032 [-0.0042, -0.0024]);
  70/30 0.8363 (dAUC -0.0012 [-0.0017, -0.0007]); three-way with RC-02 0.8322. Worse in every
  test year (2010: 50/50 dAUC -0.0033 [-0.0049, -0.0020]).
- Interpretation: averaging helps when models make different errors; the EBM is a restricted
  version of the same information (main effects of the same features), so it only dilutes GB-01.
  Further gains must come from new information (features) or better tuning, not from blending.
- Source(s): R-10 Sec. 4.4 p. 22 (simple averaging of the RiskCalc and PFM outputs did not add
  stable power); otherwise standard ensemble reasoning.
- Human reviewer (initials) and date:

### D-087 fs_year 2011 is a one-time lockbox for the final model choice
- Date / phase / agent: 2026-11-02 / round 2 / orchestrator
- Status: ADOPTED
- Decision: from round 2 on, no experiment reads fs_year >= 2011 rows for selection. All tuning is on
  inner splits inside the training window; the outer folds (test 2008/09/10) are scored once per
  final configuration. The shortlisted finalists are scored once on fs_year 2011 (11-month label,
  D-077) before the final refit that includes 2011.
- Reason: round 2 runs many more experiments on the same three test years; the best of many
  scores is optimistically biased (selection overfitting; cf. D-062). A sealed year gives one
  less-biased check.
- CORRECTION (2026-11-02, found by report-writer): 2011 is NOT untouched. The calibration round
  already scored it: `reports/calibration/q6_2011_pseudo_test.csv` (11-month label AUC GB-01
  0.8497, RC-02 0.8345; the seven level rules tested on 2011) and
  `reports/calibration/q5_2011_other_labels_fold4.csv`. No model was chosen on these AUCs, so 2011
  still works as a lockbox for the round-2 model choice; it does NOT work as an independent check
  of the level rule (D-073 was informed by it). Say this openly in the update.
- Source(s): R-07 Sec. 3.2 p. 95 (re-using test years during development violates out-of-time
  testing); walk-forward design as in R-33 Sec. 4.8 p. 23.
- Human reviewer (initials) and date:


---

## Round 1b referee diagnostics (merged by orchestrator from reports/tournament/r1b_diagnostics.md, 2026-11-02)

### D-110 Referee verdict on the EBM challengers (GAM-01, GAM-02)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED
- Decision: GAM-01 and GAM-02 are CLEAN (no leakage). Contract PASS (506,159 rows, 0 missing/NaN/out-of-range);
  saved fold models reproduce the submitted own, raw and +CAL PDs exactly (max |diff| 0) on test rows with every
  label column removed and rows shuffled. Process notes: D-082 missing-bin reset was added after a full OOT run
  (robustness rule, training data only, OOT effect +0.0004 AUC, not used as evidence); GAM-02 reused GAM-01's
  in-window grid choice; +CAL inherits rule-e selection optimism (D-071); shared D-062 caveat.
- Alternatives considered: CONCERN by analogy with RC-03 (D-063) - rejected because the rule was triggered by
  training shapes, is not reported as a performance gain, and changes nothing measurable.
- Evidence: r1b_contract.csv, r1b_gam_repro.csv; code read of run_gam.py, gam_lib.py, explain_gam.py.
- Source(s): CLAUDE.md sec. 5 rule 5; data/splits/README.md.
- Risks / what would change our mind: a refit from current code not reproducing the saved models (not run:
  ~10 min per fold at 2 threads; saved-model reproduction done instead).
- Human reviewer (initials) and date:

### D-111 EBM face validity confirmed (D-084)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED
- Decision: 0 violations in 20 sign-expected terms x 3 folds x 2 models (step-function against-share 0); GAM-02
  pairs involve only unsigned features. Flag for officers: rev_nonpositive lowers the score (-0.19), OTHER levels
  of legal_group (-0.87) / macro_area (+0.65) are thin pools that unseen codes inherit.
- Evidence: r1b_gam_face_validity.csv (fold-3 GAM-01 shapes in section 1.5).
- Source(s): docs/FEATURE_SPEC.md expected-sign table.
- Human reviewer (initials) and date:

### D-112 GAM-01 not promoted over GB-01; GAM-01 becomes the transparent fallback (replaces RC-03 in D-067)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (referee); human sign-off requested for the fallback change
- Decision: GB-01 remains champion. GAM-01 replaces RC-03 as the documented transparent alternative.
- Evidence (1,000-draw firm bootstrap, r1b_gam_paired_bootstrap.csv): GAM-01 vs GB-01 dAUC -0.0120 [-0.0139,
  -0.0102], dLL +0.00102 [+0.00088, +0.00115], worse in every year and firm group; GAM-01+CAL vs CAL-01 -0.0119 /
  +0.00098. GAM-01 vs RC-02 dAUC +0.0061 [+0.0044, +0.0078], dLL -0.00063 [-0.00078, -0.00048] (2008 AUC and
  new-firm LL ties), 0 vs 2 face-validity violations. GAM-01 vs SC-03 +0.0089 / -0.00139. GAM-02 vs GAM-01
  +0.0008 [+0.0004, +0.0012] / -0.00008 [-0.00010, -0.00005]: small, GAM-01 kept (D-083).
- Alternatives considered: RC-03 as fallback (CONCERN, D-063, and dominated); GAM-02 (not worth ten heat maps).
- Business reason: if the team or Roger judges SHAP reason codes insufficient, the fallback with readable
  curves is also the stronger transparent model.
- Risks / what would change our mind: the 2011 lockbox reverses the GB-01 vs GAM-01 ranking.
- Human reviewer (initials) and date:

### D-113 Power by segment (lesson L4)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (reporting)
- Decision: report AR by year, size quintile (training-fold asst_tot edges), ATECO section (< 30 test defaults
  pooled; L separate) and seen/new, with firm-clustered CIs and default counts, for GB-01, GAM-01+CAL, RC-02, SC-03.
- Evidence (r1b_segment_power.csv): GB-01 best or tied in every cell; AR Q1 0.626 -> Q5 0.735; C 0.745, I 0.756,
  F 0.559, A 0.550, L 0.643 (997 defaults); seen 0.688, new 0.573; years 0.688 / 0.672 / 0.661. GB-01 minus
  GAM-01 CI excludes 0 in every year, size quintile, firm group and in sections C, E, F, G, J, L, M.
- Source(s): R-33 Tables 9-14 pp. 19-21 (AR by industry, size, year); R-33 Sec. 4.6 fn. 14 p. 22 (real estate).
- Risks: section cells with < 100 defaults have single-model AUC CIs 0.08-0.23 wide; do not rank models there.
- Human reviewer (initials) and date:

### D-114 Missing-input check (lesson L1): no extra under-prediction for GB-01; no n_core_imputed flag
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (no change to GB-01); OPEN for GAM-01 if it ever becomes the submission model
- Evidence (r1b_missing_tests.csv; relative obs/pred = group vs same-year average, Poisson CI): GB-01
  debt_split_missing 1.04 [0.98, 1.09] (88,593 rows, 1,446 defaults); core ratio NaN 1.01 [0.84, 1.22] (1,800 rows,
  114 defaults, DR 6.3%). GAM-01+CAL 1.21 [1.00, 1.46] (missing bins reset to average, D-082); SC-03 1.30
  [1.07, 1.56]; RC-02 0.75 [0.62, 0.90].
- Decision: keep GB-01 as is. For GAM-01, replace "reset sparse missing bins to 0" by a pooled in-window
  "any core missing" bin before any use.
- Source(s): R-34 (mean fills understate PDs), R-05 pp. 60-61, via docs/LIT_moodys_feature_cleaning_2026-10-30.md L1.
- Human reviewer (initials) and date:

### D-115 Re-split grade grid: 16 grades, no grade above ~10% of firms (supersedes the 11 bands of D-078 for reporting)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (proposal for the deck and harness documentation; human sign-off requested)
- Decision: edges 0.03, 0.10, 0.17, 0.25, 0.36, 0.50, 0.62, 0.75, 1.0, 1.2, 1.5, 1.9, 2.5, 5, 10, 20, 99 (%).
  Chosen on fold-3 GB-01 in-sample PDs on 2007-2009 training rows at the CAL-01 level (no 2010 data), by splitting
  any grade above 12% at its rounded median.
- Evidence (r1b_grade_summary.csv): max share 9.9% (edge sample), 10.3% (OOT 2008-09), 10.0% (2010); old grid had
  19.6% / 19.4% in grades 2-3. 2010 observed default rates are monotone across all 16 grades.
- Alternatives considered: equal-share quantile grades (edges not round, move with every refit); keeping 11 grades.
- Source(s): R-33 Sec. 4.9 p. 24 (< 12% of observations per category).
- Risks: the edge sample is in-sample (slightly more spread than OOT); shares on the holdout will differ.
- Human reviewer (initials) and date:

### D-116 The 2010 under-prediction, quantified on the new grid
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (finding; no group-level correction, per D-078)
- Evidence (r1b_grade_table.csv, CAL-01, 2010): 2,968 observed vs 2,511.4 expected (obs/pred 1.18, +457).
  Grades 1-5 (PD < 0.5%, 42.5% of firms): 174 vs 186.1, on target. Grades 6 and 8-14 (PD 0.5-10%, 50.0% of firms):
  above the acceptance band, +501.6 defaults, obs/pred 1.26-1.41. Grade 16 (>= 20%): 451 vs 498.9, over-predicted.
  GB-01 at its own level: obs/pred 1.43, grades 5-15 flagged.
- Reading: a cycle-driven level miss concentrated in the middle of the scale plus a slightly compressed top
  (2010 slope 0.957); ranking intact. Bands ignore default correlation (R-16 pp. 33-34): diagnostic only.
- Business reason: in a 2010-like year mid-grade PDs were 20-30% too low (observed 1.26-1.41x predicted), so pricing would not cover expected loss there.
- Human reviewer (initials) and date:

### D-117 Overfitting and validation evidence for the deck
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (reporting)
- Evidence: OOT calibration slopes 0.96-1.01 by year for GB-01 (new firms 0.88, all models 0.83-0.89); per-year
  AUC range 0.0135 (GB-01, smallest); pooled AUC CI width ~0.009; same-year in-sample minus OOT AUC: GB-01
  +0.015 to +0.026, RC-02 +0.004 to +0.006, GAM-01 +0.006 (r1b_same_year_overfit.csv, r1b_insample_vs_oot.csv,
  r1b_stability_summary.csv).
- Decision: the deck quotes only OOT numbers, shows the same-year gap honestly (GBM fits training years harder;
  R-33 reports -0.5 AR points for RiskCalc France, Sec. 4.7-4.8) and uses the OOT slope ~1 as the evidence that
  the PDs are not over-spread.
- Source(s): R-33 Sec. 4.7-4.8 pp. 22-23; R-10 Sec. 4.5 p. 22.
- Human reviewer (initials) and date:

### D-118 Selection ledger and correction to D-087 (fs_year 2011 is partially spent)
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: ADOPTED (correction of fact in D-087)
- Finding: 30 OOT prediction files and 62 calibration configurations have been scored on 2008-2010
  (r1b_selection_ledger.json). D-087 says 2011 "has so far been used only for a level statistic"; in fact GB-01
  and RC-02 fold-4 models were scored on fs2011 with AUC (reports/calibration/q5_2011_other_labels_fold4.csv,
  q6_2011_pseudo_test.csv; D-076 quotes 0.885 / 0.869 / 0.850) and tau is anchored to the fs2011 rate (D-073).
  GAM-01/02 and SC-03 have not been scored on 2011. (Referee read only the headers/model names of those files.)
- Decision: treat 2011 as valid for ONE ranking decision between GB-01 and GAM-01; never as a level test.
- Human reviewer (initials) and date:

### D-119 Pre-registered one-time 2011 lockbox protocol
- Date / phase / agent: 2026-11-02 / round 1b / referee
- Status: OPEN (proposal; needs orchestrator + human sign-off before anyone opens 2011)
- Decision: freeze finalists (GB-01, GAM-01), code, seeds; fit fold 4 on fs 2007-2010 ok rows; label y11 (D-077)
  primary, s0/s3 sensitivity; metrics AUC and log loss after one common per-model prior shift; same decision
  rule, 1,000-draw firm bootstrap; one run by the referee, one file, result reported whatever it says; then
  refit the chosen model on 2007-2011 (D-077). No other model is scored on 2011.
- Alternatives considered: no lockbox (selection optimism unchecked); a random-firm holdout inside 2010 (not
  out-of-time, and 2010 is already used).
- Risks: one year, one cycle point; a reversal would be a tie-break signal, not proof.
- Source(s): assumption (standard validation practice); R-33 Sec. 4.8 p. 23 (walk-forward).
- Human reviewer (initials) and date:
```

---

## Round 2 features (merged by orchestrator from reports/features/round2_decisions.md, 2026-11-02)

### D-090 Feature set `full2` (opt-in; v1 unchanged)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED (for the GBM challenger; NOT for the probit/logit core)
- Decision: `FULL2_FEATURES` = the 26 v1 features (unchanged) + 19 new features = 45. It is opt-in:
  `B.fit(train, feature_set="full2")` adds `params["v2"]`, and `B.transform(rows, params, feature_set="full2")`.
  The default `fit`/`transform` calls give byte-identical v1 params and outputs (test `test_v1_unchanged`).
  `CORE_FEATURES` and `ALL_FEATURES` are unchanged. Two candidates were evaluated and rejected
  (D-092, D-096). They can still be computed with `feature_set="full2_candidates"`, which is used only by
  the evidence scripts.
- Alternatives considered: changing the full set in place. Rejected because round-1 experiments depend on it.
- Evidence (inner LightGBM, fixed settings, no tuning, `round2_lgb_ablation.csv`, `round2_ablation_seeds.csv`):
  v1 full AUC 0.8313 (3-seed mean, s.d. 0.0006). All 21 candidates: 0.8415 (+0.0102), log loss −0.00063.
  **Final 45-feature set** (interest_to_sales unconstrained; `round2_lgb_ablation.csv`, seeds 0/1/2):
  AUC 0.8403 / 0.8415 / 0.8412, mean 0.8410, vs v1 0.8314 / 0.8307 / 0.8318, mean 0.8313 → **+0.0096 AUC**.
  Log loss mean 0.06664 vs 0.06735 → **−0.00071**. Dropping the two duplicates costs about 0.0005 AUC against all
  21 candidates (within 1 s.d.).
  **No single new feature explains the gain.** Adding any one of them to v1 adds ≤ +0.0025, and dropping any
  one from the full candidate set loses ≤ 0.0033 (tfr_ratio). Part of the +0.010 may therefore come from
  the fixed, untuned LightGBM gaining capacity with more columns rather than from new information. The GBM
  challenger must compare v1 and full2 under its own tuned settings on the frozen folds before claiming the gain.
- Source(s): see D-091..D-097 per feature.
- Risks / what would change our mind: the gain disappears under the challenger's tuned model, or fold-3
  (2010) does not confirm it (D-062 rule).
- Human reviewer (initials) and date:

### D-091 Test of the D-010 hypothesis "`goodwill` = value added"; personnel cost / sales is adopted as hypothesis-dependent
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED (hypothesis supported by four checks; the feature is labelled HYPOTHESIS-DEPENDENT)
- Test (`round2_data_checks.csv`): if goodwill = value added = EBITDA + personnel cost, then
  goodwill − ebitda = personnel cost, and it should be (1) never negative, (2) zero only for firms with
  no staff, (3) linked to the staff-severance fund (TFR, `liab_lt_emp`), which only exists when there are
  employees, and (4) plausible in size. Results:
  (1) negative on 0.012% of rows. (2) zero on 26.2% of rows; of these only 0.94% have TFR > 0, versus
  93.2% of rows with a positive value; 52.5% of the zero rows are real estate (section L, 19.3% of all
  rows), i.e. property companies without employees. (3) Spearman(goodwill − ebitda, liab_lt_emp) = 0.93.
  (4) median personnel cost / sales = 9.3%. Every check is consistent with the hypothesis, and none
  contradicts it. So the feature is kept.
- Decision: `personnel_to_sales` = slog((goodwill − ebitda) / rev_operating); a negative value → NaN (data
  error); the revenue rule is the same as for ar_days. Expected sign + (R-33 p. 12: upward-sloping transform).
  Data: AUC 0.570 (2009 inner 0.562); deciles flat at about 1.1-1.3%, then D10 2.58% → + confirmed.
- Risks: if the bank's field definition says otherwise, drop the feature. It is not in the core.
  Spearman with tfr_ratio is 0.75 (both measure labour intensity).
- Source(s): R-33 Table 3 p. 11, p. 12; field meaning: D-010 (data-auditor), tested here.

### D-092 RiskCalc 4.0 France ratios (F1): what was adopted, and what was rejected because our data already contain it
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: operating_margin, payables_to_sales, personnel_to_sales, interest_to_sales ADOPTED;
  net_debt_coverage and gross_margin REJECTED
- Decisions and evidence (`round2_univariate_stats.csv`, `round2_rho_pairs.csv`):
  - **net_debt_coverage** = cf_operations / (total liabilities − cash), total liabilities = asst_tot − eqty_tot.
    Rule for a denominator ≤ 0 (cash covers every liability, 2.9% of rows, default rate 0.70% vs 1.30%):
    +inf if the firm generates cash (clipped to p99 = the safest value), 0 if not. The ratio works
    (AUC 0.247, monotone deciles 5.75% → 0.40%), but its Spearman with v1 `cfo_to_debt` is **0.990**. In our data,
    netting cash and widening the denominator from debt to all liabilities hardly changes the ranking.
    Adding it to v1 gives +0.0003 AUC (noise; s.d. 0.0006). → REJECTED as a duplicate. Our `cfo_to_debt` already
    *is* RiskCalc France's debt-coverage family, which answers L7(i).
  - **gross_margin** = (rev − COGS)/rev is **identical** to operating_margin on 99.993% of rows, because
    prof_operations = rev_operating − COGS exactly (01_profile.md §5; COGS = all operating costs). → REJECTED
    (not implemented).
  - **operating_margin** = slog(prof_operations / rev), rev ≤ 0 → NaN. AUC 0.291, deciles 4.53% → 0.55%. Spearman
    0.87 with ebitda_margin (the measure RiskCalc 4.0 France replaced). ADOPTED for full2 (trees tolerate the overlap, D-025).
  - **interest_to_sales** = slog(exp_financing / rev); rev = 0 → +inf if interest > 0, else 0; rev < 0 → NaN. AUC 0.580,
    but the deciles are hump-shaped (0.93 … 2.14, 2.20, then 1.27, 1.43 in D9-D10). The top deciles are firms
    with small sales relative to debt (holding / real-estate type). R-33 p. 12 reports an upward slope.
    **Our data contradict a monotone '+' at the top, so expected sign = none** (no monotone constraint), the same
    handling as asset_turnover in D-022. ADOPTED.
  - **payables_to_sales** = slog(AP_st / rev); same revenue rule; NaN if the debt-split block is missing (impute:
    sector share × debt_st, D-021). AUC 0.636, monotone deciles 0.79% → 2.43%. Spearman 0.93 with ap_days (COGS ≈
    revenue here). Added alone it adds nothing measurable (−0.0003). ADOPTED for full2 because it is the R-33
    definition and its sales denominator is defined for more firms (COGS ≤ 0 → NaN in ap_days). A GLM user should take one of the two.
  - Zero / negative revenue: the same rule as v1 `ar_days` / `rev_nonpositive` (D-019).
- Source(s): R-33 Table 3 p. 11 (inputs), Table 4 p. 14 (weights), p. 12 (directions of the transforms), p. 10 Table 2 (AP/sales example).
- Human reviewer (initials) and date:

### D-093 Sector-relative percentiles (F2)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED for full2 (GBM); not for GLMs next to their base ratio
- Decision: for equity_ratio, roa_net, interest_coverage, cash_ratio, st_debt_ratio and tangible_ratio, add
  `<ratio>_sect_pct`, the firm's percentile within its ATECO section. Each section has a 101-point quantile grid
  of the training-window (winsorised) feature, stored in params. Linear interpolation is used, ties in the grid
  get their mid-percentile, values outside the grid → 0 / 1. A section with < 2,000 training rows, or a
  missing / unseen section, → pooled grid. NaN base → NaN (impute view: computed from the imputed base, so ≈ 0.5).
- Evidence: Spearman with the base ratio is 0.88-0.98 (tangible 0.88, st_debt 0.91, the rest ≥ 0.96), so the
  correction is modest. The univariate AUC is close to the base. In the 3-seed ablation, adding single
  percentiles to v1 gives between −0.0002 and +0.0008. Dropping the whole block from the 21-candidate set
  loses 0.0009 AUC and 0.00013 log loss (about 1.5 s.d.). This is small but consistent.
- Alternatives: sector-demeaned ratios (z-scores) need a scale per sector and are more fragile with tails;
  percentiles are the RiskCalc-style "percentile score" (R-33 p. 12).
- Source(s): R-01 Sec. 3.3 pp. 17-18 (industry corrects ratios whose meaning differs by sector).
- Risks: VIF up to 11.7 (st_debt_ratio_sect_pct) in a GLM. Do not use a percentile and its base together in a GLM.

### D-094 Shrunk default-rate encodings for ATECO division, province and region (F3)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED for full2
- Decision: `ateco_div_te` (83 divisions), `province_te` (110 HQ_city codes) and `region_te` (20 regions).
  Each value = logit((d_l + m·p0)/(n_l + m)) − logit(p0): the level's default rate pulled toward the
  training base rate p0, on the log-odds scale, centred so that 0 = average. **Shrinkage strength m by
  empirical Bayes** (method of moments: between-level variance of default rates minus binomial noise
  gives tau²; m = p0(1−p0)/tau²), fitted in-window, with guard rails [50, 50,000]. Fitted values on 2007-09:
  m = 555 (division), 1,834 (province), 3,879 (region) rows. A level with n rows keeps the share n/(n+m)
  of its own rate; none hit a guard rail. Missing / unseen level → 0 (the prior).
  **Out-of-fold:** training rows get encodings from 5 folds grouped by firm id (fold = crc32(id) mod 5,
  deterministic, stored in params as per-fold tables): `transform(..., oof=True)`, training only.
  Scoring / harness: the full-window table (`oof=False`, the default).
- Evidence (`round2_target_encoding.csv`, `round2_lgb_ablation.csv`): training AUC in-sample vs OOF: division
  0.592 vs 0.580, province 0.564 vs 0.547, region 0.536 vs 0.530. The in-sample values are inflated by the
  rows' own labels. 2009 inner AUC: 0.572 / 0.537 / 0.529. LightGBM on 2009: v1 + OOF encodings 0.8356, but
  v1 + IN-SAMPLE (leaky) encodings 0.8330, and log loss even worse than v1 (+0.00007). Leakage hurts out of
  time, which confirms the OOF design. 3-seed single additions: province +0.0017, region +0.0017, division +0.0010.
- Alternatives: one-hot of 83 / 110 levels (many with < 30 defaults, D-023); fixed m chosen by hand (arbitrary).
- Source(s): assumption (standard empirical-Bayes / m-estimate encoding; no VERIFIED reference in REFERENCES.md;
  literature-scout could add one). R-01 Sec. 3.1 p. 14 supports industry information in general.
- Risks: geography encodes local conditions *in 2007-09*, which may not persist. Province and region are nested (Spearman 0.45).

### D-095 Missingness / data-quality counts (F4, part 1)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: n_core_imputed ADOPTED; n_missing_raw REJECTED
- Evidence (`round2_count_rates.csv`):
  - **n_missing_raw** (count of 32 missing raw amounts; roe and margin_fin excluded because they are blank
    by construction when equity ≤ 0): values 0 / 3 / 6 cover 99.7% of rows (the debt-split block). Spearman with
    v1 `debt_split_missing` = **0.994**, AUC 0.505, single addition −0.0001. → REJECTED as a duplicate.
  - **n_core_imputed** (core numeric features still NaN after identity reconstruction, i.e. those a GLM fills
    with a sector median): 0 on 474,379 rows (1.29% default), **1 on 1,169 rows (6.33% default)**, 2 on 278 rows
    (2.16%). The "1" group is almost all `ar_days` undefined because **revenue is negative** (1,078 rows with
    rev < 0, default rate 6.8%). The v1 flag `rev_nonpositive` mixes zero and negative revenue and has no
    sign (D-022). This is exactly the L1 / R-34 effect: a median fill would make these firms look average when
    they are 5× riskier. Sign +. AUC 0.505 only because of its low prevalence. In the GBM ablation its single
    addition is within noise (−0.0004), so the case rests on the per-group default rates and on harness safety
    (messy holdout rows), not on AUC. ADOPTED.
- Source(s): R-34 (median fill understates PDs), R-05 pp. 60-61 (less real data, worse estimate), via L1.
- Risks: on the holdout the count may also pick up formatting junk that a training row never had; it is
  never extrapolated beyond the training values by trees.

### D-096 Balance-sheet gap and rounding flag (F4, part 2)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: bs_gap ADOPTED (sign none); rounding flag REJECTED
- **Units check first:** all amounts are whole euros (100% integer-valued; median assets EUR 3.44MM), so
  "rounded to thousands" is a meaningful test. Only 0.26% of asst_tot values are multiples of 1,000 (chance
  would be 0.1%).
- **Rounding flag** (all 5 key amounts non-zero multiples of 1,000): 503 rows (0.11%) with **2 defaults**
  (0.40% vs 1.30%). A looser version (≥ 3 of 5) has 702 rows and 7 defaults. Far too few defaults to estimate
  anything, so → REJECTED. R-01 p. 22 found rounded rows informative in a much larger sample. Here the flag
  would be noise.
- **bs_gap** = (asst_tot − eqty_tot − debt_st − debt_lt − liab_lt − liab_lt_emp) / asst_tot: the liability-side
  item the file does not classify (data-auditor: residual ≥ −€2 on 99.84% of rows; most likely accruals /
  deferred income [hypothesis]). Distribution: median 0.1%, p90 2.5%, p99 24%. AUC 0.482 (weakly −); deciles
  are non-monotone (1.48 … 1.11 … 1.41, 1.27). Sign none. In GBM it is one of the stronger single additions
  (+0.0018). ADOPTED for full2 without a monotone constraint. In the deck, describe it as "unclassified
  liabilities", not as a data error.
- Source(s): R-01 Sec. 3.4.1 p. 22 (rounding); R-33 Sec. 2.2 p. 6 (balance-sheet plausibility); assumption for the meaning.

### D-097 Unused fields (F5) and their signs
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED for full2
- nonbank_fin_debt = (debt_fin_st + debt_fin_lt)/asst_tot, sign + (more fixed claims). Data: AUC 0.523, mass
  at 0 then 1.54% / 1.64% in the top deciles → + confirmed. NaN when the block is missing (15.7%). Impute: sector share.
- ap_lt_ratio = AP_lt/asst_tot, sign + (suppliers turned into long-term creditors). Mostly zero; AUC 0.501, top
  decile 1.41% vs 1.30% → weak + (not contradicted). Single GBM addition +0.0006 (about 1 s.d.).
- fin_fixed_ratio = asst_fixed_fin/asst_tot (holding / group structure), sign **none**: group support vs.
  illiquid stakes. Deciles non-monotone (1.21-1.52%); AUC 0.503; GBM single addition +0.0014, log loss −0.00014.
- tfr_ratio = liab_lt_emp/asst_tot (staff-severance fund). In round 1 it was rejected for having "no expected
  sign" (D-024). Now it is **adopted with sign none**. Two opposite stories (labour-intensive firms with rigid
  costs vs. a cheap internal source of funding owed to staff); deciles hump-shaped (1.15, 1.29, 1.65, 1.53 …
  1.09, 1.37). Univariate AUC 0.510, **but it is the largest single GBM contribution**: +0.0025 added alone,
  −0.0033 when dropped from the candidate set (5 s.d.). The tree uses it in interaction (labour intensity is
  a sector / business-model descriptor). It is unconstrained, so face validity rests on the shape plot.
- Source(s): assumption (no VERIFIED source for these fields).
- Risks: tfr_ratio is Italy-specific; the Spearman with personnel_to_sales is 0.75. Explaining it to a
  credit officer: "how labour-intensive the firm is", not "more TFR = more risk".

### D-098 Collinearity of full2 (Roger M8)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED
- Decision: full2 is a tree-model set. D-025 applies: in GBM, high VIF is tolerated and SHAP is reported by
  family. Pairs with |rho| ≥ 0.70 that involve an adopted new feature (`round2_rho_pairs.csv`): the 6 sector
  percentiles with their bases (0.88-0.98); operating_margin–ebitda_margin 0.87; payables_to_sales–ap_days 0.93;
  personnel_to_sales–tfr_ratio 0.75; roa_oper / cfo_to_debt–roa_net_sect_pct 0.73 / 0.78; cfo_to_debt–
  interest_coverage_sect_pct 0.72. The two duplicates at ≥ 0.99 were dropped (D-092, D-095). VIF (imputed, all
  v1 + candidates): st_debt_ratio 48, tangible_ratio 37, wc_ratio 36, ebitda_margin / operating_margin 15.5,
  st_debt_ratio_sect_pct 11.7. Any GLM using full2 must use L2 regularisation or drop one member of each listed pair.
- Source(s): R-01 Sec. 4.2 p. 26 (VIFs).

### D-099 Rejected / not built in round 2, and open caveats
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: REJECTED (items 1-4), OPEN (5-7)
1. gross_margin: identical to operating_margin (D-092).
2. net_debt_coverage: duplicate of cfo_to_debt, rho 0.990 (D-092).
3. n_missing_raw: duplicate of debt_split_missing, rho 0.994 (D-095).
4. rounding flag: 503 rows / 2 defaults (D-096).
5. NLP features: still REJECTED (D-003). The data have no text field. Macro / cycle features: still not
   features (D-004); this is left to calibration.
6. OPEN: the sign of interest_to_sales was changed from + to none **after** seeing the 2007-09 deciles. That is
   the same post-hoc pattern the referee flagged in D-063. Report it that way.
7. OPEN: the round-2 choices were made on 2007-09 (D-062): quote fold 3 (2010) separately. `docs/FEATURE_SPEC.md`
   is regenerated by `src.features.univariate` (v1), which **overwrites** the round-2 section. Rerun
   `src.features.univariate_v2` afterwards to restore it.

---

## Round 2 phase A: challenger-gbm (merged by orchestrator from experiments/gbm/decisions.md, 2026-11-02)

### D-100 Round-2 anti-overfitting protocol, and where it was not fully met
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: ADOPTED
- Decision: every choice is made on the fold's inner slices (fold 3: train 2007-08, validate 2009, firm halves A/B; fold 2: train 2007, validate 2008; fold 1: firm split 70/15/15 inside 2007). Outer test years are scored once per final configuration. fs_year >= 2011 is excluded by a pyarrow read filter in `r2_lib.load_panel`, `compare_r2.py`, and the round-1 scripts `explain.py`, `run_experiment.py`, `compare.py` (those three were patched; only the read changes).
- Alternatives considered: a single choice made once on the fold-3 inner split and applied to all folds. It leaks: 2009 labels would inform the fold-2 score.
- Business reason: a level or ranking gain that exists only because the test year was peeked at would disappear on the holdout.
- Evidence: GB-01 rebuilt through `r2_lib` (GB-01r) matches GB-01 row by row, max |PD diff| = 0.0. The 15:12 edit of src/features/build.py did not change the 26 round-1 features (`check_feature_stability.py`: same list, identical fitted params, fold-3 OOT diff 0.0). Known departures: (1) the GB-08 and GB-08b constraint sets are fixed across folds and were informed by inner splits of later folds (D-102), so their 2008 and 2009 scores are not strictly clean; 2010 is. (2) GB-08b was conceived after GB-08's outer score was seen. (3) The sklearn RF replaced the LightGBM RF after the latter's outer score was seen (D-106). (4) Round-1 outer results, including GB-02 and GB-06, were known before round 2 started.
- Source(s): assumption (protocol set by the orchestrator); R-07 and R-01 for walk-forward as the preferred design.
- Risks / what would change our mind: the lockbox check later on is the real test of any promotion.
- Human reviewer (initials) and date:

### D-101 GB-07 overfitting diagnostics: GB-01 shows no harmful overfitting
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: ADOPTED (finding)
- Decision: keep GB-01's regularisation and the 1.2 x best_iter rule (D-052). The in-sample vs out-of-time gap is mostly not reducible by tuning.
- Alternatives considered: more shrinkage or fewer trees, tested through GB-09 (D-104).
- Business reason: a model that memorises the training window would rank new applicants worse than its validation suggests.
- Evidence (`GB-07/gap_by_fold.csv`, `learning_curve_flatness.csv`, `calibration_by_year.csv`, `learning_curve.png`, `gap.png`): GB-01 in-sample vs next-year AUC 0.8476/0.8441, 0.8590/0.8359, 0.8498/0.8307, an AR gap of 0.7, 4.6, 3.8 points. GB-02 AR gap 5.2, 8.7, 6.5 points, yet GB-02 scores higher out of time, so a bigger gap is not by itself worse. Inner log loss at 1.2 x best_iter is +0.00009, +0.00004, +0.00003 above the minimum; at 2x +0.00033, +0.00022, +0.00016. Calibration slope 0.969, 1.005, 0.957. Firm-halves A and B of the same year differ by up to 0.017 AUC (fold 2: 0.8533 vs 0.8364), far more than any variant gap here, which is why only paired tests are used.
- Source(s): R-33 Sec. 4.7-4.8 pp. 22-23 (in- vs out-of-sample AR gap -0.5 points, N-33). Not like for like: France, different model and design, and our gap includes a move across years and into new firms.
- Risks / what would change our mind: the split of the gap between drift and overfitting is inferred, not measured. Evidence for drift: fold 2's inner model shows no 2007 to 2008 decay (0.8528 vs A 0.8533, though B is 0.8364).
- Human reviewer (initials) and date:

### D-102 GB-08 ablation: the cost of the constraints is real but overlapping, and the "denominator" rule was post hoc
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: ADOPTED (finding); the denominator rule itself is REJECTED as an economic justification
- Decision: report the ablation as a measure of cost. Do not use "denominator near zero" as the economic reason for dropping constraints.
- Alternatives considered: dropping the costliest constraints regardless of economics (roa_net, roa_oper); rejected because these are primary reason codes.
- Business reason: each dropped constraint must carry a reason a credit officer would accept.
- Evidence (`GB-08/ablation_summary.csv`, `checks_additivity.csv`, `checks_tail_dr.csv`, `checks_sector_mix.csv`, `shapes_costly.png`, `freed_shapes.png`): mean A+B log-loss cost, largest first: roa_net 0.00024, cfo_to_debt 0.00022, roa_oper 0.00019, interest_coverage 0.00018, ebitda_margin 0.00010. All dropped 0.00052. Sum of single costs 0.00069, 0.00177, 0.00161 vs all-dropped 0.00021, 0.00052, 0.00084, so the costs overlap and do not add. No CI on single-drop costs. Raw default rates do rise in the top tail, in both years checked and also for the assets-denominated ROA ratios: e.g. cfo_to_debt 2007 lowest ventile 0.21% [0.13, 0.34] vs top 0.40% [0.28, 0.58]; roa_net 2009 0.46% [0.34, 0.63] vs 0.89% [0.71, 1.12]. The lowest ventile is picked after the fact, so these contrasts flatter the rise. Because ROA shows the same rise, a near-zero denominator does not explain it. The property-vehicle story holds only for ebitda_margin (section L 46% of the top quintile vs 10% of the rest, 2007); for interest_coverage and cfo_to_debt section L is under-represented (13% vs 19%, 14% vs 19%).
- Source(s): assumption. No verified source on the top-tail rise. Cause unresolved.
- Risks / what would change our mind: a verified source or a data explanation for why very strong ratios default more often.
- Human reviewer (initials) and date:

### D-103 GB-08 and GB-08b: statistical winners, not adopted on face validity
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: REJECTED for GB-08; OPEN for GB-08b (human decision requested, agent recommends against)
- Decision: keep all 20 constraints. GB-08 drops interest_coverage, cfo_to_debt, ebitda_margin. GB-08b drops ebitda_margin only.
- Alternatives considered: GB-02 (D-051); reclassifying ebitda_margin as "no expected sign" like asset_turnover, which would be a post-hoc FEATURE_SPEC change.
- Business reason: reason codes such as "strong cash cover raises PD" cannot be defended in a decline conversation.
- Evidence: GB-08 dAUC +0.0032 [+0.0024, +0.0042], d log loss -0.00035 [-0.00043, -0.00027]; AUC 0.8465, 0.8386, 0.8354. GB-08b dAUC +0.0015 [+0.0009, +0.0021], d log loss -0.00016 [-0.00019, -0.00011]; AUC 0.8451, 0.8374, 0.8328; every per-year CI excludes zero on both metrics. Both meet the win rule. Face validity: GB-08 cfo_to_debt CONTRADICTS (58%), interest_coverage CONTRADICTS (56%), ebitda_margin mostly agrees (44%); at the 99th percentile of cfo_to_debt the log-odds are -4.66, as high as cash burners at -4.64; 5.2% of 40,000 sampled 2009 firms get a freed ratio in their top-3 PD-raising reasons while above its PD minimum (`GB-08/checks_wrong_way_reason_codes.csv`). GB-08b ebitda_margin CONTRADICTS (70%): PD multiple vs median margin is 0.87-0.90x for margins of -6% and below, and 1.18-1.22x for margins of 56% and above; 5.7% of firms get it as a top-3 raiser with a top-quintile margin (`GB-08b/checks_ebitda_shape.csv`, `checks_ebitda_reason_codes.csv`). The top-tail side has a supported story (section L, median turnover 0.06). The loss-making side does not. Inner B agrees with GB-08 in all folds but is not independent: A+B costs informed the set.
- Source(s): assumption (credit-officer face validity, D-051, D-064).
- Risks / what would change our mind: a referee face-validity rule that accepts limited non-monotonicity, or Kiet deciding that a margin effect capped at about +/-20% PD is acceptable. An untested alternative for feature-architect: give the model the business-model signal directly so ebitda_margin can stay constrained.
- Human reviewer (initials) and date:

### D-104 GB-09 random search: formally passes on AUC, not recommended for promotion
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: REJECTED as a replacement (agent judgment stricter than the win rule; orchestrator may overrule)
- Decision: keep GB-01's pre-declared small grid.
- Alternatives considered: 37 configs (GB-01's 5 + 32 random, sklearn ParameterSampler, seed 0): learning_rate log-uniform 0.015-0.10; max_depth 2, 3, 4; num_leaves 4-16 capped at 2^depth; min_data_in_leaf 200-4000; feature and bagging fraction 0.5-1.0; lambda_l1 0-10; lambda_l2 log-uniform 0.1-100; min_gain_to_split 0-0.5. Same lightgbm early stopping on A.
- Business reason: a more complex config should earn its place on the holdout, not on one year.
- Evidence: dAUC +0.0009 [+0.0004, +0.0014]; by year +0.0001 [-0.0008, +0.0010], +0.00003 [-0.0009, +0.0010], +0.0021 [+0.0012, +0.0028]. d log loss -0.00006 [-0.00011, -0.00000]; by year +0.000003, +0.00011, -0.00028, so WIN_ll is false. Slope 0.968 vs 0.976. Chosen configs differ widely (lr 0.024, 0.050, 0.016; depth 3, 4, 4; final trees 322, 198, 1052). Selection check (`GB-09/selection_check.csv`): Spearman A vs B log loss 0.31, 0.69, 0.68; the fold-1 A gain reverses on B; the top 10 configs lie within 0.00008-0.00015 A log loss.
- Source(s): assumption.
- Risks / what would change our mind: if the 2010 gain were confirmed on the lockbox, the fold-3 config might be worth testing as a fixed config for the final refit.
- Human reviewer (initials) and date:

### D-105 GB-10 recency weights: no gain in ranking
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: REJECTED
- Decision: no recency weighting of training rows.
- Alternatives considered: half-life none, 4, 2, 1, 0.5 years, crossed with GB-01's candidates.
- Business reason: test whether recent years describe next year better.
- Evidence: dAUC +0.0002 [-0.0003, +0.0006], d log loss +0.000002 [-0.00003, +0.00004]. Folds 1-2 are identical to GB-01 by construction: with one inner-train year weights cannot be tested, so the tie goes to no weighting. Hence this experiment tests only fold 3 and cannot meet the all-years rule. Fold 3 chose half-life 0.5, the edge of the grid, on an A gain of 0.00012; B log loss improved 0.00005 but B AUC fell (0.8414 vs 0.8433); 2010 AUC +0.0005 [-0.0009, +0.0017]. The intercept stays anchored to the unweighted training rate, so any level benefit is excluded by design; level belongs to calibration (D-071).
- Source(s): assumption.
- Risks / what would change our mind: one fold is thin evidence; a longer window at the final refit could be retested.
- Human reviewer (initials) and date:

### D-106 GB-11 random-forest benchmark: GB-01 does not clear it on ranking
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: ADOPTED as a benchmark, not a candidate. LightGBM boosting='rf' implementation (GB-11lgbrf) REJECTED.
- Decision: report sklearn RandomForest (300 trees, sqrt features, 50% rows per tree, one-hot categoricals, D-053 calibration) as the benchmark and state plainly that the monotone GB-01 does not beat it.
- Alternatives considered: LightGBM RF: per-year AUC 0.8429, 0.8340, 0.8229 but pooled 0.7896 and raw inner log loss 0.09-0.17, with raw 99th-percentile PD 0.95-0.998. Attribution to specific LightGBM settings (lambda_l2=0 etc.) is untested.
- Business reason: R-17 says beating logistic regression alone no longer signals progress, so the model needs a stronger reference point.
- Evidence: GB-11 AUC 0.8395 (0.8449, 0.8396, 0.8335), log loss 0.06392. vs GB-01: dAUC +0.0020 [+0.0001, +0.0038], d log loss -0.00023 [-0.00039, -0.00007]; WIN_auc true (lower bound barely above zero), WIN_ll false (2008 +0.00001). GB-02 vs GB-11: +0.0045 [+0.0028, +0.0064]. GB-08 vs GB-11: +0.0012 [-0.0005, +0.0030]. So boosting beats RF when both are unconstrained; GB-01's shortfall is the price of monotonicity. RF in-sample AUC 0.953-0.962 is not a useful overfit signal because leaves memorise rows. min_samples_leaf 50 was the grid edge in all folds, so the benchmark may be understated.
- Source(s): R-17 Conclusions, MS p. 25 (VERIFIED; retail credit scoring, not corporate).
- Risks / what would change our mind: RF face validity was not measured and no fold-3 model was saved, so GB-11 has no explain output. RF is unconstrained and is not eligible for submission on current rules.
- Human reviewer (initials) and date:

### D-107 GB-12 combination not run
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: REJECTED (not run; no outer scoring used)
- Decision: skip GB-12. The spec remains in `run_r2.py` for later use.
- Alternatives considered: GB-08 constraints + GB-09 top-5 configs per fold + weights.
- Business reason: a combination is only useful if it could be submitted.
- Evidence: of the pieces, GB-08 and GB-08b fail face validity (D-103), GB-10 is null (D-105), GB-09 is not recommended (D-104). No two eligible pieces exist to combine, and another outer look would not change any decision.
- Source(s): n/a.
- Risks / what would change our mind: if Kiet accepts GB-08b (D-103), run GB-08b + GB-09 once.
- Human reviewer (initials) and date:

### D-108 Pipeline accepts a feature-set name; round-1 scripts carry the lockbox filter
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: ADOPTED
- Decision: `run_r2.py <exp> --features full|core|<list name in build.py>`; `r2_lib.py` holds the general fitter, `gbm_lib.py` is untouched.
- Alternatives considered: editing gbm_lib, rejected because GB-01 must stay reproducible for the harness.
- Business reason: a new feature set can be tested without code changes.
- Evidence: the feature-set route was run only with "full". FULL2_FEATURES now exists in build.py but was not run from this pipeline. One limit to check: `B.transform` returns `ALL_FEATURES`, so a list outside it would need the caller to confirm columns exist.
- Source(s): n/a.
- Risks / what would change our mind: not exercised with FULL2.
- Human reviewer (initials) and date:

### D-109 Recommendation, round 2 phase A: GB-01 stays the GBM incumbent
- Date / phase / agent: 2026-11-02 / round 2 phase A / challenger-gbm
- Status: OPEN (recommendation; promotion belongs to the referee, GB-08b to the team)
- Decision: keep GB-01 as the constrained GBM entry. Look for further gains in information (phase B, FULL2), not tuning or weighting.
- Alternatives considered: GB-02, GB-08, GB-08b, GB-09, GB-10, GB-11 (above).
- Business reason: approval needs ordering and pricing needs level; none of these variants moves the level (mean PD 1.14-1.15% vs observed 1.54%), which stays with calibration.
- Evidence: comparison.md round-2 section. Tuning plus weighting yields at most +0.0009 AUC. Constraints cost up to +0.0065 and are the only large lever, but using it breaks face validity.
- Source(s): R-17 (benchmark); otherwise assumption.
- Risks / what would change our mind: the win rule counts a +0.00003 year as "better" and has no multiplicity correction across six outer-scored variants. Suggest the referee also require per-year CIs or a minimum effect size.
- Human reviewer (initials) and date:

---

## Round 2 phase B: challenger-gbm (merged by orchestrator from experiments/gbm/decisions_r2b.md, 2026-11-02)

### D-120 Feature set full2 beats v1 in the GBM (GB-20 meets the win rule vs GB-01)
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED by the GBM challenger as its round-2 entry; promotion over GB-01 is OPEN for the referee
- Decision: GB-20 = GB-01's grid, inner scheme, calibration and constraint rule on the 45 full2 features.
  Training rows use out-of-fold encodings; slices A/B and test rows use the fold's full-window table.
- Alternatives considered: stay on v1 (GB-01); GB-23 (D-127); partial drops (D-122).
- Business reason: better ranking for approve/decline and better log loss for pricing, with no monotone violation.
- Evidence: pooled AUC 0.8435 vs 0.8375, dAUC +0.0061 [+0.0047, +0.0074]; d log loss -0.00048 [-0.00059, -0.00038].
  Per year dAUC: 2008 +0.0034 [+0.0006, +0.0062], 2009 +0.0086 [+0.0067, +0.0109], 2010 +0.0062 [+0.0040, +0.0084].
  d log loss: 2008 -0.00018 [-0.00036, +0.00001], 2009 -0.00066 [-0.00085, -0.00048], 2010 -0.00059 [-0.00078, -0.00041].
  Calibration slope pooled 0.996 vs 0.976; 2010 0.957 vs 0.957. Common calibration (CAL-01 method): GB-20+CAL vs CAL-01
  dAUC +0.0060 [+0.0049, +0.0075], d log loss -0.00050 [-0.00060, -0.00038] (200 draws).
  Recipe check: run on v1 features, the driver reproduces GB-01's best_iter (408, 322) and B log loss (0.05579, 0.06518).
- Source(s): R-38 (algorithm), R-39 (monotone_constraints option); D-062 reporting rule.
- Risks / what would change our mind: the referee's refit check; a reversal on the lockbox (D-119 currently freezes only
  GB-01 and GAM-01, so GB-20 must be added before anyone opens 2011).
- Human reviewer (initials) and date:

### D-121 The feature-architect's +0.0096 is confirmed in direction, smaller on the clean year
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (reporting)
- Decision: quote +0.0062 (2010) next to the pooled +0.0061; do not quote +0.0096 as the expected gain.
- Alternatives considered: quoting the inner number, or the 2009 number (+0.0086). Both use a year the features were designed on (D-062).
- Business reason: the deck must state the gain a new year is likely to deliver.
- Evidence: fold-3 inner on the frozen scheme +0.0093 [+0.0066, +0.0119], close to the architect's check (D-090). Outer: 2009 +0.0086,
  2010 +0.0062. The feature-architect's concern that the gain is untuned capacity is not supported: v1 and full2 each stop early
  on their own best tree count (fold-3 inner mean 350 vs 595), and the gain survives GB-09's tuning (D-127).
- Source(s): D-062, D-090.
- Risks / what would change our mind: one clean year only; 2010 is also one point in the cycle.
- Human reviewer (initials) and date:

### D-122 Which groups carry the gain: F1 and F5 are robust, F3 only in the longer window, F2 and F4 add little
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (finding); the set stays as designed, so that GB-20 stays a test of the pre-declared full2
- Decision: report by group. Do not read SHAP share as value.
- Alternatives considered: pruning F2 now. Not done: inner gain from pruning is not established, and pruning after seeing the outer result needs its own run.
- Business reason: fewer, explainable drivers are easier to defend to a credit officer.
- Evidence (GB-21, dAUC = drop group minus full2, fold-2 inner; fold-3 inner): F1 -0.0025 [-0.0041, -0.0006]; -0.0016 [-0.0030, -0.0006].
  F5 -0.0020 [-0.0041, -0.0003]; -0.0022 [-0.0033, -0.0011]. F3 -0.0003 [-0.0030, +0.0025]; -0.0028 [-0.0044, -0.0013].
  F4 +0.0001 [-0.0012, +0.0012]; -0.0011 [-0.0018, -0.0003]. F2 +0.0004 [-0.0011, +0.0019]; -0.0008 [-0.0019, +0.0004]
  (log loss +0.00003, +0.00014). All 19 together: -0.0053, -0.0093. The single drops sum to far less than the total, so the groups
  substitute for each other. F2 is the largest SHAP family (sum of mean |SHAP| 0.56) with no measurable AUC value: its attribution is
  taken from equity_ratio (0.212 to 0.091) and roa_net (0.208 to 0.128).
- Source(s): R-41 (TreeSHAP); D-025 (attribution among correlated features).
- Risks / what would change our mind: inner slices are 2008 and 2009, which the feature design saw (D-062).
- Human reviewer (initials) and date:

### D-123 tfr_ratio: conditional effect is "more fund, lower PD"; D-097's interaction wording is corrected
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (reading); stays unconstrained
- Decision: describe TFR to a credit officer as "a large staff-severance fund for a given wage bill goes with lower risk, other things equal", not as "labour intensity".
  We do not impose a minus sign: that would be chosen after seeing the shape (the D-063 pattern).
- Alternatives considered: D-097's reading (labour intensity; "the tree uses it in interaction"); sign minus; dropping it.
- Business reason: this is the most robust new single feature, so Kiet must be able to explain it.
- Evidence: drop tfr_ratio alone -0.0023 [-0.0042, -0.0007] and -0.0020 [-0.0030, -0.0010], the only single drop whose CI excludes zero and that is negative in both inner windows among those tested.
  SHAP by band (2010 sample): 0 +0.112, 0-2% +0.094, 2-5% -0.096, 5-10% -0.283, >10% -0.199.
  (1) Sector mix hides it: 48% of zero-TFR rows in the 2010 sample are real estate. Training-row AUC 0.510 all, 0.473 excluding L, 0.448 in manufacturing.
  (2) Suppression: standardised logit coefficient (excl. L) -0.043 alone, -0.197 next to personnel_to_sales (+0.325).
  (3) Fund / wage bill: default rate falls from 2.3% (decile 1) to 1.0% (decile 9), then 1.7% in decile 10.
  (4) Equity: tfr SHAP is the same across equity terciles (5-10% band: -0.27, -0.29, -0.28), so no equity interaction, although raw rates rise with TFR among low-equity firms (2.1% to 4.5%).
  (5) Turnover: a real interaction. Low-turnover, TFR > 5% firms: raw default 5.7-11.5% on 880 rows, 62 defaults; high-turnover 0.7-1.1%. SHAP in the 5-10% band: -0.09 low, -0.22 mid, -0.32 high turnover.
- Source(s): assumption. No VERIFIED source for TFR. The readings "tenure / maturity" and "patient funding owed to staff" are hypotheses [UNVERIFIED]. The wage bill uses D-091's goodwill hypothesis.
- Risks / what would change our mind: the bank's field definition; TFR rules are Italy-specific; found after looking at the data, so the lockbox should confirm it.
- Human reviewer (initials) and date:

### D-124 Encodings: division persists, province only partly, region is erratic; the group's value needs more training years
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (kept in full2); province_te flagged as the weakest of the three on persistence
- Decision: keep all three. Say plainly that the evidence for province alone is not significant.
- Alternatives considered: dropping province_te and region_te (inner: +0.0003 [-0.0023, +0.0027]; -0.0013 [-0.0026, +0.0001]). Not significant either way, so no post-hoc removal.
- Business reason: industry and location add context that a firm's own statement lacks, with unseen levels mapped to the prior.
- Evidence: single drops, fold-2; fold-3 inner: division -0.0000; -0.0009 [-0.0016, -0.0002]. Province +0.0011 [-0.0017, +0.0034]; -0.0007 [-0.0018, +0.0002]. Region +0.0005; -0.0001 (ns).
  Group: -0.0003 (one fit year); -0.0028 (two fit years). Slope of next-year default on the encoding (1 = taken at face value, diagnostic, `encoding_persistence.csv`):
  division 1.22 / 0.87 / 0.85; province 0.50 / 0.78 / 0.48; region 0.06 / 1.31 / 0.95. Rank correlation with next-year level rates: division 0.65 / 0.64 / 0.72; province 0.20 / 0.27 / 0.40.
  Correction to the brief: the encodings carry expected sign + (D-094), so under the "signs from expected_signs" rule they ARE constrained in GB-20; their rising shapes are imposed, not evidence.
  The unconstrained new features are interest_to_sales, bs_gap, fin_fixed_ratio and tfr_ratio.
- Source(s): R-47 Sec. 1.2, 2.7, 4.2-5 (regularised, cross-fitted target encoding; generic ML datasets, not credit). R-46 abstract only (origin of the idea; our m is our own design, R-46's formula is not verified).
- Risks / what would change our mind: geography was fitted on 2007-09 conditions; level names rely on the [UNVERIFIED] ISTAT table and an assumed NACE mapping.
- Human reviewer (initials) and date:

### D-125 Face validity: zero violations, but interest_to_sales can undo a constraint (decision needed)
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: zero-violation finding ADOPTED; interest_to_sales sign OPEN for the team
- Decision: propose restoring R-33's + sign on interest_to_sales in the final configuration, subject to one outer scoring.
- Alternatives considered: leave free (GB-20); drop it. Inner: constrained + -0.0012 [-0.0025, -0.0001] and -0.0010 [-0.0021, -0.0000]; d log loss +0.00017, +0.00004. Drop -0.0008 and -0.0023.
- Business reason: "more interest per euro of sales lowers PD" cannot be defended to a credit officer.
- Evidence: GB-22, fold-3 model: 32 constrained features agree, 3 have 0 splits, 0 violations; ICE 500 rows x 35 features 0 violating rows (folds 1-2 by construction). A first run flagged n_core_imputed; that was a bug in our check (a one-point grid gave NaN). The grid now spans the observed min-max.
  interest_to_sales SHAP: deciles 3-4 about -0.09, 7-8 about +0.10, 10 -0.15, while raw default in decile 10 is 2.4%. Spearman 0.64 with interest_to_assets (constrained +), so this free near-duplicate can partly reverse a direction we impose elsewhere; D-099 item 6 already flags the sign as post hoc.
  bs_gap (+0.06 at exact 0 to -0.08) and fin_fixed_ratio (+0.02 to -0.04) are small; the first is "hypothesis only", and drop fin_fixed_ratio is +0.0016 [+0.0005, +0.0027] and -0.0004 (ns), so it earns nothing.
  n_core_imputed has 0 splits in fold 3: D-095's "adds back risk" does not happen in this model (native NaN already handles it). Reason codes: 42% of firms' top 3 repeat a family, so codes must be merged by family.
- Source(s): R-33 p. 12 (upward slope); R-39; R-41.
- Risks / what would change our mind: the outer score of that configuration; the AUC cost is real, about 0.001.
- Human reviewer (initials) and date:

### D-126 About 40% of full2's AUC gain buys back the cost of the constraints; the constraints stay
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (constraints kept; unconstrained twin is a diagnostic only, not outer scored)
- Decision: report both parts of the gain. full2 + GB-08's relaxation is left for a joint decision with phase A.
- Alternatives considered: unconstrained full2 (fails face validity, as GB-02 did, D-064); full2 + GB-08 relax.
- Business reason: separates new information from a looser effective constraint, which a validator would ask about.
- Evidence (inner, fold 2; fold 3): gain from full2 with constraints +0.0053; +0.0093. Without constraints on both +0.0033 [-0.0005, +0.0073]; +0.0054 [+0.0024, +0.0075]. Log loss -0.00022; -0.00062 vs -0.00005; -0.00031.
  Constraint cost on full2 +0.0032; +0.0048 AUC (v1: +0.0052; +0.0088). So constraints cost about 40% less on full2, not zero. GB-08 relax adds about the same on either set: v1 +0.0024, +0.0037; full2 +0.0023, +0.0028.
- Source(s): assumption (face validity, D-051); R-39.
- Risks / what would change our mind: the relaxed set frees interest_coverage but leaves interest_coverage_sect_pct constrained, an inconsistency to settle first.
- Human reviewer (initials) and date:

### D-127 GB-23 (full2 + GB-09's tuned settings): not adopted
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: REJECTED as the entry; tuning remains an option for the final-window refit
- Decision: keep GB-20's pre-declared grid.
- Alternatives considered: per-fold A-selected mix (GB-20 fold 1, GB-23 folds 2-3): not built, because it was never a pre-declared configuration.
- Business reason: simpler, and no reliable gain.
- Evidence: inner choice recorded before GB-23's outer run (`GB-23/inner_choice.csv`). A gain on full2: -3e-7, +5e-8, +1.0e-4 (GB-09 cfg "wins" folds 2-3, effectively a tie in 2 of 3). B check +2.1e-4, +2.5e-4, +1.9e-4. So the v1 tuning does not hurt on full2, but its settings differ widely by fold (learning rate 0.024 / 0.05 / 0.016; final trees 674 / 253 / 1,271).
  Outer, reported only: vs GB-20 pooled +0.0007 [+0.0002, +0.0012], 2009 -0.0004 with log loss +0.00015 [+0.00003, +0.00026], 2010 +0.0020 [+0.0011, +0.0026]. Fails "better in all 3 years". vs GB-01 it wins, like GB-20.
- Source(s): D-052; assumption.
- Risks / what would change our mind: fold-3 prefers the tuned cfg on A, B and 2010, on v1 too (GB-09 2010 +0.0022). It may be that low learning rate and stronger penalties pay off once data are plentiful. Testable in the final window, on inner slices.
- Human reviewer (initials) and date:

### D-128 More features did widen the same-year gap; the new-firm shortfall is 2008 only
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: ADOPTED (reporting)
- Decision: state the wider gap in the deck, with the out-of-time evidence next to it.
- Alternatives considered: reporting only the OOT slope, which would hide the gap.
- Business reason: a validator will look for overfitting; PD spread drives pricing.
- Evidence: same-year in-sample minus OOT AUC, GB-01 vs GB-20 (out-of-fold encodings): 2008 fold 2 0.0262 vs 0.0349; 2008 fold 3 0.0173 vs 0.0259; 2009 fold 3 0.0146 vs 0.0217. So +0.007 to +0.009. Full-table encodings add about 0.001. Final trees 338/490/386 vs 444/722/737. Inner Platt slopes dropped (1.017/0.943/0.998 to 0.953/0.919/0.961), the map corrects this; OOT slopes 0.996 pooled, 2010 unchanged at 0.957.
  In-sample AUC rises more than OOT AUC, but OOT still rises. Reading: extra fit is partly noise, and does not cost out-of-time performance.
  New firms: pooled dAUC -0.0011 [-0.0061, +0.0040] vs seen +0.0069. By year new: 2008 -0.0100 [-0.0174, -0.0021] (log loss +0.00088 [+0.00039, +0.00134]), 2009 +0.0083 [+0.0000, +0.0167], 2010 +0.0079 [-0.0025, +0.0189]. Seen-minus-new gap: 2008 +0.0166 [+0.0085, +0.0244], 2009 +0.0002, 2010 -0.0015 (ns).
  So only fold 1 (one training year, 1,379 defaults) loses on new firms. Our hypothesis: too little data for 45 features and one-year encodings. Not tested further; this is one fold.
- Source(s): D-117; R-33 Sec. 4.7-4.8 (same-year gap reporting).
- Risks / what would change our mind: if the holdout is new-firm heavy and resembles 2008; final training has more years, like folds 2-3.
- Human reviewer (initials) and date:

### D-129 What didn't work, what was not done, and process disclosures
- Date / phase / agent: 2026-11-02 / round 2 phase B / challenger-gbm
- Status: REJECTED (items 1-3); OPEN (items 4-5); disclosures for the referee (item 6)
1. Isotonic: B log loss worse than Platt in every fold (0.04700 vs 0.04410; 0.05585 vs 0.05522; 0.06414 vs 0.06380). Fold-1 B has 203 defaults, not eligible anyway. Platt kept.
2. full2 does not fix the level: 2010 obs/pred at common level 1.19 vs CAL-01 1.18. Level stays the calibration round's job.
3. Sector percentiles as a source of power, and "the tree uses TFR in interaction with leverage": neither holds (D-122, D-123).
4. NOT run on full2: native NaN vs shared imputation, the depth/grid search beyond GB-09's cfg, XGBoost. Round-1 tie (D-056) is assumed to carry over.
5. Prediction cost measured under CPU contention: 4.7-5.3 s per 100k rows at 4 threads vs 1.0 s for GB-01 (8 threads, D-059). The harness-engineer must re-measure; most of it is the feature transform.
6. Disclosures. Outer folds scored once each for GB-20 and GB-23; bootstraps only re-read saved predictions. Lockbox: all reads use the fs_year < 2011 filter. GB-24 (inner-only) and GB-23's inner choice were run after GB-20's outer result was known, and the interest_to_sales proposal comes from those runs, so they are not blind. 2010 labels appear in diagnostics (encoding persistence, TFR cells, SHAP tables), no configuration choice uses them. TFR and suppression readings were developed after looking at training data and 2010 cells. Fold 1 inner is a firm split within 2007, as D-061 notes for GB-01. The driver gained subcommands after GB-20/GB-21 ran; their code path is unchanged. `score_r1.py` functions are exec'd from source (its body would rescore everything), with an L1 column mirrored and then discarded. gbm_lib.py and phase A's files were not edited.
- Source(s): D-056, D-059, D-061, D-072.
- Human reviewer (initials) and date:
