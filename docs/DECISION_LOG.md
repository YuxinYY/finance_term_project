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
  of noisy ratios) supports bounding outliers in general.
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
  under the harness rules without approval.
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
- Source(s): R-01 Sec. 4.2 p. 26 (VIFs used to limit collinearity).
