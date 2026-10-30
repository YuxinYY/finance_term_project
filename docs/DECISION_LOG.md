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

---

## Tournament round 1: challenger-scorecard (merged by orchestrator from experiments/scorecard/decisions.md)

### D-030 Benchmark family and pre-declared settings (SC-01..SC-04)
- Date / phase / agent: 2026-10-31 / Round 1 / challenger-scorecard
- Status: ADOPTED
- Decision: four benchmarks on the frozen walk-forward folds, each refit per fold on `fs_year <= t`, `label_status == ok`, with `B.fit/B.transform(impute=True)`: SC-01 single best core numeric ratio (chosen by in-window |AUC-0.5|; it was roa_net in all 3 folds) -> logit; SC-02 unpenalised logit on the 10 core features + 3 categoricals; SC-03 monotone-WoE scorecard; SC-04 Z'-style logit on wc_ratio, equity_ratio, roa_oper, asset_turnover.
- Tuning: none needed. GLMs are unpenalised (no hyper-parameter). SC-03 settings were declared before looking at results and are identical in all folds, including fold 1 (single training year): 20 initial quantile bins, pool-adjacent-violators to monotone in the expected-sign direction, minimum final bin 2% of training rows. No inner validation split was used; no fold uses a default instead of a tuned value because nothing was tuned.
- Evidence: `experiments/scorecard/SC-0*/metrics.md`. Pooled AUC 0.746 / 0.803 / 0.817 / 0.770; log loss 0.0713 / 0.0672 / 0.0666 / 0.0699.
- Source(s): Stein (benchmarking logic, course); Altman lineage only via R-32; Ohlson (R-12) cited only as existence of logit lineage (content unread).

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

### D-057 LightGBM monotone method 'intermediate' (vs 'advanced')
- Date / phase / agent: 2026-10-31 / Phase 3 / challenger-gbm
- Status: ADOPTED 'intermediate' ('advanced' GB-05 REJECTED: no material gain)
- Evidence: GB-05 vs GB-01 dAUC +0.0002 [-0.0000, +0.0004], d log loss -0.00002 [-0.00004, +0.00000].
- Business reason: same face validity, simpler and faster; the difference is below any business relevance.
- Source(s): assumption.
- Human reviewer (initials) and date:

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
