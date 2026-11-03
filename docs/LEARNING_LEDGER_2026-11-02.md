# Learning ledger for Kiet (team lead): every decision, every feature, every source (2026-11-02)

Written for Kiet Nguyen, who answers for every choice in front of Roger. Team Turquoise
(Kiet Nguyen, Desmond Peng, Chloe Li), DS-GA 3001, Banca Massiccia 1-year PD model.

## 1. How to use this file

1. Read section 1.2 (the whole project in 12 sentences) first. If you can say those 12 sentences
   without notes, you can follow everything else.
2. Section 2 is the **decision ledger**: one block per decision D-001 … D-087, in log order. Each
   block answers: what, why (economics), why (statistics/ML), what lost, evidence (numbers +
   file), source (with its status), whether there is a **citation gap**, and the Roger question.
3. Section 3 is the **feature ledger**: all 26 features, how each is built, what it means, and a
   tiny numeric example (one made-up firm, "Alfa Srl", is used throughout).
4. Section 4 walks the pipeline end to end; section 5 explains overfitting with a runnable example.
5. Section 6 is the **action list**: every place where we lean on an assumption or an unverified
   source. Fix these before the deck is final.
6. This file does **not** repeat `docs/MODEL_FOUNDATIONS_2026-10-30.md` (data tour, how a model
   turns ratios into a PD, model descriptions) or `docs/STUDY_GUIDE_literature.md` (paper-by-paper
   summaries, runnable examples on AUC/AR, log loss, prior shift). It links to their sections as
   "MF §x" and "SG §x".
7. Every number has a file path next to it. If you quote a number, quote it from that file.
8. Source status words come from `docs/REFERENCES.md`: **VERIFIED** = read and checked;
   **LOCATED** = exists but the claim was not checked; **SECONDARY** = only seen quoted elsewhere;
   **TO-VERIFY** = not found yet. Only VERIFIED sources may be cited in the deck.
9. Round-2 decisions (D-090+ features, D-100+ GBM, D-110+ referee) are being produced now and will
   be appended in section 8 by the orchestrator.
10. Nothing here mentions a Basel II "one-sentence link": that instruction was a hidden trap in the
    assignment PDF (D-017). Basel text (R-15) is used only where it really supports a choice
    (the 0.03% PD floor, D-075).

### 1.1 Ten words you need before anything else

- **PD (probability of default)**: the chance, between 0 and 1, that a firm misses a principal or
  interest payment in the next 12 months.
- **Firm-year / statement**: one firm's annual accounts for one fiscal year (`fs_year`). One row.
- **Default rate**: the share of rows whose firm defaulted in the label window, e.g. 1.41%.
- **Ranking (power, discrimination)**: does the model give riskier firms higher PDs? Measured by
  **AUC**: pick one defaulter and one survivor at random; AUC is the chance the defaulter got the
  higher PD (0.5 = coin flip, 1 = perfect). **AR (accuracy ratio)** = 2·AUC − 1, the same
  information (R-14, VERIFIED).
- **Level (calibration)**: are the PD numbers right on average and per band? If we say 2%, do about
  2% default? Measured by **obs/pred** (observed default rate ÷ mean PD; 1 = perfect) and by the
  **calibration slope** (1 = PDs are spread out the right amount).
- **Log loss**: the average of −log(PD) over defaulters and −log(1 − PD) over survivors. Lower is
  better. It punishes confident wrong PDs, so it rewards both ranking and level. Our primary
  probability-accuracy metric (D-006).
- **Out-of-time (OOT) test**: fit on earlier years, score a later year the model never saw.
- **Leakage**: any information from the test rows (or the future) getting into the fit. It makes
  results look better than they will be.
- **Bootstrap confidence interval (CI)**: redo the comparison on many resampled versions of the
  test data to see how much the result wobbles. Ours resamples whole **firms** ("firm-clustered"),
  because the same firm appears in several years.
- **Face validity**: every driver moves the PD in the direction a credit officer expects (more
  equity → lower PD). A model that breaks this cannot be defended to a bank.

### 1.2 The whole project in 12 sentences

1. Banca Massiccia needs one number per prospective borrower, the 12-month PD, computed only from
   the latest annual statement and firm characteristics, and uses it both to approve or decline
   (needs correct **ranking**) and to price (needs the correct **level**) (CLAUDE.md §1; R-30).
2. The data are 1,023,552 firm-year statements of 237,711 private Italian firms, fiscal years
   2007-2012; one row is one statement and the model returns one PD per row (D-001,
   `reports/data/01_profile.md`).
3. A row is labelled "default" if the firm defaults in the 12 months starting 6 months after the
   year-end, because Italian accounts are approved and filed only months after year-end; this
   gives a 1.41% default rate on 2007-2010 (D-002, `reports/data/00_labels.md`).
4. We test like a bank lives: train on years up to t, score year t+1, for t = 2007, 2008, 2009
   (506,159 test rows, 7,784 defaults; D-015, `reports/tournament/scorecard_r1.md`).
5. We built 26 features: scale-free ratios in eight families (size, leverage, liquidity,
   profitability, coverage, activity, asset structure, flags) plus industry, legal form and region,
   with documented rules for zero denominators, training-window clipping of outliers, and missing
   values rebuilt from accounting identities first (D-018, D-019, D-021; `docs/FEATURE_SPEC.md`).
6. Every feature got an expected direction (sign) **before** any model was fitted, and models
   are not allowed to contradict it (face validity; D-022, D-051, D-064).
7. Thirteen round-1 models competed on the same frozen splits: four scorecard benchmarks, three
   RiskCalc-style probits (Roger's own published design, R-01) and six LightGBM variants; two
   glass-box EBMs were added afterwards (D-030, D-040, D-050, D-080).
8. The winner, GB-01 (monotone-constrained gradient-boosted trees), has pooled AUC 0.8375 vs 0.8166
   for the best scorecard, better in every test year and for new and seen firms, with zero
   wrong-signed drivers (D-065, `reports/tournament/r1_paired_bootstrap.csv`).
9. An unconstrained version ranks even better (+0.0065 AUC) but uses ratios against their economic
   direction, so we rejected it: we pay that much AUC for a model a credit officer can defend (D-051).
10. Every model under-predicts the level by about a third (obs/pred 1.34 for GB-01), because the
    default rate rose every year 2007-2010; we fix the level with one frozen shift of every firm's
    log-odds toward a target of 1.49% (D-071 to D-073, `experiments/calibration/harness_level_params.json`).
11. The fully readable alternatives cost power: the EBM gives up 0.012 AUC and the RiskCalc-style
    probit 0.018 AUC against GB-01 (D-081, D-065), and averaging them with GB-01 makes it worse (D-086).
12. Our known weaknesses are: the feature set was designed on years that later served as test years
    (D-062), new firms are ranked less well (AUC 0.7865 vs 0.8438), the holdout level is a judgment,
    and the label window is still unconfirmed by Roger/TA (D-002a, D-073, D-079).

---

## 2. Decision ledger (D-001 … D-087)

Notes on numbering. IDs D-026..D-029, D-036..D-039 and D-048..D-049 were never used (agents were
given blocks of ten). The log has **D-002 and a separate heading D-002a** for the same topic (the
window choice, and the open question to Roger about it): they are merged below into one block.
D-086 and D-087 were added by the orchestrator on 2026-11-02 and are included because section 5
depends on D-087.

Path shorthand: DL = `docs/DECISION_LOG.md`; SC1 = `reports/tournament/scorecard_r1.md`;
UNI = `reports/features/univariate.md`; CAL = `reports/calibration/`.

### D-001 One row = one firm-year statement (ADOPTED)
- **Decided:** the unit of analysis is a firm's annual statement; the model returns one PD per row.
- **Economics:** at application the bank has the firm's latest accounts and nothing else; that is
  exactly one row. A firm that applies again next year is scored again on its new accounts.
- **ML/stats:** rows of the same firm are not independent (same managers, same business), so
  uncertainty is measured by resampling whole firms, and results are reported for "seen" firms
  (in training) and "new" firms separately.
- **Alternatives:** loan application as the unit (the proposal) — impossible, the file has no loan,
  application or pricing fields; one label per firm — throws away the year-to-year change.
- **Evidence:** 1,023,552 rows, 237,711 firms, 0 duplicate (id, fs_year) pairs; 12.6% of firms
  appear once, 42.6% in all 6 years (`reports/data/01_profile.md` §1-2).
- **Source:** R-30 assignment, Appendix B "Each row is one firm-year" (VERIFIED).
- **Citation gap?** NO.
- **If Roger asks** "Your proposal said loan applications. Why the change?" → "The data contain
  statements, not loans. The bank scores the latest statement at application, so the firm-year is
  the decision unit. Repeated firms are handled with a firm-clustered bootstrap and a seen/new split."

### D-002 + D-002a Default label window: 12 months starting 6 months after year-end (ADOPTED, provisional; question OPEN)
- **Decided:** y = 1 if `def_date` falls in [statement date + 6 months, statement date + 18 months).
  Rows that defaulted before the window are dropped ("already defaulted"); rows whose window is not
  complete by 2013-05-31 are dropped ("censored"), not set to 0. D-002a: we have asked Roger/TA which
  window the holdout uses (email Q1); the panel carries labels for start 0, 3 and 6 months.
- **Economics:** a lender can only use accounts that exist. Italian law allows up to 120 days
  (180 if the statute allows) to approve the accounts and 30 more days to file them, so a 31 Dec
  statement is usable from roughly mid-year. A default in the first six months after year-end
  happens before the bank could have used that statement.
- **ML/stats:** the label must match the moment of prediction; otherwise the model learns from
  defaults it could not have predicted with that information. Dropping censored rows avoids
  counting unknown outcomes as "no default", which would bias the rate down.
- **Alternatives:** start 0 (2007-2010 rate 1.07%, one more test fold 2011); start 3; keep censored
  rows as 0 (fs2012 at start 0 would show 0.35%); the shared code's default of y = 1 for observed
  defaults in incomplete windows (gives a 100% default rate in that year: rejected, D-011).
- **Evidence:** start 6: 650,373 usable rows, 9,163 defaults, 1.41% (2007-2010); start 0:
  837,913 rows, 9,328 defaults, 1.11% (2007-2011); same years 2007-2010: 1.41% vs 1.07% (ratio
  1.32). 4,237 rows default 0-6 months after year-end (`reports/data/00_labels.md` §2, §5;
  `reports/data/labels_sensitivity_summary.csv`).
- **Source:** R-28 (art. 2364, 120/180 days) and R-29 (art. 2435, filing within 30 days): VERIFIED
  for the **current** text; the 2007-2012 wording is SECONDARY. Actual filing dates are an
  assumption (legal maxima ≠ real dates; REFERENCES "must not claim" #6).
- **Citation gap?** YES (partial). The law is verified only in today's wording; R-01 Sec. 3.2 p. 16
  (VERIFIED: statements reach lenders 3-4 months after the statement date, US data) supports the
  general idea of a lag. Needs literature-scout for the 2007-2012 wording of art. 2364/2435.
- **If Roger asks** "How do you know the bank only sees the accounts six months later?" → "We don't
  observe filing dates. The Civil Code's maximum deadlines (120 days to approve, 30 to file) put
  availability around mid-year; that is an assumption, which is why we built start-0 and start-3
  labels too and asked you which window the holdout uses. The ranking is robust to the window
  (D-076); the level moves by about a third on 2007-2010."

### D-003 No NLP / text features (REJECTED)
- **Decided:** no text features.
- **Economics:** the proposal promised sentiment from management reports; the bank's data file has
  no text, so the promise cannot be delivered.
- **ML/stats:** nothing to model.
- **Alternatives:** none possible.
- **Evidence:** the data dictionary (assignment Appendix B) lists no text field; re-confirmed in D-024.
- **Source:** R-30 (VERIFIED).
- **Citation gap?** NO.
- **If Roger asks** "What happened to the NLP?" → "There is no text in the data. We dropped it and
  said so, rather than promise something we could not build."

### D-004 No external macro data (OPEN, default: not used)
- **Decided:** no macro series (unemployment, GDP) in the model; cycle effects are handled in the
  calibration step.
- **Economics:** the assignment allows external data only if cleared "several weeks" before
  submission; we had no clearance. The bank still needs a level that tracks the cycle, so the
  level is set from the training data's own default-rate history (D-071, D-073).
- **ML/stats:** macro variables vary only by year, so with 3-4 training years a model cannot
  estimate their effect reliably anyway; they mostly move the level, not the ranking.
- **Alternatives:** Italian unemployment (what Moody's uses for Italy, R-04); no-cycle model.
- **Evidence:** none needed (rule); the calibration study shows the level, not the ranking, is
  what drifts (D-071).
- **Source:** R-05 p. 41 (macro forecasting deliberately left outside RiskCalc, VERIFIED);
  R-10 Sec. 4.1 p. 21 (firm-specific information dominates for middle-market firms, VERIFIED);
  cost: Moody's does use a cycle adjustment and external data for the level (R-33 Sec. 2.4 p. 9,
  Sec. 3.3 p. 15; R-04 slides 13-15; all VERIFIED).
- **Citation gap?** NO.
- **If Roger asks** "Moody's adjusts for the cycle. Why don't you?" → "We could not get clearance
  for external data in time. We handle the cycle as a level adjustment on top of a
  statements-only score, which is the same structure Moody's uses (R-04, R-33), but our level
  comes only from the training data."

### D-005 Evaluation design: expanding-window walk-forward + paired firm-clustered bootstrap (ADOPTED, provisional)
- **Decided:** train on fs_year ≤ t, test on t+1; compare models with a paired bootstrap that
  resamples firms.
- **Economics:** this is what deployment looks like: the bank fits on the past and lends in the
  future, through whatever the economy does next.
- **ML/stats:** a random split would let the model see the same firm and the same year's economy
  in training and test, which flatters results. "Paired" means both models are scored on the same
  resampled data in each draw, so the CI is for the **difference**, which is much tighter than two
  separate CIs.
- **Alternatives:** random k-fold (leaks time and firm); single train/test split (one number, no
  sense of variation).
- **Evidence:** see D-015 (fold sizes) and D-060 (bootstrap method).
- **Source:** the log lists R-07 (VERIFIED: walk-forward Sec. 3 pp. 91-96, six steps p. 94;
  resampling Sec. 4) and R-06 (only LOCATED: do not cite). R-01 Sec. 4.2-4.3 pp. 26-28 (VERIFIED)
  and R-33 Sec. 4.7-4.8 pp. 22-23 (VERIFIED) also use walk-forward. Firm clustering: R-01 Sec. 3.4.2
  p. 22 (panel dependence can mislead significance tests, VERIFIED).
- **Citation gap?** YES (small): the log still names R-06 (LOCATED). Replace it by R-07 Sec. 3 p. 94
  and R-01 Sec. 4.3 p. 28.
- **If Roger asks** "How, specifically, did you implement walk-forward?" → "One frozen file,
  `data/splits/walkforward.json`: fold k trains on 2007..2006+k and tests on 2007+k. Every
  parameter, including clipping bounds and calibration, is refitted inside the training window."

### D-006 Log loss is the primary probability-accuracy metric (ADOPTED, provisional)
- **Decided:** log loss is the headline accuracy metric; Brier and calibration-in-the-large are
  secondary; AUC (with AR = 2·AUC − 1 noted) measures ranking.
- **Economics:** pricing uses the PD level directly; a PD of 0.2% on a firm that defaults is a
  badly underpriced loan. Log loss punishes exactly that kind of confident error.
- **ML/stats:** log loss is the negative average log-likelihood, a "proper" score (it is minimised
  by the true probabilities). Brier is mainly driven by discrimination according to BCBS (R-16
  p. 32), so it is reported as "overall probability accuracy", not as calibration.
- **Alternatives:** Brier as primary; AUC only (ignores level); Hosmer-Lemeshow-type tests (R-16
  pp. 33-34 says they understate error when defaults are correlated).
- **Evidence:** used in every scorecard (SC1 T1).
- **Source:** the log entry gives no R-id. Covered by R-07 Sec. 2.2 pp. 88-89 (likelihood compares
  calibration of models, relatively; VERIFIED), R-01 Tables 5-6 pp. 27, 30 (RiskCalc reports
  calibration as log-likelihood, VERIFIED), R-16 p. 32 (Brier, VERIFIED).
- **Citation gap?** YES: add R-07 Sec. 2.2 and R-01 Tables 5-6 to the log entry.
- **If Roger asks** "Is log loss a calibration test?" → "No. As you wrote (R-07 p. 89), likelihood
  ranks models relative to each other; it does not certify calibration. We pair it with obs/pred,
  calibration slope and a binned observed-vs-predicted table."

### D-007 Compress the schedule to three days (ADOPTED)
- **Decided:** one tournament round, 200 bootstrap draws instead of 1,000; keep referee, red team
  and decision log.
- **Economics:** none; team scheduling.
- **ML/stats:** 200 draws give noisier CI endpoints (each 2.5% tail rests on about 5 draws, D-060);
  fine for clear differences, not for borderline ones.
- **Alternatives:** the 18-day plan in `docs/PLAYBOOK.md`.
- **Evidence:** n/a.
- **Source:** R-30 grading table (VERIFIED).
- **Citation gap?** NO.
- **If Roger asks** "Are 200 bootstrap draws enough?" → "For the promotion decisions, yes: those
  CIs are far from zero. Borderline pairs (GB-05 vs GB-01, RC-01 vs SC-03) should be re-run with
  1,000 draws before being quoted (D-060)."

### D-008 Drop the GAM/EBM challenger (REJECTED for time; SUPERSEDED by D-080)
- **Decided (then):** do not run the glass-box challenger. Later reversed (D-080).
- **Economics/ML:** interpretability was to be covered by the probit and scorecard.
- **Evidence:** none; it was a time decision.
- **Source:** assumption.
- **Citation gap?** NO (no claim made).
- **If Roger asks** "Why did you first drop the GAM?" → "Time. We reinstated it once the first
  round finished, and it is now our best fully readable model (D-081)."

### D-009 `def_date` values in 2013: keep them, data end = 2013-05-31 (ADOPTED, provisional; email Q2)
- **Decided:** treat 2013 default dates as real; the data end at 2013-05-31. Sensitivity: blank them.
- **Economics:** if defaults were tracked until May 2013, ignoring them would undercount defaults
  for the latest statements.
- **ML/stats:** the cutoff date decides which windows are complete (censoring).
- **Alternatives:** follow Appendix B literally (no information after 2012-12-31).
- **Evidence:** `def_date` is firm-level (0 firms with >1 date); 16,863 defaulting firms; 1,416
  firms with 2013 dates (5,570 rows), at 229-377 firms/month vs 261-372 in Jan-May 2012; 0 rows
  dated after their default. Under the start-6 window the two treatments give identical labels for
  fs_year 2007-2011; no walk-forward fold depends on it (`reports/data/00_labels.md` §1-2).
- **Source:** data evidence (no citation needed).
- **Citation gap?** NO.
- **If Roger asks** "Appendix B says no defaults after 2012. Why do you use 2013 dates?" → "They are
  in the file at the same monthly intensity as 2012 and stop sharply at 31 May 2013, which looks
  like continued tracking, not errors. It changes no label in our test folds; we asked to confirm."

### D-010 Data dictionary vs file names; fields that are not what they say (ADOPTED)
- **Decided:** use the file's column names, accept either spelling; reinterpret fields by checking
  identities: `roa` = 100 × operating profit / assets (exact on 100% of rows); `goodwill` behaves
  like value added; `inc_financing` looks like the net financial result; `eqty_corp_family_tot` is
  100% empty.
- **Economics:** a credit officer reads "goodwill" as an intangible asset; using it under that name
  would give nonsense explanations.
- **ML/stats:** the harness must not crash on the dictionary spelling (assignment rule: no errors on
  messy input).
- **Alternatives:** trust the dictionary.
- **Evidence:** dictionary 41 names; file 44 columns (3 spelled differently + def_date + stmt_date +
  index); `goodwill` median EUR 420,706, EBITDA ≤ goodwill on 99.99% of rows
  (`reports/data/01_profile.md` §1, §5).
- **Source:** data evidence.
- **Citation gap?** NO (state them as "we found that", not as facts).
- **If Roger asks** "How do you know `goodwill` is value added?" → "We don't know it; EBITDA is below
  it on 99.99% of rows, which is what value added would do. So we did not use it (D-024)."

### D-011 + D-068 The shared label function mislabels incomplete windows (OPEN: proposed fix)
- **Decided:** the panel censors any row whose window is incomplete, whatever the outcome; the
  shared `make_labels` does not and would mark only the observed defaults as valid (y = 1). Fix
  proposed, not applied (shared code may only change with human approval).
- **Economics:** a year where only the defaulters are kept has a 100% default rate: any model fitted
  on it would be absurdly pessimistic.
- **ML/stats:** classic censoring bias: you can see the default but not the survival.
- **Alternatives:** use the shared code as is.
- **Evidence:** fs_year 2011 (start 6): 2,357 rows would be "ok" with y = 1; panel and make_labels
  agree exactly on 2007-2010 (0 mismatches; SC1 T12, `reports/tournament/r1_d011_label_check.csv`).
- **Source:** none needed (code correctness).
- **Citation gap?** NO.
- **If Roger asks** "Did a labelling bug affect your results?" → "No. We found it, showed that
  2007-2010 labels are identical either way, and built the 2011 training label from the panel."

### D-012 HQ_city codes read as ISTAT province codes → region → macro-area (ADOPTED, hypothesis)
- **Decided:** map the 110 city codes to 20 regions and 5 macro-areas; the code stays a category.
- **Economics:** regional conditions differ; five areas are populated enough to estimate.
- **ML/stats:** 45 of 110 codes have < 30 defaults: a one-hot of raw codes would fit noise.
- **Alternatives:** raw HQ_city one-hot; no geography.
- **Evidence:** exactly 110 codes 1-110; most frequent 15, 58, 17, 1, 63 = Milano, Roma, Brescia,
  Torino, Napoli under ISTAT coding; region default rates 0.69%-1.89%, macro-area 1.30%-1.62%
  (`reports/data/01_profile.md` §3c).
- **Source:** the ISTAT code table is [UNVERIFIED] (agent knowledge, no document read; D-023).
- **Citation gap?** YES: needs literature-scout (an ISTAT province-code list valid 2009-2016).
- **If Roger asks** "How do you know code 15 is Milan?" → "It is a hypothesis: the frequencies match
  the ISTAT coding. If it is wrong, the grouping is still a fixed partition of codes and the model
  uses fold-fitted effects, not the names."

### D-013 ATECO division → section mapping (ADOPTED)
- **Decided:** `ateco_sector` = 2-digit ATECO 2007 division, mapped to sections A-U by NACE Rev. 2
  ranges.
- **Economics:** industries differ in risk (real estate 0.92% vs admin services 1.99%) and in what
  a "normal" ratio looks like (R-01 Sec. 3.3, Table 3).
- **ML/stats:** 83 divisions are too many for thin data; sections are a coarser, standard grouping.
- **Alternatives:** raw divisions (36 of 83 have < 30 defaults); a hand-made 7-group scheme (proposal).
- **Evidence:** 83 divisions, none in 64-66 (financials absent, consistent with "non-financial firms");
  sections O, P, U < 30 defaults; L 0.92%, C 1.67%, G 1.62%, F 1.45%, N 1.99%
  (`reports/data/01_profile.md` §3a, `profile_by_ateco_section.csv`).
- **Source:** NACE ranges [UNVERIFIED]; `docs/2603_-CODES_ATECO_(003).pdf` confirms only the code format.
- **Citation gap?** YES: needs literature-scout (Eurostat NACE Rev. 2 / ISTAT ATECO 2007 structure).
- **If Roger asks** (margin note M7) "How did you group ATECO?" → "Division → official section;
  sections with fewer than 30 training defaults pooled into OTHER, refitted in every fold; default
  rates by group are in `reports/features/univariate.md` §5."

### D-014 Missing values: NA tokens → NaN; no masking of −999-like values (ADOPTED)
- **Decided:** every NA token and non-finite number becomes NaN; values like −999 are kept.
- **Economics:** −999 euros of extraordinary income is a plausible small amount, not a code.
- **ML/stats:** masking real values would delete information; the harness still accepts many NA
  spellings because the holdout may be messier.
- **Alternatives:** mask sentinel values.
- **Evidence:** training uses only the empty string as a missing code; 148 rows of
  inc_extraord = −999 or similar (`reports/data/01_profile.md` §4).
- **Source:** data evidence.
- **Citation gap?** NO.
- **If Roger asks** "How do you handle odd missing codes in the holdout?" → "A broad NA-token list
  maps them to NaN; then the same identity rebuild and missing rules as in training apply."

### D-015 Frozen walk-forward splits (ADOPTED)
- **Decided:** `data/splits/walkforward.json`: fold 1 train 2007 → test 2008; fold 2 2007-08 → 2009;
  fold 3 2007-09 → 2010. Read-only; everyone uses the same file.
- **Economics:** mirrors deployment (D-005).
- **ML/stats:** one shared split prevents each team member from finding a "lucky" split.
- **Alternatives:** test 2011 (window incomplete at start 6); random splits.
- **Evidence:** test sizes 161,261 / 170,386 / 174,512; defaults 2,064 / 2,752 / 2,968; new-firm
  rows 29,835 / 17,172 / 16,737 (`reports/data/walkforward_folds.csv`, DL D-015).
- **Source:** none in the log entry; R-07 Sec. 3 p. 94 and R-01 Sec. 4.3 p. 28 (VERIFIED) cover it.
- **Citation gap?** YES (small): add R-07/R-01.
- **If Roger asks** "Fold 1 trains on one year. Is that a problem?" → "It is the weakest fold; that
  is why we report per-year results and why the clean fold 3 (2010) is quoted next to pooled numbers."

### D-016 Missingness findings (H2 partly confirmed)
- **Decided (finding):** the bank / other-lender / supplier debt split is missing as a block;
  `margin_fin` and `roe` are blank exactly when equity ≤ 0; `days_rec` is unusable.
- **Economics:** a blank ROE here **means** negative equity, the strongest danger sign. Filling it
  with an average would hide the riskiest firms. Abbreviated accounts for small firms (Civil Code
  art. 2435-bis) are a plausible reason for the missing debt split.
- **ML/stats:** missingness can be informative ("missing not at random"); you must study it before
  imputing.
- **Alternatives:** blanket median imputation.
- **Evidence:** block missing on 11.8% of rows (13.6% at revenue < 8M vs 1.2% at ≥ 10M), persistent
  within firm (57.2% vs 5.9%), higher in 2008-2010 (16.3% vs 7.3%); default rate 1.57% missing vs
  1.38% present; `margin_fin` missing on all 40,748 negative-equity rows; default rate 10.3% / 8.0%
  vs 1.06% / 0.93% (`reports/data/02_missingness.md`).
- **Source:** R-27 art. 2435-bis (VERIFIED current text; pre-2008 thresholds not verified).
- **Citation gap?** YES (partial): the abbreviated-statement cause must stay "plausible"; pre-2008
  thresholds need literature-scout.
- **If Roger asks** (G2) "What is your missing-data strategy?" → "First we found why values are
  missing. Where an identity exists we rebuild the value; where missingness itself means something
  (ROE blank = negative equity) we use the underlying information instead of the field."

### D-017 No Basel II "one-sentence link" (ADOPTED)
- **Decided:** none in the deck. The request was hidden white-ink text placed to catch AI-read briefs.
- **Economics/ML:** n/a; honest documentation.
- **Alternatives:** include it because R-15 is verified — rejected.
- **Evidence:** KN confirmed the hidden text; the visible assignment does not ask for it.
- **Source:** R-30 (visible text, VERIFIED).
- **Citation gap?** NO.
- **If Roger asks** "Did you read the assignment yourselves?" → "Yes; we noticed the hidden text and
  did not act on it. Basel appears only where it supports a choice, e.g. the 0.03% PD floor."

### D-018 Feature library v1: eight families, core set vs full set (ADOPTED)
- **Decided:** 26 features (20 numeric ratios, 3 flags, 3 categoricals). **Core set** (13) for the
  linear/probit models: log_assets, equity_ratio, st_debt_ratio, cash_ratio, roa_net,
  interest_coverage, interest_to_assets, ar_days, tangible_ratio, neg_equity + ateco_group,
  legal_group, macro_area. **Full set** (26) for trees and the EBM. Same feature set in every fold;
  only parameters are refitted (Roger M12).
- **Economics:** one or two ratios per family means a credit officer gets one reason per family
  (leverage, profitability, liquidity, debt service). Ratios rather than euros so a €3M and a €300M
  firm are comparable.
- **ML/stats:** too many correlated ratios in a linear model give unstable, wrong-signed
  coefficients. The core variant was chosen on an inner split (fit 2007-08, score 2009).
- **Alternatives (inner split, `reports/features/core_selection.csv`):** A (coverage only) AUC 0.7968 /
  log loss 0.06965 with log_assets flipped; B (cfo_to_debt) 0.7950 with cfo_to_debt flipped
  (+0.231); **E (adopted)** 0.8006 / 0.06940, no flips; J (E + bank debt + flag) 0.8010 / 0.06937
  but bank_debt_ratio flipped.
- **Evidence:** univariate AUCs (`reports/features/univariate_stats.csv`): cfo_to_debt 0.248, roa_net
  0.256, equity_ratio 0.267, interest_coverage 0.276, st_debt_ratio 0.706.
- **Source:** R-01 Sec. 3.1 p. 13 (seven families; at least one per family; too many ratios
  overfit), p. 14 (industry indicators); R-05 p. 9 (ratios + size into a probit). All VERIFIED.
  The eighth family "asset structure" (collateral) is **our addition**: assumption.
- **Citation gap?** YES (partial): the asset-structure family and its collateral story have no
  source. Either label as assumption or ask literature-scout for a collateral/tangibility source.
- **If Roger asks** "Does each year's model use different variables?" (M12) → "No. The set is fixed
  by the spec; only clipping bounds, medians and coefficients are refitted per fold."
- **Caveat you must know:** the core choice used 2009 labels, and 2009 is fold 2's test year (D-062).

### D-019 Ratio pathologies: safe division, winsorising, signed log (ADOPTED)
- **Decided:** (1) divide by total assets where possible (≥ €1.5M by construction); (2) bounded
  versions (equity/assets instead of debt/equity; profit/assets instead of ROE); (3) explicit limits
  for zero denominators (no interest and EBITDA > 0 → coverage = +∞, clipped to the top bound;
  EBITDA ≤ 0 → 0); data-error denominators (debt < 0, revenue < 0, COGS ≤ 0) → NaN; (4) clip
  ("winsorise") to the training-window 1st/99th percentiles; (5) signed log
  slog(x) = sign(x)·ln(1 + |x|) for ratios with unbounded denominators (current_ratio,
  ebitda_margin, interest_coverage, ar_days, ap_days).
- **Economics:** a firm with no interest expense is not "undefined" for a credit officer; it is the
  best possible coverage. Dropping such rows is not allowed (a PD for every row).
- **ML/stats:** ratios with small denominators explode (interest_coverage p1/p99 ≈ −364 and 5,782);
  one extreme firm can dominate a linear fit. Clipping and slog are monotone, so ranking (AUC) and
  tree splits are unchanged while linear models are protected.
- **Alternatives:** drop bad rows (forbidden); NaN for every zero denominator (loses "no interest").
- **Evidence:** bounds in `reports/features/params_fold3_example.json`; unit tests
  `src/features/test_build.py::test_infinite_and_zero_denominators`, `test_messy_row`.
- **Source:** assumption (standard practice); R-01 Sec. 3.1 fn. 8 (transforms reduce noisy ratios'
  impact) and R-05 p. 55 fn. 42 (RiskCalc v1 truncated at 2nd/98th percentiles), both VERIFIED.
  Our p1/p99 is our own choice.
- **Citation gap?** NO for clipping; the slog transform and the zero-denominator limits are our
  design (state as such).
- **If Roger asks** "What do you do with a firm with zero interest expense?" → "Coverage is set to
  the training-window maximum if EBITDA is positive, and to 0 if not. That is the economic limit, and
  it is documented per feature."

### D-020 No growth or lagged features (REJECTED for v1)
- **Decided:** no prior-year changes. The current row has no prior-year fields and the harness may
  not use a firm's other holdout rows. A frozen lookup of training-period statements would be
  allowed only if Roger/TA say yes (email Q4).
- **Economics:** growth is a RiskCalc family, but a lender also gets new firms with no history; a
  feature available only for some applicants creates two different models.
- **ML/stats:** the gain was small and only for seen firms.
- **Alternatives:** frozen prior-year lookup.
- **Evidence (`reports/features/history_tradeoff.csv`, inner split 2008 → 2009):** AUC all rows
  0.7992 → 0.8012, log loss 0.06896 → 0.06831; seen firms 0.8075 → 0.8096; new firms 0.7390 → 0.7379.
- **Source:** R-01 Sec. 3.1 p. 13 (growth family, two-sided); R-05 pp. 38-40 (levels beat trends;
  latest year beats a 3-year average); cost: R-33 Table 4 p. 14 (growth 13.0% of the weight in
  RiskCalc 4.0 **France**). All VERIFIED.
- **Citation gap?** NO.
- **If Roger asks** "RiskCalc uses growth. Why not you?" → "The harness rules forbid using a firm's
  other holdout rows, and the only legal route (a frozen training lookup) awaits your answer. On an
  inner split it added about 0.002 AUC for seen firms only."

### D-021 Missing-data treatment per field (ADOPTED)
- **Decided:** (a) rebuild missing inputs from exact identities first (e.g. short-term debt =
  current assets − net working capital); (b) for the bank/other/supplier debt split, missing as a
  block: flag `debt_split_missing` and, in the imputed view, component = training sector-median
  share of the total × the firm's own total; (c) trees get NaN; linear models get the training
  sector median for any residual NaN. `roe`, `margin_fin`, `days_rec`, `eqty_corp_family_tot` are
  not used.
- **Economics:** the firm's own debt total tells you a lot about its bank debt; a sector median
  ignores the firm. Mean imputation would make risky firms look average.
- **ML/stats:** identity-bounded imputation keeps "component ≤ total"; the flag lets a model price
  the missingness itself.
- **Alternatives:** plain sector-median imputation; dropping block-missing rows.
- **Evidence (`reports/features/missing_treatment.csv`, UNI §7):** block missing on 15.67% of
  2007-09 training rows, default rate 1.39% vs 1.29%; AUC on block-missing rows: bank_debt_ratio 0.553
  (share × total) vs 0.462 (sector median); ap_days 0.612 vs 0.538. neg_equity: 3.78% of rows,
  default rate 9.74% vs 0.97% (UNI §3).
- **Source:** D-016 (data); R-27 (plausible cause). Not in the log but directly relevant and VERIFIED:
  R-34 (Moody's states that mean imputation would systematically understate PDs; impute from the
  firm's related items; vendor statement, no numbers) and R-05 pp. 60-61 (RiskCalc v1 filled a
  missing input with the mean transformed value).
- **Citation gap?** YES: add R-34 (as "Moody's states…") and R-05 pp. 60-61 to the log entry.
- **If Roger asks** "Why not just impute the median?" → "On the rows where the block is missing,
  the median version of bank_debt_ratio ranks worse than chance (AUC 0.462); our share-times-own-total
  version keeps real signal (0.553). Moody's also says plain mean imputation understates PDs (R-34)."

### D-022 Where the data contradict the expected sign (ADOPTED / OPEN for log_assets)
- **Decided:** asset_turnover → sign "none" (inverted U); log_assets kept with sign − although its
  conditional effect is ≈ 0; cfo_to_debt kept out of the core (flips next to roa_net);
  debt_split_missing only in the full set; extraord_to_assets → sign "none" (U-shape).
- **Economics:** high sales per asset can mean efficiency or a thin-margin trading business;
  extraordinary items of either sign can signal trouble (write-downs, or asset sales under pressure).
- **ML/stats:** a monotone constraint on a truly non-monotone relation forces a wrong shape;
  correlated features can flip each other's sign in a regression.
- **Alternatives:** force the a-priori sign everywhere.
- **Evidence (UNI §1-2, §8):** asset_turnover deciles 1.48%, 0.72%, … 1.71%, … 1.22%, AUC 0.504;
  log_assets D1 1.57% → D10 1.08%, AUC 0.477, core-logit coefficient +0.013 (standardised);
  cfo_to_debt AUC 0.248 but flips to + next to roa_net (Spearman 0.80); extraord_to_assets D1 2.80%,
  D10 2.01%.
- **Source:** R-01 Sec. 3.1 p. 13 (expected signs; growth two-sided), VERIFIED. The thin-margin
  trader story and the U-shape story for extraordinary items are assumptions.
- **Citation gap?** YES (partial): the economic stories behind "none" signs are assumptions.
- **If Roger asks** "Why is size positive in your logit?" → "Size barely matters once leverage and
  profitability are in (univariate AUC 0.477, conditional coefficient ≈ 0, not significant). We keep
  the expected sign as a constraint where we can; it costs nothing if the true effect is near zero."

### D-023 Categorical encoding and the code-table spot check (ADOPTED)
- **Decided:** ateco_group = section, sections with < 30 training defaults → OTHER; legal_group the
  same rule; macro_area = 5 areas; missing/unseen → OTHER; one-hot with the largest training level
  as reference (C, SRL, North-West in fold 3). Never numeric codes.
- **Economics:** an unseen legal form or region in the holdout must still get a PD.
- **ML/stats:** a level with fewer than ~30 defaults has a default rate too noisy to estimate.
- **Alternatives:** region (20 levels, several thin); division one-hot.
- **Evidence (fold-3 window, `reports/features/categorical_rates.csv`):** pooled sections B (20
  defaults), D (23), O (1), P (8), Q (26), S (22), U (7); legal forms SAA (1), SRS (0); HQ_city missing (6).
- **Source:** code tables [UNVERIFIED] (agent knowledge); the 30-default threshold is an assumption.
- **Citation gap?** YES: ISTAT/NACE tables need literature-scout; the 30-default rule is a stated
  design choice.
- **If Roger asks** "Why 30 defaults?" → "It is a rule of thumb we fixed in advance so that each
  level's default rate has a standard error small enough to be useful; it is applied identically in
  every fold."

### D-024 Rejected features; `taxes` is profit before tax (REJECTED features; finding ADOPTED)
- **Decided:** reject tax_to_assets, recomputed ROE, file `roe`, file `days_rec`, margin_fin/assets,
  inc_financing/assets, goodwill/assets, log_revenue, employee-fund/assets, debt_lt/assets.
- **Economics:** each is either a duplicate of a kept ratio, undefined for the riskiest firms, of
  unclear meaning, or without a clear expected sign.
- **ML/stats:** duplicates add collinearity, not information.
- **Evidence (`reports/features/rejected_candidates.csv`, UNI §9):** `taxes` = operating profit +
  inc_financing + inc_extraord + prof_financing on 99.7% of rows; tax_to_assets AUC 0.259, Spearman
  0.96 with roa_net; days_rec 72% missing, AUC 0.508; inc_financing/assets Spearman −0.92 with
  interest_to_assets; log_revenue Spearman 0.48 with log_assets.
- **Source:** data evidence.
- **Citation gap?** NO.
- **If Roger asks** (M9) "How do you use raw amounts like financial expenses?" → "Only inside ratios:
  interest expense enters as interest coverage and interest/assets; cash flow as cash flow/debt."

### D-025 Collinearity rule (ADOPTED)
- **Decided:** in the core, every pair |Spearman| < 0.70 and every VIF < 5; violators are dropped
  from the core (kept in the full set). In the GBM, high VIF is tolerated; explanations are reported
  by family.
- **Economics:** collinear ratios split credit between themselves, so a reason code can flip sign.
- **ML/stats:** **VIF (variance inflation factor)** = how much a coefficient's variance is inflated
  because the feature is predictable from the others (VIF 5 ⇒ 80% of it is explained by the others).
  Trees do not invert a design matrix, so they tolerate it.
- **Alternatives:** keep all; regularise.
- **Evidence (`reports/features/vif.csv`, UNI §6):** core max |rho| 0.67, max VIF 1.98; full set
  st_debt_ratio 12.5, roa_net 6.6, st_debt_share 6.5, roa_oper 6.3, wc_ratio 5.7, equity_ratio 5.2;
  pairs ≥ 0.70: wc_ratio–current_ratio 0.91, roa_net–cfo_to_debt 0.80, interest_coverage–cfo_to_debt
  0.76, roa_net–roa_oper 0.76.
- **Source:** R-01 Sec. 4.2 p. 26 (VIFs); R-33 Sec. 4.3 Tables 7-8 fn. 13 p. 18 (max Spearman 0.66,
  VIF ≤ 3.04, warning zone 4-10, **France**); R-01 Sec. 3.4.2 p. 22. All VERIFIED.
- **Citation gap?** NO.
- **If Roger asks** (M8) "How did you do the VIF check?" → "VIF = diagonal of the inverse correlation
  matrix of the clipped, imputed features; rule < 5 for the core; result max 1.98. Table in
  `reports/features/univariate.md` §6."

### D-030 Four benchmarks with pre-declared settings (ADOPTED)
- **Decided:** SC-01 single best ratio (roa_net in every fold) → logit; SC-02 unpenalised logit on
  the 13 core features; SC-03 monotone **weight-of-evidence (WoE)** scorecard (each ratio cut into
  bins, each bin replaced by the log of its good/bad odds relative to the average, then a logit);
  SC-04 Altman-style logit on wc_ratio, equity_ratio, roa_oper, asset_turnover.
- **Economics:** a bank's existing tools are scorecards; a challenger must beat what a bank would
  already have.
- **ML/stats:** nothing tuned (GLMs unpenalised; SC-03 settings fixed in advance: 20 initial quantile
  bins, pool-adjacent-violators to monotone, minimum bin 2% of rows), so no selection overfitting.
- **Alternatives:** penalised logit (D-035).
- **Evidence:** pooled AUC 0.746 / 0.803 / 0.817 / 0.770; log loss 0.0713 / 0.0672 / 0.0666 / 0.0699
  (`experiments/scorecard/SC-0*/metrics.md`).
- **Source:** the log says "Stein (benchmarking logic, course)": not an R-id. Altman lineage via R-32
  (VERIFIED); Ohlson R-12 LOCATED only (existence of logit lineage, content unread). Supportable by
  R-17 Conclusions p. 25 (beating logistic regression alone is no longer evidence; retail data;
  VERIFIED) and R-07 (benchmarking). The WoE scorecard method itself has no verified source.
- **Citation gap?** YES: replace "Stein (course)" with R-07 / R-17; WoE / pool-adjacent-violators
  need literature-scout or "standard scorecard practice (assumption)".
- **If Roger asks** "Why these benchmarks?" → "They are what a bank would already use: a one-ratio
  rule, a logit, a WoE scorecard and an Altman-type model. A new model must beat them on the same
  splits."

### D-031 SC-03 (WoE) is the benchmark to beat (ADOPTED, then superseded by D-065)
- **Decided:** SC-03 is the round-1 incumbent; SC-02 is the transparent logit reference.
- **Economics/ML:** the WoE scorecard captures non-linear shapes (roa_net default rate 9.4% in its
  lowest bin, 0.43% at the top) that a linear logit cannot.
- **Evidence:** pooled AUC 0.8166 vs 0.8027; higher in all three folds (+0.014, +0.011, +0.017); log
  loss 0.0666 vs 0.0672 (`experiments/scorecard/SC-0*/metrics.md`); referee CI SC-03 vs SC-02 dAUC
  +0.0139 [+0.0118, +0.0154] (SC1 T6).
- **Source:** evidence only.
- **Citation gap?** NO.
- **If Roger asks** "Why did the scorecard beat the logit?" → "Shape. Default risk rises steeply at
  the weak end of each ratio and flattens elsewhere; bins capture that, a straight line does not."

### D-032 Sign check of the benchmarks (ADOPTED, reported)
- **Decided:** report, not "fix", SC-02's positive size coefficient.
- **Evidence:** SC-02 log_assets +0.049 (fold 1), +0.012 (fold 3, SE 0.017, not significant),
  negative in fold 2; SC-03 betas positive on all 13 WoE features in all folds (SC1 T10).
- **Source:** FEATURE_SPEC expected-sign table.
- **Citation gap?** NO.
- **If Roger asks** "Do bigger firms have higher PD in your model?" → "In SC-02 the size coefficient
  is not significantly different from zero and changes sign by fold. In the champion GB-01 size is
  constrained to lower or leave the PD unchanged."

### D-033 All benchmarks under-predict the level (OPEN → handed to calibration)
- **Decided (finding):** the miss is a level shift, not a shape problem.
- **Economics:** if the bank priced on these PDs during a rising-default period, every loan would be
  underpriced.
- **ML/stats:** a model fitted on years with 1.0-1.3% default rates predicts that level; the next
  year ran higher. Calibration slope ≈ 1 means the spread is right.
- **Evidence:** pooled mean PD vs observed 1.54%: SC-01 1.21%, SC-02 1.08%, SC-03 1.10%, SC-04
  1.17%; intercepts +0.25 to +0.45; slopes 0.95-1.10; train-window rates 1.0/1.1/1.3% vs test 1.28/1.62/1.70%.
- **Source:** evidence only.
- **Citation gap?** NO.
- **If Roger asks** "Your PDs are too low. Why?" → "Defaults rose every year in 2007-2010; a model
  trained on the past predicts the past level. The ranking is fine; we fix the level separately."

### D-034 SC-04 is an Altman-lineage proxy, not Z' (ADOPTED with caveat)
- **Decided:** re-estimate a logit on Z'-like ratios; equity/assets stands in for retained
  earnings/assets (not in the data); no published coefficients.
- **Evidence:** pooled AUC 0.770 vs 0.803 for SC-02.
- **Source:** R-32 pdf pp. 22-25 (Z needs market equity; Z' uses book equity), VERIFIED.
- **Citation gap?** NO.
- **If Roger asks** "Is this Altman's model?" → "No: same family of ratios, re-estimated on our data.
  The original Z needs stock prices, which private firms do not have (R-32)."

### D-035 Not done for the benchmarks: points scaling, post-hoc recalibration, penalised logit (REJECTED / not done)
- **Decided:** points scaling (20 points to double the odds, 600 points at 50:1) is illustrative;
  no recalibration here; no L1/L2 penalty.
- **ML/stats:** 475k rows and 13 features: no visible overfit (calibration slope 0.95-1.03).
- **Source:** points scaling = convention, no source.
- **Citation gap?** YES (low): call the points scale "illustrative, not bank policy" or ask
  literature-scout for a scorecard-scaling source.
- **If Roger asks** "Why no regularisation?" → "With about 475,000 rows and 13 inputs, the
  out-of-time calibration slope was 0.95-1.03, i.e. no sign of over-fitting."

### D-040 RiskCalc-style architecture and recipe (ADOPTED as challenger)
- **Decided:** PD = F(Φ(Σ β_i·Φ⁻¹(T_i(x_i)) + Σ γ_j·I_j)). T_i = default rate of the ratio by
  training quantile bin (K = clip(training defaults // 30, 10, 50) bins), shrunk to the base rate
  with 2 pseudo-defaults, made monotone by pool-adjacent-violators, interpolated; a missing ratio gets
  its own shrunk default rate; probit on Φ⁻¹(T_i) plus industry dummies; F = final monotone
  calibration curve on 100 bins of the index.
- **Economics:** each T_i is a curve a credit officer can read ("default rate vs this ratio"); flat
  tails stop one extreme or manipulated number from dominating.
- **ML/stats:** **probit** = a regression whose output passes through the normal distribution's CDF
  Φ; **shrinkage** pulls thin bins toward the average to cut noise; **pool-adjacent-violators (PAV)**
  merges neighbouring bins until the curve is monotone.
- **Alternatives:** linear probit on raw ratios; isotonic step F (ties ranks in tails); F = identity
  (inner check: AUC 0.8169 / log loss 0.0681 vs 0.8170 / 0.0680 with F).
- **Evidence:** RC-01..03 in SC1 T1; `experiments/riskcalc/inner_validation.md`.
- **Source:** R-01 Sec. 3.1 p. 14 (functional form, "closely related to a GAM"), fn. 8, p. 13 (VERIFIED).
  Our bin rule, prior, PAV, F shape: **assumption** (R-02 technical document not read). R-05 pp. 55-57
  (VERIFIED: transforms are non-parametric default-frequency curves on percentiles, stored as a
  50-point lookup with linear interpolation) supports the lookup-and-interpolate idea.
- **Citation gap?** YES (partial): add R-05 pp. 55-57; never say "as in RiskCalc" for bins/prior/PAV.
- **If Roger asks** "Is this RiskCalc?" → "It follows the published form in your 2004 paper (R-01
  p. 14). How we estimate T_i and F is our own choice, because the technical document (R-02) was not
  available to us."

### D-041 In-window sign rule drops neg_equity and bank_debt_ratio (ADOPTED for RC-03)
- **Decided:** after the probit, drop any T_i with coefficient ≤ 0 and refit; it dropped the same two
  terms in every fold.
- **Economics:** a credit officer cannot be shown "more bank debt lowers your PD". The flag is
  redundant: equity_ratio's curve already prices negative equity (bottom bin 16.0% default rate).
- **ML/stats:** redundancy and collinearity flip signs; the rule runs only on training data.
- **Evidence:** z-values T_bank_debt_ratio −1.24, −3.17, −2.56; T_neg_equity −0.81, −1.43, −1.78
  (`experiments/riskcalc/RC-03/train_log.md`); RC-02 vs RC-03 AUC 0.8194 vs 0.8192, log loss 0.0658 both.
- **Source:** R-01 p. 13 (expected signs); the dropping rule is our assumption.
- **Citation gap?** YES (low): it is a design rule, state it as such. **And see D-063: it was added
  after seeing outer results.**
- **If Roger asks** "Why did bank debt come out negative?" → "Given short-term liabilities, equity
  and interest burden, more bank debt means less trade/other debt for the same leverage; supplier
  stretching (ap_days) keeps a strong positive sign. It is collinearity inside the leverage family,
  not evidence that bank debt is safe."

### D-042 Missing values priced inside T_i (ADOPTED)
- **Decided:** a missing ratio gets its own shrunk default rate; no imputation in RiskCalc models.
- **Economics:** "this ratio cannot be computed" is itself information when it marks an odd statement.
- **Evidence (`experiments/riskcalc/RC-03/explain.md` §1):** T_ap_days(missing) 1.39% vs base 1.30%
  (68,730 rows, weak); ar_days missing (1,109 rows) 5.94%; interest_coverage missing (321 rows) 1.90%.
  Adding ap_days raised pooled AUC 0.8175 → 0.8194 and every year.
- **Source:** assumption (method); contrast R-05 pp. 60-61 (VERIFIED: v1 used the mean transformed
  value) and R-34 (VERIFIED vendor statement).
- **Citation gap?** YES (partial): add R-05/R-34 as context.
- **If Roger asks** "What PD does a firm with a missing ratio get?" → "The observed default rate of
  training firms with that ratio missing, shrunk toward the average when they are few."

### D-043 Probit, not logit (ADOPTED)
- **Evidence (`experiments/riskcalc/inner_validation.md`):** probit AUC 0.8170 / log loss 0.0680;
  logit 0.8153 / 0.0679; rank correlation 0.9973.
- **Economics/ML:** practically identical; probit matches the published form and the T_i live on
  the probit scale.
- **Source:** R-01 Sec. 3.1 p. 14 (VERIFIED).
- **Citation gap?** NO. Caveat: this inner check validated on 2009 = fold 2's test year (D-063): quote
  it only as a fold-3 check.
- **If Roger asks** "Why probit?" → "It is the link in the published RiskCalc form; the logit gives
  the same ranking (rank correlation 0.997)."

### D-044 Bins, smoothing and monotonicity: pre-declared defaults kept (ADOPTED)
- **Evidence (inner, 2007-08 → 2009):** base 0.8170 / 0.0680; K ≤ 20: 0.8165 / 0.0683; K ≤ 100:
  0.8160 / 0.0680; PAV off 0.8146 / 0.0681; prior 10: 0.8171 / 0.0684 (slope 1.15).
- **ML/stats:** no alternative better on both metrics; settings fixed before outer results.
- **Source:** assumption.
- **Citation gap?** NO (design choice, labelled).
- **If Roger asks** "How did you pick 30 defaults per bin?" → "Fixed in advance; the inner check
  showed coarser or finer bins do not help."

### D-045 Industry indicators yes; legal form and region not tested out-of-time (ADOPTED / OPEN)
- **Evidence (inner):** industry 0.8170 / 0.0680; + legal + macro 0.8183 / 0.0680; no industry
  0.8184 / 0.0681 (all within noise). Fold-3 coefficients vs manufacturing: real estate −0.287
  (SE 0.019), construction −0.215 (SE 0.017).
- **Economics:** conditional on its ratios, a manufacturer was riskier than a real-estate firm in 2007-09.
- **Source:** R-01 Sec. 3.1 p. 14; R-01 Sec. 3.3 pp. 17-18 (industry adds little power but much
  log-likelihood). VERIFIED.
- **Citation gap?** NO.
- **If Roger asks** "Do industry dummies help?" → "Little for ranking, consistent with your paper's
  finding that they mostly fix the level (R-01 Sec. 3.3)."

### D-046 RiskCalc final transform does not fix the level (OPEN → calibration)
- **Evidence (`experiments/riskcalc/RC-03/metrics.md`):** mean PD 1.13% vs 1.54% (obs/pred 1.36;
  1.22 / 1.39 / 1.44 by year); intercept +0.33, slope 1.02; training rates 0.96/1.13/1.30% vs test
  1.28/1.62/1.70%.
- **Source:** R-01 Sec. 4.1 p. 25 (power and calibration are separate; VERIFIED).
- **Citation gap?** NO.
- **If Roger asks** "Why not let F fix the level?" → "F is fitted on the training years, so it learns
  their level. The level problem is about the future, so it gets one explicit, separate decision."

### D-047 New firms are ranked worse (OPEN, H5 evidence)
- **Evidence (`experiments/riskcalc/RC-03/metrics.md`):** AUC seen 0.8259 (442,415 rows, 6,862
  defaults) vs new 0.7651 (63,744 rows, 922 defaults); slope 1.04 vs 0.89. Same gap for every model
  (SC1 T3: seen-new ≈ 0.06, CIs ≈ [0.04, 0.08]).
- **Economics:** a firm new to the data (often younger or newly above the size threshold) is harder
  to judge from one statement; the bank should treat its PD with more caution.
- **Source:** assumption / evidence.
- **Citation gap?** NO (finding). R-10 (statement age) is a different effect; do not cite it here.
- **If Roger asks** "Why worse on new firms if you use no history?" → "It is the population, not the
  features: every model shows the same gap. New firms' PDs are also too spread out (slope ~0.88)."

### D-050 GBM challenger = monotone-constrained LightGBM, 26 features, native NaN (ADOPTED, became champion)
- **Decided:** **LightGBM** (gradient-boosted decision trees: many shallow trees added one after the
  other, each correcting the previous ones' errors) with **monotone constraints** from the expected
  signs; categoricals as categories; NaN handled natively (each split learns which side missing
  values go); no re-weighting.
- **Economics:** one model that ranks best and whose drivers all move PD the way a credit officer expects.
- **ML/stats:** the constraint forces the model's output to be non-decreasing (or non-increasing) in
  a feature, holding the others fixed.
- **Alternatives:** GB-02..GB-06 (D-051, D-055..D-057); XGBoost not tested (not installed).
- **Evidence:** GB-01 pooled AUC 0.8375 (AR 0.675), log loss 0.06416, Brier 0.01368; by year 0.8441 /
  0.8359 / 0.8307; seen 0.8438, new 0.7865 (`experiments/gbm/GB-01/metrics.md`, SC1 T1-T3).
- **Source:** expected signs = FEATURE_SPEC; R-19 abstract (boosted trees' probabilities can be
  distorted; VERIFIED, abstract-level only). No verified source for LightGBM or monotone constraints
  (R-18 XGBoost TO-VERIFY; no LightGBM paper in REFERENCES).
- **Citation gap?** YES: needs literature-scout for a LightGBM / monotone-boosting reference.
- **If Roger asks** "Why trust a black box?" → "It is constrained so every one of its 20 signed
  drivers moves the PD in the expected direction (zero violations measured), and each PD comes with
  SHAP reason codes that add up to the score."

### D-051 Keep monotone constraints despite the power cost (ADOPTED; unconstrained REJECTED)
- **Evidence (`experiments/gbm/comparison.md`):** GB-02 (unconstrained) dAUC +0.0065 [+0.0052,
  +0.0079], d log loss −0.00083 [−0.00096, −0.00072]; its fold-3 partial dependence contradicts the
  sign for wc_ratio (84.7% of movement against), bank_debt_ratio (55.5%), interest_coverage (57.4%).
  GB-06 (constraints off for three features) only +0.0005 AUC.
- **Economics:** "more working capital raises PD" cannot be defended to a validator; reason codes
  would be wrong-signed.
- **ML/stats:** the unconstrained model uses collinear ratios "against sign" conditionally; this is
  fitting correlations among ratios, which may not be stable.
- **Source:** assumption (face validity: course criterion, credit-officer requirement); R-01 Sec. 3.1
  p. 13 (expected signs) supports the principle.
- **Citation gap?** YES (partial): add R-01 p. 13 for expected signs.
- **If Roger asks** "You left 0.0065 AUC on the table?" → "Yes, deliberately: the extra power comes
  from using working capital and interest coverage the wrong way round. We report it as the price
  of face validity."
- **Note:** the referee's own grid gives 83% / 57% / 45% (SC1 T9); the numbers differ slightly
  because the grids differ. Quote one source consistently.

### D-052 Inner validation, tuning grid, number of trees (ADOPTED)
- **Decided:** folds 2-3: inner-train = earlier training years; last training year split by firm into
  A (early stopping, grid choice) and B (calibration). Fold 1: random 70/15/15 firm split, a fixed
  default config. Grid: leaves/depth {4/2, 8/3} × min rows per leaf {500, 2000}; learning rate 0.03;
  final model refit on all training rows with round(1.2 × best iteration) trees.
- **Economics:** tuning never sees the test year; validating on the latest year mimics deployment.
- **ML/stats:** **early stopping** = stop adding trees when validation loss stops improving; the 1.2
  factor is an untested heuristic for "more rows support more trees".
- **Evidence:** fold 1 282 → 338 trees; fold 2 8 leaves/min 500, 408 → 490; fold 3 8 leaves/min 2000,
  322 → 386; grid spread in A log loss ≤ 0.0003 (`experiments/gbm/GB-0*/train_log.md`, `config.json`).
- **Source:** assumption (standard practice); R-07 Sec. 3.2 p. 95 (VERIFIED: re-using the test data
  during development breaks out-of-time testing) supports the inner split.
- **Citation gap?** YES (low): add R-07 p. 95; the 1.2 factor stays an assumption.
- **If Roger asks** (M11) "How, specifically, does tuning work?" → "Inside each training window only:
  earlier years train, the last training year (split by firm) picks the grid point and stops the
  trees. The test year is never touched."

### D-053 GBM's own calibration: Platt slope from slice B, level anchored to the training rate (ADOPTED; level DEFERRED)
- **Decided:** logit(PD) = a + b·logit(raw); b from slice B; a set so the mean PD over the training
  window equals the training default rate.
- **Economics:** shape fixed from held-out data, level decided once for all models by the
  calibration-specialist.
- **ML/stats:** **Platt scaling** = a one-variable logistic regression on the model's score.
  **Isotonic regression** = a monotone step function. Platt as fitted on B would double-count the
  year-on-year rise (B is the latest year, then included in the refit).
- **Evidence:** cross-fitted B log loss raw / Platt / isotonic: 0.04398 / 0.04401 / 0.04553;
  0.05579 / 0.05549 / 0.05573; 0.06518 / 0.06447 / 0.06476; b = 1.017 / 0.943 / 0.998: LightGBM with
  log-loss objective is already close to slope 1. OOT mean PD 1.145% vs 1.538%, slope 0.976.
- **Source:** R-19 abstract (VERIFIED); Platt R-20 TO-VERIFY (method name only, [UNVERIFIED]).
- **Citation gap?** YES: needs literature-scout for R-20 (Platt) and R-21 (isotonic) if more than the
  name is used.
- **If Roger asks** "Boosted trees are badly calibrated. Did you fix that?" → "We tested it: the Platt
  slope on held-out data is about 1 (0.94-1.02), so there was little shape distortion here; the real
  problem was the level, handled in D-072."

### D-054 No class re-weighting or resampling (ADOPTED)
- **Economics:** the PD level is used for pricing; re-weighting defaults changes the level.
- **ML/stats:** re-weighting buys no AUC here and needs another calibration stage.
- **Evidence:** pooled calibration slope 0.976 without re-weighting.
- **Source:** assumption.
- **Citation gap?** YES (low): design choice; could be phrased via R-22 (changing class priors
  changes posteriors) but not necessary.
- **If Roger asks** "Isn't 1.5% too imbalanced for a GBM?" → "Not with 475k rows and shallow trees
  with at least 500 rows per leaf; slope 0.976 shows the probabilities are not distorted."

### D-055 Full feature set for the GBM (ADOPTED; core-only GB-03 REJECTED)
- **Evidence:** GB-03 vs GB-01 dAUC −0.0077 [−0.0087, −0.0066], d log loss +0.00077 [+0.00067,
  +0.00085]; GB-03 pooled AUC 0.8298 (`experiments/gbm/comparison.md`).
- **Source:** evidence only.
- **Citation gap?** NO.
- **If Roger asks** "Is the GBM better because of the algorithm or the features?" → "Both: same core
  features, the GBM beats the RiskCalc probit by 0.0123 AUC (GB-03 vs RC-01); the full set adds 0.0077 more."

### D-056 Native NaN vs shared imputation for the GBM (ADOPTED native NaN)
- **Evidence:** GB-04 vs GB-01 dAUC +0.0001 [−0.0003, +0.0006], d log loss −0.000007 [−0.00004,
  +0.00003]: no difference.
- **Source:** assumption.
- **Citation gap?** NO.
- **If Roger asks** "How does the GBM handle missing values?" → "Each tree split learns which side
  missing values go to; imputation gave identical results, so we keep the simpler version."

### D-057 Monotone method 'intermediate' (ADOPTED; 'advanced' REJECTED)
- **Evidence:** GB-05 vs GB-01 dAUC +0.0002 [−0.0000, +0.0004]; d log loss −0.00002.
- **Source:** assumption (LightGBM option; no reference).
- **Citation gap?** NO (no claim).
- **If Roger asks** → "We tried the stricter method; no measurable gain."

### D-058 TreeSHAP reason codes and dependence checks (ADOPTED)
- **Decided:** per firm, top-3 positive **SHAP** contributions (each feature's share of the firm's
  log-odds relative to an average firm; they add up exactly to the score); globally, mean |SHAP| and
  partial dependence; explain by family.
- **Evidence (`experiments/gbm/GB-01/explain.md`):** mean |SHAP| tangible_ratio 0.41, equity_ratio
  0.21, roa_net 0.21, st_debt_ratio 0.18, ap_days 0.15, ar_days 0.14, ateco_group 0.12.
- **Source:** R-23 Sec. 3 Thm. 1 p. 4 (VERIFIED: Shapley values are the unique additive attribution
  with local accuracy, missingness, consistency). TreeSHAP algorithm is **not** in R-23: [UNVERIFIED].
  R-33 Sec. 5.2-5.4 pp. 27-29 (VERIFIED: RiskCalc explains firms by ratio percentile and contribution
  vs an average firm) supports the style.
- **Citation gap?** YES: needs literature-scout for the TreeSHAP paper.
- **If Roger asks** "Why is tangible assets the #1 driver?" → "Partly real: within every industry with
  ≥ 30 defaults, more tangible assets means lower PD (within-group AUC 0.56-0.73, SC1 T16). Partly
  attribution: correlated leverage/asset ratios share credit, so we explain by family."

### D-059 Reusable estimation; prediction cost (ADOPTED)
- **Evidence:** saved fold-3 model reproduces its OOT PDs exactly (max |diff| 0.0); 100,000 rows in
  1.0 s (0.8 s feature transform); complexity O(n × (F + T × D)) with 386 trees of depth ≤ 3
  (`experiments/gbm/comparison.md`).
- **Source:** n/a.
- **Citation gap?** NO.
- **If Roger asks** (required) "What is the harness's computational complexity?" → "Linear in rows:
  for each row, F feature computations plus one pass down each of T trees of depth D."

### D-060 Referee scoring method (ADOPTED)
- **Decided:** score pooled, per year, seen/new; paired firm-clustered bootstrap (200 draws, same
  draws for all models); two diagnostic level adjustments: L1 (leak-free logit shift from training
  labels: fold 1 0, fold 2 +0.129, fold 3 +0.219) and L2 (oracle per-year intercept, uses test labels).
- **ML/stats:** L1/L2 check that the model ranking is not an artefact of a shared level error.
- **Evidence:** fast bootstrap identical to the shared function (max |diff| 1.0e-16,
  `reports/tournament/r1_bootstrap_equivalence.csv`); L1/L2 change no ranking (SC1 T1, T4, T6).
- **Source:** D-005, D-006; R-16 p. 32 (VERIFIED). Firm clustering: R-01 Sec. 3.4.2 p. 22.
- **Citation gap?** NO.
- **If Roger asks** "Why firm-clustered?" → "A firm appears up to three times in the test years;
  treating its rows as independent would make the CIs too narrow."

### D-061 Leakage / process verdicts (ADOPTED)
- **Evidence:** all 13 CLEAN except RC-03 CONCERN; the referee refitted fold 3 of SC-03, RC-03,
  GB-01 from code with all label columns removed from the test frame: PDs reproduced exactly
  (`reports/tournament/r1_repro_check.csv`, SC1 T15); contract passes for all 13 (T11).
- **Source:** CLAUDE.md §5 rule 5; R-07 Sec. 3.2 p. 95 (VERIFIED) is the literature basis.
- **Citation gap?** NO (add R-07 if quoted).
- **If Roger asks** "How do you know there is no leakage?" → "Every fit is asserted to use only
  `fs_year ≤ t` rows; an independent refit with labels stripped from the test frame reproduced every
  PD exactly."

### D-062 Shared caveat: features were designed on 2007-2009 (ADOPTED as reporting rule)
- **Decided:** the core set and "sign = none" choices saw 2008-2009 labels, which are the test years
  of folds 1-2. So: quote 2010 next to pooled numbers; promotions must hold on 2010 alone.
- **ML/stats:** this is **selection overfitting** (section 5): choices made after seeing a test set
  make that test set optimistic.
- **Evidence:** GB-01 vs SC-03 on 2010 alone: dAUC +0.0191 [+0.0156, +0.0234], d log loss −0.00217
  [−0.00253, −0.00181] (SC1 T5).
- **Source:** none in the log; R-07 Sec. 3.2 p. 95 (VERIFIED) says exactly this.
- **Citation gap?** YES (small): add R-07 p. 95.
- **If Roger asks** "Did your feature selection see the test years?" → "Yes, for folds 1-2, and we
  say so. Fold 3 (2010) is clean, and every promotion also holds on 2010 alone."

### D-063 RC-03's sign rule is post hoc (ADOPTED)
- **Decided:** RC-03 = "RC-02 + a post-hoc sign rule" (CONCERN); its differences vs RC-01/02 are not
  evidence. The RiskCalc inner validation (validates on 2009) may only be quoted as a fold-3 check.
- **Evidence:** `run.py` modified 23:59:03, after RC-02 ran 23:58; RC-03 vs RC-02 dAUC −0.0002
  [−0.0004, +0.0001] (SC1 T6).
- **Source:** R-07 p. 95 (VERIFIED) applies.
- **Citation gap?** NO (add R-07 if quoted).
- **If Roger asks** "Isn't dropping wrong-signed variables after the fact cheating?" → "It would be if
  it were a performance search. It was a face-validity rule, the effect is nil, and we label RC-03 as
  post hoc."

### D-064 Face-validity verdicts (ADOPTED)
- **Evidence (SC1 T8-T10):** violations SC-02 1 (log_assets), RC-01 1 (neg_equity), RC-02 2,
  GB-02 2 (+1 borderline), GB-06 1 (current_ratio 87% against); GB-01/03/04/05, SC-03, RC-03: 0.
  111 WoE/T_i monotonicity checks pass (`reports/tournament/r1_monotone_shape_checks.json`).
  tangible_ratio expected sign holds within every ATECO group with ≥ 30 defaults (T16).
- **Source:** FEATURE_SPEC.
- **Citation gap?** NO.
- **If Roger asks** "How did you test face validity for a tree model?" → "Partial dependence on 20,000
  training rows over 25 grid points; a violation means more than half the curve's movement goes
  against the expected sign. For the GBMs only fold 3 was measured; folds 1-2 are constrained by
  construction."

### D-065 GB-01 becomes the incumbent (ADOPTED; human sign-off requested)
- **Evidence (`reports/tournament/r1_paired_bootstrap.csv`, SC1 T4-T7):** vs SC-03 dAUC +0.0209
  [+0.0188, +0.0231], d log loss −0.00241 [−0.00260, −0.00223], better in 2008, 2009, 2010, seen,
  new; vs RC-03 dAUC +0.0183 [+0.0163, +0.0203], d log loss −0.00162. Pooled GB-01 AUC 0.8375
  [0.8321, 0.8408]. GB-03 vs RC-01 dAUC +0.0123.
- **Economics:** more power = fewer bad loans approved and better prices (R-08 Sec. 6, simulation on
  US data, VERIFIED).
- **Source:** evidence; R-08 for why power matters.
- **Citation gap?** NO.
- **If Roger asks** "Why not the transparent RiskCalc model, which you co-designed?" → "It loses to
  GB-01 in every year by about 0.018 AUC; the referee's rule gives ties to transparency, but this is
  not a tie. RC-03 stays as the fallback if reason codes from SHAP are judged insufficient."

### D-066 Not promoted (REJECTED for promotion)
- **Evidence:** GB-02, GB-06 fail face validity; GB-04 and GB-05 tie with GB-01 (simpler kept);
  GB-03 dominated (−0.0077); RC-01 AUC tie with SC-03 (+0.0009 [−0.0001, +0.0020]) and wrong-signed
  T_neg_equity; RC-02 two wrong signs; RC-03 CONCERN; SC-01/02/04 worse than SC-03.
- **Source:** evidence. **Citation gap?** NO.
- **If Roger asks** "What didn't work?" → this list, with numbers (also `reports/final/what_didnt_work.md`).

### D-067 Hand-off to calibration (ADOPTED)
- **Evidence:** obs/pred pooled 1.34 (GB-01), 1.36 (RC-03); by year 1.21/1.36/1.43 and
  1.22/1.39/1.44; slopes 0.976, 1.024; new-firm slopes 0.88 / 0.89; GB-01 log loss 0.06415 → 0.06343
  with the oracle level, 0.06370 with L1.
- **Source:** evidence. **Citation gap?** NO.
- **If Roger asks** "How much does the level matter?" → "A perfect per-year level would cut GB-01's
  log loss from 0.06415 to 0.06343: real, but smaller than the gap between models (D-074)."

### D-068 (merged with D-011 above)

### D-069 Faster paired bootstrap proposed for shared code (OPEN)
- **Decided:** propose replacing per-firm index lists with multiplicity weights; identical results
  (max |diff| 1e-16), fast enough for 1,000 draws.
- **Source:** n/a. **Citation gap?** NO.
- **If Roger asks** → "Same estimator, computed faster; checked on a 15,000-firm subsample."

### D-070 What a bank knows when it sets the level (ADOPTED)
- **Decided:** under start 6, fs_year t's label is complete on 30 June t+2, when decisions on t+1
  statements begin; so a partly observed next-year rate is **not** available. All level parameters
  use training rows or GB-01's in-window slice B only.
- **Economics:** you cannot set today's price with defaults that have not happened yet.
- **Evidence:** `reports/calibration/ct_targets.csv`, `monthly_hazard.csv`.
- **Source:** D-002 (R-28, R-29 VERIFIED current text); decision timing is an assumption.
- **Citation gap?** YES (partial): same as D-002.
- **If Roger asks** "Did your level use anything from the test year?" → "No; only defaults observed
  by the time a bank would be pricing that year's applications."

### D-071 Level rules: recency wins, long-run average loses, trend is fragile (ADOPTED finding)
- **Decided:** "follow the latest observed level" beats "training-window average" in 2008-2010; trend
  extrapolation ties in the rise but fails at the 2011 turn.
- **Economics:** **PIT (point-in-time)** PDs follow current conditions (what pricing needs);
  **TTC (through-the-cycle)** PDs average over the cycle (what stable underwriting limits need)
  (R-16 Sec. II p. 11).
- **Evidence (`reports/calibration/methods_by_year.csv`, GB-01, Δ log loss vs submitted, ×1e-4):**
  e last-6-months −6.31 [−7.09, −5.60]; c2 half-trend −6.28; c trend −6.16; b last year −4.50;
  f mean-last-2 −1.57; a pooled 0; d TTC proxy +0.37. Pooled log loss 0.06415 → 0.06352 (rule e);
  obs/pred 1.21/1.36/1.43 → 1.02/1.05/1.18. 2011 pseudo-test level-only extra log loss (×1e-4): a 0.33,
  d 0.48, f 0.48, e 0.74, b 0.80, c2 3.85, c 9.42 (`reports/calibration/q6_2011_pseudo_test.csv`).
- **Source:** R-16 Sec. II p. 11 (VERIFIED).
- **Citation gap?** NO. **Caveat:** rule e was chosen among 7 after seeing 2008-2010 (selection
  overfitting, section 5).
- **If Roger asks** "PIT or TTC?" → "Pricing needs PIT (R-16 p. 11); in 2008-2010 recency won by
  6.3e-4 log loss. At the 2011 turn, trend rules failed badly, so we did not ship a trend."

### D-072 Level method: prior shift (one logit intercept shift) (ADOPTED; others REJECTED)
- **Decided:** PD' = expit(logit(PD) + δ), δ = logit(τ) − logit(r_w) (τ = target level, r_w =
  training default rate). One frozen number.
- **Economics:** moves every firm's PD by the same odds factor: ranking untouched, average moved.
- **ML/stats:** this is the **prior-shift** correction: if only the base rate changes and the
  profile of defaulters/survivors stays the same, posteriors change by the ratio of prior odds
  (R-22 eq. 4). That assumption is doubtful over 2008-12, so we use it as a level tool, not a claim.
  Monotone recalibration cannot add power (R-07 pp. 89-91).
- **Evidence:** M1 0.06352 (−6.31e-4), slope 0.98, within-year |dAUC| = 0; M2 −6.16e-4; M4 isotonic
  −4.82e-4, slope 0.92; Saerens EM estimated 2.18% / 1.79% / 0.45% vs observed 1.28 / 1.62 / 1.70% and
  needs the holdout batch (`reports/calibration/methods_by_year.csv`, `saerens_em_diagnostic.csv`).
- **Source:** R-22 Sec. 2.2 eq. (4) pp. 5-6; R-07 pp. 88-91 (both VERIFIED); Platt/isotonic named only.
- **Citation gap?** NO for M1 (R-20/R-21 only if named in detail).
- **If Roger asks** "Why not isotonic?" → "The miss is level, not shape: isotonic was worse
  (−4.82e-4 vs −6.31e-4) and flattened the slope to 0.92."
- MF §6.2 and SG §2.8 have runnable prior-shift examples.

### D-073 Holdout level τ = 1.49% (ADOPTED by specialist; human sign-off requested)
- **Decided:** τ = 1.493% = expected rate two years after the fs2011 anchor (1.431%) under a
  driftless logit random walk with σ = 0.21/yr; δ = +0.059 (fit 2007-2010, r_w 1.409%) or +0.079
  (fit incl. fs2011, r_w 1.381%) (`experiments/calibration/harness_level_params.json`).
- **Economics:** the holdout year is unknown; under uncertainty the expected rate minimises expected
  log loss, and under-prediction costs more than over-prediction (D-074), so a slight upward hedge.
- **Alternatives (`reports/calibration/final_level_candidates.csv`):** pooled 1.41%; last complete
  year 1.70%; fs2011 1.43%; latest half-year 1.38%; trend 2.21% / 1.90%; minimax regret 1.49%.
- **Evidence:** hedge cost if truth is 1.43%: 0.17e-4; saving if truth is 1.70%: 1.1e-4
  (`reports/calibration/scenario_regret.csv`); realised-rate p10-p90 at 2 years 0.98%-2.08%
  (`recommended_level_random_walk.csv`).
- **Source:** R-16 p. 11; R-22 eq. (4) (VERIFIED). Random walk and σ: **assumption**.
- **Citation gap?** YES: the hedge model has no citation possible; keep it labelled as a judgment.
- **If Roger asks** "Where does 1.49% come from?" → "Latest estimated rate 1.43%, nudged up because
  the holdout is about two years ahead and under-pricing costs more than over-pricing. It is a
  judgment; the alternatives and their costs are in `scenario_regret.csv`."

### D-074 Asymmetry and scenario costs (ADOPTED finding; H4 NOT confirmed)
- **Evidence (`reports/calibration/asymmetry_gb01.csv`, ×1e-4):** 30% too low 8.17 / 10.35 / 10.93;
  30% too high 5.57 / 7.02 / 7.40; scenarios for ~1.5%: truth 1.28% 1.7, 1.43% 0.2, 1.70% 1.3, 2.13%
  11.8. Best leak-free level gain 6.3e-4 vs algorithm gains 16e-4 (GB-01 vs RC-02) and 24e-4 (vs SC-03).
- **Economics (assumption):** under-predicted PDs price loans below expected loss and attract
  adversely selected borrowers; over-prediction loses volume.
- **Source:** R-08 Sec. 6 (adverse selection, argued for model power; VERIFIED) by analogy.
- **Citation gap?** YES (partial): the pricing reading of asymmetry is an assumption.
- **If Roger asks** "Does the level matter more than the model (H4)?" → "No, not in our walk-forward:
  the algorithm gap is 2.5-4 times the best level gain. But a 30% level miss costs half the GB-01 vs
  RC-02 gap, so it must be set carefully."

### D-075 PD floor 0.03%, cap 99% (ADOPTED); 50% cap (REJECTED)
- **Evidence (`reports/calibration/floor_cap_effect.csv`):** 31 PDs < 0.03%, 0 defaults, log loss
  unchanged; 855 PDs > 50% with 553 defaults (65%): a 50% cap raises log loss 0.063524 → 0.063621.
- **Source:** R-15 para. 285 p. 67 (VERIFIED: Basel corporate PD floor 0.03%), **by analogy only**;
  we do not claim the bank is an IRB bank.
- **Citation gap?** NO.
- **If Roger asks** "Why a floor?" → "No firm is riskless; 0.03% is the Basel corporate floor, used by
  analogy. In our data it never changed a result."

### D-076 Label-window sensitivity: ship start 6, switchable (ADOPTED; OPEN pending Roger)
- **Evidence (`reports/calibration/q5_label_window_levels.csv`, `q5_label_window_cost_matrix.csv`,
  `q5_2011_other_labels_fold4.csv`):** τ s0 1.37%, s3 1.39%, s6 1.49%; cost of shipping s6 if truth
  is s0: 0.57e-4; fold-4 GB-01 AUC on 2011 0.885 (s0), 0.869 (s3), 0.850 (s6, 11-month).
- **Source:** evidence. **Citation gap?** NO.
- **If Roger asks** "What if the holdout uses a different window?" → "We change one number (τ); the
  ranking is the same. The cost of being wrong is small at the end of the data."

### D-077 Include fs2011 in the final fit with an 11-month label (ADOPTED recommendation)
- **Evidence (`reports/calibration/q6_partial_cohort_test.csv`, fold-3 analogue):** GB-01 full
  0.8307 / 0.07051; newest cohort cut to 11 months 0.8301 / 0.07063; dropped 0.8263 / 0.07086.
- **Economics:** the newest behaviour is the most relevant for the next loans.
- **ML/stats:** a slightly short label beats no data for the newest year; the level is set separately.
- **Source:** evidence. **Citation gap?** NO.
- **If Roger asks** "2011's window isn't complete. Why train on it?" → "In the 2010 analogue, an
  11-month newest cohort lost almost nothing vs a full label, while dropping it lost 0.004 AUC."
- **Note:** this interacts with D-087 (2011 is first used once as a lockbox, then added to the fit).

### D-078 Grade table and group calibration (ADOPTED reporting; group corrections REJECTED)
- **Evidence (`reports/calibration/grade_table_2010.csv`):** default rate rises across all 11 grades;
  grades 1-3 hold 42.5% of firms and 5.9% of defaults; 9-11 hold 3.8% and 33.8%; grades 4-9 run
  20-40% above their PD in 2010. Group calibration (`subgroup_calibration.csv`): new firms obs/pred
  1.06, slope 0.89; size slopes 0.88 to 1.10.
- **Source:** R-16 pp. 33-34 (binomial bands understate error with correlated defaults; VERIFIED).
- **Citation gap?** NO.
- **If Roger asks** "What would a loan officer use?" → "The 11-grade table: each grade's PD band,
  share of firms and observed default rate."

### D-079 Calibration hand-off; harness chain (ADOPTED)
- **Decided:** harness = GB-01 final fit (2007-2011) → own calibrator → prior shift δ → clip
  [0.0003, 0.99]. Open: hedge sign-off; label window; holdout period; new-firm over-dispersion.
- **Source:** n/a. **Citation gap?** NO.

### D-080 EBM/GAM challenger reinstated (ADOPTED, supersedes D-008)
- **Decided:** run an **EBM (Explainable Boosting Machine)**: a GAM (generalised additive model, PD's
  log-odds = sum of one curve per feature) whose curves are learned by boosting.
- **Economics:** every driver is one readable curve.
- **Source:** R-01 p. 14 fn. 11 ("closely related to a GAM", VERIFIED); EBM papers R-24, R-25 TO-VERIFY.
- **Citation gap?** YES: needs literature-scout for R-24/R-25. (Also: the entry is dated 2026-10-30 but
  cites D-072 from 10-31; the date is probably wrong.)
- **If Roger asks** "Why an EBM?" → "It is the modern version of the additive structure in RiskCalc;
  it tells us what full readability costs in power."

### D-081 GAM-01: EBM main effects with monotone constraints (ADOPTED as glass-box entry)
- **Evidence (`experiments/gam/GAM-01/metrics.md`, `experiments/gam/paired_bootstrap.csv`):** pooled
  AUC 0.8254, by year 0.8348 / 0.8233 / 0.8177; log loss 0.06517; new 0.774 vs seen 0.832; +CAL log
  loss 0.06451; vs GB-01 dAUC −0.0120 [−0.0140, −0.0104], d log loss +0.00102.
- **Source:** R-01 fn. 11; method [UNVERIFIED].
- **Citation gap?** YES (as D-080).
- **If Roger asks** "What does readability cost?" → "About 0.012 AUC vs GB-01; only part of it is the
  sign constraints, the rest is the lack of interactions."

### D-082 Reset sparse missing-bin scores to zero (ADOPTED)
- **Evidence:** asset_turnover missing (19 rows, 0 defaults) scored −5.10 log-odds, cash_ratio missing
  (15 rows) −0.90 — a messy-input row would get a PD ~150× too low; rule: reset bins with < 30
  defaults; pooled AUC 0.82504 → 0.82545.
- **ML/stats:** a tiny cell with zero defaults is learned as "riskless": classic small-sample overfit.
- **Source:** assumption. **Citation gap?** NO (design rule).
- **If Roger asks** "What happens to a holdout row with a value missing that was never missing in
  training?" → "It gets the average-firm contribution for that feature, not an extreme one."

### D-083 GAM-02 (+10 interaction pairs) not adopted (REJECTED)
- **Evidence:** AUC 0.8262 (+0.0008 [+0.0004, +0.0012]), log loss −0.00008; all 10 pairs involve
  only unconstrained features (largest: asset_turnover × ateco, 0.046).
- **Source:** GA2M idea R-24 [UNVERIFIED]. **Citation gap?** YES (as D-080).
- **If Roger asks** → "Statistically detectable, economically small: 10 more heat maps for 0.0008 AUC."

### D-084 Face validity of EBM shapes (ADOPTED finding)
- **Evidence (`experiments/gam/GAM-01/explain.md`):** all 20 signed features monotone as expected;
  asset_turnover rises then dips; extraord_to_assets V-shaped; rev_nonpositive = 1 **lowers** the
  score (−0.19) — wording must be fixed before use; macro_area OTHER +0.65, legal_group OTHER −0.87
  (thin pooled levels; unseen levels map there); negative-equity jump +0.8.
- **Source:** none needed. **Citation gap?** NO.
- **If Roger asks** "Any odd shapes?" → "Yes: 'no revenue' lowers the score, because it overlaps
  with zero asset turnover and industry; read together they offset. We would not show that reason
  code to an officer."

### D-085 Reusable EBM estimation and cost (ADOPTED)
- **Evidence:** prediction 0.8-0.9 s per 100,000 rows; complexity O(n × (F + T)) table lookups.
- **Citation gap?** NO.

### D-086 Ensemble of GB-01 and the EBM rejected (REJECTED)
- **Evidence (DL D-086; 100 draws):** 50/50 AUC 0.8343 (dAUC −0.0032 [−0.0042, −0.0024]); 70/30
  0.8363; three-way with RC-02 0.8322; worse in every year.
- **ML/stats:** averaging helps when errors differ; the EBM is a restricted version of the same
  information, so it only dilutes GB-01.
- **Source:** log says "assumption". R-10 Sec. 4.4 p. 22 (VERIFIED: simple averaging of RiskCalc and
  PFM outputs did not add stable power) is close support.
- **Citation gap?** NO (fixed 2026-11-02: R-10 Sec. 4.4 p. 22 added to the log). Script and output:
  `experiments/ensemble/ens01.py`, `experiments/ensemble/ens01_output.txt`.
- **If Roger asks** "Why not blend models?" → "We tried; every blend was worse in every year."

### D-087 fs_year 2011 is a one-time lockbox (ADOPTED)
- **Decided:** from round 2 on, no experiment reads fs_year ≥ 2011 rows for selection; finalists are
  scored once on 2011 (11-month label) before the final refit that includes 2011.
- **ML/stats:** see section 5.3.
- **Source:** assumption; R-33 Sec. 4.8 p. 23 (walk-forward, VERIFIED). Add R-07 Sec. 3.2 p. 95.
- **Citation gap?** NO (fixed 2026-11-02: R-07 Sec. 3.2 p. 95 added; the log's wording corrected).
- **Honesty point you must know:** the original log text said 2011 was used only for a level
  statistic. That was incomplete (now corrected in D-087): `reports/calibration/q5_2011_other_labels_fold4.csv` and
  `q6_2011_pseudo_test.csv` already contain GB-01 and RC-02 **AUCs on 2011** (GB-01 0.8497 with the
  11-month label, RC-02 0.8345) and the 2011 test of the seven level rules. No model was *chosen*
  with those numbers, so the lockbox is still clean for round-2 model choice, but say "2011 was
  looked at once for two round-1 models and for the level rules".
- **If Roger asks** "How do you know round 2 didn't overfit your three test years?" → "We sealed
  2011 and score the finalists on it once."

---

## 3. Feature ledger (all 26 features in `docs/FEATURE_SPEC.md`)

### 3.0 How every numeric feature is built (the same seven steps)

Code: `src/features/build.py`. The order matters; each step is a defence against a specific problem.

1. **Coerce** (`coerce_input`): accept either column spelling (D-010), turn every NA token and
   non-finite number into NaN, keep codes as strings. *Defends against:* a crash on messy holdout rows.
2. **Identity rebuild** (`_reconstruct`, row by row): if a raw field is missing but an exact
   accounting identity gives it, fill it. Identities used: debt_st = asst_current − wc_net (and the
   two rearrangements); eqty_tot = margin_fin + fixed assets; total assets ≤ 0 or missing → sum of
   fixed + current assets; prof_operations = rev_operating − COGS, or roa × assets / 100;
   profit = cf_operations − ebitda + prof_operations, or roe × equity / 100; ebitda and cf_operations
   from the same cash-flow identity. *Defends against:* throwing away a row's information.
3. **Safe division** (`_raw_features`): divide by total assets where possible (always > 0 after
   step 2); for other denominators a written rule says what zero or negative means (+∞, 0 or NaN).
   *Defends against:* division by zero and economically wrong values.
4. **Winsorise** (clip) to the 1st and 99th percentiles of the **training window** (stored in the
   parameter file; ±∞ lands on the bound). *Defends against:* one extreme firm dominating a fit.
5. **Signed log** for five ratios with unbounded denominators: slog(x) = sign(x)·ln(1 + |x|),
   applied after clipping. It keeps order (so AUC and tree splits are unchanged) and tames tails
   for linear models.
6. **Flags and categoricals**: neg_equity, debt_split_missing, rev_nonpositive; ATECO section,
   legal form, macro-area, with levels that have < 30 training defaults pooled into OTHER.
7. **Imputation** (only in the `impute=True` view used by logit/probit-type models): the debt-split
   block is rebuilt as training sector-median share × the firm's own total; any remaining NaN takes
   the training sector median. Trees and the EBM get the NaN instead. *Defends against:* GLMs
   failing on NaN, while letting trees learn what "missing" means.

Nothing in `transform` reads labels or any row other than the current one, and every parameter
(bounds, medians, shares, category levels) comes from `fit` on the training window. That is what
makes it legal for the harness. MF §4.5 has the same rules in plain words.

**Where the "evidence" numbers come from.** Univariate AUC, decile default rates (D1 = lowest 10%
of the feature, D10 = highest) and winsor bounds: `reports/features/univariate.md` §1-2 (fold-3
training rows, fs_year 2007-2009, 475,861 rows, 6,195 defaults). AUC < 0.5 means the PD falls as the
feature rises. Multivariate sign: core logit (UNI §8, in-sample, standardised coefficients) for core
features; for every signed feature, the share of the **unconstrained** GBM's (GB-02) partial
dependence that moves against the expected sign (SC1 T9; below 0.5 = the data agree conditionally).

**Model codes.** SC-01 roa_net only; SC-02 / SC-03 / GB-03 = 13 core; SC-04 = wc_ratio, equity_ratio,
roa_oper, asset_turnover; RC-01 = 10 core numeric incl. neg_equity + ateco_group; RC-02 = RC-01 +
bank_debt_ratio + ap_days; RC-03 = RC-02 − neg_equity − bank_debt_ratio; GB-01/02/04/05/06,
GAM-01/02 = all 26.

**The example firm "Alfa Srl"** (made up, all euros): total assets 10.0M (tangible fixed 3.0M,
intangible fixed 0.4M, financial fixed 0.6M, current 6.0M of which receivables 2.5M and cash 0.5M);
equity 2.0M; short-term debt 5.0M, long-term debt 3.0M; bank debt 2.0M short + 1.5M long; trade
payables due within a year 1.8M; net working capital 1.0M; operating revenue 12.0M; COGS 11.4M;
operating profit 0.6M; EBITDA 1.2M; interest expense 0.3M; net profit 0.2M; accounting cash flow
0.8M; extraordinary items −0.05M; ATECO division 25 (section C, manufacturing); legal form SRL;
HQ_city 15 (Milano → North-West, hypothesis D-012). Its values are computed below.

**Raw fields used** (file names; meanings from MF §1.2-1.3): `asst_tot` total assets;
`eqty_tot` book equity (assets − liabilities); `debt_st` / `debt_lt` liabilities due within / after
12 months; `debt_bank_st/lt` bank debt; `AP_st` trade payables due within 12 months; `asst_current`
current assets; `wc_net` net working capital (current assets − short-term liabilities); `cash_and_equiv`
cash; `AR` trade receivables; `rev_operating` operating revenue (sales); `COGS` operating costs
(the identity prof_operations = rev − COGS holds, so it behaves like **all** operating costs, not just
cost of goods); `prof_operations` operating profit; `ebitda` operating profit before depreciation;
`exp_financing` interest expense; `profit` net profit; `cf_operations` net profit + depreciation
(an accounting cash flow, not a cash-flow statement); `inc_extraord` extraordinary items;
`asst_tang_fixed` / `asst_intang_fixed` tangible / intangible fixed assets.

---

### F-01 `log_assets` — size (core, expected −)
- **Formula:** ln(asst_tot). **Units:** natural log of euros (16.1 ≈ €10M; +0.69 = double the size).
- **Construction:** step 2 (missing/≤ 0 total assets rebuilt from components) → ln → clip to
  [14.23, 18.44] (≈ €1.5M to €102M). No slog.
- **Economic story:** larger firms are more diversified and have more funding options. Size enters
  once, here; every other feature is a ratio.
- **Sign vs data:** univariate AUC 0.477 (agrees, weak), D1 1.57% → D10 1.08%. Multivariate: core logit
  +0.013 (disagrees, ≈ 0); SC-02 positive in folds 1, 3 (not significant, D-032); GB-02 19% against
  (agrees, mostly). Constrained − in GB-01/EBM, where its importance is tiny (EBM 0.016, D-084).
- **Models:** SC-02, SC-03, RC-01/02/03, GB-all, GAM.
- **Source:** size family R-01 Sec. 3.1 p. 13 (VERIFIED); RiskCalc uses CPI-deflated assets (R-05
  p. 61) and France "real total assets" (R-33 Table 3 p. 11), VERIFIED; we use nominal (CPI is
  external data). **Citation gap:** NO.
- **Example:** Alfa: ln(10,000,000) = **16.118**.

### F-02 `equity_ratio` — leverage (core, expected −)
- **Formula:** eqty_tot / asst_tot. **Units:** share of assets (0.20 = 20%). Can be negative.
- **Construction:** identity rebuild of equity (margin_fin + fixed assets) → divide by assets → clip
  [−0.0809, 0.9806]. Negative equity stays negative (that is why it is preferred to debt/equity,
  which flips sign and explodes when equity → 0).
- **Economic story:** the equity cushion: the share of assets that can be lost before creditors lose
  anything.
- **Sign vs data:** AUC 0.267 (strong, agrees); D1 4.94% → D10 0.49% (D9 0.41%: flat at the top).
  Core logit −0.076 (agrees); GB-02 5% against (agrees). RC: bottom bin 16.0% default rate (D-041).
- **Models:** SC-02, SC-03, SC-04, RC-all, GB-all, GAM. 2nd largest GB-01 driver (mean |SHAP| 0.21).
- **Source:** leverage family R-01 p. 13; equity/assets is the leverage input of RiskCalc 4.0 France
  (R-33 Table 3 p. 11). VERIFIED. **Citation gap:** NO.
- **Example:** 2.0M / 10.0M = **0.20**. A firm with equity −0.5M → −0.05 (and neg_equity = 1).

### F-03 `st_debt_ratio` — leverage (core, expected +)
- **Formula:** debt_st / asst_tot. **Units:** share of assets.
- **Construction:** debt_st rebuilt as asst_current − wc_net if missing → divide → clip [0.0019, 1.0008].
- **Economic story:** everything that must be paid or refinanced inside the 12-month PD horizon
  (bank, other lenders, suppliers), relative to assets.
- **Sign vs data:** AUC 0.706 (agrees, strong); D1 0.45% → D10 4.33%, monotone. Core logit +0.360;
  GB-02 2% against. Full-set VIF 12.5 (redundant with wc_ratio, equity_ratio): fine for trees.
- **Models:** SC-02, SC-03, RC-all, GB-all, GAM.
- **Source:** leverage family R-01 p. 13 (VERIFIED); the "refinancing inside the horizon" story is an
  assumption. **Citation gap:** YES (partial; family covered, specific ratio not in a source).
- **Example:** 5.0M / 10.0M = **0.50**.

### F-04 `bank_debt_ratio` — leverage (full, expected +)
- **Formula:** (debt_bank_st + debt_bank_lt) / asst_tot. **Units:** share of assets.
- **Construction:** NaN if the debt-split block is missing (trees); imputed view: each component =
  training sector-median share of debt_st (or debt_lt) × the firm's own debt_st (debt_lt), and the
  flag `debt_split_missing` = 1 → clip [0, 0.9073].
- **Economic story:** the borrowing a bank can call or refuse to roll over; the lender-facing leverage.
- **Sign vs data:** AUC 0.579 (agrees); D1 0.98%, D2 0.75%, … D10 2.20% (not monotone at the bottom).
  **Conditionally it flips:** RC-02 coefficient negative in all folds (z −1.24, −3.17, −2.56); GB-02
  45% against (borderline). Explanation: collinearity inside leverage (D-041).
- **Models:** RC-02 (dropped in RC-03), GB-all, GAM.
- **Source:** assumption. **Citation gap:** YES (needs literature-scout or labelled assumption).
- **Example:** (2.0M + 1.5M) / 10.0M = **0.35**. If Alfa's split were missing and the sector median
  bank share of short-term debt were 0.4 (illustrative), bank_st would be imputed as 0.4 × 5.0M = 2.0M.

### F-05 `st_debt_share` — leverage / maturity (full, expected +)
- **Formula:** debt_st / (debt_st + debt_lt). **Units:** share of total debt (0-1).
- **Construction:** total debt ≤ 0 → NaN (no debt, or sign error) → clip [0.00475, 1].
- **Economic story:** maturity structure: more of the debt due within a year means more refinancing risk.
- **Sign vs data:** AUC 0.570 (agrees); D1 0.51% … D7 1.64%, D8 1.38% (non-monotone at the top; D10
  undefined because many firms have share = 1 exactly). GB-02 12% against. Flagged as a "decile
  break" feature (D-051); its EBM shape differs between folds (D-084).
- **Models:** GB-all, GAM.
- **Source:** assumption. **Citation gap:** YES.
- **Example:** 5.0M / 8.0M = **0.625**.

### F-06 `wc_ratio` — liquidity (full, expected −)
- **Formula:** wc_net / asst_tot (= (asst_current − debt_st) / asst_tot). **Units:** share of assets.
- **Construction:** wc_net rebuilt from the identity if missing → divide → clip [−0.8071, 0.9661].
- **Economic story:** what is left of current assets after paying all short-term liabilities.
- **Sign vs data:** AUC 0.384 (agrees); D1 3.15% → D8 0.75%, then D9 0.92%, D10 1.00% (flat/upturn at
  the top). **GB-02 83% against** (the main face-validity failure of the unconstrained model, D-051).
  Spearman 0.91 with current_ratio.
- **Models:** SC-04, GB-all, GAM.
- **Source:** Altman lineage (R-32, VERIFIED for the Z-score in general; the specific WC/TA line is
  used in D-034 but not recorded in the claim map). **Citation gap:** YES (small): literature-scout to
  record the R-32 page for WC/TA.
- **Example:** 1.0M / 10.0M = **0.10**.

### F-07 `current_ratio` — liquidity (full, expected −)
- **Formula:** slog(asst_current / debt_st). **Units:** signed log of a multiple (1.2× → 0.788).
- **Construction:** debt_st = 0 → +∞ → clipped to 109.58 → slog 4.71; debt_st < 0 → NaN; clip
  [0.0299, 109.58] on the raw scale, then slog.
- **Economic story:** the classic liquidity ratio: current assets per euro of short-term liabilities.
- **Sign vs data:** AUC 0.383 (agrees); D1 1.54% < D2 2.57% (the very lowest decile is safer than
  the next: a "decile break"), then falls to 0.72%. GB-02 16% against; with constraints dropped on it
  (GB-06) the model used it 87% against the sign once wc_ratio was constrained.
- **Models:** GB-all, GAM.
- **Source:** assumption (classic ratio). **Citation gap:** YES (low).
- **Example:** 6.0M / 5.0M = 1.2 → slog = ln(2.2) = **0.788**.

### F-08 `cash_ratio` — liquidity (core, expected −)
- **Formula:** cash_and_equiv / asst_tot. **Units:** share of assets.
- **Construction:** divide → clip [0, 0.4997].
- **Economic story:** money available right now for the next payments.
- **Sign vs data:** AUC 0.441 (agrees, modest); D1 1.55% → D10 0.83%. Core logit −0.035; GB-02 46%
  against (borderline: conditionally weak).
- **Models:** SC-02, SC-03, RC-all, GB-all, GAM.
- **Source:** liquidity family R-01 p. 13; RiskCalc 4.0 France uses cash and marketable
  securities/assets (R-33 Table 3 p. 11). VERIFIED. **Citation gap:** NO.
- **Example:** 0.5M / 10.0M = **0.05**.

### F-09 `roa_net` — profitability (core, expected −)
- **Formula:** profit / asst_tot. **Units:** share of assets per year (0.02 = 2% return).
- **Construction:** profit rebuilt from the cash-flow identity or roe × equity if missing → divide →
  clip [−0.2319, 0.2456].
- **Economic story:** net return after interest, extraordinary items and tax; losses eat equity.
- **Sign vs data:** AUC 0.256 (2nd strongest); D1 5.90% → D9 0.38%, D10 0.53% (slight upturn at the
  top). Core logit −0.541 (largest core coefficient); GB-02 10% against.
- **Models:** SC-01 (the single best core ratio in every fold), SC-02, SC-03, RC-all, GB-all, GAM.
- **Source:** profitability family R-01 p. 13 (VERIFIED). **Citation gap:** NO.
- **Example:** 0.2M / 10.0M = **0.02**.

### F-10 `roa_oper` — profitability (full, expected −)
- **Formula:** prof_operations / asst_tot (= the file's `roa` / 100). **Units:** share of assets.
- **Construction:** operating profit rebuilt as rev − COGS or roa × assets / 100 → divide → clip
  [−0.2123, 0.3171].
- **Economic story:** operating return before financing and extraordinary items.
- **Sign vs data:** AUC 0.283; D1 5.61% → D10 0.51%. GB-02 29% against. Spearman 0.76 with roa_net.
- **Models:** SC-04, GB-all, GAM.
- **Source:** profitability family R-01 p. 13 (VERIFIED). **Citation gap:** NO.
- **Example:** 0.6M / 10.0M = **0.06** (file `roa` would show 6.0).

### F-11 `ebitda_margin` — profitability (full, expected −)
- **Formula:** slog(ebitda / rev_operating). **Units:** signed log of a margin.
- **Construction:** revenue ≤ 0 → NaN (see rev_nonpositive) → clip [−9.562, 0.9169] → slog.
- **Economic story:** operating cash margin per euro of sales.
- **Sign vs data:** AUC 0.279 (0.291 imputed; missing on 3.3%); D1 5.08% → D10 0.47%. GB-02 28% against.
- **Models:** GB-all, GAM.
- **Source:** profitability family R-01 p. 13; RiskCalc 4.0 France replaced EBITDA/sales with
  operating margin (R-33 Sec. 1.2 p. 4), VERIFIED. **Citation gap:** NO.
- **Example:** 1.2M / 12.0M = 0.10 → slog = ln(1.10) = **0.0953**.

### F-12 `interest_coverage` — debt coverage (core, expected −)
- **Formula:** slog(ebitda / exp_financing). **Units:** signed log of "times covered".
- **Construction:** interest ≤ 0: +∞ if EBITDA > 0 (→ clipped to 5,782 → slog 8.66), 0 if EBITDA ≤ 0;
  negative EBITDA with positive interest gives a negative ratio (worst); clip [−364.2, 5,782] → slog.
- **Economic story:** how many times operating cash earnings cover the interest bill: the direct
  measure of debt-service capacity.
- **Sign vs data:** AUC 0.276; D1 4.86% → D9 0.45%, D10 0.56%. Core logit −0.161; **GB-02 57%
  against** (fails face validity unconstrained, D-051).
- **Models:** SC-02, SC-03, RC-all, GB-all, GAM.
- **Source:** debt-coverage family R-01 p. 13 (VERIFIED). **Citation gap:** NO.
- **Example:** 1.2M / 0.3M = 4.0 → slog = ln(5) = **1.609**. "Beta Spa" with no interest and EBITDA
  0.5M → +∞ → 5,782 → **8.663**. "Gamma Srl" with EBITDA −0.2M and interest 0.05M → −4 → **−1.609**.

### F-13 `cfo_to_debt` — debt coverage (full, expected −)
- **Formula:** cf_operations / (debt_st + debt_lt). **Units:** share of debt per year.
- **Construction:** debt = 0: +∞ if cash flow > 0 else 0; debt < 0 → NaN; clip [−0.3947, 2.372].
- **Economic story:** internal cash flow per euro of debt; its inverse is "years to repay from
  internal funds".
- **Sign vs data:** **best univariate feature** (AUC 0.248; D1 5.75% → D9 0.29%). But it flips to +
  next to roa_net (Spearman 0.80) in a logit (inner variant B, +0.231), so kept out of the core.
  GB-02 14% against.
- **Models:** GB-all, GAM.
- **Source:** debt-coverage family R-01 p. 13; RiskCalc 4.0 France uses (profit + D&A)/(liabilities −
  cash and securities), close to this (R-33 Table 3 p. 11). VERIFIED. **Citation gap:** NO.
- **Example:** 0.8M / 8.0M = **0.10** (ten years to repay).

### F-14 `interest_to_assets` — debt coverage / burden (core, expected +)
- **Formula:** exp_financing / asst_tot. **Units:** share of assets per year.
- **Construction:** divide → clip [0, 0.0649].
- **Economic story:** interest burden = cost of debt × leverage; also the way the raw euro amount
  exp_financing is used (Roger M9).
- **Sign vs data:** AUC 0.634; D1 0.99%, D3-D4 0.75%, D10 3.54% (flat then steep). Core logit +0.232;
  GB-02 5% against.
- **Models:** SC-02, SC-03, RC-all, GB-all, GAM.
- **Source:** RiskCalc 4.0 France uses interest expense / **sales** (R-33 Sec. 1.2 p. 4, Table 3), VERIFIED;
  the per-assets version is ours. **Citation gap:** NO (cite as "related to").
- **Example:** 0.3M / 10.0M = **0.03**.

### F-15 `asset_turnover` — activity (full, expected none)
- **Formula:** rev_operating / asst_tot. **Units:** sales per euro of assets per year.
- **Construction:** divide → clip [0, 3.769]. No monotone constraint.
- **Economic story:** a priori − (idle assets), but high turnover also marks thin-margin traders, and
  near-zero turnover marks holding/real-estate firms, which were safer here.
- **Data:** AUC 0.504; deciles 1.48%, 0.72%, 1.00%, 1.46%, 1.71%, 1.70%, … 1.22% (inverted U, D-022).
  GBM rises with PD over its range (D-058); EBM rises to a peak at 0.5-1 then dips (D-084).
- **Models:** SC-04, GB-all, GAM (and in all 10 GAM-02 pairs' neighbourhood, D-083).
- **Source:** activity family R-01 p. 13 (VERIFIED); the inverted-U story is an assumption.
  **Citation gap:** YES (partial).
- **Example:** 12.0M / 10.0M = **1.2**.

### F-16 `ar_days` — activity (core, expected +)
- **Formula:** slog(365 × AR / rev_operating). **Units:** signed log of days.
- **Construction:** revenue = 0: +∞ if AR > 0 (→ 59,982 days cap) else 0; revenue < 0 → NaN; clip
  [2.96, 59,982] → slog. Replaces the file's `days_rec` (72% missing, inconsistent).
- **Economic story:** days of sales outstanding: slow collection ties up cash and can signal weak
  customers or overstated receivables.
- **Sign vs data:** AUC 0.583; D1 0.87% → D8 1.90%, then flat. Core logit +0.137; GB-02 13% against.
  Missing (1,109 rows) has a 6.6% default rate: missing marks "almost no revenue" (D-042, D-082).
- **Models:** SC-02, SC-03, RC-all, GB-all, GAM.
- **Source:** activity family R-01 p. 13 (VERIFIED); specific ratio: assumption. **Citation gap:** YES (low).
- **Example:** 365 × 2.5M / 12.0M = 76.0 days → slog = ln(77.04) = **4.344**. Reason-code wording for
  capped firms must say "almost no revenue", not "60,000 days" (D-058).

### F-17 `ap_days` — activity (full, expected +)
- **Formula:** slog(365 × AP_st / COGS). **Units:** signed log of days.
- **Construction:** COGS ≤ 0 → NaN; NaN if the debt-split block is missing (trees); imputed view via
  share × own total (step 7) → clip [0, 4,884] → slog.
- **Economic story:** days payable outstanding: stretching suppliers is a classic symptom of
  liquidity stress.
- **Sign vs data:** AUC 0.591; D1 0.98% → D9 2.09%. GB-02 29% against; RC-02/03 strong + coefficient.
  On block-missing rows: AUC 0.612 with share × total vs 0.538 with sector median (D-021).
- **Models:** RC-02, RC-03, GB-all, GAM (5th GB-01 driver, 0.15).
- **Source:** RiskCalc 4.0 France uses accounts payable / sales (R-33 Table 3 p. 11), VERIFIED.
  **Citation gap:** NO (cite as "related to").
- **Example:** 365 × 1.8M / 11.4M = 57.6 days → slog = ln(58.6) = **4.071**.

### F-18 `tangible_ratio` — asset structure (core, expected −)
- **Formula:** asst_tang_fixed / asst_tot. **Units:** share of assets.
- **Construction:** divide → clip [0, 0.9876].
- **Economic story:** buildings and machines can be pledged and keep value in distress.
- **Sign vs data:** AUC 0.384; D1 1.75% < D2 2.18% (decile break), then down to D10 0.37%. Core logit
  −0.320; GB-02 5% against; holds within every ATECO group with ≥ 30 defaults (within-group AUC
  0.56-0.73, SC1 T16). **#1 GB-01 driver** (mean |SHAP| 0.41), partly an attribution artefact (D-064).
- **Models:** SC-02, SC-03, RC-all, GB-all, GAM.
- **Source:** assumption (collateral argument). **Citation gap:** YES: it is our top driver and has
  no source. Highest-priority item for literature-scout.
- **Example:** 3.0M / 10.0M = **0.30**.

### F-19 `intangible_ratio` — asset structure (full, expected +)
- **Formula:** asst_intang_fixed / asst_tot. **Units:** share of assets.
- **Construction:** divide → clip [0, 0.4504].
- **Economic story:** capitalised costs and brands lose most of their value in distress.
- **Sign vs data:** AUC 0.550; D2 1.24% … D5 0.93% … D10 2.27% (D1 undefined: many zeros). GB-02 37%
  against (conditionally weak); EBM shape unstable across folds (D-084).
- **Models:** GB-all, GAM.
- **Source:** assumption. **Citation gap:** YES.
- **Example:** 0.4M / 10.0M = **0.04**.

### F-20 `extraord_to_assets` — profitability (full, expected none)
- **Formula:** inc_extraord / asst_tot. **Units:** share of assets.
- **Construction:** divide → clip [−0.0466, 0.1092]. No monotone constraint.
- **Economic story:** two-sided: big write-downs and big one-off gains (asset sales under pressure)
  can both precede default.
- **Data:** AUC 0.462; U-shape D1 2.80%, middle ≈ 0.9-1.1%, D10 2.01%; EBM V-shape around 0 (D-084).
- **Models:** GB-all, GAM.
- **Source:** assumption. **Citation gap:** YES (low; a stated hypothesis consistent with the data).
- **Example:** −0.05M / 10.0M = **−0.005**.

### F-21 `neg_equity` — leverage flag (core, expected +)
- **Formula:** 1 if eqty_tot < 0, else 0 (NaN equity → 0 after the rebuild attempt).
- **Economic story:** liabilities exceed assets: creditors are already under-covered on paper. Adds
  a jump that the clipped equity_ratio (lower bound −0.081) cannot show.
- **Sign vs data:** AUC 0.624; 3.78% of rows; default rate 9.74% vs 0.97% (UNI §3). Core logit +0.159.
  **Flips in RiskCalc** (T_neg_equity < 0 in all folds; redundant with T_equity_ratio, D-041). EBM jump +0.8.
- **Models:** SC-02, SC-03, RC-01, RC-02 (dropped in RC-03), GB-all, GAM.
- **Source:** assumption. Related data fact: `margin_fin`/`roe` blank exactly when equity ≤ 0 (D-016).
  **Citation gap:** YES (low).
- **Example:** Alfa: equity 2.0M ≥ 0 → **0**.

### F-22 `debt_split_missing` — missing indicator (full, expected +)
- **Formula:** 1 if any of debt_bank_st, debt_bank_lt, AP_st is missing.
- **Economic story:** the split is missing as a block, mostly for small firms (abbreviated accounts
  are a plausible cause, R-27); weakly riskier.
- **Sign vs data:** AUC 0.505 (none); 15.67% of rows; default rate 1.39% vs 1.29%. GB-01 constrained +.
- **Models:** GB-all, GAM.
- **Source:** keeping a flag next to an imputed value follows Moody's statement in R-34 (VERIFIED,
  vendor assertion). **Citation gap:** NO.
- **Example:** Alfa reports the split → **0**.

### F-23 `rev_nonpositive` — activity flag (full, expected none)
- **Formula:** 1 if rev_operating ≤ 0.
- **Economic story:** no revenue: holding, start-up or wind-down firms; marks where revenue ratios
  are undefined. Direction mixed.
- **Data:** AUC 0.507; 3.28% of rows; default rate 1.86% vs 1.28%. In the EBM it **lowers** the score
  (−0.19), offsetting asset_turnover = 0 (D-084): the reason-code wording must be fixed.
- **Models:** GB-all, GAM.
- **Source:** assumption. **Citation gap:** YES (low).
- **Example:** Alfa: revenue 12.0M → **0**.

### F-24 `ateco_group` — industry (core, categorical)
- **Formula:** 2-digit ATECO 2007 division → section A-U; sections with < 30 training defaults →
  OTHER; missing/unseen → OTHER. Reference level = largest (C in fold 3).
- **Economic story:** industries differ in risk and in what a normal ratio is (R-01 Sec. 3.3).
- **Data (fold-3 window, UNI §5):** C 1.61%, F 1.29%, G 1.46%, L 0.83%, N 1.78%, OTHER 0.90%.
  Conditional on ratios, manufacturing is the riskiest large sector (RC fold-3: L −0.287, F −0.215, D-045).
- **Models:** SC-02, SC-03, RC-all (the only categorical in RC), GB-all, GAM.
- **Source:** R-01 Sec. 3.1 p. 14 and Sec. 3.3 (VERIFIED); section ranges [UNVERIFIED].
  **Citation gap:** YES (code table: literature-scout).
- **Example:** Alfa: division 25 → section C → **C** (reference level, contribution 0 in a logit).

### F-25 `legal_group` — firm type (core, categorical)
- **Formula:** legal_struct; forms with < 30 training defaults → OTHER.
- **Economic story:** single-member S.r.l., S.r.l. and S.p.A. differ in governance and capital rules
  [code meanings UNVERIFIED].
- **Data (UNI §5):** SRL 1.26% (reference), SRU 1.87%, SPA 0.91%, SAU 0.90%, OTHER 0.82%. EBM: OTHER
  −0.87 (thin; unseen forms land there, flag for the harness, D-084).
- **Models:** SC-02, SC-03, GB-all, GAM.
- **Source:** assumption; code meanings UNVERIFIED. **Citation gap:** YES.
- **Example:** Alfa: **SRL**.

### F-26 `macro_area` — geography (core, categorical)
- **Formula:** HQ_city (ISTAT province code, hypothesis) → region → 5 macro-areas; missing → OTHER.
- **Economic story:** regional economic conditions; five levels keep each well populated.
- **Data (UNI §5):** North-West 1.28% (reference), North-East 1.22%, Centre 1.42%, South 1.31%,
  Islands 1.29%, OTHER 4.0% (150 rows, 6 defaults). Small differences; we must not tell a
  North/South story without a source (REFERENCES "must not claim" #13).
- **Models:** SC-02, SC-03, GB-all, GAM.
- **Source:** assumption; ISTAT table UNVERIFIED. **Citation gap:** YES.
- **Example:** Alfa: code 15 → Milano → Lombardia → **North-West**.

---

## 4. Pipeline walkthrough: raw file → PD (and what could go wrong at each step)

For each step: what happens, the file/code, the risk, and the guard.

**Step 1. Raw file → clean columns.** `data/raw/` (read-only) is read by `src/data/load_raw.py`:
both column spellings accepted, every NA token → NaN, the index column kept as `src_index`.
*Risk:* a renamed or missing column crashes the harness; a sentinel like −999 is deleted.
*Guard:* alias map (D-010), sentinel values kept (D-014), unit test `test_messy_row`.

**Step 2. Labels.** `src/data/build_panel.py::label_one`: window [statement + 6m, statement + 18m),
data end 2013-05-31; statuses ok / already_defaulted / censored / after_default.
*Risk 1 — leakage:* `def_date` is the answer; any feature derived from it is cheating.
*Guard:* `def_date` is never a feature; the referee strips every label column from the test frame
and still reproduces the PDs (SC1 T15). *Risk 2 — censoring bias:* counting unknown outcomes as
survivors, or keeping only observed defaults (the shared-code bug, D-011/D-068).
*Guard:* censored rows dropped; panel vs shared code checked equal on 2007-2010 (SC1 T12).

**Step 3. Panel.** `data/interim/panel.parquet`: one row per firm-year with `y` (start 6), `y_s0`,
`y_s3`, `y_s6`, sensitivity labels (`y_s6_d2012` …), `label_status`, window dates.
*Risk:* the label window turns out different in the holdout (D-002a). *Guard:* all windows stored;
the level parameter τ is switchable per window (D-076).

**Step 4. Frozen splits.** `data/splits/walkforward.json` (+ `walkforward_test_rows.parquet`):
three expanding folds, row_id = id × 10000 + fs_year.
*Risk:* each challenger picking its own favourable split. *Guard:* one read-only file for everyone
(CLAUDE.md §5 rule 6); every output file must match the test-row list (contract check SC1 T11).

**Step 5. Features.** `src/features/build.py`: `fit` on the training window only (clip bounds,
medians, debt shares, category pooling), `transform` row by row (section 3.0).
*Risk 1 — parameter leakage:* bounds or medians computed on all years. *Guard:* `fit` is called
inside each fold on `fs_year ≤ t` rows (asserted in each runner; refit reproduced, D-061).
*Risk 2 — design leakage:* the **choice** of features saw 2008-2009 labels (D-062).
*Guard:* report 2010 next to pooled numbers; promotions must hold on 2010.

**Step 6. Models.** Each model is refitted per fold on training rows only.
- Scorecards (SC-01..04): nothing tuned → no tuning leakage possible (D-030).
- RiskCalc-style (RC-01..03): settings pre-declared; *risk:* the post-hoc sign rule (D-063).
- GBM (GB-01..06): inner time-ordered split (A: early stopping/grid; B: calibration slope) inside
  the training window (D-052). *Risk:* too many trees / too deep → overfit. *Guard:* early stopping,
  depth ≤ 3, ≥ 500 rows per leaf, OOT calibration slope 0.976.
- EBM (GAM-01/02): same inner scheme; *risk:* tiny missing-value bins learned as "riskless".
  *Guard:* reset rule (D-082).
Code: `experiments/{scorecard,riskcalc,gbm,gam}/`; models in MF §5.

**Step 7. Referee scoring and promotion.** `reports/tournament/score_r1.py`: pooled / per year /
seen-new metrics, paired firm-clustered bootstrap (200 draws), face validity, promotion rule (SC1 §1).
*Risk:* picking the best of many variants on the same three years (selection overfitting, §5).
*Guard:* pre-declared promotion rule; ties go to the simpler model; 2010-only check; lockbox (D-087).

**Step 8. Calibration (level).** `experiments/calibration/s1..s8_*.py`: one logit shift δ from the
training default rate r_w to the target τ (D-072, D-073); floor 0.03%, cap 99% (D-075).
*Risk 1:* using information from the test year (e.g. the first months of its defaults).
*Guard:* D-070's "what the bank knows" rule. *Risk 2:* the level rule itself was chosen after seeing
2008-2010 (D-071). *Guard:* 2011 turning-point check; hedge costs in `scenario_regret.csv`.

**Step 9. Grade table.** `reports/calibration/grade_table_2010.csv`: 11 pre-declared PD bands with
observed default rates. *Risk:* treating binomial bands as a pass/fail test. *Guard:* reported as
diagnostic only (R-16 pp. 33-34).

**Step 10. Harness (future, owned by harness-engineer).** Planned chain (D-079): load frozen
params → `transform` → GB-01 final booster (refit on 2007-2011 after the lockbox check, D-077,
D-087) → its own calibrator → + δ → clip [0.0003, 0.99] → one PD per row, same order.
*Risks:* unseen categories (map to OTHER: check what OTHER means for each model, D-084); a column
missing entirely; a LightGBM version different from 4.7.0 (D-059); re-estimating anything from
the holdout (forbidden). *Guards:* row-wise `transform` with no batch statistics; stress tests;
complexity O(n × (F + T × D)), about 1 s per 100,000 rows (D-059).

---

## 5. Overfitting and validation, from scratch

**Overfitting** = the model (or the modeller) learns features of the particular data it was tuned
on that will not repeat. The symptom is always the same: results look better on the data used for
decisions than on new data. There are four kinds in this project.

### 5.1 The four kinds, with our checks and numbers

**(1) Model overfitting** — the fitted model memorises noise in the training rows.
- *What it looks like:* PDs too extreme out-of-time; **calibration slope < 1** (slope = how much the
  model's log-odds must be shrunk to fit new data; 1 = right spread, < 1 = too spread).
- *Our checks:* out-of-time slopes for the main candidates are 0.98-1.02 pooled (SC1 §5; GB-01
  0.976, RC-03 1.024, SC-02 0.983); benchmark logits 0.95-1.03 (D-035). GBM capacity limited by
  depth ≤ 3, ≥ 500-2,000 rows per leaf, early stopping, L2 penalty 10; grid differences ≤ 0.0003 log
  loss (D-052). RiskCalc bins need ≥ 30 defaults each, shrunk with 2 pseudo-defaults (D-044).
- *Where it still bites:* tiny cells. The EBM learned −5.10 log-odds for a missing-value bin with
  19 rows and 0 defaults (D-082); fixed by the reset rule. New firms' slope ≈ 0.88 (SC1 T3): PDs too
  spread for firms unlike most training rows.

**(2) Selection overfitting ("garden of forking paths")** — the *people* overfit: every time we look
at test results and then choose (a feature, a rule, a model), the chosen thing looks better on that
test set than it will on new data, even if every individual model was fitted honestly.
- *Our cases:* the feature set was chosen on 2007-2009 (D-062); RC-03's sign rule was added after
  seeing outer results (D-063); level rule e was the best of 7 after seeing 2008-2010 (D-071);
  round 2 runs many more variants on the same three test years (D-087).
- *Our checks:* pre-declared promotion rule and settings (SC1 §1; D-030, D-044); "a promotion must
  hold on 2010 alone" (GB-01 vs SC-03 on 2010: +0.0191 [+0.0156, +0.0234], SC1 T5); paired CIs; ties
  go to the simpler model; and the lockbox (5.3).
- *The size of the effect* is shown by the runnable example in 5.2.

**(3) Time drift (non-stationarity)** — the world changes between training and use, so even an
honest model is wrong out-of-time.
- *Our numbers:* default rate 0.96% (2007) → 1.28% → 1.62% → 1.70% (2010) → ≈ 1.43% (2011,
  11-month estimate) (MF §1.4, `reports/calibration/rates_by_cohort.csv`). GB-01 obs/pred 1.21 / 1.36
  / 1.43 by test year (SC1 T2); GB-01 AUC 0.8441 / 0.8359 / 0.8307 (slow decline). Feature power is
  fairly stable: e.g. equity_ratio univariate AUC 0.2919 / 0.2594 / 0.2562 in 2007 / 2008 / 2009
  (UNI §4).
- *Our checks:* walk-forward testing (D-005), per-year results everywhere, the level set separately
  (D-071 to D-073), and a test of the level rules on the 2011 turn (trend rule cost 9.42e-4 vs 0.33e-4
  for the pooled average, D-071).

**(4) Calibration overfitting** — the level or the calibration curve is fitted (or chosen) on too
little or the wrong data.
- *Our cases:* isotonic calibration needs many defaults (fold 1's slice B had 203 defaults, so it was
  not eligible, D-053); Saerens EM estimated the 2010 prior at 0.45% vs 1.70% observed (D-072); the
  τ = 1.49% hedge rests on one turning point and an assumed random walk (D-073).
- *Our checks:* one-parameter level method (prior shift) instead of a flexible curve (D-072); within-
  year AUC unchanged by calibration (|dAUC| = 0); scenario costs published (D-074).

### 5.2 Runnable example: picking the best of 20 equal models

The script below builds 20 scoring models that are **equally good by construction** (each is the
true risk driver plus its own mix of pure-noise columns, all with the same noise size, so all have
exactly the same true AUC). We pick the one with the highest AUC on a test set (~450 defaults, like
a small fold) and score that same model on a fresh set. Needs only numpy and scikit-learn.

```python
# Selection overfitting ("garden of forking paths"), runnable with numpy + scikit-learn.
# We build 20 candidate scoring "models" that are EQUALLY GOOD BY CONSTRUCTION: each one is
#   score_k = true_driver + (unit-length random mix of 10 pure-noise columns),
# so every model has exactly the same true AUC. We pick the one with the best AUC on a test set,
# then score the SAME model on a fresh set drawn from the same world.
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
N_ROWS, BASE_RATE, N_MODELS, N_NOISE = 20_000, 0.015, 20, 10   # ~300 defaults per set

W = rng.normal(size=(N_MODELS, N_NOISE))
W /= np.linalg.norm(W, axis=1, keepdims=True)                  # every model gets noise of variance 1

def make_set():
    x = rng.normal(size=N_ROWS)                                 # the one real risk driver
    z = rng.normal(size=(N_ROWS, N_NOISE))                      # columns with no information
    logit = np.log(BASE_RATE / (1 - BASE_RATE)) + 0.9 * x
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return x, z, y

def aucs(x, z, y):
    return np.array([roc_auc_score(y, x + z @ W[k]) for k in range(N_MODELS)])

x, z, y = make_set();  a_test = aucs(x, z, y)
x2, z2, y2 = make_set(); a_new = aucs(x2, z2, y2)
k = int(np.argmax(a_test))
print(f"defaults: test set {y.sum()}, fresh set {y2.sum()}")
print(f"test-set AUCs of 20 equally good models: min {a_test.min():.4f}, mean {a_test.mean():.4f}, max {a_test.max():.4f}")
print(f"winner (model {k}) beats the average model by {a_test[k]-a_test.mean():+.4f} AUC on the test set")
print(f"same winner on the fresh set beats the average model by {a_new[k]-a_new.mean():+.4f} AUC")

edge_test, edge_new = [], []
for _ in range(200):                                            # repeat to show it is systematic
    x, z, y = make_set();  at = aucs(x, z, y)
    x2, z2, y2 = make_set(); an = aucs(x2, z2, y2)
    k = int(np.argmax(at))
    edge_test.append(at[k] - at.mean()); edge_new.append(an[k] - an.mean())
print(f"200 repeats: winner's edge on the test set {np.mean(edge_test):+.4f}, "
      f"on a fresh set {np.mean(edge_new):+.4f} (true edge is exactly 0)")
```

Real output (run 2026-11-02, scikit-learn 1.7.2, about 35 s on a laptop):

```
defaults: test set 449, fresh set 419
test-set AUCs of 20 equally good models: min 0.6480, mean 0.6579, max 0.6655
winner (model 15) beats the average model by +0.0076 AUC on the test set
same winner on the fresh set beats the average model by -0.0086 AUC
200 repeats: winner's edge on the test set +0.0167, on a fresh set +0.0009 (true edge is exactly 0)
```

**How to read it.** The 20 models are identical in truth, yet on one test set their AUCs spread over
0.018 (0.6480 to 0.6655) purely by chance. The "winner" looked +0.0076 better than average; on fresh
data it was −0.0086, i.e. its edge was luck. Over 200 repeats the winner's apparent edge averages
+0.0167 AUC and shrinks to +0.0009 (≈ 0) on fresh data.

**Why this matters for us.** +0.0167 AUC is the same order as real differences we act on:
GB-02 vs GB-01 +0.0065, GB-01 vs GAM-01 +0.0120, GB-01 vs RC-03 +0.0183 (SC1 T6; D-081). Our real
comparisons are far less exposed than this toy (7,784 test defaults instead of ~450, paired CIs, and
our candidates are not identical), but the lesson holds: **the more variants we try on the same test
years, the more the best one's score is inflated.** With few defaults the noise is larger; metric
noise is driven by the number of defaults, not rows (R-07 Sec. 4.1 pp. 96-99, VERIFIED; SG §2.4).

### 5.3 Why fs_year 2011 is kept as a one-time lockbox (D-087)

- **What a lockbox is:** a slice of data nobody looks at while making choices, opened once at the
  end to get one unbiased score for the final choice. Once opened, it is spent.
- **Why 2011:** it is the newest year with (almost) known outcomes: an 11-month label is available
  because defaults are tracked to 2013-05-31 (D-077). It is also a turning point (the rate fell from
  1.70% to about 1.43%), which is exactly the situation where an overfit model or level rule would
  fail. Earlier years cannot serve: 2008-2010 are the test years everyone has looked at, and 2007 is
  the only training year of fold 1.
- **Why one time only:** if we scored finalists on 2011, changed something, and scored again, 2011
  would become just another test year subject to selection overfitting (5.2).
- **Then what:** after the single check, 2011 is added to the final fit, because the newest
  behaviour matters most and the 2010 analogue shows an 11-month newest cohort beats dropping it
  (AUC 0.8301 vs 0.8263, `reports/calibration/q6_partial_cohort_test.csv`).
- **Honest caveat (say it before Roger finds it):** 2011 is not completely unseen. The calibration
  round scored the level rules on 2011 (`q6_2011_pseudo_test.csv`) and computed fold-4 AUCs of GB-01
  (0.8497 with the 11-month label) and RC-02 (0.8345) on it (`q5_2011_other_labels_fold4.csv`,
  `q6_2011_pseudo_test.csv`). No model was chosen using those numbers, so the lockbox is still valid
  for **round-2 model choice**, but not for the **level rule**, whose 2011 check has already been used.
- **What the lockbox does not fix:** the D-062 design leak in 2008-2009, and the possibility that the
  holdout year behaves unlike 2011.

---

## 6. Citation-gap summary (the action list)

Every decision or feature whose support is an assumption or not VERIFIED. "Fix" says what to do;
"needs literature-scout" means a new source must be found and verified before it can be cited.
Priority: H = in the deck's main story or a likely Roger question; M = appears in the report; L = a
stated design choice that only needs the word "assumption".

| Item | What lacks support | Current status | Suggested fix | Pri |
|---|---|---|---|---|
| D-002 / D-070 label timing | 2007-2012 wording of arts. 2364/2435; actual filing dates | R-28/R-29 VERIFIED current text only; history SECONDARY | literature-scout: historical text; meanwhile cite R-01 Sec. 3.2 p. 16 (statement lag, US) and keep "legal maxima, assumption" | H |
| D-005 evaluation design | log cites R-06 | R-06 LOCATED | replace with R-07 Sec. 3 p. 94 + R-01 Sec. 4.3 p. 28 (VERIFIED) | M |
| D-006 log loss primary | no R-id in log | — | add R-07 Sec. 2.2 pp. 88-89, R-01 Tables 5-6 (VERIFIED) | M |
| D-012, D-023, F-26 province codes | ISTAT province table | [UNVERIFIED], agent knowledge | needs literature-scout (ISTAT code list) | M |
| D-013, D-023, F-24 ATECO sections | NACE Rev. 2 / ATECO 2007 division ranges | [UNVERIFIED] | needs literature-scout (Eurostat / ISTAT) | M |
| D-015 frozen splits | no R-id | — | add R-07 Sec. 3 p. 94 (VERIFIED) | L |
| D-016 / D-021 abbreviated accounts | pre-2008 thresholds of art. 2435-bis | R-27 current text VERIFIED; history SECONDARY | keep "plausible cause"; literature-scout for history | M |
| D-018 asset-structure family | collateral argument | assumption | needs literature-scout or label assumption | H |
| F-18 tangible_ratio (#1 GB-01 driver) | why tangible assets lower PD | assumption | needs literature-scout (collateral / asset tangibility and default) | **H** |
| D-019 slog, zero-denominator limits | transform design | assumption (clipping is covered by R-05 fn. 42, R-01 fn. 8) | state "our design" | L |
| D-021 imputation | Moody's support not in log | R-34, R-05 pp. 60-61 VERIFIED | add both to the log ("Moody's states…") | M |
| D-022, F-15, F-20 "sign none" stories | thin-margin traders; two-sided extraordinary items | assumption | label as hypotheses consistent with deciles | L |
| D-023 30-default pooling rule | threshold | assumption | state as pre-declared design rule | L |
| D-030 benchmarks | "Stein (course)"; WoE / PAV method | not an R-id; no source | cite R-07 + R-17 Conclusions p. 25; WoE/PAV: literature-scout or "standard practice (assumption)" | M |
| D-035 points scaling | PDO 20 / 600 at 50:1 | convention | call it illustrative | L |
| D-040 RiskCalc recipe | bins, prior, PAV, F shape | assumption (R-02 unread) | add R-05 pp. 55-57 for lookup + interpolation; never say "as in RiskCalc" for the rest | H |
| D-041 sign-drop rule | rule | assumption (post hoc, D-063) | label design rule + post hoc | M |
| D-042 missing priced in T_i | method | assumption | add R-05 pp. 60-61 / R-34 as contrast | L |
| D-044 bin settings | defaults | assumption | label pre-declared | L |
| D-050 LightGBM, monotone constraints | algorithm and constraint method | no source (R-18 XGBoost TO-VERIFY; no LightGBM ref) | needs literature-scout (LightGBM paper; monotone boosting) | H |
| D-051 face validity principle | why constraints | assumption | add R-01 Sec. 3.1 p. 13 (expected signs) | M |
| D-052 tuning scheme, 1.2 × trees | heuristic | assumption | add R-07 p. 95 for inner split; 1.2 stays a heuristic | L |
| D-053, D-072 Platt / isotonic | method names | R-20, R-21 TO-VERIFY | needs literature-scout if described beyond the name | M |
| D-054 no re-weighting | design | assumption | label | L |
| D-058 TreeSHAP | algorithm | [UNVERIFIED] (R-23 has Shapley theory only) | needs literature-scout (TreeSHAP paper) | H |
| D-062, D-063, D-087 selection caveats | literature basis | not in log | add R-07 Sec. 3.2 p. 95 (VERIFIED) | M |
| D-073 τ hedge | random walk, σ = 0.21 | assumption (no citation possible) | label as judgment; show `scenario_regret.csv` | H |
| D-074 pricing reading of asymmetry | under-pricing → adverse selection | assumption; R-08 Sec. 6 by analogy | cite R-08 "by analogy", keep assumption label | M |
| D-080, D-081, D-083 EBM / GA2M | method papers | R-24, R-25 TO-VERIFY | needs literature-scout | M |
| D-086 ensemble rejection | reasoning | assumption | add R-10 Sec. 4.4 p. 22 (VERIFIED: averaging two models added no stable power); move the script into `experiments/` | M |
| F-03 st_debt_ratio | specific ratio | family VERIFIED (R-01 p. 13) | fine with family cite | L |
| F-04 bank_debt_ratio | story | assumption | label assumption | L |
| F-05 st_debt_share | maturity story | assumption | label assumption | L |
| F-06 wc_ratio | WC/TA in Altman | R-32 VERIFIED but WC/TA page not in claim map | literature-scout: record the R-32 page | L |
| F-07 current_ratio | classic ratio | assumption | label | L |
| F-16 ar_days | specific ratio | family VERIFIED | fine with family cite | L |
| F-19 intangible_ratio | story | assumption | label | L |
| F-21 neg_equity | flag | assumption | label | L |
| F-23 rev_nonpositive | flag | assumption | label; fix EBM reason-code wording (D-084) | L |
| F-25 legal_group | code meanings | [UNVERIFIED] | needs literature-scout (Italian legal-form codes) or describe codes neutrally | M |

Sources that are VERIFIED and already carry most of the story: R-01, R-05, R-07, R-08, R-10, R-14,
R-15 (floor only), R-16, R-17, R-19 (abstract), R-22, R-23, R-27/28/29 (current text), R-30, R-32,
R-33 (France only), R-34 (vendor statement), R-35.

---

## 7. Glossary (alphabetical)

- **AR (accuracy ratio):** 2·AUC − 1; the CAP-curve view of the same ranking information (R-14).
- **ATECO:** the Italian industry classification (2007 version), aligned with the EU's NACE Rev. 2.
- **AUC:** area under the ROC curve; probability a random defaulter gets a higher PD than a random
  survivor. 0.5 = random, 1 = perfect.
- **Bootstrap (paired, firm-clustered):** resample whole firms with replacement many times, score
  both models on each resample, read the 2.5% and 97.5% percentiles of the difference.
- **Brier score:** mean of (PD − outcome)². Overall probability accuracy; per BCBS mostly reflects
  discrimination (R-16 p. 32), so not used as a calibration measure.
- **Calibration-in-the-large / obs/pred:** observed default rate vs mean PD.
- **Calibration slope / intercept:** fit logit(P(default)) = a + b·logit(PD) on test data;
  perfect = (0, 1). b < 1: PDs too spread; a > 0: PDs too low.
- **Censoring:** the outcome window is not over yet, so the outcome is unknown.
- **Central tendency:** the average PD level a model is set to (τ in our harness).
- **Core set / full set:** the 13 features for linear/probit models vs all 26 for trees/EBM.
- **D-ID / R-ID:** decision-log entry / reference-list entry.
- **EBITDA:** earnings before interest, taxes, depreciation and amortisation; operating cash earnings.
- **EBM:** Explainable Boosting Machine; a GAM learned by boosting (D-080).
- **Early stopping:** stop adding trees when validation loss stops improving.
- **Face validity:** drivers move PD in the economically expected direction.
- **GAM:** generalised additive model; log-odds = sum of one smooth curve per feature.
- **GBM / LightGBM:** gradient-boosted trees; many small trees added in sequence, each fixing the
  previous ones' errors. LightGBM is one software implementation.
- **Inner split:** a validation split *inside* the training window, used for tuning.
- **Isotonic regression:** fitting the best monotone step function.
- **Leakage:** test or future information entering the fit.
- **Lockbox:** data kept sealed until one final check (5.3).
- **Log-odds (logit):** ln(PD / (1 − PD)); the scale on which models add up contributions.
  expit is its inverse.
- **Log loss:** mean of −[y·ln(PD) + (1 − y)·ln(1 − PD)]; lower is better.
- **Monotone constraint:** forces the model's output to move only one way as a feature rises.
- **NACE Rev. 2:** EU industry classification; sections A-U group 2-digit divisions.
- **OOT:** out-of-time.
- **PAV (pool-adjacent-violators):** merges neighbouring bins until the curve is monotone.
- **Partial dependence:** the model's average PD as one feature is varied, others held at their values.
- **PD:** probability of default in the next 12 months.
- **PIT / TTC:** point-in-time (follows current conditions; pricing) vs through-the-cycle (averaged
  over the cycle; underwriting) (R-16 p. 11).
- **Platt scaling:** recalibration by a one-variable logistic regression on the score.
- **Prior shift:** moving every PD by the same log-odds amount so the average matches a new base
  rate (R-22 eq. 4); the method we use for the level.
- **Probit:** a regression whose output passes through the standard normal CDF Φ.
- **Reason code:** the top drivers of one firm's PD, in words.
- **SHAP:** Shapley-value attribution of a prediction to features; contributions add up to the
  prediction (R-23).
- **Shrinkage:** pulling a noisy estimate toward a more stable average.
- **Signed log (slog):** sign(x)·ln(1 + |x|); order-preserving tail compression.
- **VIF:** variance inflation factor; 1/(1 − R²) of a feature regressed on the others.
- **Walk-forward:** train on years ≤ t, test on t+1, roll forward.
- **Winsorise:** clip values to fixed lower/upper bounds (here the training 1st/99th percentiles).
- **WoE (weight of evidence):** a bin's log good/bad odds relative to the overall odds; the input
  of a classic scorecard.

---

## 8. Round 2 (added 2026-11-03)

Full entries: `docs/DECISION_LOG.md` D-090..D-149. Scorecard: `reports/tournament/r2_scorecard.md`.
Status: **nothing here has been checked by a human yet.** Promotion of GB-25 needs your sign-off.

### 8.1 The result in five sentences
We added 19 inputs and got GB-25: AUC 0.8348 on 2010 (+0.0042 [0.0022, 0.0059] vs GB-01). The gain came from new information; tuning gave +0.0009 and recency weights +0.0002. The riskiest 10% in 2010 capture about the same share of defaults (51.7% vs 51.8%), so this is a modest improvement, not a new view of risk. The level problem is untouched (2010 observed/predicted 1.19; middle grades 1.33). The leakage audit is clean (D-141), though the referee is also an AI agent.

### 8.2 The new features: what each one means
AUC below 0.5 means PD falls as the feature rises. Stats are on 2007-09 training rows (`reports/features/round2_univariate.md`).

| Feature | Built as | Why it could matter for a lender | Source status |
|---|---|---|---|
| `operating_margin` | slog(prof_operations / revenue) | Thin margin leaves no buffer | R-33 VERIFIED |
| `interest_to_sales` | slog(exp_financing / revenue) | Share of sales already spent on interest. GB-25 forces + | R-33 VERIFIED |
| `payables_to_sales` | slog(AP_st / revenue) | Stretching suppliers signals cash shortage | R-33 VERIFIED |
| `personnel_to_sales` | slog((goodwill − ebitda) / revenue) | Wages are hard to cut when sales fall | R-33 for the ratio; **our inference** that `goodwill` is value added (D-091) |
| six `*_sect_pct` | Firm's percentile for equity, ROA, coverage, cash, short-term debt, tangibles within its ATECO section | 40% equity means different things in retail and real estate | R-01 Sec. 3.3 VERIFIED |
| `ateco_div_te`, `province_te`, `region_te` | Group's default rate in log-odds vs base, shrunk by n/(n+m), m = 555 / 1,834 / 3,879; out-of-fold on training rows | Sector and place conditions a single statement misses | Method R-46, R-47; economics **assumption** |
| `n_core_imputed` | Count of core inputs missing after rebuild | Mostly negative-revenue firms. **Has 0 splits in fold 3, so it does nothing there** | R-34, R-05 |
| `bs_gap` | (assets − equity − listed liabilities) / assets | Liabilities the file does not classify; no expected direction | assumption |
| `nonbank_fin_debt` | (debt_fin_st + debt_fin_lt) / assets | Fixed claims from non-bank lenders | assumption |
| `ap_lt_ratio` | AP_lt / assets | Suppliers who became long-term creditors | assumption |
| `fin_fixed_ratio` | asst_fixed_fin / assets | Holding or group structure; no expected direction | assumption |
| `tfr_ratio` | liab_lt_emp / assets | See 8.4 | **[UNVERIFIED] hypothesis** |

Worked example for the shrinkage. Run it:
```python
n, m = 1834, 1834   # rows in this province; m is the fitted province prior
print(n / (n + m))  # 0.5: it keeps half its own rate, half the average
```

### 8.3 Decisions you must be able to defend

**Why not simply use the highest score (GB-20, 0.8369 on 2010)?** (D-125, D-130, D-144.) When a firm's interest doubles, GB-20 lowers the PD for 20% of firms and GB-25 for 1.7%. GB-20 is statistically better by 0.0021 AUC on 2010. This is a face-validity judgment, and you can overrule it.
- **If Roger asks** "you gave up accuracy?": "Yes, about 0.002 AUC, to keep the direction Moody's uses (R-33) and avoid a sign we chose after seeing data."
- **Do not say** it was fully pre-specified. The fix had no outer score beforehand, but 2010 default rates by decile were already on file (D-142).

**Why keep monotone constraints when unconstrained scores higher?** (D-102, D-103.) A firm with cash flow twice its debt would score as badly as one burning cash. Our rule for relaxing them ("only where the denominator nears zero") failed its own check. High-margin firms are partly property vehicles, but that does not excuse loss-making firms getting lower PDs.
- **If Roger asks** "why do default rates rise among the most profitable firms?": "We see it and have no verified explanation; we constrain it." **Do not invent a reason.**

**Why did tuning fail?** (D-104.) Tuning changes how the same signal is fitted, not what the model knows. The top configurations are nearly tied, and chosen settings swing widely across folds, which looks like noise.

**Why are the encodings computed out-of-fold?** (D-094.) Otherwise a row sees its own default in its feature. Our test: leaky encodings scored *worse* out of time (0.8330 vs 0.8356 AUC on 2009; `reports/features/round2_target_encoding.csv`).

**Why is SHAP not value?** (D-122.) Sector percentiles have the largest SHAP share, yet dropping them changes AUC insignificantly. They take credit from the base ratios they correlate with (0.88-0.98).

**How did we treat the random forest?** (D-106 corrected.) GB-25 beats it pooled (+0.0029 [0.0005, 0.0050]) and ties on 2010 (+0.0013 [−0.0026, 0.0051]). GB-01 only ties it on AUC. The forest uses the forbidden patterns, so it is a benchmark and not a candidate.

### 8.4 Your hardest question: `tfr_ratio`
Univariate AUC is 0.51, yet it is the most robust new feature (dropping it costs about 0.002 AUC in both inner windows). Two reasons: half of zero-fund firms are low-default real estate, and its effect appears only next to the wage bill (logit coefficient −0.043 alone, −0.197 beside `personnel_to_sales`). Reading: more fund *for a given wage bill* means lower PD.
- **Why? We do not know.** Our hypothesis is that it reflects staff who stayed (a proxy for firm maturity) and patient funding. It is [UNVERIFIED] and depends on the `goodwill` inference.
- **Caveat:** firms with low turnover and a fund above 5% of assets default at 5.7-11.5% (880 rows, 62 defaults). The "labour intensity" reading in D-097 was corrected by D-123.

### 8.5 Overfitting: what round 2 changed
- **Same-year gap widened:** GB-01 0.015-0.026, GB-20 0.022-0.035, GB-25 0.017-0.029 (`experiments/gbm/comparison_r2b.md`). Only quote out-of-time numbers.
- **About 40% of the gain buys back constraint cost** (D-126). Free features recover part of what constraints blocked.
- **113 looks at the test years** (D-142). Quote 2010 first, with pooled beside it, labelled.
- **Win rule tightened from the lockbox on** (D-149). It was proposed after seeing results, so it was not used for this promotion.

### 8.6 Known defects
- **Negative-EBITDA firms** (D-147): more interest mechanically improves `interest_coverage`, so PD can fall (GB-25 40%, GB-01 30% of those firms). The constraint check cannot catch this, because it arises when the ratio is built.
- **Three "zero violations" mean nothing:** `neg_equity`, `debt_split_missing` and `n_core_imputed` have no splits in the checked model.
- **New firms did not improve** (D-145); 2008 got worse (AR −0.020 [−0.034, −0.006]).

### 8.7 Citation gaps added
Unsourced: `bs_gap`, `nonbank_fin_debt`, `ap_lt_ratio`, `fin_fixed_ratio`, `tfr_ratio`'s reading, and the economics of all three encodings. Label them as assumptions or send them to literature-scout. Also: why default rates rise among the most profitable firms.

### 8.8 Decisions waiting for you
GB-25 vs GB-20 (D-130, D-144); EBM fallback (D-112); EBITDA constraint (D-103); lockbox finalists GB-25, GB-01, GAM-01 (D-119, D-148; GB-01's side is not blind); revised win rule (D-149).
