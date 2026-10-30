# What Moody's practitioner papers say about features and data cleaning (literature-scout, 2026-10-30)

Scope: RiskCalc 4.0 and other Moody's / Moody's KMV / Moody's Analytics documents on feature
engineering and data cleaning for private-firm PD models. Everything below comes from documents
that were opened and read today. Page numbers are the documents' printed pages unless "slide" is
stated. Full bibliographic rows, URLs and notes are in `docs/REFERENCES.md` (R-ids, notes N-xx).

## 0. What was read (and what was not)

| ID | Document | Status | What it covers |
|---|---|---|---|
| R-33 | Buitrago, Makarov, Zhao & Dwyer (Dec 2015), *RiskCalc 4.0 France*, Moody's Analytics Modeling Methodology | **VERIFIED**, all 32 pp. read | **The only RiskCalc 4.0 document found and read.** France only. |
| R-01 | Dwyer, Kocagil & Stein (2004), *RiskCalc v3.1 Model* | VERIFIED (re-read Sec. 3.3–3.4) | Industry controls, "Managing data quality", panel dependence |
| R-05 | Falkenstein, Boral & Carty (2000), *RiskCalc for Private Companies* | VERIFIED (re-read pp. 25–26, 38–41, 54–61) | Statement quality, levels vs trends, truncation, missing inputs |
| R-10 | Stein, Kocagil, Bohn & Akhavein (Feb 2003), *Systematic and Idiosyncratic Risk in Middle-Market Default Prediction* | **VERIFIED** (upgraded from LOCATED) | Firm-specific vs cycle information, statement age, CRD data filtering |
| R-34 | Moody's (Dec 2021), *Advances in default detection and early warning* (web article) | VERIFIED as a web page; its claims are assertions with no numbers | Missing-input imputation |
| R-35 | Dwyer & Stein (2006), *JBF* 30, 797–810 | VERIFIED (abstract, Sec. 1, Sec. 5–6) | Default databases miss defaults, which affects calibration |
| R-04 | Buitrago & Makarov (May 2020), Moody's webinar slides | **VERIFIED** (upgraded; slides 13–15) | Italy credit-cycle adjustment |

**On RiskCalc 4.0:** we read the **France** 4.0 methodology (R-33) in full. No general RiskCalc
4.0 white paper and no RiskCalc Italy document (v3.1 or 4.0) could be found on the public web.
Everything this digest says about "4.0" means **RiskCalc 4.0 France**. Do not carry its ratio
list or weights over to Italy.

## 1. How RiskCalc 4.0 France differs from 3.1 (R-33, Sec. 1.2, p. 4)

- Same architecture as 3.1: per-ratio non-parametric transforms, then a probit with industry
  dummies, then a final non-parametric mapping to an EDF. The cycle adjustment only changes the
  final mapping (Sec. 3, p. 10, eq. 1).
- More data and "substantially improved data cleansing"; 14 industries instead of 9.
- Input changes: marketable securities added to liquid assets; interest expense/sales replaces
  financial charges/sales; personnel expenses/sales replaces change in AR turnover; operating
  margin replaces EBITDA/sales; cash flow/financial charges dropped.
- The credit-cycle adjustment (CCA) was re-specified on public-firm distance-to-default, with
  shrinkage toward a European regional factor for thin French sectors (Sec. 3.3, p. 15, fn. 11).
- Final input list (Table 3, p. 11): equity/assets; cash + marketable securities/assets;
  debt coverage = (profit + D&A)/(liabilities − cash and securities); operating margin; interest
  expense/sales; accounts payable/sales; personnel expenses/sales; sales growth; change in ROA;
  real (inflation-adjusted) total assets.
- Weights (Table 4, p. 14; share of EDF change when one input moves one s.d. from the average
  firm): debt coverage 19.2%, leverage 17.8%, liquidity 16.0%, activity 13.6%, growth 13.0%,
  size 10.4%, profitability 10.0%.
- R-33 does **not** describe how missing inputs are handled. R-34 is our only Moody's source on
  that topic (see L1).

## 2. Lessons, mapped to what we do

Ratings: **MUST** = do before Nov 1; **SHOULD** = do before Nov 1 if time allows, otherwise
log it as a known limitation; **NICE** = only if spare time; **NO** = no change (we cite the
source as support). Effort is for one agent.

### L1. Filling a missing value with a mean or median pulls PDs down. **SHOULD**, about 1 h (referee / calibration-specialist)
- **(a) Source.** Moody's says its earlier RiskCalc versions filled missing inputs with region
  and sector means, and that this would systematically understate PDs. It now estimates
  missing items from items the firm does report (sub-models for total assets and leverage
  built from sales, employees and profitability) (R-34). RiskCalc v1 filled a missing input
  with the mean transformed value and noted that less real data gives a worse result
  (R-05, pp. 60–61).
- **(b) What we do.** D-021: we rebuild missing items from accounting identities first. For the
  missing debt-split block we impute (training sector-median share) × (the firm's own total) and
  set the flag `debt_split_missing`. This is the "estimate from the firm's own related items"
  approach, so the literature **supports** it. Any remaining NaN in the GLM view gets the
  training sector median, which is exactly the mean-type fill Moody's warns about. That is
  harmless at < 0.1% of training rows, but the harness will receive messy holdout rows.
- **(c) Change?** Yes, a check rather than a redesign. In the referee scorecard, report observed
  vs mean predicted PD per fold for (i) `debt_split_missing = 1` and (ii) rows where at least
  one core input was median-filled. If (ii) is under-predicted, add an `n_core_imputed` flag
  (it can be computed from the current row alone, so it is harness-safe). Note that the
  `debt_split_missing` flag is in the full set only (D-022.4), so the core probit gets no
  missingness signal.

### L2. Plausibility checks on statements, but test whether each filter actually helps. **SHOULD**, 1–2 h (data-auditor + one challenger)
- **(a) Source.** RiskCalc 4.0 France drops statements that fail plausibility checks, e.g.
  assets ≠ liabilities + net worth, or a period shorter than 12 months (R-33, Sec. 2.2, p. 6).
  Moody's KMV filtered North American submissions for GAAP violations and for records where
  default status or the statement–loan match could not be established. Over 80% of submitted
  statements were removed, mostly interim statements (R-10, App. A, p. 30, fn. 19). Cleaner data
  alone raised the v1.0 out-of-sample one-year AR from 48.2% to 51.7% (R-01, Sec. 3.4.1, Table 4,
  p. 20). A counter-lesson: rows flagged for excessive rounding made the model weaker when
  removed, so Moody's kept them and only flagged them (R-01, p. 22).
- **(b) What we do.** We rebuild from identities (D-021) but do not filter training rows. The
  data-auditor found that the balance-sheet identity holds within 1% on only 80.05% of rows,
  but the gap comes from an omitted asset/liability item. The liability-side residual is ≥ −€2
  on 99.84% of rows (`reports/data/01_profile.md` §5), so gross violations are rare.
- **(c) Change?** In **training only**, flag rows where equity + liabilities exceed total assets
  by more than 1% (at most ~0.16% of rows: the liability-side residual is below −€2 on only
  0.16%), plus any statement shorter than 12 months if the
  data-auditor can detect one (e.g. two statements for one firm in one `fs_year`; unknown
  whether this happens). Fit the core model with and without the flagged rows and keep whichever
  does better out-of-time. Log the result either way. The harness still scores every row
  (CLAUDE.md §2.1). Do **not** copy the R-33 rule literally: with a 1% tolerance it would drop
  20% of our rows because of the omitted item.

### L3. Cite the Moody's support we now have for D-004, D-019, D-020, D-025. **MUST**, 20 min (orchestrator, decision-log edit)
- **(a) Sources.**
  - Truncation: RiskCalc v1 truncated raw ratios at the 2nd/98th percentiles in estimation and
    testing (R-05, p. 55, fn. 42).
  - Levels vs trends: levels beat trends, and the latest year beats a 3-year average
    (R-05, pp. 38–40).
  - Macro: macro forecasting was deliberately left outside the model (R-05, p. 41).
    Firm-specific information dominates systematic information for middle-market default
    (R-10, Sec. 4.1, p. 21).
  - Collinearity: Spearman correlations on transformed inputs reach at most 0.66, and VIF 4–10
    is used as the warning zone (R-33, Sec. 4.3, Tables 7–8, fn. 13, p. 18).
  - Panel dependence: within-firm dependence widens standard errors but left RiskCalc's
    conclusions unchanged (R-01, Sec. 3.4.2, p. 22).
- **(b) What we do.**
  - D-019 winsorises at p1/p99, sourced only as "assumption (standard practice)".
  - D-020 rejects growth features.
  - D-004 uses no macro data.
  - D-025 sets the core rule at |ρ| < 0.70 and VIF < 5.
  - Our bootstrap is firm-clustered.
- **(c) Change?** No change to the methods. Add the citations to the "Source(s)" lines. This
  directly answers Roger's "statements without foundation" criticism. Wording for D-019:
  "Moody's RiskCalc v1 truncated at p2/p98 (R-05 fn. 42); we use p1/p99, an assumption."

### L4. Report power by size band, sector and year, and the in-sample vs out-of-time gap. **SHOULD**, 1–2 h (referee)
- **(a) Source.** RiskCalc 4.0 France reports AR by industry, size band and statement year
  (Tables 9–14, pp. 19–21). It reports real estate, dealers and not-for-profits separately, and
  advises against scoring single-project real-estate developers (Sec. 4.6, fn. 14, p. 22). It
  shows a k-fold and walk-forward in- vs out-of-sample AR gap of −0.5 points as evidence that the
  model is not overfitted (Sec. 4.7–4.8, pp. 22–23). R-10 also uses out-of-time performance as
  its no-overfitting evidence (Sec. 4.5, p. 22).
- **(b) What we do.** Per-year AUC and default counts are planned (N-07). Per-size and
  per-sector tables and the in-sample vs OOT gap are not yet in the scorecard (to our knowledge;
  the latest scorecard was not checked today).
- **(c) Change?** Add three small tables (AUC by `log_assets` quintile, by `ateco_group` incl.
  L real estate, by test year) and one in-sample vs OOT AUC line for the winning model. Show
  default counts in every cell (R-07).

### L5. Be explicit that our PD level comes from the sample only. **SHOULD**, 15 min (calibration-specialist, wording)
- **(a) Source.** Moody's sets RiskCalc France's one-year central default tendency (2.2%) by
  combining OECD provisioning data, French bank provisions, EBA default rates and the CRD
  sample, and keeps it above the sample default rate because samples miss defaults
  (R-33, Sec. 2.4, p. 9). Dwyer & Stein show that default databases can miss defaults, and that
  calibration needs a second step for the gap between database and true default rates
  (R-35, p. 798; ~10% of small-firm defaults missed in their example, pp. 806–807). Calibration
  in RiskCalc is a final mapping adjusted to an estimated population default rate
  (R-10, App. C.2.4, p. 34).
- **(b) What we do.** D-004: no external data. The cycle and level are handled in the
  calibration step using training-window default rates.
- **(c) Change?** No method change; this is wording. In the deck and decision log, write: "the
  level is set from the training data. Moody's triangulates with external data (R-33), which the
  assignment does not allow us to use, so our level is correct only to the extent that the
  provider's default flag is complete (cf. R-35)." This contradicts any claim that we "set a
  central tendency" in the Moody's sense.

### L6. Explain each firm the way RiskCalc does. **NICE**, 1–2 h (report-writer / harness-engineer)
- **(a) Source.** For each firm, RiskCalc 4.0 France shows each ratio's percentile in the
  development sample, coloured by risk, plus a relative sensitivity and a relative contribution
  versus an average firm (Sec. 5.2–5.4, pp. 27–29).
- **(b) What we do.** CLAUDE.md §5.8 requires per-firm top drivers. SHAP is planned by family
  (FEATURE_SPEC).
- **(c) Change?** For the deck's example firms, show "ratio value, training percentile,
  contribution to log-odds vs the average firm". Credit officers already know this format.
  Percentiles must come from **training** quantiles stored in the params file, never from the
  holdout.

### L7. Ratio candidates from RiskCalc 4.0 France that we do not use. **NICE** (full set only), 1 h; **NO** for the core before Nov 1
- **(a) Source.** Debt coverage is the most heavily weighted family in R-33 (Table 4, p. 14).
  It is defined as (profit + D&A)/(liabilities − cash and securities): cash is netted out and the
  denominator is all liabilities. Profitability enters as operating margin, not net ROA.
  Personnel expenses/sales and interest expense/sales are activity inputs (Table 3, p. 11).
- **(b) What we do.**
  - `cfo_to_debt` = (profit + D&A)/(debt_st + debt_lt) is in the full set only. It was kept out
    of the core because its sign flips next to `roa_net` (D-022.3).
  - Interest enters as `interest_to_assets` and `interest_coverage`.
  - Personnel cost is not a field, but under the D-010 *hypothesis* that `goodwill` = value
    added, (value added − EBITDA) ≈ personnel cost.
- **(c) Change?** Not in the core before Nov 1 (the feature set is frozen per Roger M12). If
  time allows, the GBM challenger can add (i) a net-of-cash debt-coverage variant and
  (ii) personnel cost/revenue, the latter labelled as hypothesis-dependent. In the deck, one
  sentence defends the divergence: "RiskCalc France lets debt coverage carry its biggest weight
  next to operating margin; in our data the cash-flow coverage ratio duplicates net ROA
  (Spearman 0.80), so the core uses interest coverage instead."

### L8. Growth / trend features. **NO** change
- **(a) Source.** Levels dominate trends, and the latest level beats a 3-year mean
  (R-05, pp. 38–40). On the other hand, RiskCalc France still gives growth (sales growth,
  change in ROA, both U-shaped) 13% of the weight (R-33, Table 4, Fig. 5).
- **(b) What we do.** D-020 rejects growth features: a measured +0.002 AUC on seen firms, plus
  an unanswered rule question.
- **(c) Change?** None. Cite R-05 as support, and R-33 as the honest cost: Moody's does use
  growth.

### L9. Industry handling and excluded firm types. **NO** change (one sentence in the deck)
- **(a) Source.** Industry dummies add little power but improve calibration a lot, and they
  correct ratios whose meaning differs by sector, e.g. zero inventory in services or
  construction (R-01, Sec. 3.3, pp. 17–18). RiskCalc France's development data exclude firms
  with sales < €500k, financial institutions, real-estate developers, public/not-for-profit
  entities and start-ups in their first two years (R-33, Sec. 2.2, p. 6).
- **(b) What we do.** D-023: ATECO section dummies, pooled when a section has < 30 defaults.
  We must score every firm.
- **(c) Change?** None to the model. Report sector-level AUC (L4) and say that RiskCalc excludes
  some of these firm types where we cannot.

### L10. Statement age and new firms. **NICE** (limitation text only)
- **(a) Source.** Model power decays as statements get older (R-10, Highlight 6, Sec. 4.1, p. 21).
  Start-ups' first two statements are too volatile to use (R-33, p. 6).
- **(b) What we do.** The label starts 6 months after the statement date (00_labels.md). New
  firms have lower AUC (D-020 side finding: 0.739 vs 0.807).
- **(c) Change?** List both as known limitations. No model change.

### L11. Size in real terms. **NO** change
- **(a) Source.** Size is deflated by CPI (R-05, p. 61; R-33, "real total assets").
- **(b) What we do.** `log_assets` is nominal.
- **(c) Change?** None. CPI would be external data (not cleared, D-004), and over 2007–2012 the
  effect is small relative to the cross-section. Say so in one line if asked.

## 3. Where the literature contradicts or pushes against our choices

1. **Median fill (D-021c) is the practice Moody's says biases PDs down (R-34).** It is mitigated
   by the identity rebuild and the debt-split flag, but the core model has no missingness flag.
   Action: L1.
2. **Moody's cleans its training data by excluding implausible statements and some firm types
   (R-33, R-10). We train on everything.** This is a defensible choice, because the harness
   must score everything, but it is untested. Action: L2.
3. **Moody's sets the PD level with external data (R-33, p. 9). We cannot.** State this as a
   limitation. Action: L5.
4. **Debt coverage (a cash-flow/liabilities ratio) is RiskCalc France's top-weighted family, but
   our core excludes `cfo_to_debt`.** We have data-based reasons (D-022.3). Defend this in one
   sentence (L7).
5. Minor: Moody's truncated at p2/p98 (R-05) and we use p1/p99. Moody's uses real assets and we
   use nominal. Both are fine if labelled as our choices.

## 4. What we could not verify

- **No general RiskCalc 4.0 methodology document and no RiskCalc Italy document (3.1 or 4.0).**
  Several searches found only the France 4.0 paper (R-33), a US 4.0 press mention and an
  Australia 4.0 event page (neither read). R-03 stays SECONDARY. Do not state Italy's ratios,
  weights or version history.
- The only Italy-specific Moody's statement verified is R-04, slide 13: the Italy (and Spain,
  Portugal) models combine public-firm data with the national unemployment rate in the cycle
  adjustment. No formula or weights are given.
- **Dwyer & Stein (2004) Technical Document (R-02):** still no copy. Its content on
  transformations and missing data is unknown.
- **A Dwyer paper on overfitting or ratio selection:** searched, not found. Do not cite one.
- Dwyer, Wang & Fang (2012) *RiskCalc v3.1 France* (R-36) and Dwyer & Eggleton (2009) *Level and
  Rank Order Validation* (R-37) are known only from R-33's reference list.
- R-34's claim that mean imputation underestimates PDs has **no numbers** behind it in the
  article. Quote it as Moody's statement.
- Whether our data contain sub-12-month statements or interim statements: unknown. This is a
  data-auditor question, not a literature one.

## 5. AI-use row (for the orchestrator to merge into `docs/AI_USE_LOG.md`)

| Date | Tool / agent | Task | What the AI produced | How a human verified it (who) |
|---|---|---|---|---|
| 2026-10-30 | Claude Code (literature-scout sub-agent) | Targeted search and reading of Moody's / Moody's KMV / Moody's Analytics private-firm PD documents on feature engineering and data cleaning (RiskCalc 4.0 France; RiskCalc v3.1 and v1 re-read; Stein et al. 2003; Dwyer & Stein 2006; Moody's 2020 webinar; Moody's 2021 web article) | New/updated rows R-04, R-10, R-33..R-37, notes N-01b, N-04, N-05b, N-10, N-33..N-35, claim-map rows and "claims we must not make" 14–18 in `docs/REFERENCES.md`; this digest with lessons L1–L11 | Pending: a team member should open R-33 (pp. 4, 6, 9, 11, 14) and R-34 and check the paraphrases before any is used in the deck (initials + date) |
