# REFERENCES — verified bibliography and claim map

Status meanings (see `.claude/agents/literature-scout.md`):
- **VERIFIED**: located (URL/DOI recorded), bibliographic details match, and each claim we use
  was checked against the text, with section/page noted.
- **LOCATED**: existence and bibliographic details confirmed from a reliable page, but the
  claims we want to use have NOT yet been checked against the full text. Do not cite claims yet.
- **SECONDARY**: only a secondary mention found (citing source named). Do not cite claims.
- **TO-VERIFY / UNVERIFIED**: not yet located, or located but nothing checked.

Status as of **Oct 30, 2026** (literature-scout, Day 1 pass; **Day 1b pass** on Moody's feature
engineering / data cleaning added R-33..R-37, upgraded R-04 and R-10, and added notes N-01b, N-04,
N-05b, N-10, N-33..N-35; digest in `docs/LIT_moodys_feature_cleaning_2026-10-30.md`). Page numbers
are the printed page numbers of the document at the recorded URL unless "pdf p." is stated.

## Contradictions with the team's plan (read first)

1. **"RiskCalc is the industry standard" is contradicted by R-05.** Falkenstein et al. (2000,
   p. 10) say commercial lending then had *no* widely used benchmark model, and the stated
   *goal* of RiskCalc was to create one (p. 2 box). Drop "industry standard". The only usage
   claim we can support: Moody's KMV *reported* RiskCalc v1.0 in use at >200 institutions
   (R-01, p. 5) — a vendor statement, attribute it as such.
2. **Brier score is NOT a calibration measure according to BCBS WP14 (R-16, p. 32).** WP14
   says Brier should be read as a residual sum of squares that mainly reflects discriminatory
   power, and lacks test procedures. CLAUDE.md §1 lists Brier under "Calibration". Reword:
   report Brier as "overall probability accuracy"; carry the *level* story with
   calibration-in-the-large, binned observed-vs-predicted, and log-likelihood. Also note Stein
   (2007, p. 89): likelihood compares models *relative* to each other; it does not certify a
   model as well calibrated.
3. **Stein (2005) does not recommend cut-offs** (p. 1233): cut-offs are widely practised but
   sub-optimal; probability-based pricing is preferred. Our "approve/decline" framing is fine
   only if presented as the coarse special case of pricing, and the bank's use of a cut-off is
   an *assumption* (or taken from the assignment's transaction flow, R-30).
4. **Basel default ≠ our label.** Basel II para. 452 defines default as unlikely-to-pay OR >90
   days past due; our label is the assignment's "missed principal or interest payment". Say
   our PD is *aligned with* the IRB PD concept, not identical. Para. 452 fn. 89 lets "one
   member country" use up to 180 days for corporates (transitional) — the text does not name
   the country; do not say it was Italy.
5. **Prior-shift correction (R-22) assumes class-conditional densities p(x|default) are
   unchanged** (Sec. 2.2). Over a credit cycle firms' ratios themselves move, so a pure
   prior-shift recalibration is an approximation; calibration-specialist should state this.
6. **Altman's original Z-score needs market value of equity** (R-32 pdf p. 9, 22–25); it is a
   public-firm model. For private Italian firms use the book-equity Z' re-estimation, or just
   cite Altman as lineage.
7. **Lessmann et al. (2015) data are retail credit scoring** (R-17, Sec. 1), not corporate.
   They also say beating LR alone no longer shows progress; RF should be a benchmark (p. 25).
8. **Filling missing inputs with a mean/median biases PDs downward, per Moody's (R-34).** Moody's
   says its earlier RiskCalc versions imputed missing inputs from region/sector means and that
   this would systematically understate default probabilities (asserted, no numbers given).
   RiskCalc v1 itself filled a missing input with the mean transformed value (R-05, pp. 60–61).
   Our D-021 fills residual NaN with the training sector median in the GLM view: acceptable for
   fields missing on < 0.1% of rows, but the harness will meet messy holdout rows. Check
   calibration on rows with imputed core inputs (digest lesson L1).
9. **Moody's sets the level (central default tendency) with external data we cannot use.**
   RiskCalc 4.0 France triangulates the one-year CDT from OECD provisioning, French bank
   provisions and EBA default rates, and forces it to exceed the development-sample default rate
   because samples miss defaults (R-33, Sec. 2.4, p. 9; R-35, p. 798). Under D-004/CLAUDE.md §2.6
   our level must come from the training data only; say so explicitly and do not imply we
   "triangulated" a central tendency.
10. **"RiskCalc 4.0" was read only in its France edition (R-33).** No general RiskCalc 4.0
    methodology and no RiskCalc Italy document (3.1 or 4.0) was found. Never attribute the France
    ratio list, weights or CCA formula to Italy or to "RiskCalc 4.0" in general.

## Bibliography

| ID | Reference | Status | Where located | Intended use / claim to check |
|---|---|---|---|---|
| R-01 | Dwyer, D. W., Kocagil, A. E., & Stein, R. M. (2004, April 5). *Moody's KMV RiskCalc v3.1 Model: Next-Generation Technology for Predicting Private Firm Credit Risk.* Moody's KMV. | **VERIFIED** | https://www.rogermstein.com/wp-content/uploads/RiskCalc-v3-1-Model.pdf (author's site; 36 pp.) | See notes N-01, **N-01b** (industry, data quality, panel dependence — Day 1b) |
| R-02 | Dwyer, D. W., & Stein, R. M. (2004). *Technical Document on RiskCalc v3.1 Methodology.* Moody's KMV. | LOCATED (listed in R-01 ref. 7, p. 35; in R-33 references p. 31; on rogermstein.com publication list without link). Re-searched Day 1b: still no copy | no copy found | Detailed methodology — do not cite specifics |
| R-03 | Moody's KMV. *RiskCalc v3.1 Italy* (model documentation). | SECONDARY (TOC of a private-company valuation book lists "9.1 RiskCalc Italy"; R-04 slide 13 now confirms an Italy model exists in the European suite; a press release (insidearm.com) says Capitalia used "RiskCalc 3.1 Italia" — press release not read) | no readable copy (re-searched Day 1b) | Italy ratio families — **not available; do not claim which ratios the Italy model uses** |
| R-04 | Buitrago, M., & Makarov, U. (2020, May 6). *Tracking the Impact of Covid-19 on Credit Risk: European Private Firms* (webinar slides). Moody's Analytics. | **VERIFIED** (Day 1b re-read) | https://ma.moodys.com/rs/961-KCJ-308/images/Moody%27s%20Analytics%20Webinar%20Tracking%20the%20Impact%20of%20Covid-19%20on%20Credit%20Risk%20-%20Europe.pdf | See note N-04 (slides 13–15) |
| R-05 | Falkenstein, E., Boral, A., & Carty, L. V. (2000, May). *RiskCalc for Private Companies: Moody's Default Model.* Moody's Investors Service, Rating Methodology. | **VERIFIED** | https://efalken.com/papers/riskcalc.pdf (first author's site; listed at efalken.com/papers/index2.html) | See notes N-05, **N-05b** (statement quality, levels vs trends, truncation, missing inputs — Day 1b) |
| R-06 | Sobehart, J. R., Keenan, S. C., & Stein, R. M. (2000). *Benchmarking Quantitative Default Risk Models: A Validation Methodology.* Moody's Investors Service. | LOCATED (author's site link found; not read) | http://www.rogermstein.com/wp-content/uploads/53621.pdf | Walk-forward; CAP/AR origin |
| R-07 | Stein, R. M. (2007). Benchmarking default prediction models: Pitfalls and remedies in model validation. *Journal of Risk Model Validation*, 1(1), 77–113. (orig. Moody's KMV TR #030124, 2002) | **VERIFIED** | http://www.rogermstein.com/wp-content/uploads/BenchmarkingDefaultPredictionModels_TR030124.pdf (JRMV typeset version) | See notes N-07 |
| R-08 | Stein, R. M. (2005). The relationship between default prediction and lending profits: Integrating ROC analysis and loan pricing. *Journal of Banking & Finance*, 29(5), 1213–1236. doi:10.1016/j.jbankfin.2004.04.008 | **VERIFIED** | http://www.rogermstein.com/wp-content/uploads/JBF_2026.pdf (published JBF typeset) | See notes N-08 |
| R-09 | Bohn, J. R., & Stein, R. M. (2009). *Active Credit Portfolio Management in Practice.* Wiley. ISBN 978-0-470-08018-4. | LOCATED | Wiley product page | Not read |
| R-10 | Stein, R. M., Kocagil, A. E., Bohn, J., & Akhavein, J. (2003, February). *Systematic and Idiosyncratic Risk in Middle-Market Default Prediction: A Study of the Performance of the RiskCalc and PFM Models.* Moody's Special Comment (cover: Moody's Investors Service, Global Credit Research; R-01 lists it as Moody's KMV). | **VERIFIED** (Day 1b; read pp. 1–10, 14–15, 21–22, 26–30, 33–34) | http://www.rogermstein.com/wp-content/uploads/riskcalcpfm.pdf (author's site; 40 pp.) | See note N-10. NB first-page author spelling is "Jalal Akhavein"; Stein's site spells "Akhavain" |
| R-11 | Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance*, 23(4), 589–609. doi:10.1111/j.1540-6261.1968.tb00843.x | LOCATED (Crossref metadata matches; full text paywalled) | api.crossref.org/works/10.1111/j.1540-6261.1968.tb00843.x | For method/sample details cite R-32 (Altman's own restatement) |
| R-12 | Ohlson, J. A. (1980). Financial ratios and the probabilistic prediction of bankruptcy. *Journal of Accounting Research*, 18(1), 109–131. doi:10.2307/2490395 | LOCATED (Crossref metadata matches; JSTOR paywalled) | api.crossref.org/works/10.2307/2490395 | "Logit on ratios; size effect" **not checked** — cite only as existence of logit lineage |
| R-13 | Shumway, T. (2001). Forecasting bankruptcy more accurately: A simple hazard model. *Journal of Business*, 74(1), 101–124. | TO-VERIFY | — | Not attempted today |
| R-14 | Engelmann, B., Hayden, E., & Tasche, D. (2003). *Measuring the Discriminative Power of Rating Systems.* Deutsche Bundesbank Discussion Paper, Series 2: Banking and Financial Supervision, No. 01/2003. | **VERIFIED** | https://bundesbank.de/resource/blob/704150/b9fa10a16dfff3c98842581253f6d141/472B63F073F071307366337C94F8C870/2003-10-01-dkp-01-data.pdf | See notes N-14. **Cite this DP, not the *Risk* (2003) article** — the *Risk* piece "Testing rating accuracy" was not located (UNVERIFIED); Stein (2007, p. 83 fn. 7) cites "Engelmann et al (2003)" for the identity |
| R-15 | Basel Committee on Banking Supervision (2006, June). *International Convergence of Capital Measurement and Capital Standards: A Revised Framework — Comprehensive Version.* BIS. | **VERIFIED** | https://www.bis.org/publ/bcbs128.htm → https://www.bis.org/publications/200606-standards-basel-ii-international-convergence-capital-measurement-and-capital-standards-revised-framework.pdf | See notes N-15 |
| R-16 | BCBS (2005, May, revised). *Studies on the Validation of Internal Rating Systems.* Working Paper No. 14. | **VERIFIED** | https://www.bis.org/publ/bcbs_wp14.htm → https://www.bis.org/publications/studies-validation-internal-rating-systems-revised.pdf | See notes N-16 |
| R-17 | Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *EJOR*, 247(1), 124–136. doi:10.1016/j.ejor.2015.05.030 | **VERIFIED (manuscript version)** | Crossref metadata; text read in manuscript "…A ten-year update" at https://lirias.kuleuven.be/retrieve/319952 (same authors). Accepted MS also at eprints.soton.ac.uk/377196 (403 today) | See notes N-17. Page refs are manuscript pages, not EJOR pages |
| R-18 | Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*. | TO-VERIFY | — | Not attempted |
| R-19 | Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML '05*, 625–632. doi:10.1145/1102351.1102430 | **VERIFIED (abstract-level claims only)** | https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting (cs.cornell.edu PDF failed: certificate error) | See notes N-19 |
| R-20 | Platt, J. (1999). Probabilistic outputs for support vector machines… | TO-VERIFY | — | Not attempted |
| R-21 | Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD '02*. | TO-VERIFY | — | Not attempted |
| R-22 | Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. *Neural Computation*, 14(1), 21–41. doi:10.1162/089976602753284446 | **VERIFIED** | https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf (ULB repository preprint; preprint page numbers) | See notes N-22 |
| R-23 | Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NIPS 2017* (31st Conf.). arXiv:1705.07874 | **VERIFIED** | https://arxiv.org/abs/1705.07874 (v2) | See notes N-23. **Does not contain TreeSHAP** |
| R-24 | Lou, Y., Caruana, R., & Gehrke, J. (2012). Intelligible models for classification and regression. *KDD '12*. | TO-VERIFY | — | Not attempted |
| R-25 | Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML. arXiv:1909.09223. | TO-VERIFY | — | Not attempted |
| R-26 | Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: a review. *JRSS A*, 160(3), 523–541. doi:10.1111/j.1467-985X.1997.00078.x | LOCATED (Crossref metadata + abstract) | api.crossref.org | Abstract does not mention reject inference; **reject-inference content not checked** |
| R-27 | Italian Civil Code, art. 2435-bis (*bilancio in forma abbreviata*). | **VERIFIED (current text)**; 2007–2012 thresholds SECONDARY | https://www.filodiritto.com/codici/codice-civile/capo-v-societa-azioni/art-2435-bis ; thresholds history: ecnews.it / uniupo.it (secondary) | See notes N-27 |
| R-28 | Italian Civil Code, art. 2364 co. 2 (ordinary meeting within 120 days of year-end; statute may allow up to 180). Srl: art. 2478-bis refers to the same limits. | **VERIFIED (current text)**; in-force-since-2003 SECONDARY | https://brocardi.it/codice-civile/libro-quinto/titolo-v/capo-v/sezione-vi/art2364.html ; history: fiscoetasse.com/approfondimenti/12173 (secondary) | See notes N-28 |
| R-29 | Italian Civil Code, art. 2435 co. 1 (directors file approved accounts with the Registro delle imprese within 30 days of approval). | **VERIFIED (current text)**; historical wording not checked | https://www.brocardi.it/codice-civile/libro-quinto/titolo-v/capo-v/sezione-ix/art2435.html | See notes N-28 |
| R-30 | Assignment: *Term Project Instructions*, DS-GA 3001, Oct 11, 2026. | VERIFIED | course file | Business need, transaction flow, harness rules, evaluation criteria |
| R-31 | Team Turquoise proposal with RMS feedback. | VERIFIED | course file | Feedback items (FEEDBACK_RESPONSE.md) |
| R-33 | Buitrago, M. A., Makarov, U., Zhao, J. Y., & Dwyer, D. (2015, December). *RiskCalc 4.0 France.* Modeling Methodology, Moody's Analytics Quantitative Research Group (ed. C. Crossen). | **VERIFIED** (all 32 pp. read Day 1b) | http://ma.moodys.com/rs/961-KCJ-308/images/RiskCalc_40_France_FINAL.pdf | See note N-33. **The only RiskCalc 4.0 document read. France, not Italy** |
| R-34 | Moody's (2021, December 2). *Advances in default detection and early warning.* Credit Risk Insights web article (no named author). | **VERIFIED (web page; claims are assertions without numbers)** | https://www.moodys.com/web/en/us/insights/credit-risk/advances-in-default-detection-and-early-warning.html | See note N-34 |
| R-35 | Dwyer, D. W., & Stein, R. M. (2006). Inferring the default rate in a population by comparing two incomplete default databases. *Journal of Banking & Finance*, 30(3), 797–810. doi:10.1016/j.jbankfin.2005.11.001 | **VERIFIED** (abstract, Sec. 1 p. 798, Sec. 5–6 pp. 806–807) | http://www.rogermstein.com/wp-content/uploads/InferringDefaultRate_JBF1.pdf (published typeset). Stein's site dates it 2005; journal says 2006 (received Oct 2005) | See note N-35 |
| R-36 | Dwyer, D., Wang, Y.-J., & Fang, F. (2012). *Moody's Analytics RiskCalc v3.1 France.* Moody's Analytics. | LOCATED (R-33 references, p. 31 only) | no copy found | Not read; do not cite |
| R-37 | Dwyer, D., & Eggleton, D. (2009). *Level and Rank Order Validation of RiskCalc v3.1 United States.* Moody's KMV. | LOCATED (R-33 references, p. 31 only) | no copy found | Not read; do not cite |
| R-32 | Altman, E. I. (2000, July). *Predicting Financial Distress of Companies: Revisiting the Z-Score and ZETA Models.* Working paper, NYU Stern (adapted from Altman 1968 and Altman, Haldeman & Narayanan 1977). | **VERIFIED** | Author's page pages.stern.nyu.edu/~ealtman/Zscores.pdf (HTTP 500 today); text read from mirror https://moodle2.units.it/pluginfile.php/322514/mod_resource/content/0/Zscores.pdf (unpaginated; pdf pages cited) | See notes N-32 |

## Verified notes (claims, location, what we take, what we do differently)

**N-15 Basel II (R-15)**
- Para. 211 (p. 52): IRB banks may use their own estimates of the risk components; these are
  PD, LGD, EAD and effective maturity M.
- Para. 271 (p. 63): risk-weighted assets for corporates depend on PD, LGD, EAD (and sometimes M).
- Para. 273 (p. 64): optional firm-size adjustment for SME corporates (consolidated sales < €50m).
- Para. 285 (p. 67): corporate PD = the one-year PD of the borrower's internal grade, with a
  floor of 0.03%; defaulted grades get PD = 100%. NB: the PD is per **grade**, not per obligor.
- Para. 444 (p. 98): for IRB banks, internal ratings and default/loss estimates must play an
  essential role in credit approval, risk management, capital allocation and governance; it
  also notes pricing models may use PDs that differ from the IRB ones.
- Para. 447 (p. 99): PD estimates must be a long-run average of one-year default rates per grade.
- Para. 452 (p. 100): default = bank judges obligor unlikely to pay in full without recourse to
  security, and/or obligor >90 days past due on a material obligation. Fn. 89: one (unnamed)
  member country may use up to 180 days for corporates for a 5-year transition.
- Para. 461–463 (p. 102): PD may be estimated by internal default experience, external mapping,
  or statistical default models (average of borrower-level model PDs within a grade); at least
  five years of data for at least one source.
- *What we take:* the one-year horizon, the PD/LGD/EAD split, and the link to grade-level
  long-run PDs. *What we differ on:* we estimate firm-level PDs with the assignment's
  missed-payment label, not the Basel default definition; we are not building a capital model.

**N-01 RiskCalc v3.1 (R-01)**
- p. 5: Moody's KMV says RiskCalc v1.0 was in use at more than 200 institutions (vendor claim).
- Sec. 3.1, p. 12: FSO mode uses only financial statements (+ industry) and updates only when
  new statements arrive; the complete version adds forward-looking market information.
- Sec. 3.1, p. 13: ratios grouped into profitability, leverage, debt coverage, growth,
  liquidity, activity, size; at least one ratio per group; too many ratios overfit.
  Expected signs given (e.g. leverage raises PD; growth is two-sided).
- Sec. 3.1, p. 14 (eq. + fn. 11): FSO EDF = F(Φ(Σ β_i T_i(x_i) + Σ γ_j I_j)); T_i are
  non-parametric univariate transforms, I_j industry indicators, F a final transform; the
  authors say this is closely related to a GAM. Fn. 8: transforms reduce the impact of
  manipulated or noisy ratios.
- Sec. 3.2, p. 16: private-firm statements usually reach lenders 3–4 months after the
  statement date (US/CRD context); the full model uses sector distance-to-default to lead them.
- Sec. 4.1, p. 25: validation is along two separate dimensions, power and calibration.
- Sec. 4.2–4.3, pp. 26–28: walk-forward testing (fit up to year t, score t+1, roll forward,
  pool out-of-time predictions) plus k-fold; VIFs used to limit collinearity (p. 26).
- Tables 5–6 (pp. 27, 30): power reported as AR, calibration as log-likelihood lead.
- Sec. 4.4, Table 7 (p. 30): AR reported per year across the cycle.
- Sec. 5, p. 31: more powerful models more profitable; banks using pricing gain more than
  those using cut-offs; weaker models suffer adverse selection in competition (summarising
  Stein 2003 / Stein & Jordão 2003).
- *What we take:* the architecture (per-ratio transforms → probit → final calibration), the
  ratio families, the per-year AR table, walk-forward design. *What we do differently:* no
  market data (FSO-like only), Italian data with the assignment's label, and we compare the
  RiskCalc-style model against ML challengers on frozen splits.

**N-01b RiskCalc v3.1 — industry, data quality, panel dependence (R-01, Day 1b)**
- Sec. 3.3, pp. 17–18 + Table 2: adding industry controls to the FSO model raised one-year AR
  only modestly (54.4% → 55.1%) but improved log-likelihood strongly; industry dummies also
  correct ratios whose meaning differs by sector. Table 3 example: in services, construction,
  mining/transport/utilities 40%+ of firms carry no inventory, so "low inventory" should not
  earn credit there.
- Sec. 3.4.1, pp. 19–20 + Table 4: re-estimating the v1.0 variable set on the larger, cleaner
  CRD raised out-of-sample one-year AR from 48.2% to 51.7% (5-year 40.1% → 45.5%) with no
  change of variables — cleaner data alone improved the model.
- p. 21: default misclassification (defaults recorded as non-defaults after mergers, system
  migrations, subsidiary/parent mapping) was detected with Hausman-type checks: e.g. many
  defaults among firms the model rates safest is a red flag; one lender's post-acquisition
  defaults were missing and were corrected.
- pp. 21–22 + fn. 20: Benford's-law screening found excessive rounding at some banks; the model
  was estimated with and without flagged rows and performed *better* keeping them, so they were
  kept but flagged.
- Sec. 3.4.2, p. 22: with panel data, ignoring within-firm dependence can mislead significance
  tests; GEE estimates enlarged standard errors but left conclusions and rank ordering
  unchanged. pp. 22–23: a Cox duration model ranked firms very similarly; the one-year probit
  was slightly more powerful at one year.
- *Take:* data cleaning is worth power; test each filter by out-of-time performance rather than
  assuming that removing suspect rows helps; industry dummies mainly fix the level. *Differ:* we
  cannot drop rows at prediction time (harness must score every row).

**N-04 Moody's 2020 webinar (R-04)** — slide numbers
- Slide 13: the RiskCalc suite has fifteen European country models for middle-market firms plus
  a Large Firm Europe model (> €50MM assets); most use a public-firm-based credit cycle
  adjustment (CCA); **Spain, Italy and Portugal models combine public-firm data with the
  national unemployment rate**.
- Slide 14: CCA compares a sector's current public-EDF credit quality with its historical
  average and moves the FSO EDF up or down accordingly.
- Slide 15: CCA is a transformation of public-firm distance-to-default by industry, made
  region/country/industry-specific; inputs are refreshed monthly with a lag (e.g. end-March
  DDs and February unemployment reflected from 1 May).
- *Take:* the cycle is handled as a level adjustment on top of a financial-statement-only score,
  which is our structure too (D-004, calibration step). *Differ:* we have no market or
  unemployment data (not cleared, CLAUDE.md §2.6); our cycle handling is in-sample only.

**N-05 RiskCalc for Private Companies (R-05)**
- p. 1 (summary): ratio–default relationships differ between public and private firms, so
  public-firm models may misstate private-firm risk; model uses 10 ratios from 17 inputs.
- p. 2 (box): the stated aim was to build a *benchmark* for commercial loans.
- p. 9: 9 ratios + size are transformed and combined in a probit; output mapped to 1- and
  5-year EDFs; private firms differ from public ones (smaller, less leverage, etc.); the model
  supports, not replaces, analyst judgement. Fn. 1: the public/private split is the existence
  of liquid equity prices.
- p. 10: at the time, middle-market lending was largely subjective and **no widely used
  benchmark existed**; earlier academic default studies used a median of ~40 defaults.
- *What we take:* private-firm-specific estimation and the transform-then-probit design; the
  "supplement, not substitute" framing for credit officers. *What we do differently:* see N-01.

**N-05b RiskCalc for Private Companies — data and features (R-05, Day 1b)**
- p. 25 (Exhibit 3.5): CRD statements carry a quality type, grouped as audit 28%, compile 48%,
  review 15%, tax return 7%, unknown 2%; lenders' private-firm data are mostly unaudited.
- pp. 38–39 (Exhibits 4.9–4.10): for prediction, ratio *levels* beat their *trends*; in probits
  with level + trend, the level dominates (e.g. NI/assets trend insignificant). Sales growth and
  NI growth are U-shaped. Trends are useful for monitoring, less for prediction; 8 of the 10
  RiskCalc inputs are levels.
- p. 40 (Exhibits 4.11–4.12): the latest year's ratio predicts better than the 3-year average;
  the authors stress getting the most recent statement. (Public-firm data.)
- pp. 40–41 (Exhibits 4.13–4.15): audited private firms default less (≈30% lower), but audit
  status is highly correlated with size and its power is mostly absorbed by size in the
  multivariate model, so it was left out.
- p. 41: macroeconomic data and (then) industry were deliberately left outside the model;
  macro forecasting is a time-series problem, separate from cross-sectional ranking.
- p. 55 fn. 42: raw ratios are truncated at the 2nd and 98th percentiles in estimation and
  testing, which reduces outlier influence and improves linear models.
- pp. 55–57: transforms are non-parametric univariate default-frequency curves (on percentiles),
  checked by regressing probit residuals on polynomial terms (Exhibit 6.1); the stored
  transform is a lookup of 50 points with linear interpolation (p. 57).
- pp. 60–61 (Appendix 6B): none of the inputs is mandatory; a missing input takes the **mean
  transformed value**; the authors say less real data gives a worse output.
- p. 61: size = total assets deflated by CPI.
- *Take:* levels-only features (D-020), truncation at fixed training percentiles (D-019), a
  frozen lookup transform for the harness, and that a missingness/quality signal can be
  absorbed by size. *Differ:* we keep a missing flag next to the imputed value (cf. R-34); we
  use nominal assets (CPI is external data, not cleared).

**N-07 Stein (2007) JRMV (R-07)**
- Sec. 1, pp. 77–78: validation has two dimensions, power and calibration.
- Sec. 2.1.2, p. 81: CAP and ROC carry the same information, framed for different questions.
- p. 83 + fn. 7: area under ROC (A) summarises power and equals a Mann–Whitney statistic;
  AR = 2(A − 0.5), citing Engelmann et al. (2003). Two models can share A but differ in shape.
- p. 82 fn. 6: cut-offs suit simple decisions but are usually sub-optimal for profit; Stein
  (2005) prices along the whole ROC.
- Sec. 2.2, pp. 88–89: a likelihood criterion can prefer a useless but well-centred model over
  a powerful but mis-levelled one; a simple prior adjustment can fix the level of the powerful
  model; likelihood is a relative, not an absolute, calibration check.
- Sec. 2.3, pp. 89–91: power limits how much resolution calibration can achieve; monotone
  recalibration cannot add power.
- Sec. 3, pp. 91–96: out-of-sample *and* out-of-time testing preferred; walk-forward
  procedure in six steps (p. 94); reusing the same out-of-time data while developing breaks
  the out-of-time principle (p. 95).
- Sec. 4–4.1, pp. 96–99: metric variability is driven by the number of **defaults**, not total
  observations (Figs. 6–7); resampling recommended to size it; Bamber bound on Var(A) (p. 99).
- *What we take:* report AUC only (state AR = 2·AUC − 1), walk-forward splits, bootstrap CIs,
  per-year results, and default counts next to every metric. *What we do differently:*
  firm-clustered bootstrap because firms repeat across years.

**N-08 Stein (2005) JBF (R-08)**
- Sec. 1, p. 1214: many institutions find it helpful to define a lending cut-off; others want
  a rational pricing scheme (author's statement, not a survey).
- Sec. 2–3, pp. 1215–1226: optimal cut-off is where an iso-performance line (set by default
  rate, LGD, spread, fees) is tangent to the ROC; higher loss potential → more conservative.
- Sec. 3.2, p. 1224: relationship revenue justifies a looser cut-off for those clients.
- Sec. 4, p. 1227: the same machinery gives a break-even price for every score (no loan refused).
- Sec. 6, pp. 1231–1232: simulation on Moody's CRD, prototypical US bank (~$4.25B annual
  lending): switching to a more powerful model gains ~4.1 bps; competing banks with different
  models face adverse selection; gains larger under pricing than cut-offs.
- Sec. 7, pp. 1232–1233: powerful models cost less to operate; author does **not** recommend
  cut-offs but notes they are widely practised.
- *What we take:* the economic case for power, the cut-off vs pricing link, the competitive
  adverse-selection argument. *What we do differently:* we do not have spreads/fees/LGD data,
  so any profit illustration is a stylised assumption, labelled as such.

**N-14 Engelmann, Hayden & Tasche (2003) (R-14)**
- Abstract & Sec. III.2 eq. (17), p. 12; proof App. A, p. 23: AR = 2·AUC − 1, so AR and AUC
  carry the same information.
- Sec. III.1.3 and III.1.5 (pp. 9–11), IV.2–IV.3: confidence intervals for AUC and a test for
  comparing two models' AUCs on the same data.
- *Take:* answer to Roger's M13. *Differ:* we use a firm-clustered bootstrap rather than the
  analytic variance because firms repeat.

**N-16 BCBS WP14 (R-16)**
- Sec. II, p. 11: banks using ratings mainly for underwriting tend toward through-the-cycle
  (TTC) systems; banks using them for pricing or current risk tracking tend toward
  point-in-time (PIT); hybrids exist. p. 12: a PD is a forward-looking forecast over a fixed
  horizon, usually one year.
- Sec. III, p. 32: AR and ROC preferred for power, with CIs whose width grows as defaults fall;
  Brier score is **not** a calibration-accuracy measure (see Contradiction 2).
- Sec. III, pp. 32–34: calibration is harder to validate than power; binomial and
  Hosmer–Lemeshow tests understate type I error when defaults are correlated; no really
  powerful calibration tests exist.
- p. 35: open question of minimum default counts for validation.
- *Take:* frame calibration evidence as diagnostic, not a pass/fail test; the TTC/PIT framing
  for the central-tendency decision. *Differ:* we report binned calibration by year rather than
  per-grade supervisory tests.

**N-17 Lessmann et al. (2015) (R-17)** — manuscript pages
- Sec. 1, p. 2: credit scoring here concerns the retail sector; PD scorecards are built with
  classifiers.
- Conclusions, p. 25: recommend random forest as the benchmark; heterogeneous ensemble
  (HCES-Bag) may be harder to beat; LR is the industry standard but outperforming LR alone no
  longer signals progress; lab results may overstate real-world gains.
- *Take:* justification for an ML challenger and for benchmarking beyond LR. *Differ:*
  corporate, single-country panel with time-ordered splits and calibration focus.

**N-19 Niculescu-Mizil & Caruana (2005) (R-19)** — abstract (p. 625)
- Boosted trees/stumps push predicted probabilities away from 0 and 1 (sigmoid distortion);
  neural nets and bagged trees are reasonably calibrated; Platt scaling and isotonic
  regression are compared; after calibration boosted trees, RF and SVMs give the best
  probabilities.
- NOT checked: body text on calibration-set size (isotonic vs Platt with small data).
- *Take:* why the GBM challenger needs a post-hoc calibration step. *Differ:* calibration
  fitted only inside the training window (inner split).

**N-22 Saerens, Latinne & Decaestecker (2002) (R-22)** — preprint pages
- Sec. 2.2, eq. (4), pp. 5–6: if the within-class densities are unchanged, posteriors can be
  re-weighted by the ratio of new to old class priors and renormalised; no refitting needed.
- Sec. 2.3 / abstract: EM procedure estimates the new priors when unknown.
- *Take:* a principled central-tendency adjustment. *Differ:* the invariance assumption is
  doubtful across the 2008–12 cycle (Contradiction 5); we treat it as a sensitivity tool.

**N-23 Lundberg & Lee (2017) (R-23)**
- Sec. 2, Def. 1, p. 2: additive feature attribution = explanation is a linear function of
  binary feature-presence variables.
- Sec. 3, Theorem 1, p. 4: Shapley values are the only additive attribution satisfying local
  accuracy, missingness and consistency.
- *Take:* per-firm reason codes that add up to the model output. *Differ:* we explain in
  log-odds of PD; TreeSHAP needs a separate citation (Lundberg et al., later work — not verified).

**N-27/N-28 Italian Civil Code (R-27, R-28, R-29)** — current text
- Art. 2364 co. 2: ordinary meeting at least yearly, within the statute's term and no later
  than 120 days after year-end; statute may allow up to 180 days for groups preparing
  consolidated accounts or when structure/object require it, with reasons in the art. 2428
  report. Secondary source (fiscoetasse) dates the 120/180 rule to D.Lgs. 6/2003 and the
  "ovvero" wording to 2005; Srl follow the same limits via art. 2478-bis (secondary).
- Art. 2435 co. 1: directors file the approved accounts with the Registro delle imprese within
  30 days of approval.
- Art. 2435-bis (current): companies with no listed securities below two of three thresholds
  may file abbreviated accounts — balance sheet only at capital-letter/Roman-numeral item
  level, aggregated income-statement items, reduced notes, and possible exemption from the
  management report. Current thresholds are later than our data; secondary sources say
  4.4m assets / 8.8m revenue / 50 employees were set by D.Lgs. 173/2008. **Pre-2008 values
  not verified.**
- *Take:* for a December year-end, approval by about end-April (end-June if extended) and
  filing about 30 days later supports a label window starting months after fiscal year-end;
  abbreviated statements are a plausible cause of structurally missing line items. *Differ:*
  legal maxima ≠ actual filing dates; data-auditor must check against data (`stmt_date`).

**N-10 Stein, Kocagil, Bohn & Akhavein (2003) (R-10)**
- p. 2 Highlights + Sec. 2.1, p. 8: on North American CRD data RiskCalc (financial-statement
  model) beat PFM (public-equity-based comparables model) on every size, sector and period
  cut; median AR gap ~8 points; both beat Z-score.
- Highlight 6 + Sec. 4.1, p. 21: both models lose power as statements get older; PFM decays
  faster.
- Sec. 4.1, p. 21 + Conclusion p. 27: firm-specific (idiosyncratic) information appears to
  dominate systematic (industry/market) information for middle-market default prediction;
  systematic information matters more in cyclical industries (Sec. 2.2.2, p. 9; Fig. 5 p. 14).
- Sec. 4.2, p. 21: localized (country-specific) RiskCalc models beat the North American model
  abroad — local accounting/credit conventions matter.
- Sec. 2.3, p. 9 + Sec. 4.3, p. 22: after re-fitting each model's prior (base rate) to the test
  data, RiskCalc's likelihood was still better; the authors attribute part of that to power
  (weaker models have flatter calibration curves).
- Sec. 4.4, p. 22: simple averaging/thresholding of the two models' outputs did not add stable
  power.
- Sec. 4.5, p. 22: out-of-time performance after the data cut-off is offered as evidence
  against overfitting.
- App. A, p. 30 + fn. 19: Moody's KMV filters CRD submissions for GAAP violations,
  undeterminable default status and unmatched statements/obligations; of ~907k North American
  statements submitted only ~192k survived (> 80% filtered, mostly interim statements).
- App. C.2.1–C.2.5, p. 34: stepwise selection ranked by univariate power and checked in a
  multivariate setting; non-parametric transforms; probit/logit on transforms (GAM-like); final
  mapping adjusted to a population default-rate estimate to correct sample bias.
- *Take:* financial-statement-only modelling is the right core for private firms; cycle effects
  are second-order for ranking; strict input filtering; out-of-time evidence for no
  overfitting. *Differ:* we cannot filter holdout rows; our "localization" is the Italian data
  itself.

**N-33 RiskCalc 4.0 France (R-33)** — printed pages
- Sec. 1.2, p. 4: 4.0 keeps the 3.1 structure ("ratios and inputs proved robust"); changes:
  more data and better data cleansing; 14 industries instead of 9; marketable securities added
  to liquid assets; interest expense/sales replaces financial charges/sales; personnel
  expenses/sales replaces change in AR turnover; operating margin replaces EBITDA/sales; cash
  flow/financial charges dropped; CCA re-specified on public-firm distance-to-default with
  smoothing for thin sectors.
- Sec. 2.1, p. 6: default = start of legal proceedings (liquidation, receivership, cessation of
  payments, continuation plan) — local definition.
- Sec. 2.2, p. 6: development data exclude firms with sales < €500k, financial institutions,
  real-estate developers, public-sector/not-for-profit entities and start-ups (first two years'
  statements too volatile); statements failing plausibility checks (e.g. assets ≠ liabilities +
  net worth; period shorter than 12 months) are dropped.
- Sec. 2.3, p. 7 (Table 1): 4.0 sample 1993–2014, 643k+ firms, 76k+ defaults, 3.8m statements.
- Sec. 2.4, p. 9: development samples tend to show lower default rates than the population
  (purged defaulters, missed events); one-year CDT of 2.2% triangulated from OECD loan-loss
  provisions, French bank provisions, EBA default rates and the CRD sample, set to exceed the
  sample rate. p. 10: CCA-mode CDT equals FSO CDT when the cycle is neutral.
- Sec. 3, p. 10 eq. (1): FSO EDF = F(Φ(Σβ_i T_i(x_i) + Σγ_j I_j)); T_i and F non-parametric
  (local regression / density estimation, fn. 7); CCA changes only the final transform F.
- Sec. 3.1, p. 11, Table 2: seven groups (activity, debt coverage, growth, leverage, liquidity,
  profitability, size) with expected directions; growth two-sided; selection questions:
  availability, intuitive meaning, power, unambiguous definition, not-too-high correlation.
- Table 3, p. 11: France 4.0 inputs — activity: interest expense/sales, accounts
  payable/sales, personnel expenses/sales; debt coverage: (profit + D&A)/(liabilities − cash
  and marketable securities); growth: sales growth, change in ROA; leverage: equity/assets;
  liquidity: cash and marketable securities/assets; profitability: operating margin; size:
  real total assets.
- pp. 11–12 + Fig. 5 p. 13: transforms drawn against the ratio's percentile; shapes are
  monotone except growth (U-shaped); equity/assets and debt coverage are convex (steepest at
  the weak end).
- Sec. 3.2, p. 14, Table 4: weight = share of total EDF change when each input moves 1 s.d.
  from the average firm; 4.0 France weights: debt coverage 19.2%, leverage 17.8%, liquidity
  16.0%, activity 13.6%, growth 13.0%, size 10.4%, profitability 10.0%.
- Sec. 3.3, p. 15 + fn. 11: CCA = weighted average of French-industry and European-region DD
  factors; weight on France = 1/(1 + 20/N) with N French public firms in the industry
  (Vasicek-style shrinkage).
- Sec. 4.1, p. 16 + fn. 12: one-year AR 69.1% (CCA) / 70.5% (FSO) vs 3.1's 67.8% / 69%.
- Sec. 4.3, p. 18, Tables 7–8: Spearman correlations on transformed inputs, max 0.66
  (equity/assets vs debt coverage); VIFs ≤ 3.04; fn. 13 uses VIF 4–10 as the warning zone.
- Sec. 4.4–4.6, pp. 18–22, Tables 9–15: AR by industry, by size band, by year; real estate,
  dealers and not-for-profits reported separately (one-year AR 59.3% / 70.2% / 62.3%) and not
  recommended for scoring; fn. 14 warns against scoring single-project real-estate developers.
- Sec. 4.7–4.8, pp. 22–23: 5-fold and walk-forward (from 2000) out-of-sample tests; in- vs
  out-of-sample AR gap −0.5 points (one-year).
- Sec. 4.9, p. 24 + Fig. 11, p. 25: FSO EDF averages roughly flat over time; CCA EDF follows
  the cycle (peaks 2009). Fig. 12, p. 26: equity/assets and cash/assets drift upward over
  1993–2012.
- Sec. 5.2–5.4, pp. 27–29: per-firm explanation tools — percentile of each ratio in the
  development sample, relative sensitivity, and relative contribution versus an average firm.
- *Take:* see digest. *Differ:* Italy, no CCA, in-sample level only, no growth inputs.

**N-34 Moody's 2021 web article (R-34)**
- Moody's says a missing input is now estimated so as to neither penalise nor reward the firm;
  earlier RiskCalc versions imputed missing inputs from region and sector means, which Moody's
  says would systematically understate PDs — the reason it kept minimum-input rules. Two
  sub-models estimate total assets and leverage from sales, employees and profitability;
  incomplete statements go to a "limited financials" adjustment or a benchmark model; a
  "Near-Neighbors" ML adjustment is announced as forthcoming. No numbers are given.
- *Take:* plain mean/median imputation is a known downward bias; impute from the firm's own
  related items (our share × own-total rule is this idea) and keep a flag. *Differ:* we do not
  have their sub-models; this is a vendor web article, so cite as Moody's statement only.

**N-35 Dwyer & Stein (2006) (R-35)**
- Abstract + Sec. 1, p. 798: calibration needs the true prior default rate, but default
  databases are often incomplete (hidden or unreported defaults); calibration has two steps —
  map score to default rate, then adjust for the gap between database and true default rates.
  Incomplete data also confound validation of calibration.
- Sec. 5–6, pp. 806–807: capture–recapture on two merged legacy databases; with independence
  the combined database misses ~10% of small-firm defaults; the estimate is very sensitive to
  correlation between the two databases' capture rates (Table 2: 10% at ρ = 0 to 26% at ρ = 0.2).
- *Take:* name "the sample may under-record defaults" as a risk in the level decision.
  *Differ:* our target is the assignment's label in the same provider's data, so the in-sample
  rate is the operative base rate; we cannot run capture–recapture (one source).

**N-32 Altman (2000) (R-32)** — pdf pages
- pdf p. 2: Z-score (1968) and ZETA (1977) still used by practitioners (author's statement).
- pdf pp. 5–7: MDA chosen; sample of 66 manufacturers (33 bankrupt 1946–65, 33 matched
  non-bankrupt, assets $1–25m); data from one statement before bankruptcy.
- pdf pp. 8–9: 22 candidate ratios in liquidity, profitability, leverage, solvency, activity;
  five chosen; X4 uses market value of equity.
- pdf pp. 22–25: original model needs stock prices so it is a public-firm model; a re-estimated
  Z' with book equity is offered for private firms; author notes limited private-firm testing.
- *Take:* ratio-based lineage; a book-equity Z' benchmark for the scorecard challenger.
  *Differ:* tiny matched sample vs. our full panel; we estimate PDs, not group classification.

## Claim map

| Claim used in deck/report | Ref | Section/page | Verified by / date |
|---|---|---|---|
| Our PD corresponds to the IRB risk component "PD" (with LGD, EAD, M) | R-15 | para. 211, p. 52 | lit-scout 2026-10-30 |
| Basel corporate PD is the one-year PD of the grade, floored at 0.03% | R-15 | para. 285, p. 67 | lit-scout 2026-10-30 |
| Basel default = unlikely to pay or >90 days past due | R-15 | para. 452, p. 100 | lit-scout 2026-10-30 |
| IRB PD = long-run average of one-year default rates per grade | R-15 | para. 447, p. 99 | lit-scout 2026-10-30 |
| IRB banks must use ratings/PDs in credit approval (use test); pricing PDs may differ | R-15 | para. 444, p. 98 | lit-scout 2026-10-30 |
| Statistical default models are an accepted way to quantify grade PDs; ≥5 years of data | R-15 | paras. 462–463, p. 102 | lit-scout 2026-10-30 |
| Power and calibration are distinct validation dimensions | R-07; R-01; R-16 | R-07 Sec. 1 pp. 77–78; R-01 Sec. 4.1 p. 25; R-16 Sec. III | lit-scout 2026-10-30 |
| AR = 2·AUC − 1 (same information; report one) | R-14; R-07 | R-14 eq. 17 p. 12, App. A p. 23; R-07 p. 83 fn. 7 | lit-scout 2026-10-30 |
| CAP and ROC answer slightly different questions with the same information | R-07 | Sec. 2.1.2, p. 81 | lit-scout 2026-10-30 |
| Metric variability is driven by the number of defaults → report CIs and default counts | R-07; R-16 | R-07 Sec. 4.1 pp. 97–99; R-16 p. 32 | lit-scout 2026-10-30 |
| Walk-forward (out-of-sample, out-of-time) testing is the preferred design | R-07; R-01 | R-07 Sec. 3.1–3.2 pp. 92–96; R-01 Sec. 4.3 p. 28 | lit-scout 2026-10-30 |
| Re-using the test years while developing violates out-of-time testing | R-07 | Sec. 3.2, p. 95 | lit-scout 2026-10-30 |
| Monotone recalibration cannot add power; power bounds calibration resolution | R-07 | Sec. 2.3, pp. 89–91 + fn. 15 | lit-scout 2026-10-30 |
| Likelihood compares models' calibration relatively, not absolutely | R-07 | Sec. 2.2, p. 89 | lit-scout 2026-10-30 |
| No really powerful calibration tests exist; binomial/HL understate type I error with correlated defaults | R-16 | Sec. III, pp. 33–34 | lit-scout 2026-10-30 |
| Brier score reflects mainly discrimination, not calibration accuracy | R-16 | Sec. III, p. 32 | lit-scout 2026-10-30 |
| Underwriting-oriented ratings tend TTC; pricing-oriented tend PIT | R-16 | Sec. II, p. 11 | lit-scout 2026-10-30 |
| Optimal cut-off depends on default rate, LGD, spreads, fees (ROC tangency) | R-08 | Secs. 2–3, pp. 1215–1226 | lit-scout 2026-10-30 |
| ROC machinery also yields break-even prices for every score | R-08 | Sec. 4, p. 1227 | lit-scout 2026-10-30 |
| More powerful models are more profitable, more so under pricing than cut-offs (simulation, US data) | R-08; R-01 | R-08 Sec. 6 pp. 1231–1232, Sec. 7 p. 1233; R-01 Sec. 5 p. 31 | lit-scout 2026-10-30 |
| With competing banks, a weaker model attracts adversely selected borrowers (model/simulation argument) | R-08; R-01 | R-08 pp. 1231–1232; R-01 p. 31 | lit-scout 2026-10-30 |
| Cut-offs are common in practice but sub-optimal; pricing preferred | R-08; R-07 | R-08 p. 1214, p. 1233; R-07 fn. 6 p. 82 | lit-scout 2026-10-30 |
| RiskCalc architecture: per-ratio non-parametric transforms → probit with industry → final transform (GAM-like) | R-01; R-05 | R-01 Sec. 3.1 p. 14; R-05 p. 9 | lit-scout 2026-10-30 |
| RiskCalc ratio families: profitability, leverage, debt coverage, growth, liquidity, activity, size | R-01 | Sec. 3.1 p. 13; Appendix p. 34 | lit-scout 2026-10-30 |
| Too many ratios → overfitting; one per family | R-01 | Sec. 3.1, pp. 12–13 | lit-scout 2026-10-30 |
| FSO mode = statements only; updates only with new statements | R-01 | Sec. 3.1, p. 12 | lit-scout 2026-10-30 |
| Private-firm ratio–default relationships differ from public firms | R-05 | p. 1 summary; p. 9 | lit-scout 2026-10-30 |
| Private firms have no liquid equity prices, so market-based models do not apply directly | R-05; R-01; R-32 | R-05 p. 9 fn. 1; R-01 Sec. 3.2 p. 15; R-32 pdf pp. 22–25 | lit-scout 2026-10-30 |
| Statements reach lenders with a lag (3–4 months in Moody's US data) | R-01 | Sec. 3.2, p. 16 | lit-scout 2026-10-30 |
| Italian law: accounts approved ≤120 days after year-end (≤180 if statute allows) | R-28 | art. 2364 co. 2 (current text) | lit-scout 2026-10-30 |
| Italian law: approved accounts filed within 30 days | R-29 | art. 2435 co. 1 (current text) | lit-scout 2026-10-30 |
| Smaller companies may file abbreviated accounts with aggregated items | R-27 | art. 2435-bis (current text) | lit-scout 2026-10-30 |
| Altman Z: MDA, 66 manufacturers, five ratios incl. market equity | R-32 | pdf pp. 5–9 | lit-scout 2026-10-30 |
| Boosted trees produce distorted probabilities; post-hoc calibration (Platt/isotonic) fixes them | R-19 | abstract, p. 625 | lit-scout 2026-10-30 |
| Prior-shift adjustment of posteriors when class priors change | R-22 | Sec. 2.2 eq. 4 | lit-scout 2026-10-30 |
| SHAP values: unique additive attribution with local accuracy, missingness, consistency | R-23 | Sec. 3, Thm. 1, p. 4 | lit-scout 2026-10-30 |
| Beating LR alone is no longer evidence of progress; use RF-type benchmarks (retail data) | R-17 | Conclusions, MS p. 25 | lit-scout 2026-10-30 |
| RiskCalc 4.0 (France) kept the 3.1 structure: transforms → probit + industry dummies → final transform; cycle only moves the final transform | R-33 | Sec. 1.2 p. 4; Sec. 3 p. 10 eq. 1 | lit-scout 2026-10-30 (1b) |
| RiskCalc 4.0 France inputs: equity/assets, cash+securities/assets, (profit+D&A)/(liabilities−cash), operating margin, interest exp./sales, AP/sales, personnel exp./sales, sales growth, ΔROA, real assets | R-33 | Table 3, p. 11 | lit-scout 2026-10-30 (1b) |
| In RiskCalc 4.0 France, debt coverage carries the largest weight (19.2%), then leverage (17.8%) | R-33 | Sec. 3.2, Table 4, p. 14 | lit-scout 2026-10-30 (1b) |
| Moody's drops statements failing plausibility checks (assets ≠ liabilities + net worth; < 12-month period) from development data | R-33; R-10 | R-33 Sec. 2.2 p. 6; R-10 App. A p. 30 | lit-scout 2026-10-30 (1b) |
| Moody's excluded tiny firms, financials, real-estate developers, public/NFP and start-ups from RiskCalc France development | R-33 | Sec. 2.2, p. 6 | lit-scout 2026-10-30 (1b) |
| Cleaner/larger data alone raised RiskCalc v1.0 out-of-sample one-year AR 48.2% → 51.7% (US data) | R-01 | Sec. 3.4.1, Table 4, p. 20 | lit-scout 2026-10-30 (1b) |
| Removing suspect (rounded) rows made RiskCalc v3.1 weaker, so they were kept and flagged | R-01 | Sec. 3.4.1, p. 22 | lit-scout 2026-10-30 (1b) |
| Industry controls add little power but much calibration (log-likelihood); ratio meaning differs by sector | R-01 | Sec. 3.3, pp. 17–18, Tables 2–3 | lit-scout 2026-10-30 (1b) |
| Ignoring within-firm dependence can overstate significance; GEE left RiskCalc conclusions unchanged | R-01 | Sec. 3.4.2, p. 22 | lit-scout 2026-10-30 (1b) |
| Ratio levels predict default better than trends; latest year beats a 3-year average | R-05 | pp. 38–40, Exhibits 4.9–4.12 | lit-scout 2026-10-30 (1b) |
| RiskCalc v1 truncated raw ratios at the 2nd/98th percentiles | R-05 | p. 55 fn. 42 | lit-scout 2026-10-30 (1b) |
| RiskCalc v1 filled a missing input with the mean transformed value | R-05 | Appendix 6B, pp. 60–61 | lit-scout 2026-10-30 (1b) |
| Moody's says mean imputation of missing inputs would systematically understate PDs (vendor statement, no numbers) | R-34 | web article, 2021-12-02 | lit-scout 2026-10-30 (1b) |
| Audit/quality status of private-firm statements is predictive but mostly absorbed by size | R-05 | pp. 40–41, Exhibit 4.15 | lit-scout 2026-10-30 (1b) |
| Macroeconomic forecasting was deliberately left outside RiskCalc | R-05 | p. 41 | lit-scout 2026-10-30 (1b) |
| For middle-market firms, firm-specific (financial statement) information dominates systematic information | R-10 | Sec. 4.1 p. 21; Conclusion p. 27 | lit-scout 2026-10-30 (1b) |
| Model power decays as financial statements age | R-10 | Highlight 6 p. 2; Sec. 4.1 p. 21 | lit-scout 2026-10-30 (1b) |
| Development samples tend to under-record defaults, so calibration needs a central-tendency adjustment | R-33; R-35; R-10 | R-33 Sec. 2.4 p. 9; R-35 Sec. 1 p. 798; R-10 App. C.2.4 p. 34 | lit-scout 2026-10-30 (1b) |
| Moody's European RiskCalc models adjust for the cycle via public-firm DD; Italy, Spain, Portugal also use national unemployment | R-04; R-33 | R-04 slides 13–15; R-33 Sec. 3.3 p. 15 (France) | lit-scout 2026-10-30 (1b) |
| Per-firm explanation as ratio percentiles + contribution vs an average firm (RiskCalc tools) | R-33 | Sec. 5.2–5.4, pp. 27–29 | lit-scout 2026-10-30 (1b) |
| Collinearity check on transformed inputs: Spearman (max 0.66) and VIF (warning zone 4–10) | R-33 | Sec. 4.3, Tables 7–8 + fn. 13, p. 18 | lit-scout 2026-10-30 (1b) |

## Claims we must NOT make (or must word as assumptions)

1. "RiskCalc is the industry standard" — contradicted by R-05 p. 10. Use: "a widely documented
   practitioner approach for private firms; Moody's reported >200 institutional users of v1.0
   (R-01, p. 5)".
2. "Banca Massiccia uses an acceptance cut-off" — no source. Write: "We assume the bank
   approves/declines against a threshold (assignment transaction flow, R-30); Stein (2005)
   shows this is a special case of pricing."
3. "Banca Massiccia is an IRB bank / subject to the 0.03% floor / must meet the use test" — not
   in the assignment. Say the Basel link is conceptual.
4. "Our default label is the Basel default definition" — it is not (Contradiction 4).
5. "Italy used a 180-days-past-due rule" — Basel fn. 89 does not name the country.
6. "Statements are published March–June" / "available from July" (proposal) — only legal
   maxima are verified (R-28, R-29); actual availability is an assumption to test in data.
7. "The data contain only firms the bank lent to" (selection bias) — unverified; data-auditor
   must check; R-26 not read for reject inference.
8. Specific Stein (2005) or RiskCalc dollar/bps figures as expectations for our bank — they
   are simulations on US Moody's data; quote only with that context.
9. "Brier measures calibration" (Contradiction 2).
10. Anything about Ohlson's variables, size effect or results (R-12 not read); Altman 1968
    details except via R-32; Shumway (R-13), XGBoost monotone constraints (R-18), Platt (R-20),
    isotonic/Zadrozny-Elkan (R-21), EBM/GA2M (R-24, R-25) — all unverified.
11. "Isotonic overfits with small calibration sets" — not checked in R-19 body.
12. RiskCalc Italy's ratio list (R-03 SECONDARY). For the Italian credit-cycle method, say only
    what R-04 slides 13–15 show: public-firm DD-based CCA combined with the national
    unemployment rate (no formula, weights or version number given).
13. Claims on Italian regional (North/South) credit differences or 2011–12 sovereign-crisis
    credit conditions — no source located yet; phrase as assumption or drop.
14. "RiskCalc 4.0 uses ratios X/Y" without saying **France** (R-33). No general or Italian 4.0
    document was read. Also do not say what changed in RiskCalc Italy 4.0 (no source).
15. "Moody's showed mean imputation underestimates PDs by N%" — R-34 gives no numbers; it is a
    vendor assertion. Quote as "Moody's states…".
16. Anything from the Dwyer & Stein (2004) Technical Document (R-02), Dwyer, Wang & Fang
    (2012) (R-36) or Dwyer & Eggleton (2009) (R-37) — not read.
17. "Moody's found ≈80% of private-firm statements are bad" as a general rate — R-10 p. 30 is one
    North American CRD example, mostly interim statements (fn. 19).
18. A Dwyer paper on overfitting in ratio selection — searched, not found; do not cite.
