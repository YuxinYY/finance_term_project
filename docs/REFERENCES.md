# REFERENCES — verified bibliography and claim map

Status meanings (see `.claude/agents/literature-scout.md`):
- **VERIFIED**: located (URL/DOI recorded), bibliographic details match, and each claim we use
  was checked against the text, with section/page noted.
- **LOCATED**: existence and bibliographic details confirmed from a reliable page, but the
  claims we want to use have NOT yet been checked against the full text. Do not cite claims yet.
- **SECONDARY**: only a secondary mention found (citing source named). Do not cite claims.
- **TO-VERIFY / UNVERIFIED**: not yet located, or located but nothing checked.

Status as of **Oct 30, 2026** (literature-scout, Day 1 pass). Page numbers are the printed page
numbers of the document at the recorded URL unless "pdf p." is stated.

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

## Bibliography

| ID | Reference | Status | Where located | Intended use / claim to check |
|---|---|---|---|---|
| R-01 | Dwyer, D. W., Kocagil, A. E., & Stein, R. M. (2004, April 5). *Moody's KMV RiskCalc v3.1 Model: Next-Generation Technology for Predicting Private Firm Credit Risk.* Moody's KMV. | **VERIFIED** | https://www.rogermstein.com/wp-content/uploads/RiskCalc-v3-1-Model.pdf (author's site; 36 pp.) | See notes N-01 |
| R-02 | Dwyer, D. W., & Stein, R. M. (2004). *Technical Document on RiskCalc v3.1 Methodology.* Moody's KMV. | LOCATED (listed in R-01 ref. 7, p. 35; on rogermstein.com publication list without link) | no copy found | Detailed methodology — do not cite specifics |
| R-03 | Moody's KMV. *RiskCalc v3.1 Italy* (model documentation). | SECONDARY (TOC of a private-company valuation book lists "9.1 RiskCalc Italy"; Moody's 2020 webinar R-04) | no readable copy | Italy ratio families — **not available; do not claim which ratios the Italy model uses** |
| R-04 | Moody's Analytics (2020). *Tracking the Impact of COVID-19 on Credit Risk — Europe* (webinar slides). | LOCATED | ma.moodys.com PDF | Italy/Spain/Portugal credit-cycle adjustment uses national unemployment rate — not re-read today |
| R-05 | Falkenstein, E., Boral, A., & Carty, L. V. (2000, May). *RiskCalc for Private Companies: Moody's Default Model.* Moody's Investors Service, Rating Methodology. | **VERIFIED** | https://efalken.com/papers/riskcalc.pdf (first author's site; listed at efalken.com/papers/index2.html) | See notes N-05 |
| R-06 | Sobehart, J. R., Keenan, S. C., & Stein, R. M. (2000). *Benchmarking Quantitative Default Risk Models: A Validation Methodology.* Moody's Investors Service. | LOCATED (author's site link found; not read) | http://www.rogermstein.com/wp-content/uploads/53621.pdf | Walk-forward; CAP/AR origin |
| R-07 | Stein, R. M. (2007). Benchmarking default prediction models: Pitfalls and remedies in model validation. *Journal of Risk Model Validation*, 1(1), 77–113. (orig. Moody's KMV TR #030124, 2002) | **VERIFIED** | http://www.rogermstein.com/wp-content/uploads/BenchmarkingDefaultPredictionModels_TR030124.pdf (JRMV typeset version) | See notes N-07 |
| R-08 | Stein, R. M. (2005). The relationship between default prediction and lending profits: Integrating ROC analysis and loan pricing. *Journal of Banking & Finance*, 29(5), 1213–1236. doi:10.1016/j.jbankfin.2004.04.008 | **VERIFIED** | http://www.rogermstein.com/wp-content/uploads/JBF_2026.pdf (published JBF typeset) | See notes N-08 |
| R-09 | Bohn, J. R., & Stein, R. M. (2009). *Active Credit Portfolio Management in Practice.* Wiley. ISBN 978-0-470-08018-4. | LOCATED | Wiley product page | Not read |
| R-10 | Stein, R. M., Kocagil, A. E., Bohn, J., & Akhavein, J. (2003). *Systematic and Idiosyncratic Risk in Middle-Market Default Prediction.* Moody's KMV. | LOCATED (listed in R-01 ref. 25, p. 36) | — | Not read |
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
12. RiskCalc Italy's ratio list or Italian credit-cycle method (R-03 SECONDARY, R-04 not re-read).
13. Claims on Italian regional (North/South) credit differences or 2011–12 sovereign-crisis
    credit conditions — no source located yet; phrase as assumption or drop.
