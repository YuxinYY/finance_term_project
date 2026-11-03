# Team Turquoise: Project Update

**Banca Massiccia: a one-year probability of default for private Italian borrowers**
DS-GA 3001 (Prof. Roger Stein) · Team Turquoise: Kiet Nguyen, Desmond Peng, Chloe Li · 3 November 2026

<!-- Numbers come from files named in the hidden comments. D-xxx = docs/DECISION_LOG.md; R-xx = docs/REFERENCES.md (VERIFIED rows only). Longer version with more detail: reports/final/2026-11-02_Turquoise_Project_Update.md (pre round 2). -->

**Status.** The formulation is fixed and two modelling rounds are complete. Our leading candidate is a monotone gradient-boosted tree model on 45 inputs (GB-25). Out of time on 2010, the cleanest test year, it has AUC 0.835 [0.827, 0.842], against 0.812 for our weight-of-evidence scorecard. Our referee step found no leakage in the new pipeline (D-141). The new inputs improve ranking modestly and give no new view of risk: the riskiest 10% of firms in 2010 capture about the same share of defaults as before (51.7% vs 51.8%). The PD **level** is the main open problem, because defaults rose faster than any training window could show. All results are preliminary, and the model choice is not final.

## 1. Formulation

The bank needs one number per borrower: the probability of a missed payment within 12 months, using only what is known at the decision. It serves two uses: approve/decline needs correct **ordering** (AUC; AR = 2·AUC − 1, R-07 p. 83), and pricing needs a correct **level** (log loss, observed/predicted). A cut-off rule is our assumption from the assignment's transaction flow.

- **Unit:** one firm's annual statement. The data contain no loan records (D-001).
- **Label:** a statement for year *t* (always dated 31 December) is a default if the firm defaults between 30 June *t*+1 and 30 June *t*+2 (D-002). Civil Code arts. 2364 and 2435 (R-28, R-29; current text) imply the statement is not public until about mid-*t*+1. Filing dates are not in the data, so the 6-month start remains an assumption.
- **No** default-date features, later years, firm history or external data (D-004, D-020).

## 2. Data

1,023,552 statements from 237,711 firms, 2007-2012. <!-- reports/data/01_profile.md -->

| Fiscal year | 2007 | 2008 | 2009 | 2010 | 2011 |
|---|---|---|---|---|---|
| Default rate | 0.96% | 1.28% | 1.62% | 1.70% | 1.28% over an incomplete 11-month window |

<!-- reports/data/00_labels.md §2, §5 -->

What we found:
- Default dates run to 31 May 2013, past the stated end of the data. We use them as given and test dropping them (D-009).
- The label window moves the level by about a third: 1.41% with the 6-month start, 1.07% from year-end (D-002a). <!-- reports/data/00_labels.md §5 -->
- Accounting identities hold, so we rebuild missing items from them first (D-021).
- The bank/financial/trade debt split goes missing as a block (98.2% of cases), mostly for small firms. It gets a flag plus a share-based rebuild (D-021). <!-- reports/data/02_missingness.md -->
- `taxes` is really pre-tax profit (D-024). `goodwill` behaves like value added and passed the plausibility checks we could run on it (D-091). That reading is still an inference, not a documented fact.
- Province and ATECO code tables match ISTAT's official lists (R-53, R-54). That the provider uses this coding is an inference from the data.

## 3. Features

**v1: 26 inputs** in the RiskCalc v3.1 families (R-01 Sec. 3.1 p. 13) plus asset structure, each with an expected direction set before fitting (D-018, D-022).

**Round 2 adds 19 inputs, fitted in-window** (D-090 to D-099): cost-structure ratios from RiskCalc 4.0 France (R-33 Table 3 p. 11), sector percentiles (R-01 Sec. 3.3), shrunk out-of-fold default-rate encodings of industry division, province and region (R-46, R-47), data-quality signals, and previously unused fields.

What held up on inner splits (D-122): the cost-structure ratios and unused fields gained in both windows. Encodings gained only with two training years. Sector percentiles added no significant power despite large SHAP shares. RiskCalc France's net debt coverage duplicates our cash-flow ratio (Spearman 0.99), so it was not added.

**Rejected:** text features (no text in the data), growth features (+0.002 AUC on seen firms only, and a rule question to Roger/TA; D-020), and duplicate ratios (D-024).

## 4. Preliminary results

Walk-forward on frozen folds: train up to year *t*, test on *t*+1, for 2008-2010 (506,159 rows, 7,784 defaults; R-07 Sec. 3, R-33 Sec. 4.8). Comparisons use a paired, firm-clustered bootstrap.

<!-- reports/final/update_table.csv (script update_table.py), own level before the level shift -->

| Model | AUC pooled | AR | AUC 2010 | Log loss |
|---|---|---|---|---|
| **GB-25** monotone LightGBM, 45 inputs, Moody's sign on interest/sales | **0.8424** | 0.685 | **0.8348** | 0.06380 |
| GB-20 same, interest/sales unconstrained | 0.8435 | 0.687 | 0.8369 | 0.06367 |
| GB-01 monotone LightGBM, 26 inputs (round-1 winner) | 0.8375 | 0.675 | 0.8307 | 0.06415 |
| GB-11 random forest benchmark (R-17 p. 25) | 0.8395 | 0.679 | 0.8335 | 0.06392 |
| GB-02 unconstrained LightGBM, 26 inputs (fails face validity) | 0.8440 | 0.688 | 0.8402 | 0.06332 |
| GAM-01 explainable boosting machine | 0.8254 | 0.651 | 0.8177 | 0.06517 |
| RC-02 RiskCalc-style probit | 0.8194 | 0.639 | 0.8115 | 0.06580 |
| SC-03 weight-of-evidence scorecard | 0.8166 | 0.633 | 0.8115 | 0.06656 |

- **GB-25 vs GB-01:** ΔAUC +0.0049 [0.0037, 0.0061] pooled, +0.0042 [0.0022, 0.0059] on 2010. Log loss is better pooled and in 2009-2010, and tied in 2008 (D-144). <!-- reports/tournament/r2_paired_bootstrap.csv -->
- **GB-25 vs the random forest:** +0.0029 [0.0005, 0.0050] pooled, but **tied on 2010**: +0.0013 [−0.0026, 0.0051]. Round-1's GB-01 does not separate from the forest on AUC, [−0.0002, 0.0042]. <!-- r2_paired_bootstrap.csv -->
- **GB-20 vs GB-25:** freeing the interest/sales sign is worth +0.0021 AUC on 2010, but gives high-interest firms a *lower* score, and that setting was chosen after seeing the data. Choosing between them is still open.

**What did not work:**
- Hyper-parameter search on v1 (37 configurations): +0.0009 AUC, with log loss worse in two years (D-104).
- Recency weights: +0.0002, no effect (D-105).
- Blending GB-01 with the EBM: −0.0032 (D-086; R-10 Sec. 4.4 p. 22 found the same for simple averaging).
- Dropping constraints: up to +0.0065 AUC, but rejected. A firm with cash flow twice its debt would score as badly as one burning cash (D-103).
- Our "drop a constraint only where the denominator nears zero" rule failed its own check (D-102).

## 5. Validation

**Where power is weak** (GB-01 → GB-25): smallest firms AR 0.626 → 0.638; construction 0.559 → 0.568, but no gain on 2010 (+0.001); new firms 0.573 → 0.569, and significantly worse in 2008, when only one training year was available (−0.020 [−0.034, −0.006]) (D-145). The largest firms remain strongest (0.735 → 0.745). <!-- reports/tournament/r2_scorecard.md segments -->

**Level.** Every model under-predicts: GB-01's observed/predicted is 1.21 / 1.36 / 1.43 across 2008-2010, while calibration slopes stay near 1, so the spread is right and the level is wrong. We apply a prior shift on the log-odds (R-22 eq. 4), anchored to the latest default speed available in training (D-071, D-072). This brings observed/predicted to 1.02 / 1.05 / 1.18 for GB-01. That rule was chosen after seeing these years, so the gain is optimistic. New features do not fix the level: 2010 stays at 1.19 for GB-25 (D-146). Proposed holdout target: 1.49%, which rests on an assumed random walk in the default rate (D-073). PDs are floored at 0.03% and capped at 99% (D-075).

**2010, on a 16-grade scale** with edges set on training years (no grade above about 10% of firms; design target from R-33 p. 24): the safest 42.5% of firms are on target, but grades covering half the firms (PD 0.5-10%) saw observed/predicted of 1.26-1.41, about 500 extra defaults (GB-01; D-116). GB-25 does not change this: 1.33 vs 1.32 across grades 6-14 (D-146). In a 2010-like year, mid-grade loans would be priced too low.

**Overfitting.**

| Check | Evidence | Reading |
|---|---|---|
| Same-year in-sample minus out-of-time AUC | GB-01 0.015-0.026; GB-20 0.022-0.035; GB-25 0.017-0.029 | More features fit training years harder. We quote only out-of-time numbers. R-33 reports about 0.5 AR points for RiskCalc France, which is not like for like. |
| Calibration slope, out of time | GB-25 1.005 / 1.014 / 0.957 | PDs are not over-spread overall. New firms' slopes are 0.83-0.89 for every round-1 model. |
| Learning curve (GB-07) | Flat at its minimum | Tree count is not a sensitive choice. |
| Constraint buy-back (D-126) | About 40% of the new features' inner gain recovers what constraints blocked | Part of the gain is not new information. |
| Selection | 113 looks at 2008-2010 so far (D-142) | Absolute scores carry unquantified optimism. Margins and the 2010 rows matter more. |

**Fiscal 2011 is a partly spent check.** The calibration round already scored GB-01 and RC-02 on it, and the level target uses its default rate (D-118). Planned use (D-119, D-148, not yet signed off): freeze the finalists (proposed: GB-25, GB-01, GAM-01), score them once on 2011, report the result whatever it says, and use it only for the model choice, never to test the level.

## 6. Hypotheses

| Hypothesis | Status |
|---|---|
| Statement information carries most of the ranking signal (R-10 Sec. 4.1) | Supported indirectly; we have no market or macro data to test against |
| Monotone boosting beats transparent models, at a cost in readability | Supported: GB-01 beats the EBM by 0.012 AUC |
| New information beats more tuning | Supported: +0.0049 from features vs +0.0009 from tuning |
| New firms are harder to rank | Supported: about 0.057 AUC gap for GB-01, and similar across models |
| The cycle moves the level more than the ranking | Mostly supported: level drifts strongly, but AUC also falls about 0.013 from 2008 to 2010 |
| Average-value imputation biases PDs down (R-34) | Only for models that fill with averages: scorecard 1.30, EBM 1.21, GB-01 1.01 relative observed/predicted (D-114) |

## 7. Issues

1. **Unanswered questions for Roger/TA:** label window, 2013 default dates, holdout period, use of training-data firm history. Each has a stated default assumption. They move the level (target 1.49% vs 1.37%), not the ranking.
2. **The level comes from the sample only.** Moody's sets its central tendency above the sample rate using external data, because samples miss defaults (R-33 Sec. 2.4 p. 9). We cannot, so a holdout more stressed than 2010 would be under-predicted.
3. **Unsourced economics.** Our top driver, tangible-asset share, has verified support only for creditor recovery and borrowing capacity (R-48), not for lower PD; we present the link as our data finding. The staff-severance fund's effect (more fund, lower PD, given the wage bill) is our unverified hypothesis, possibly reflecting firm maturity (D-123). Its reading also depends on the `goodwill` inference.
4. **Unexplained pattern.** Actual default rates rise among the most profitable firms. We constrain this away and do not yet have an explanation (D-102).
5. **Process limits.** Some round-2 choices followed outer scores and are disclosed (D-100, D-129). The GB-25 sign fix was not fully blind, because 2010 default rates by interest/sales decile were already on file; GB-25 is nonetheless the lowest of the three new-feature configurations on 2010 (D-142). Features were designed on 2007-09 data, so 2010 is the clean year (D-062).
6. **A known defect.** For firms with negative EBITDA, more interest mechanically improves `interest_coverage`, so the PD can fall (GB-25 40%, GB-01 30% of those firms; D-147). A fix needs its own inner-only test.
7. **Thin cells.** Several sectors have 30-66 test defaults, so we neither rank models nor correct levels there (D-078).

## 8. Next steps to 16 November

1. Test the negative-EBITDA fix on inner splits.
2. Team sign-off on the finalists and the one-time 2011 check; then choose the final model (GB-25 vs GB-20; EBM as transparent fallback, which needs its missing-value rule repaired first, D-114).
3. Final fit on 2007-2011 with the 11-month label (D-077); freeze all parameters.
4. Harness and stress tests on messy input; document computational complexity.
5. Deck, with reason codes for example borrowers, what did not work, contributions appendix and bibliography; then red-team review and claim audit.

## 9. AI use

We used Anthropic's Claude (Claude Code) as a set of agents: data audit, literature verification, feature, model and calibration code, referee scoring, and drafting, including this update. The referee is an AI agent, not an independent human review. No language model is inside the PD model. Numbers are generated by repository code and traced to files, and citations are limited to sources an agent located and read (`docs/REFERENCES.md`). **Most AI output in `docs/AI_USE_LOG.md` is still pending human verification.** Before 16 November we will re-run the key scripts, spot-check the cited pages and initial each check.

## References (verified sources only)

- R-01 Dwyer, Kocagil & Stein (2004). *Moody's KMV RiskCalc v3.1 Model.* Moody's KMV.
- R-07 Stein (2007). Benchmarking default prediction models. *Journal of Risk Model Validation*, 1(1), 77-113.
- R-10 Stein, Kocagil, Bohn & Akhavein (2003). *Systematic and Idiosyncratic Risk in Middle-Market Default Prediction.* Moody's.
- R-17 Lessmann, Baesens, Seow & Thomas (2015). Benchmarking state-of-the-art classification algorithms for credit scoring. *EJOR*, 247(1), 124-136 (manuscript read).
- R-22 Saerens, Latinne & Decaestecker (2002). Adjusting the outputs of a classifier to new a priori probabilities. *Neural Computation*, 14(1), 21-41.
- R-28, R-29 Italian Civil Code, arts. 2364 co. 2, 2435 co. 1 (current text).
- R-33 Buitrago, Makarov, Zhao & Dwyer (2015). *RiskCalc 4.0 France.* Moody's Analytics.
- R-34 Moody's (2021). *Advances in default detection and early warning.* Web article.
- R-46 Micci-Barreca (2001). A preprocessing scheme for high-cardinality categorical attributes. *SIGKDD Explorations* (abstract read).
- R-47 Pargent et al. (2022). Regularized target encoding outperforms traditional methods (preprint read).
- R-48 Almeida & Campello (2007). Financial constraints, asset tangibility, and corporate investment. *RFS*, 20(5) (NBER WP 12087 read).
- R-53 ISTAT. *Elenco dei comuni italiani.* R-54 ISTAT (2009). *Classificazione delle attività economiche Ateco 2007.*
