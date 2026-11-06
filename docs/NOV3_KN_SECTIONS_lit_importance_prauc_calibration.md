# KN's sections for the Nov 3 team update

Draft by Claude Code for KN, 2026-11-03. Text is written to paste into the team report. Every source is in
`docs/REFERENCES.md` with its check status. Open each source yourself before the deck cites it.
Numbers come from `reports/calibration/summary.md` and `experiments/gbm/GB-25/` (our model, not the teammates').

---

## A. Literature review (supports the teammates' claims)

Short version: most claims are supported. Four need rewording, and one is contradicted by a source we read.

| Teammate claim | Source | Status | What it supports, and the limit |
|---|---|---|---|
| Walk-forward testing (train on earlier years, test on the next) | Stein (2007) R-07 Sec. 3.1-3.2; Dwyer, Kocagil & Stein (2004) R-01 Sec. 4.3 | **Supported** | Out-of-time testing is the preferred design. Re-using test years while developing breaks it. |
| Fit all parameters inside the training window (imputation, scaling) | R-07 Sec. 3.2 | **Supported** | Same source. Leakage rule. |
| Statements reach lenders 3-4 months late; we use 5 months | R-01 Sec. 3.2 p. 16; Italian Civil Code art. 2364 (120 days), art. 2435 (30 days) R-28, R-29 | **Supported, with a limit** | 120 + 30 days is about 5 months. Art. 2364 lets a statute extend to 180 days, so 5 months is the base case, not a guarantee. Cite the *current* code text; the 2007-12 wording was not checked. |
| Features come from RiskCalc (ROA change, sales growth, margins, etc.) | R-01 p. 13; RiskCalc 4.0 France R-33 Table 3 p. 11 | **Supported** | R-33 is **France**, not Italy. Say "RiskCalc 3.1 and the French 4.0 edition". Check that "change in AR turnover" is in R-01/R-33 before writing "from RiskCalc"; I did not find it in the ratio lists. |
| VIF > 10 as the collinearity cutoff | R-33 Sec. 4.3, Tables 7-8, fn. 13 p. 18 | **Partly** | R-33 treats VIF 4-10 as a *warning zone*. It does not say "drop above 10". Word it as "a common rule of thumb, consistent with R-33's warning zone". |
| Sector target encoding with K-fold and smoothing | Pargent et al. (2022) R-47 | **Supported** | Unregularised target encoding overfits rare levels. Shrinkage plus cross-fitting fixes it. Micci-Barreca (2001) R-46 is abstract-level only: do not quote its formula. |
| Missing flags plus sector-median fill | Bryzgalova et al. (2025) R-55; Moody's R-34 | **Reword** | R-55 is about *stock-market firm fundamentals* and argues simple fixes are unreliable. It does **not** endorse sector medians. Cite it for "missingness is systematic, so keep a flag". I could only read its abstract. Moody's (R-34) says filling missing inputs with means understates PDs (a vendor statement, no numbers). So test calibration on rows with imputed inputs. |
| Decaying weighted sum of past years (halving weights) | Kim, Cho & Ryu (2021) R-56 | **Weak, partly contradicted** | R-56 (abstract only; paywalled) says geometric-lag variables help machine-learning default models. But Falkenstein et al. (2000) R-05 pp. 38-40 find the **latest year beats a 3-year average**. Our own test: adding lags gave +0.002 AUC (`reports/features/history_tradeoff.csv`). Present it as "tested, small gain", not "the literature recommends". |
| Weight of evidence (WoE) for logistic regression | Seitshiro & Govender (2024) R-57; Raymaekers et al. (2021) R-52 | **Reword** | R-57 (abstract only) found WoE did **not** improve accuracy on commercial bank data. Its value is explanation of each bin. R-52 gives the definition and the small-bin problem. |
| LightGBM, EBM/GAM | Ke et al. (2017) R-38; Lou et al. (2012) R-24; Nori et al. (2019) R-25 | **Supported** | R-38 does not cover monotone constraints or NaN handling. Cite the docs R-39 for those. |
| Calibration: bucket, fit a curve, then prior-shift | R-01 (final transform); Saerens et al. (2002) R-22 eq. 4; Niculescu-Mizil & Caruana (2005) R-19 | **Supported, one flaw** | See Section D. The formula the teammates wrote equals Saerens eq. 4 (I checked the algebra). |
| Imbalanced data and PR-AUC | Saito & Rehmsmeier (2015) R-58; Davis & Goadrich (2006) R-59 | **Supported as a diagnostic only** | See Section C. |

**Gap in the teammates' review.** They cite Moody's papers for features. The assignment asks for finance *and* ML history,
and what the bank does today. We still need one sentence each on: RiskCalc's structure (transform each ratio, probit,
final calibration; R-01 Sec. 3.1 p. 14), Altman-style scores (R-32, public-firm lineage), and why private-firm models
differ from public ones (R-05 p. 1). Do not call RiskCalc "the industry standard" (R-05 p. 10 contradicts it).

**Do not write:** "tangible assets lower PD, per the literature" (R-48 supports recovery, not PD); any Basel II
sentence in the Business Problem section (the PDF trap, D-017).

---

## B. Feature importance (our GB-25 model, for reference)

The teammates have not built a final model, so this describes ours. Use it as a template and a check, not as their result.
Source file: `experiments/gbm/GB-25/shap_all_features_2010.csv` (fold-3 model, 2010 rows).

**Method.** SHAP (Lundberg & Lee 2017 R-23; TreeSHAP, Lundberg et al. 2020 R-41). Each firm's PD (in log-odds) is split into
one contribution per feature plus a base value, and the parts add up exactly. Average the absolute values for global
importance. I use this, not the built-in "gain" or "split count" importance, because R-41 shows those are inconsistent.

**Top drivers by mean |SHAP| (log-odds, 2010):**

| Rank | Feature | Mean abs SHAP | Reading |
|---|---|---|---|
| 1 | tangible_ratio | 0.363 | more tangible assets, lower PD (in our data) |
| 2 | asset_turnover | 0.165 | free (no sign forced) |
| 3 | st_debt_ratio | 0.148 | |
| 4 | interest_coverage_sect_pct | 0.143 | sector-relative |
| 5 | equity_ratio_sect_pct | 0.142 | sector-relative |
| 6 | roa_net | 0.138 | higher profit, lower PD |

**Three cautions to state with it:**
1. The tangible-assets mechanism is **our data only**. No source says tangibility lowers PD (REFERENCES Contradiction 11).
2. Importance is not causation, and correlated features share credit.
3. We forced signs (monotone constraints) on 36 of 45 features. That helps explanation, not accuracy: the unconstrained twin was
   better by +0.0065 AUC out of time (D-051; `docs/MONOTONE_SIGN_AUDIT_2026-11-03.md`).

**What the teammates should do:** SHAP on their LightGBM per walk-forward round, then check the top features keep the same
rank across rounds. For logistic regression, standardized coefficients. For EBM, the shape plots. Per-firm "top 3 drivers" is
the credit-officer view (CLAUDE.md rule 8).

---

## C. PR-AUC: is it allowed, what does it give, what does it cost

**Allowed?** Yes. The assignment (p. 3) says models are judged on "accuracy of the probabilities" and "discriminatory power".
It names no metric list. PR-AUC is an extra diagnostic. It is not a calibration metric and not a substitute for either judged item.

**What it is.** Precision = share of flagged firms that really default. Recall = share of defaulters caught. PR-AUC averages
precision over all recall levels (average precision).

| | Benefit | Cost |
|---|---|---|
| **PR-AUC** | Focuses on the rare defaulters. Does not reward the huge pool of easy non-defaults (Saito & Rehmsmeier 2015). Shows early-ranking quality: are the riskiest 1-5% really bad? | **Depends on the default rate.** Their paper: "the baseline of PRC is determined by the ratio of positives and negatives". Our default rate moves 1.28%, 1.62%, 1.70% across 2008-10, so PR-AUC is not comparable across years. Noisier with about 2,000-3,000 defaults per year. Says nothing about PD *level*. |
| **ROC-AUC / AR** | Insensitive to the default rate, so comparable across years and portfolios. The standard in the sources we cite (R-07, R-14). AR = 2*AUC - 1. | Can look flat when defaults are very rare (Saito & Rehmsmeier). Treats a top-1% error like a bottom-50% error. |

**Link between the two.** A curve is better in ROC space exactly when it is better in PR space (Davis & Goadrich 2006). But
the *areas* can rank models differently, so optimizing one does not guarantee the other.

**Recommendation.** Keep AUC/AR and log loss as the headline (they match what Roger will grade and what R-07 uses). Report PR-AUC
in an appendix with the default rate printed beside it, and compare models **within the same year only**. Add one line:
"PR-AUC is reported because defaults are about 1.5% of rows; it is not comparable across years." Do not use it for tuning
without also checking log loss, because PR-AUC ignores the PD level.

Open for the teammate who proposed it: why PR-AUC and not top-decile capture? Ask them what decision it serves.

---

## D. Calibration: what the teammates did, and the plan

**Their method.** Score the training data, sort into k buckets, take each bucket's default rate, fit a curve from score
quantile to default rate, then apply the prior-shift formula to move from the training rate pi_S to the population rate pi_T.
This is RiskCalc's final-transform idea (R-01) plus Saerens et al. eq. 4 (R-22). The formula matches eq. 4 exactly.

**Four issues to fix or state:**

1. **Calibrating on the same data the model was fit on.** A boosted model scores its own training rows too confidently, so the
   bucket rates are biased. R-19 (Sec. 2.1-2.2) says calibration maps must be fitted on data independent of training.
   Our fix (D-053): a held-out slice inside the training window. Ask them to use their validation split for this.
2. **Where does pi_T come from?** The report says "true population default rate" but never gives it. It cannot come from the
   holdout (constraint 2). We used a judgment: tau = 1.493%, from a random-walk on yearly rates (D-073). They need a stated rule.
   Rate rise was real: observed/predicted was 1.21, 1.36, 1.43 for 2008-10 (`reports/calibration/summary.md`).
3. **Prior shift assumes the firms' ratios look the same when the default rate changes** (R-22 Sec. 2.2). Across 2008-12 that is
   doubtful. Use it as a level tool and say so.
4. **Their "How do our models perform" box is empty.** Fill it with: calibration-in-the-large (mean PD vs observed),
   calibration slope, log loss, and binned observed vs predicted, per walk-forward year. Brier is "overall accuracy", not
   calibration (BCBS WP14, R-16 p. 32). Note Stein (2007, p. 89): log-likelihood compares models; it does not certify one.

**Our results to cite as the reference run** (GB-01, pooled 2008-10, `reports/calibration/summary.md`):

| Step | Pooled log loss | Slope | Note |
|---|---|---|---|
| No level fix (M0) | 0.06415 | 0.976 | mean PD 1.15% vs observed 1.54% |
| Prior shift to target (M1) | 0.06352 | 0.98 | gain 6.3e-4; AUC unchanged |
| Isotonic + level | n/a | 0.92 | rejected: slope worse, AUC -0.0008 |
| Saerens EM | n/a | n/a | rejected: needs the holdout batch |

Caveat: the target rule was chosen after seeing 2008-10, so the gain is optimistic. On the 2011 pseudo-test trend rules failed.

**Plan for the final report (concise):**
1. Shape: each model's own calibrator, fitted on a held-out slice of the training window (not the rows it trained on).
2. Level: one prior-shift number, frozen in the parameter file (harness constraint 2).
3. Target level: written rule, assumption stated, sensitivity shown (30% too low costs about 1.5x a 30% miss too high).
4. Evidence: calibration table per year and a binned plot per model; paired bootstrap for log-loss differences.
5. Limits: new-firm slope 0.89 (over-dispersed); floor 0.03% and cap 99% (D-075); level is a judgment, not an estimate.

---

## What I need from you before this goes in the report

- Confirm which of the "Reword" rows you accept. Rows with abstract-only sources (R-55, R-56, R-57) should be softened
  or read in full via NYU access.
- Decide whether to raise the decay-weight contradiction (R-05) with Chloe/Desmond now or after the team update.
