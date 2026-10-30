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
