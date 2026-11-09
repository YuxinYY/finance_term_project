# Calibration study guide: timeline, methods, literature, what we tried (2026-11-08)

For Kiet, to learn in one hour and then teach Chloe and Desmond. Every number comes from a file in this repo
(cited inline). Runnable example: `docs/2026-11-08_study_calibration_example.py`.

**Suggested hour:** Part 1 (timeline, 10 min) → Parts 2–4 (concepts and methods, 15 min) → run the example
(5 min) → Parts 5–7 (what we chose and why, 15 min) → Part 8 (literature, 5 min) → Part 10 (quiz, 10 min).

---

## Part 1. The timeline, untangled

### 1.1 Four words that mean nearly the same thing

| Term | Meaning | In the data |
|---|---|---|
| **Fiscal year (FY)** | The accounting year a statement covers. All statements here run Jan 1 – Dec 31. | column `fs_year` |
| **Statement date** | The last day of that year | `stmt_date` = 31 Dec of `fs_year` on every row |
| **Firm-year** | One row = one firm's statement for one FY. "Firm-year" is the *row*; "fiscal year" is the *year label on the row*. | key (`id`, `fs_year`) |
| **Cohort FY t** | All firm-year rows with `fs_year = t` | e.g. 174,512 usable rows for FY2010 |

So "FY2010 rows" and "the 2010 firm-years" are the same thing: every firm's 2010 statement.

### 1.2 The key fact: a FY's label lives 1.5–2.5 years *after* the FY

A FY t statement is approved and filed by about mid-year t+1 (Civil Code: 120 days to approve, 30 to file;
R-28, R-29). The bank decides with it from then on, so the 12-month PD window starts then:

```
label window for FY t  =  [1 Jul t+1 , 30 Jun t+2)        (our D-002, "start-6")
```

Defaults are tracked in the file only up to **31 May 2013** (last `def_date`; D-009). Lay every FY on that line:

```
           2008      2009      2010      2011      2012      2013      2014
FY2007     |Jul=====Jun|                                                      complete
FY2008               |Jul=====Jun|                                            complete
FY2009                         |Jul=====Jun|                                  complete
FY2010                                   |Jul=====Jun|                        complete
FY2011                                             |Jul=====May|.Jun          11 of 12 months seen
FY2012                                                       ┊ |Jul=====Jun|  NOTHING seen
                                                       data ends 31 May 2013 ┘
```

Counts from `data/interim/panel.parquet` (label status by FY):

| FY | Usable rows ("ok") | Defaults | 12-month rate | Censored (window not fully observed) |
|---|---|---|---|---|
| 2007 | 144,214 | 1,379 | 0.96% | 0 |
| 2008 | 161,261 | 2,064 | 1.28% | 0 |
| 2009 | 170,386 | 2,752 | 1.62% | 0 |
| 2010 | 174,512 | 2,968 | 1.70% | 0 |
| 2011 | 0 | – | – | 183,953 |
| 2012 | 0 | – | – | 184,989 |

(A few hundred rows per year are `already_defaulted`: the firm defaulted before its window opened, so the bank
would not be lending to it.)

**Answer to "the last FY in the data is 2012, so why leave out 2011?"**
- **FY2012 has statements but no labels.** Its window (Jul 2013–Jun 2014) is entirely after the data end. We
  can compute features for it but not whether those firms defaulted, so it cannot train or test anything.
- **FY2011 is only 11/12 labelled** (Jul 2012–May 2013), and those 11 months rely on the 2013 default dates
  that contradict Appendix B (D-009). A firm with no `def_date` might default in June 2013, which we cannot see.
- **We did not throw 2011 away.** We (1) kept it *sealed* as a one-time "lockbox" test (D-087, D-119), and (2)
  include it in the *final* model fit with an 11-month label (D-077), because including the newest cohort beat
  dropping it on a fold-3 analogue (AUC 0.8301 vs 0.8263).
- **UPDATE 2026-11-08 (D-151): the team now uses Chloe's 5-month window** (1 Jun t+1 to 31 May t+2) for all
  future work. The tables in this Part still show the old start-6 counts until the panel is rebuilt. See Part 14.
- Chloe/Desmond start the window one month earlier (1 Jun t+1). That makes FY2011's window Jun 2012–May 2013,
  fully inside the data, so they treat FY2011 as a complete test year. Same data, different label assumption.

### 1.3 Five kinds of data (the source of most confusion)

| Name | What it is | Who sees labels | Used for | Our case |
|---|---|---|---|---|
| **Training set** | Rows the model is fitted on | us | fitting the model, imputation, encodings, bins | FY ≤ t in each fold |
| **Validation (inner) set** | A slice *inside* the training window, held back from fitting | us | early stopping, tuning, Platt slope ("slice B") | inside FY ≤ t |
| **Test set (outer fold)** | The next FY after the training window | us, but only to *score* | honest out-of-time estimate per fold | FY2008, 2009, 2010 |
| **Lockbox** | A test year we refuse to look at during development | us, once | one clean check of the finalists after all choices are made | FY2011 |
| **Holdout** | Roger's vault: "drawn from a future time period", `def_date` = NA, may contain new firms | Roger only | grading our harness (5 pts) | unknown years, probably FY2012+ statements |

Rule of thumb: **training/validation = we fit with them; test/lockbox = we only score them; holdout = we never
see it.** The harness must give each holdout row a PD from that row alone plus frozen parameters (CLAUDE.md §2.2).

### 1.4 Our walk-forward folds (`data/splits/walkforward.json`)

| Fold | Train on FY | Test on FY | Train rate | Test rate | New firms in test |
|---|---|---|---|---|---|
| 1 | 2007 | 2008 | 0.96% | 1.28% | 29,835 rows |
| 2 | 2007–2008 | 2009 | 1.13% | 1.62% | 17,172 |
| 3 | 2007–2009 | 2010 | 1.30% | 1.70% | 16,737 |
| Lockbox | 2007–2010 | 2011 (11-month label) | | | |
| Final fit | 2007–2011 | → Roger's holdout | | | |

**Why no gap between train and test?** The decision on FY t+1 statements starts about 1 Jul t+2. By then FY t's
label window (ends 30 Jun t+2) is complete. So training on FY ≤ t and testing on FY t+1 is exactly what the bank
could have done in real time (D-070).

**Chloe's "temporal" design differs:** train FY2007 / validate FY2008 / test FY2009, i.e. the year just before the
test year is used for validation and Platt, not for fitting. Ours fits on all years up to t and carves the
validation slice from *inside* those years.

---

## Part 2. Why calibrate at all

The bank uses one PD for two jobs (CLAUDE.md §1):

| Job | Needs | Measured by | Changed by calibration? |
|---|---|---|---|
| Approve/decline | right **order** | AUC / accuracy ratio | **No.** Any increasing map keeps the order (R-07 pp. 89–91) |
| Pricing | right **level** | log loss, obs/pred, calibration slope | **Yes**, that is its whole purpose |

What we found (`reports/calibration/summary.md` §1): GB-01 ranked well (calibration slope 0.976, close to 1) but its
mean PD was 1.15% against 1.54% observed; obs/pred 1.21 / 1.36 / 1.43 in 2008 / 2009 / 2010. Default rates rose
each year and every model was fitted on earlier, calmer years. **The miss was in the level, not the shape.**

Plain meaning: a loan priced at 1.15% expected default that actually defaults at 1.54% loses money on average.

---

## Part 3. Three separate questions (the framework to teach)

| Question | Plain words | Our answer | Decision |
|---|---|---|---|
| **Shape** | Is the spread of PDs right, i.e. are high PDs high enough relative to low ones? | Platt slope from held-out slice B. Here it came out ≈ 1 (1.017 / 0.943 / 0.998 by fold) | D-053 |
| **Method** | Once we know the target level, how do we move all PDs to it? | Prior shift: add one number δ to every logit | D-072 |
| **Target** | What average default rate should the holdout year have? | τ = 1.493% (start-6 label) | D-071, D-073 |

Keeping these apart is the main teaching point. Chloe's Platt-on-next-year answers shape and target in one
step, so the target it uses is just "whatever the validation year was" without anyone choosing it.

---

## Part 4. The methods, explained

Notation: `logit(p) = ln(p/(1−p))`, `expit` is its inverse. A PD of 1% has logit −4.60; 1.5% has −4.18.

### 4.1 Platt scaling
`PD_cal = expit(a + b·logit(PD_raw))`, with a and b fitted by logistic regression of the label on
`logit(PD_raw)` **on data not used to fit the model** (R-19 Sec. 2.1, eqs. 1–3, VERIFIED).
- b = slope: b < 1 means PDs are too spread out (shrink them), b > 1 means too compressed.
- a = intercept: the level.
- **Our finding:** b ≈ 1 for LightGBM with log-loss objective and no reweighting, so Platt's gain was nearly all
  intercept, meaning year-to-year level drift (D-053). R-19 reports that boosted trees often need Platt; we
  tested that and found little distortion here.
- **Trap:** Platt fitted on year t−1 copies year t−1's level into year t. In a rising cycle that helps by
  accident (an implicit trend rule); at a turn it hurts (D-072, "M3raw").

### 4.2 Isotonic regression
A free-form increasing step function fitted by PAV (pool-adjacent-violators) (R-44 VERIFIED). It is flexible,
but its flat steps create **ties**, so AUC can drop slightly. It also needs many defaults. R-19 Sec. 5 shows it
overfitting small calibration sets on non-credit data. Our "≥ 500 defaults" eligibility rule is our own heuristic,
not a published threshold.
- **Our result:** −4.82e-4 log loss vs −6.31e-4 for prior shift, slope 0.92, within-year AUC −0.0008. **Rejected.**

### 4.3 Prior shift (adopted)
If the default rate changes from r_w (the training rate) to τ but the profile of a defaulter versus a survivor
stays the same, Bayes' rule says every PD moves by the same amount in logit units:

```
PD' = expit( logit(PD) + δ ),   δ = logit(τ) − logit(r_w)
```

This is Saerens, Latinne & Decaestecker (2002), eq. 4 (R-22 VERIFIED), the "simple prior adjustment" in Stein
(2007) pp. 88–89 (R-07).
- It is one frozen number, needs no data at prediction time, and leaves AUC unchanged (`auc_invariance_check.csv`:
  |ΔAUC| = 0).
- **Honest caveat:** the "same profiles" assumption is doubtful across 2008–12. We use the formula as a level
  tool, not as a claim that the assumption holds (D-072).

### 4.4 Saerens EM (rejected)
EM re-estimates the new prior from the *unlabelled* test batch. It gave 2.18% / 1.79% / 0.45% against observed
1.28 / 1.62 / 1.70% (`saerens_em_diagnostic.csv`). It is also illegal for us, because it uses other holdout rows
(constraint 2).

### 4.5 Example output (run `docs/2026-11-08_study_calibration_example.py`)

Simulated year 1 at 0.87% defaults, year 2 at 1.44%, model fitted on year 1:

| Variant | AUC | obs/pred | Lesson |
|---|---|---|---|
| raw | 0.8422 | 1.66 | under-predicts the worse year |
| prior shift to τ = 1.5% | 0.8422 | 0.98 | level fixed, order untouched |
| Platt on a year-1 slice | 0.8422 | 1.66 | cannot see year 2's rise |
| isotonic on a year-1 slice | 0.8409 | 1.65 | ties lower AUC, no level gain |

(τ = 1.5% landed close to the truth here by construction. In real life τ is a forecast, and Part 5 is about
choosing it.)

---

## Part 5. Choosing the target τ: the real judgment

### 5.1 PIT vs TTC
- **Point-in-time (PIT):** PDs follow the current cycle. Pricing-oriented systems tend to be PIT.
- **Through-the-cycle (TTC):** a long-run average. Underwriting-oriented systems tend to be TTC.
  (R-16 Sec. II p. 11, VERIFIED.)
- Since the PD is used for pricing, we lean PIT. A true TTC level cannot be estimated from 2007–2010 anyway:
  there are no benign years after the turn (D-071).

### 5.2 What the bank knows when it sets the level (D-070)
For test year t+1 the bank has every training label up to 30 Jun t+2 and **nothing** of the test window. The
freshest legal signal is how fast defaults arrived in the *last six months* of the last complete cohort.

### 5.3 Seven candidate rules, tested in the walk-forward (D-071, `ct_targets.csv`, `methods_by_year.csv`)

| Rule | Definition | Δ log loss pooled (e-4, negative = better) | 2011 pseudo-test extra loss (e-4) |
|---|---|---|---|
| **e** | default speed in months 7–12 of the last training cohort's window, ×2 | **−6.31** [−7.09, −5.60] | 0.74 |
| c2 | halfway between the last year and a logit trend | −6.28 | 3.85 |
| c | logit trend extrapolated one year | −6.16 | 9.42 |
| b | last training year's rate | −4.50 | 0.80 |
| f | mean of the last 2 years | −1.57 | 0.48 |
| a | pooled training rate (= no shift) | 0 | 0.33 |
| d | equal-weight average of the years (TTC proxy) | +0.37 | 0.48 |

Targets for 2008/09/10: e 1.17 / 1.50 / 1.61%, observed 1.28 / 1.62 / 1.70%. Rule e recovers 88% of the
gain an oracle (perfect per-year level) would get (0.06352 vs 0.06343).

**Read it carefully:**
- In the rise (2008–10), recency wins and the long-run average loses.
- At the 2011 turn, trend rules fail badly (c: 9.42), while every rule except c and c2 lands within 0.8e-4.
- e was picked among 7 rules after seeing 2008–2010, so its gain is optimistic. Say so (D-071).

### 5.4 The shipped τ = 1.493% (D-073)
- **Anchor:** the FY2011 12-month rate estimated from its 11 observed months is 1.431%.
- **Hedge:** the holdout year is unknown, probably about 2 years later. Under a driftless logit random walk
  (σ = 0.21/yr, estimated from 2007–2011 cohort changes), the expected rate 2 years ahead is 1.493%. It sits
  above the anchor because the rate is convex in logit and because of the asymmetry below.
- **Shift:** δ = logit(τ) − logit(r_w) = **+0.079** for the final fit on 2007–2011 (r_w = 1.381%), or +0.059 if fit
  on 2007–2010 (`experiments/calibration/harness_level_params.json`).
- **This is a judgment.** The random-walk persistence and σ are assumptions; human sign-off is still requested.

### 5.5 Why we lean slightly high: asymmetry (D-074)
- Being 30% too low costs 8.17 / 10.35 / 10.93e-4 of log loss (2008 / 09 / 10); 30% too high costs
  5.57 / 7.02 / 7.40e-4. **Too low costs about 1.5×.**
- Business reading (stated as an assumption): under-priced loans lose money and attract the borrowers that
  better-priced competitors turn away. Over-pricing loses volume, not money on booked loans.
- Scenario costs of shipping about 1.5%: truth 1.43% → 0.2e-4; 1.70% → 1.3e-4; a stressed 2.13% → 11.8e-4.

### 5.6 Floor and cap (D-075)
- PDs are clipped to [0.03%, 99%].
- The 31 PDs below 0.03% had 0 defaults.
- A 50% cap was rejected: the 855 firms above 50% defaulted 65% of the time, and capping them raised log loss.
- The 0.03% floor is the Basel corporate floor (R-15 para. 285), used *by analogy only*. We do not claim the bank
  is IRB.

### 5.7 Label window sensitivity (D-076)
- τ by window: s0 1.37%, s3 1.39%, s6 1.49%. Switching costs at most 0.59e-4.
- The harness reads `tau[s6]` and switches if Roger/TA confirm another window (D-002a is still open).
- **For Chloe's start-5 label, τ must be recomputed** (in progress, Part 9).

---

## Part 6. The full pipeline the harness runs (D-079)

```
raw row → features (frozen params) → GBM raw PD
        → own Platt map, intercept anchored so mean PD on training rows = r_w          (shape, D-053)
        → PD' = expit(logit(PD) + δ),  δ = logit(1.493%) − logit(1.381%) = +0.079       (level, D-072/073)
        → clip to [0.0003, 0.99]                                                         (D-075)
```

Results (`reports/tournament/r2_scorecard.md`):
- **GB-25+CAL**, pooled 2008–10: obs/pred 1.10 (GB-25 raw 1.36), log loss 0.06315 vs 0.06380, AUC unchanged.
- By year: obs/pred 1.03 / 1.06 / 1.19.

---

## Part 7. What worked, what didn't, what is still open

| Item | Status | Evidence |
|---|---|---|
| Prior shift (one δ) | **Worked**, adopted | −6.31e-4 pooled, AUC unchanged |
| Platt slope from in-window slice B | Worked but nearly a no-op (slope ≈ 1) | D-053 |
| Recency rule e | Best in the rise; optimistic (picked after the fact) | D-071 |
| Raw Platt on the previous year | Rejected for the harness: an implicit trend rule | D-072 |
| Isotonic | Rejected: worse loss, ties | D-072 |
| Saerens EM | Rejected: wrong estimates; needs holdout rows | D-072 |
| Trend extrapolation (c) | Rejected: fails at the 2011 turn (9.42e-4) | D-071 |
| TTC / pooled average | Rejected: worst in the walk-forward | D-071 |
| Group-level corrections (sector, size) | Rejected: would be fitted on test years | D-078 |
| Class reweighting / oversampling | Rejected: distorts the level | D-054 |
| Macro data for the cycle | Not used (external data not cleared) | D-004 |
| **2010 middle-book under-prediction** | **OPEN.** Grades 6–14 in 2010: 2,108 observed vs 1,590 expected (obs/pred 1.33) | D-116, D-146 |
| **New-firm over-dispersion** (slope 0.89) | **OPEN.** Needs a shape fix, not a level fix | D-078 |
| Human sign-off on τ hedge, label window, 2011 inclusion | **OPEN** | D-079 |

**Big-picture verdict (H4, D-074):**
- Level matters, but less than the algorithm. The best leak-free level gain is 6.3e-4; GB vs the WoE scorecard is
  24e-4.
- A 30% level miss (8–11e-4), however, is half the size of the algorithm gaps.
- So the level is second-order only if it is set carefully.

---

## Part 8. Literature map: claim → source (all VERIFIED in `docs/REFERENCES.md`)

| Claim we make | Source |
|---|---|
| Monotone recalibration cannot add power | R-07 Stein (2007), Sec. 2.3 pp. 89–91 |
| Re-using test years during development breaks out-of-time testing (hence the lockbox) | R-07 Sec. 3.2 p. 95 |
| Likelihood compares models' calibration relatively, not absolutely | R-07 Sec. 2.2 p. 89 |
| Prior-shift formula | R-22 Saerens et al. (2002), Sec. 2.2 eq. 4 |
| Platt formula; fit on an independent calibration set; isotonic can overfit small sets | R-19 Niculescu-Mizil & Caruana (2005), Secs. 2.1, 2.2, 5 |
| Platt restated (the original was not read) | R-43 Lin, Lin & Weng (2007), Sec. 1 |
| PAV/isotonic properties | R-44 Brümmer & du Preez (2013) |
| Pricing systems tend PIT, underwriting TTC | R-16 BCBS WP14, Sec. II p. 11 |
| Calibration tests are weak; binomial bands understate error with correlated defaults | R-16 Sec. III pp. 33–34 |
| Brier mostly reflects discrimination, so it is not a calibration measure | R-16 Sec. III p. 32 |
| 0.03% corporate PD floor (analogy only) | R-15 Basel II para. 285 p. 67 |
| Statements reach lenders with a lag | R-01 RiskCalc 3.1 Sec. 3.2 p. 16; filing law R-28, R-29 |

**Do NOT say** (`REFERENCES.md` "must not claim"):
- Anything attributed to Platt's original paper (R-20 SECONDARY) or Zadrozny & Elkan (R-21 not obtained).
- "Brier measures calibration."
- That the bank is IRB.
- How RiskCalc Italy adjusts for the cycle beyond R-04 slides 13–15.
- That R-19 gives a defaults threshold for isotonic (the 500 rule is ours).

---

## Part 9. Chloe's model: the reference calibration (round 1 done, D-150)

Files: `reports/calibration/2026-11-08_chloe_lgbm_v1_round1.md`, `experiments/calibration/2026-11-08_chloe_lgbm_v1/`.
Her LightGBM was refit with her config: train FY2007, early stopping on FY2008, test FY2009.
- **Reproduces her result:** FY2009 AUC 0.84309 vs her reported 0.8431.
- **Why not her production model:** it was trained on FY2007–2011, so every year is in-sample for it. It was never
  used for evaluation.
- **Rounds 2–3:** waiting for files from Chloe.

FY2009, her label (`2026-11-08_metrics_fy2009.csv`, `2026-11-08_bootstrap_vs_b_platt.csv`; Δ in e-4, 1,000-draw
bootstrap):

| Variant | AUC | Log loss | Δ vs her Platt | Mean PD | Observed | obs/pred |
|---|---|---|---|---|---|---|
| raw | 0.8431 | 0.06459 | +11.42 [+10.19, +12.72] | 0.97% | 1.58% | 1.62 |
| her Platt (fit on FY2008) | 0.8431 | 0.06345 | ref. | 1.19% | 1.58% | 1.33 |
| our slope + training-rate anchor | 0.8431 | 0.06459 | +11.46 | 0.97% | 1.58% | 1.63 |
| **ours + rule e level (1.516%)** | 0.8431 | **0.06278** | **−6.63 [−8.47, −4.88]** | 1.58% | 1.58% | **1.00** |
| isotonic + rule e | 0.8427 | 0.06275 | −6.91 | 1.58% | 1.58% | 1.00 |

On our label (start-6, 170,386 rows), the same story: her Platt has obs/pred 1.38; ours + rule e has 1.04 and is
−8.61e-4 better.

**What this teaches (it matches Parts 4–5 exactly):**
1. **AUC is identical across every Platt/shift variant** (ΔAUC = 0). Calibration changes only the level.
2. **The slope was already ≈ 1** (0.996). Without a level choice, our recipe equals raw.
3. **Her Platt helps only because its intercept carries FY2008's higher rate**, which in effect is rule b ("next
   year = last year"). It is still 1.33× too low.
4. **All of the gain is the level choice.** Rule e's near-perfect FY2009 hit is partly luck: on her label window,
   rule e's level-only extra loss is 0.12 / 0.88 / 10.19e-4 for 2009 / 2010 / 2011, and it over-predicts 2011 by
   43%. Expect worse in rounds 2–3. This is why our shipped τ is a hedged anchor, not "rule e every year".
5. **Isotonic ties on loss but loses AUC** (−0.0004, not significant) and gives exactly 0 PD to 5,324 firms
   without the floor. It stays rejected.

**Flags for Chloe:**
- **Her saved production Platt would double-count the level if used.** It was fitted on pooled out-of-time PDs (raw mean 1.105% vs
  observed 1.536%) and then applied to a model refit on those same years. She labelled the file "(Ignore this)",
  so it is a caution, not a live bug (Part 13).
- **Her level:** τ = 1.464% on her window, δ = +0.077 (`2026-11-08_params_for_production.json`). The intercept
  anchor still needs her 2010–2011 feature rows.
- **Harness risk:** three features (`annualized_sales_growth`, `annualized_change_in_roa`, `change_in_ar_turnover`)
  need the firm's prior-year statement. That is allowed only from a frozen *training-data* lookup, never from other
  holdout rows (Part 17, D-020).
- **Label mismatch:** 163 of her FY2009 defaults (4–29 Jun 2010) fall before our window opens.

---

## Part 10. Quiz yourself (Roger-style), with model answers

1. **"Calibration improved your log loss. Did it improve the model?"** It improved the *level* (pricing). The
   ranking (AUC) is unchanged by construction: a monotone map cannot add power (R-07).
2. **"Why not just fit Platt on last year?"** Its intercept copies last year's level, which is a hidden forecast
   ("next year = last year"). We make that forecast explicit (τ), test seven versions of it, and show trend rules
   fail at the 2011 turn.
3. **"Where does 1.49% come from?"** The FY2011 rate estimated from 11 months (1.43%), projected 2 years under a
   logit random walk (σ = 0.21/yr), nudged up because under-prediction costs about 1.5× more. It is a stated
   judgment, with costs of being wrong in `scenario_regret.csv`.
4. **"Did any test-year information leak into the level?"** No: τ, r_w and δ use training rows only (D-070).
   Disclosure: rule e was chosen after seeing 2008–10, and FY2011 informed the τ anchor, so 2011 is not an
   independent test of the level (D-087).
5. **"Your model under-predicts the 2010 middle grades. Why not fix it?"** Any fix would be fitted on the test
   year. We disclose it instead: grades 6–14, obs/pred 1.33 (D-146).
6. **"Why is FY2012 not in training if it's in the data?"** Its label window (Jul 2013–Jun 2014) is after the last
   observed default (31 May 2013), so its outcomes are unknown.

---

# Deep dive (added 2026-11-08): plain words, small numbers

Figures for this section are in `reports/calibration/fig/`:
- `2026-11-08_chloe_r1_reliability_curves.png`
- `2026-11-08_chloe_r1_mapping_curves.png`
- `2026-11-08_chloe_r1_prior_shift_slide.png`

They are made by `experiments/calibration/2026-11-08_chloe_lgbm_v1/2026-11-08_r1_calibration_curves.py`. All
worked numbers below were computed in Python, not typed by hand.

## Part 11. Two tools you need first: odds and log-odds

| PD | Odds = PD ÷ (1 − PD) | Log-odds = ln(odds) = logit(PD) |
|---|---|---|
| 0.1% | 0.001001 | −6.91 |
| 1% | 0.010101 | −4.60 |
| 2% | 0.020408 | −3.89 |
| 10% | 0.1111 | −2.20 |
| 50% | 1 | 0 |

- Odds say "defaulters per survivor". A 1% PD means about 1 defaulter per 99 survivors.
- **Why use odds instead of PD?** Odds can be multiplied freely and the PD stays below 100%: PD = odds ÷ (1 + odds).
  Multiplying a PD directly cannot work: 60% × 2 = 120%.
- Adding a number to the log-odds is the same as multiplying the odds. Adding 0.703 means odds × e^0.703 = × 2.02.
- Every calibration method below is a rule of the form "log-odds in → log-odds out".

## Part 12. Every method with small numbers

### 12.1 Where the prior shift comes from: Bayes' rule with 10,000 firms

**Normal year: 10,000 firms, 100 default (1%).**
- Suppose 30% of defaulters have a "red-flag" profile (say, negative equity), but only 2% of survivors do.
- Red-flag firms who default: 30% × 100 = **30**. Red-flag firms who survive: 2% × 9,900 = **198**.
- PD of a red-flag firm = 30 ÷ (30 + 198) = **13.16%**. In odds: 30 / 198 = 0.1515.

**Bad year: same 10,000 firms, 200 default (2%).** Assumption: *what a defaulter looks like does not change*,
so still 30% of defaulters and 2% of survivors have the red flag.
- Red-flag defaulters: 30% × 200 = **60**. Red-flag survivors: 2% × 9,800 = **196**.
- PD = 60 ÷ 256 = **23.44%**. In odds: 60 / 196 = 0.3061.

**Now the shortcut.** The odds went from 0.1515 to 0.3061, i.e. × 2.02. That is exactly the ratio of the overall
odds: (2/98) ÷ (1/99) = 0.020408 ÷ 0.010101 = **2.02**. So:

```
new odds of any firm = old odds × (odds of the new overall rate ÷ odds of the old overall rate)
in log-odds:   logit(PD_new) = logit(PD_old) + δ,     δ = logit(τ) − logit(r_w) = ln 2.02 = 0.703
```

That is the whole prior shift (Saerens et al. 2002, eq. 4; R-22). The profile fractions (30%, 2%) cancel out, so
**the one number we need is the new overall rate τ**. The rest of the model stays as it is.

What the shift does to firms (r_w = 1% → τ = 2%, odds × 2.02):

| Firm's PD before | Odds before | × 2.02 | PD after |
|---|---|---|---|
| 0.1% | 0.001001 | 0.002022 | 0.20% |
| 0.5% | 0.005025 | 0.010152 | 1.005% |
| 1% | 0.010101 | 0.020408 | 2.00% |
| 5% | 0.0526 | 0.1063 | 9.61% |
| 20% | 0.25 | 0.5051 | 33.6% |
| 60% | 1.5 | 3.03 | 75.2% |

- Small PDs roughly double; big PDs rise less and never pass 100%.
- The order never changes: a firm that was above another stays above it. That is why AUC cannot move.
- The new *average* PD lands close to τ, not exactly on it, because the curve bends. On Chloe's FY2009 the target
  was 1.516% and the resulting average was 1.581%.

**Chloe FY2009 with real numbers:** r_w = 0.8735% (FY2007), τ = 1.516% (rule e).
δ = logit(0.01516) − logit(0.008735) = −4.174 − (−4.732) = **+0.558**, i.e. odds × 1.75.
In `prior_shift_slide.png` the whole histogram slides right by that amount (the plotted median shift is 0.57,
because the shift also includes the small anchoring step).

### 12.2 The rules are the guesses for τ (the "new 200 defaulters")

In 12.1 we were *told* the bad year has 200 defaulters. In real life the bank must **guess** next year's rate
before it happens. Each rule is one way of guessing, from data the bank already has on decision day.

Toy history: three past years at 1.0%, 1.2%, 1.5% (equal size), and in the most recent six months 0.9% of the
last cohort defaulted.

| Rule | How the guess τ is made | τ | δ vs pooled r_w = 1.233% | A 1% firm becomes |
|---|---|---|---|---|
| a pooled | average of all years | 1.233% | 0 | 1.00% |
| b last year | the most recent year | 1.50% | +0.198 (odds × 1.22) | 1.22% |
| c trend | straight line through the years' log-odds, one year ahead | 1.82% | +0.397 (× 1.49) | 1.48% |
| e last 6 months | 0.9% in half a year × 2 | 1.80% | +0.384 (× 1.47) | 1.46% |

**Step by step for any rule:**
1. Pick a rule and compute τ from past data.
2. Compute δ = logit(τ) − logit(r_w).
3. Add δ to every firm's log-odds, then turn the result back into a PD.

The rules affect step 1 only. Bayes (12.1) is steps 2–3.

**Rule e in real dates (Chloe's window).** The decision day for FY2009 statements is 1 Jun 2010. The most recent
fully watched cohort is FY2008, whose window ran 1 Jun 2009 to 31 May 2010. Rule e counts defaults in its *last*
six months (Dec 2009 to May 2010) and doubles the rate. That gives 1.516%, and all of it is known on decision
day. It is like reading the speedometer instead of the trip average.

### 12.3 Platt scaling: a fitted straight line on log-odds

```
logit(PD_new) = a + b × logit(PD_raw)
```

a and b are fitted by a one-variable logistic regression of 0/1 default labels on logit(PD_raw), using a year the
model was not trained on.

Toy example with a = 0.30, b = 0.95:

| Raw PD | logit | × 0.95 + 0.30 | New PD |
|---|---|---|---|
| 0.1% | −6.907 | −6.261 | 0.19% |
| 1% | −4.595 | −4.065 | 1.69% |
| 10% | −2.197 | −1.787 | 14.3% |

- b below 1 pulls extreme PDs toward the middle. b above 1 pushes them apart.
- **With b = 1, Platt is exactly a prior shift with δ = a.** The only difference is where the level comes from:
  Platt *fits* a from last year's labels, which amounts to "next year = the calibration year". The prior shift
  *chooses* τ explicitly with a rule.
- Chloe's Platt on FY2008: a = 0.2116, b = 0.9956. A raw 1% becomes 1.26%; FY2009 needed about 1.6%.
- Her saved production Platt: a = 0.263, b = 0.968. A raw 1% becomes 1.50%.

### 12.4 Isotonic regression: the staircase (PAV, "pool adjacent violators")

Six firms sorted by raw PD, with labels 0, 1, 0, 0, 1, 1:
1. Start with each firm's own label as its PD: 0, 1, 0, 0, 1, 1.
2. Firm 2 (1) is above firm 3 (0): wrong direction. Pool them into their average: 0, ½, ½, 0, 1, 1.
3. ½ is still above firm 4 (0). Pool firms 2–4: (1 + 0 + 0)/3 = ⅓. Result: **0, ⅓, ⅓, ⅓, 1, 1**. Now it never goes down.

Two problems are visible even in this tiny example:
- **Ties:** firms 2–4 get the same PD although the model ranked them differently, so AUC can fall.
- **Zeros:** firm 1 gets PD = 0%. On Chloe's FY2009, 5,324 firms would get exactly 0% without our 0.03% floor.

On 170,000 firms the steps are smaller but still visible (`mapping_curves.png`, yellow line). On FY2009 it tied
the prior shift on log loss and lost 0.0004 AUC. With no gain and more risk, we keep it rejected (D-072).

### 12.5 Saerens EM (rejected), in one paragraph

EM guesses τ from the *unlabelled* holdout batch itself:
1. Average the PDs over the batch.
2. Treat that average as the new τ and shift.
3. Average again, and repeat until it settles.

Two problems: it uses other holdout rows, which is forbidden (Part 17), and on our data it guessed
2.18% / 1.79% / 0.45% when the truth was 1.28% / 1.62% / 1.70%.

### 12.6 How to read the three figures

- **`reliability_curves.png`:** firms are sorted into groups by PD. Each dot is a group: its average PD (x) against
  the share that actually defaulted (y).
  - On the dashed line means right; above it means PDs too low.
  - Raw sits above (×1.62), Platt is still above (×1.33), isotonic + shift and our prior shift sit on the line.
- **`mapping_curves.png`:** each line shows what a method does to one firm's PD.
  - Smooth lines (prior shift, Platt) keep the order.
  - The isotonic staircase creates ties and a drop to 0 at the bottom left.
- **`prior_shift_slide.png`:** the same hill of firms slid to the right. Same shape, so same order, so same AUC.

## Part 13. Where Chloe calibrates in her code

She calibrates in two places:

1. **Per round, in the code she pushed:** `src/finance_project/model_development/run_walk_forward_models.py`,
   lines 270–273 on branch `teammate/project-update`.
   - `ProbabilityCalibrator.fit(validation_raw, y_validation, "platt")` fits Platt on that round's validation
     year, then applies it to the test year.
   - Every "calibrated" number in her `tests/*_temporary_report.md` comes from this.
2. **For production, saved but not pushed:** `production_lightgbm_v1/(Ignore this) platt_calibrator.joblib`.
   - Platt a = 0.263, b = 0.968, fitted on 529,382 pooled out-of-time predictions from her rounds 1–3 (FY2009–2011,
     raw mean 1.105%, observed 1.536%).
   - See `training_manifest.json`, fields `calibration_*`. The script that made it is not in her branch.

The file name says "(Ignore this)", so she has not adopted it. **The caution if anyone does use it:**
- Its intercept learned "add about 30% to the odds", because the *per-round* models were too low.
- The production model was retrained on 2007–2011, so it already knows those higher-default years.
- Adding the boost again counts the same correction twice, like resetting a slow watch and then adding the
  5 minutes anyway.
- The size of the effect is not measured. It needs the production model's own training rows.

## Part 14. "Already defaulted" firms, and the 5-month window (D-151)

**What it means.** The FY t statement is dated 31 Dec t. Under the 5-month window, the bank uses it from
1 Jun t+1. Some firms default **between 1 Jan and 31 May of t+1**: after the statement date, but before the bank
could have used it.

**How many** (panel, FY2007–2011): **2,897 rows**. By month: Jan 558, Feb 518, Mar 559, Apr 568, May 694.
The usable data is 835,016 rows with 11,336 defaults in-window. That matches Chloe's `labels.md` exactly.

**What Chloe does, and what we do:** drop them. They are neither "survived" (0) nor "defaulted in the window" (1).
On 1 Jun t+1 the bank would be looking at a firm that is already in default, not a loan applicant.

**Does dropping them hurt?**
- **For training: hardly.** The model still learns from the 11,336 in-window defaulters, who look similar
  (losses, negative equity).
- **Keeping them as 1** would teach the model about a period the bank cannot act on, and push the level up.
- **Keeping them as 0** would be wrong: they defaulted.
- **The real risk is in the holdout.** `def_date` is NA there, so the harness cannot tell these firms apart and
  will give them a PD anyway. If Roger counts a default from the statement date onward, those firms count as
  defaults, and our PDs for them will be too low. They would be about 1 in 5 of all defaults (2,897 of
  14,233 in training).
- That is why email question 1 (which window Roger uses) still matters. There is nothing to "treat" in the
  harness: every row must get a PD.

**What switching to 5 months changes for us (D-151):**
1. Labels change slightly, e.g. FY2009 includes June 2010 defaults.
2. **FY2011 becomes a complete 12-month cohort**, if the 2013 default dates are valid (D-009). This removes the
   11-month workaround (D-077).
3. τ becomes about 1.46% instead of 1.49% (D-150).
4. The split file must be rebuilt as v2 (`walkforward.json` stays read-only), and the referee scores again.

## Part 15. Why the lockbox, if we already have validation and test sets

**Chloe's point:** "We have validation and test sets, so we can just train on everything." Half right:
- **Training:** the final model *does* use everything, including 2011 (D-077). The lockbox costs nothing in training.
- **Scoring:** what the lockbox protects is our **claim** about how good the model is.

**The exam analogy.** The test years 2008–2010 are practice exams. Each time we try a model, look at its
2008–2010 score, and keep the better one, those exams turn into practice. We have done this **113 times**
(r2 selection ledger: 51 prediction files plus 62 calibration settings). After 113 tries, the best score is partly
skill and partly luck. Pick the best of 113 coin-flip sequences and it will look like a "skilled" coin.

A tiny example: 10 models that are *truly identical* (AUC 0.840) score 0.840 ± 0.005 on a test year just from
noise. The best of the 10 will show about 0.848. That is +0.008 from luck alone, bigger than most of our real
differences (GB-25 vs GB-01 is +0.005).

**The lockbox is a fresh exam taken once**, after every choice is frozen. It gives one number nobody tuned
against, which is what the 5 out-of-sample points are graded on. Stein (2007) calls re-using test years during
development a violation of out-of-time testing (R-07 Sec. 3.2 p. 95).

**Honest state today:**
- 2011 is already partly spent for us: the calibration round scored it (D-087, D-118).
- Chloe's pipeline used FY2011 as an ordinary test year (her round 3).
- So 2011 is clean only for "which of our finalists", and only if nobody now picks a model *because* of
  Chloe's 2011 numbers.

The rule going forward: **choose on 2008–2010 and inner splits; score 2011 once; then refit on everything.**

## Part 16. Calibration in one line per method

| Method | Formula | Level comes from | Keeps order? | Verdict |
|---|---|---|---|---|
| Prior shift | logit(PD′) = logit(PD) + δ | a chosen τ (a rule) | yes | **adopted** |
| Platt | logit(PD′) = a + b·logit(PD) | last year's labels, implicitly | yes (b > 0) | shape step only |
| Isotonic | step function by PAV | the calibration year | ties | rejected |
| Saerens EM | iterate the prior on the holdout batch | the holdout itself | yes | rejected (illegal) |

## Part 17. What the holdout rules actually say (exact citations)

Quotes are from `docs/Term Project Instructions 20261011.pdf`. Page numbers are the PDF's printed page numbers.

| # | Exact text | Page |
|---|---|---|
| Q1 | "Your prediction code should take in one or more rows of data, and process each record in any manner you find useful, provided the processing does not use the any of the test data, **besides the current record**, for any purpose (e.g., you cannot calculate the mean of a variable across the test data and use it to scale that input variable)." | p. 5, *Model codebase* |
| Q2 | "Your model should not depend in any way on the data in the hold-out sample." | p. 3, *Appendix A, Output specifications* |
| Q3 | "you will only be allowed to use data that you would have access to in real life at the time of your prediction for the firm in the prediction year." | p. 3 |
| Q4 | "Be aware that there may be borrowers in the hold-out sample that were not in the training data." | p. 3 |
| Q5 | "Each row is one firm-year" · "The value for the default date (def_date) is NA in the holdout sample" | p. 6, *Appendix B, Data Conventions* |
| Q6 | "The holdout sample is drawn from a future time period and may include both firms that are in the training data and those that are not." | p. 6 |
| Q7 | "If you use the training data dynamically (e.g., for KNN), make sure to provide a data file in the form that you need for your harness." | p. 5 |
| Q8 | "The harness should not re-estimate your model at run time." | p. 5 |

**What this means, carefully:**
- **We do not know how many rows per firm the holdout has.** Q6 says "a future time period", which could be
  one year or several (that is email question 3). So a new firm could appear with both its FY2013 and FY2014 rows.
- **Even if it does, Q1 forbids using one to score the other.** When the harness scores that firm's FY2014 row,
  "the current record" is FY2014 only. The FY2013 row is "other test data". So a new firm is scored from its
  one row alone. A firm seen in training is scored from its one row too, unless we use a training lookup (next point).
- **A frozen lookup into the *training* data is a grey zone, and probably allowed.** Q7 explicitly allows shipping a
  training data file. Q3 allows data the bank "would have access to in real life", and a bank has a known
  borrower's past statements. So for a firm seen in training, a growth feature built from its *training* rows
  does not break Q1. That is our email question 4. Our current choice is the conservative "no" (D-020): the
  measured gain was only +0.002 AUC, and only for seen firms.
- **Correction to what I said earlier:** Chloe's growth features are not automatically illegal. They are illegal
  *as currently coded*: her pipeline builds them from a `history.csv` that contains the rows being scored. If the
  holdout holds two years of the same firm, one holdout row would feed the other's growth feature. They become legal only if rebuilt to read a frozen training-data file, and new firms then get blanks.
- CLAUDE.md §2.2 ("must not use any holdout row other than the current record") is our wording of Q1 + Q2 + Q8.

---

## Glossary

- **r_w:** default rate of the exact rows the final model was fitted on.
- **τ (tau):** target default rate for the holdout.
- **δ (delta):** the logit shift between them.
- **obs/pred:** observed default rate ÷ mean PD; 1 is perfect, > 1 means under-predicting.
- **Calibration slope:** coefficient from regressing the label on logit(PD); 1 means the spread is right.
- **Slice B:** an in-window hold-back, used for early stopping and the Platt slope.
- **Lockbox:** a sealed test year.
- **e-4:** units of 0.0001 log loss.
