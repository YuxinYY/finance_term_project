# Team Turquoise: project briefing, 2026-10-30

**For:** Kiet Nguyen, Desmond Peng, Chloe Li
**Purpose:** what we did, why, what we learned, what's still open, and the decisions we need to make
together before Day 3 (harness + deck).
**Status:** Days 0–2 of the compressed plan are complete (D-007). Final submission is due **Nov 16**.
The 3-day plan finishes early, which leaves about two weeks of buffer (see Section 9).

Every number below comes from a file produced by code; the path is given next to it. Decision IDs
(D-xxx) point to `docs/DECISION_LOG.md`, where each one has alternatives, evidence and sources.
Reference IDs (R-xx) point to `docs/REFERENCES.md`.

---

## 1. One-page summary

- **Recommended model:** LightGBM with monotone constraints (GB-01). It's the only model that is
  both the most accurate *and* has every ratio moving in the economically expected direction.
- **Accuracy, pooled over test years 2008–2010** (506,159 firm-years, 7,784 defaults):

  | Model | AUC [95% CI] | Log loss | Ratios with the wrong direction |
  |---|---|---|---|
  | **GB-01 LightGBM, monotone** | **0.838 [0.832, 0.841]** | 0.0642 → **0.0635 after calibration** | 0 |
  | RC-02 RiskCalc-style probit | 0.819 | 0.0658 → 0.0651 after calibration | 1 |
  | SC-03 WoE scorecard (benchmark) | 0.817 [0.812, 0.821] | 0.0666 | 0 |

  GB-01 beats the scorecard by **+0.021 AUC [+0.019, +0.023]** and the RiskCalc-style model by
  **+0.018 [+0.016, +0.020]** (firm-clustered paired bootstrap, `reports/tournament/scorecard_r1.md`).
  On 2010 alone, the cleanest test year (see Section 5), GB-01 beats the scorecard by +0.019
  [+0.016, +0.023].
- **PD level for the holdout:** shift all PDs so they average **1.49%**, with a 0.03% floor and a
  99% cap (D-073). Reason: default rates rose every year (1.0% → 1.7%), so a model trained on the
  past under-predicts by 20–45%. Under-predicting costs about 1.5× more than over-predicting
  (D-074).
- **Biggest open risks:** (1) which label window Roger uses (unanswered); (2) which period the
  holdout covers; (3) new firms are ranked less well (AUC 0.787 vs 0.844); (4) our feature set
  was chosen while looking at 2008–09 (so we quote 2010 as the honest number).
- **Next:** team decisions in Section 6 → Day 3: harness, deck, claim audit, red-team.

---

## 2. The problem as we formulated it (and why)

| Choice | What we did | Why (evidence / source) |
|---|---|---|
| Unit of analysis | One row = one firm's annual statement → one PD (D-001) | The data is firm-year statements, not loans (assignment App. B). This answers Roger's "What does this mean?" on "application" |
| Two uses of the PD | Ranking (approve/decline) needs **power**; pricing needs the right **level** | Power and calibration are separate validation dimensions (R-07 Sec. 1; R-01 Sec. 4.1; R-16 Sec. III) |
| Cut-off | Stated as an **assumption** from the assignment's transaction flow | Stein (2005) calls cut-offs common but sub-optimal; pricing is better (R-08 p. 1233) |
| Default label | Default in the 12 months **starting 6 months after** the statement date (D-002) | Italian law: approval ≤120 days (≤180 by statute) + filing ≤30 days (R-28, R-29). The bank can't use a statement before it's available. Pending Roger (D-002a) |
| Data window | Labels for fs_year 2007–2010; 2011 is 11/12 observed; 2012 can't be labelled | Defaults are tracked to 2013-05-31 (D-009) |
| Evaluation | Walk-forward: train on years < t, test on year t (2008, 2009, 2010) | Stein (2007) Sec. 3 p. 94; RiskCalc v3.1 Sec. 4.2–4.3 |
| Metrics | AUC (AR = 2·AUC − 1, R-14); log loss; calibration-in-the-large; binned observed vs predicted; Brier labelled "overall accuracy" | Brier is not a calibration measure (R-16 p. 32); metric noise depends on # defaults (R-07 Sec. 4.1) → CIs + default counts everywhere |
| Uncertainty | Firm-clustered paired bootstrap (200 draws) | Firms repeat across years, so row-level bootstrap would overstate certainty |
| No NLP, no macro, no growth | REJECTED (D-003, D-004, D-020) | No text in data; external data needs clearance weeks ahead; growth needs other rows and gained only +0.002 AUC, all on already-seen firms |
| No Basel sentence | Removed (D-017) | Hidden white-ink instruction in the assignment PDF |

---

## 3. What we did, step by step, and the reason for each action

### 3.1 Day 0: setup (orchestrator)

| Action | Reason |
|---|---|
| Used the `datascience` conda env; installed `pytest`, `lightgbm`, `shap` | Kiet's preference; only what the plan needed (no xgboost, interpret or pygam after the cuts) |
| `git init`, `.gitignore` excludes raw data, parquet (except frozen splits) and pickles | Undo safety with many agents writing files; the 321 MB CSV stays out of git |
| Drafted `docs/EMAIL_DRAFT_roger_ta.md` (5 questions, each with a default assumption) | Roger's answers decide the label and level; each question has a fallback, so nothing blocks on him. **Not sent by Claude** |
| Compressed the schedule to 3 days (D-007) | Team deadline. Kept the referee, red-team and decision log, because formulation + documentation = 25/30 pts |
| Dropped the GAM/EBM challenger (D-008) | Time; interpretability is covered by RiskCalc-style transforms + SHAP. Logged as "not tested", not "worse" |
| Kept LightGBM, not XGBoost | Same capability (monotone constraints), faster, already installed; XGBoost paper (R-18) unverified. Kiet agreed |
| Removed the Basel II sentence from CLAUDE.md and the report template (D-017) | Hidden trap. CLAUDE.md was read-only, so it was made writable at Kiet's request |
| Ran agents in parallel where they don't share files; challengers wrote logs to their own folders, merged by the orchestrator | Avoid edit collisions in the shared docs |
| Re-computed every agent's headline numbers from their saved files before reporting | CLAUDE.md rule 4: every number comes from a file. All checks matched |

### 3.2 Day 1: data audit (data-auditor) → `reports/data/`

| Finding | Number | What we did |
|---|---|---|
| Panel size | 1,023,552 firm-years; 237,711 firms; 16,863 defaulting firms | One row per firm-year (D-001) |
| `def_date` is firm-level; dates run to 2013-05-31 although App. B says "no info after 2012" | 1,416 firms / 5,570 rows with 2013 dates; monthly counts look like normal tracking | Use as given; sensitivity with 2013 removed. **No test fold depends on it** (D-009) |
| Window start changes the level a lot | 2007–10 default rate: 1.41% (start 6) vs 1.07% (start 0) | Start 6 adopted; the panel carries 0/3/6 labels (D-002) |
| Rows with a default in the first 6 months after the statement | 4,237 | Excluded: the bank couldn't have used that statement in time |
| Test folds | 2008: 2,064 defaults (1.28%); 2009: 2,752 (1.62%); 2010: 2,968 (1.70%) | Frozen in `data/splits/` (D-015) |
| Debt breakdown (bank / other / trade) missing as a block | 11.8% of rows; 13.6% for small firms vs 1.2% for large; persistent within firm | Plausibly abbreviated statements (art. 2435-bis, R-27, "plausible"); weak signal (1.57% vs 1.38%) |
| `margin_fin` blank on all negative-equity rows; `roe` blank beyond ±150% | default rate when blank: 10.3% / 8.0% vs ~1% | Recompute from components, don't impute (Roger G2) |
| Field meanings differ from the dictionary | `roa` = operating profit / assets; `goodwill` ≈ value added; `taxes` ≈ pre-tax profit; column spellings differ | Logged as data-derived hypotheses (D-010, D-024); the harness accepts both spellings |
| `HQ_city` = ISTAT province codes; ATECO → 2007 sections | 45/110 provinces have <30 defaults; sections O, P, U pooled | Region / macro-area + pooled sections (D-012, D-013, D-023) |
| Shared `labels.py` bug | Would mark 2,357 incomplete 2011 rows (all defaults) as usable | Not used; fix proposed (D-011 / D-068), **needs our approval** |

### 3.3 Day 1: literature (literature-scout) → `docs/REFERENCES.md`, `docs/STUDY_GUIDE_literature.md`

- 15 references checked against their text on Day 1, more on Day 2 (R-04, R-10, R-33, R-34, R-35).
- **Contradictions found in our own plan:** "RiskCalc is the industry standard" (contradicted, R-05
  p. 10); "Brier measures calibration" (contradicted, R-16 p. 32); "the bank uses a cut-off"
  (assumption only; Stein 2005 recommends pricing).
- **Roger = Stein.** RiskCalc v3.1 (R-01), Stein 2005 (R-08) and Stein 2007 (R-07) are his.
  Our design follows them: separate power/calibration, walk-forward, CIs, per-year AR, ratio
  families, per-ratio transforms.

### 3.4 Day 1: features (feature-architect) → `src/features/build.py`, `docs/FEATURE_SPEC.md`

| Decision | Reason / evidence |
|---|---|
| fit/transform design: every fitted quantity (winsor bounds, sector medians, category maps) comes from training rows only; transform works row by row on messy input | Leakage rule + harness rule "no statistics of the holdout batch". 14/14 tests pass |
| Core set of 10 ratios across RiskCalc families (size, leverage, liquidity, profitability, coverage, activity, asset structure) + 3 categoricals | R-01 Sec. 3.1: one ratio per family; too many overfit. Max correlation 0.67, max VIF 1.98 (Roger M8) |
| Full set (~20) for the GBM | Trees handle collinearity; the full set beats core-only by +0.008 AUC (GB-03, D-055) |
| Missing data: rebuild from accounting identities first; debt-breakdown block → sector share of the firm's own debt + a flag; NaN kept for trees | Beats a plain sector median on the block-missing rows (AUC 0.553 vs 0.462 for bank-debt ratio) (D-021) |
| Expected-sign table, used for monotone constraints and face-validity checks | Roger's face-validity criterion; credit-officer trust |
| Signed-log scale for heavy-tailed ratios; caps at training percentiles | Ratio pathologies (D-019); RiskCalc v1 truncated at p2/p98 (R-05 p. 55, Day-2 finding) |
| Data contradicts expected sign: `asset_turnover` inverted-U → unconstrained; `cfo_to_debt` strongest alone but flips next to `roa_net` (corr 0.80) → GBM only | D-022 |

### 3.5 Day 2: tournament round 1 (three challengers → referee) → `reports/tournament/scorecard_r1.md`

| Model family | What it is | Best result | What it teaches us |
|---|---|---|---|
| Scorecard (SC-01..04) | One-ratio logit, core logit, WoE scorecard, Altman-lineage | SC-03 WoE: AUC 0.817 | The benchmark. One ratio (ROA) alone gets 0.746; transparent models get to about 0.80–0.82 |
| RiskCalc-style (RC-01..03) | Per-ratio default-rate curves → probit + industry → final map (R-01 form) | RC-02: 0.819 | Driver curves a loan officer can read. Same 10 ratios every year. Only marginally beats the scorecard (+0.001 to +0.003) |
| LightGBM (GB-01..06) | Monotone-constrained boosting, full features, native NaN, Platt calibration in-window | GB-01: 0.838 | Clear, significant gain. Removing constraints adds +0.0065 AUC but 2–3 ratios then move the wrong way, so it's rejected (D-051) |

**Referee verdicts (D-060..D-069):**
- All experiments CLEAN except **RC-03 = CONCERN**: its "drop wrong-signed ratios" rule was
  written after seeing test results. It ties RC-02, so we use RC-02.
- The referee refit fold 3 of SC-03, RC-03 and GB-01 from code and got the submitted predictions
  back exactly, so they're reproducible.
- **Caveat affecting everyone (D-062):** the feature set was designed using 2007–2009, which
  includes the test years of folds 1–2. So **2010 is the honest headline**.
- `tangible_ratio` as GB-01's #1 driver is plausible: it keeps its direction *within* every
  industry (within-group AUC 0.56–0.73).

### 3.6 Day 2: calibration (calibration-specialist) → `reports/calibration/`, D-070..D-079

| Question | Answer | Evidence |
|---|---|---|
| Why do all models under-predict? | Each training window's default rate trails the next year (1.0/1.1/1.3% vs 1.28/1.62/1.70%). Calibration slope ≈ 1, so it's a **level** problem, not shape | `scorecard_r1.md`, `methods_by_year.csv` |
| Best way to set the level in the walk-forward | Rule "e": the default rate of the last 6 training months, annualised. Δlog loss −6.3e-4 [−7.1, −5.6]; recovers 88% of a perfect per-year level | `ct_rules_summary.csv`. **Caveat:** picked from 7 rules after seeing the years → optimistic |
| How to move PDs to the level | **Prior shift** (logit intercept shift, Saerens eq. 4). AUC unchanged (verified). Isotonic rejected (slope 0.92, AUC −0.0008); Platt rejected (hidden trend); Saerens-EM rejected (needs the holdout batch, and was wrong: 2.18% vs 1.28%) | `method_params.csv`, `auc_invariance_check.csv`, `saerens_em_diagnostic.csv` |
| GB-01 before → after, observed/predicted | 2008: 1.21 → 1.02; 2009: 1.36 → 1.05; 2010: 1.43 → 1.18 | Re-computed from `experiments/calibration/CAL-01` |
| Level for the holdout | **1.49%** = latest cohort (fs2011, 1.43%, 11/12 months) + a small hedge. Equal-weight 2008–11 gives 1.51%; worst-case optimum 1.49% | `final_level_candidates.csv`, `harness_level_params.json` |
| Too low vs too high | 30% too low costs 8.2–10.9e-4 log loss; 30% too high costs 5.6–7.4e-4 → **too low is about 1.5× worse** | `asymmetry_gb01.csv` (D-074) |
| Label-window sensitivity | Same rule gives 1.37% (start 0), 1.39% (start 3), 1.49% (start 6). Shipping start-6 when the truth is start-0 costs only 0.57e-4 | `q5_label_window_levels.csv`, `q5_label_window_cost_matrix.csv` |
| Include 2011 in the final fit? | Yes, with an 11-month label from the panel (not `make_labels`). Dropping the newest cohort cost 2.6–3.5e-4 and 0.003–0.004 AUC on the fold-3 analogue (point estimates, no CI) | `q6_partial_cohort_test.csv` (D-077) |
| Floor / cap | 0.03% floor (by analogy with the Basel corporate floor, R-15 para. 285): binds on 31 rows, 0 defaults. 50% cap rejected (rows above 50% default at 65%) | `floor_cap_effect.csv` (D-075) |
| Grade table for loan officers | 11 PD bands; observed default rate rises across all 11 | `grade_table_2010.csv` (D-078) |

**Grade table, 2010, after calibration** (`reports/calibration/grade_table_2010.csv`):

| Grade | PD band | Firms | Defaults | Mean PD | Observed DR |
|---|---|---|---|---|---|
| 1 | 0.03–0.10% | 6,158 | 4 | 0.08% | 0.06% |
| 2 | 0.10–0.25% | 34,194 | 51 | 0.17% | 0.15% |
| 3 | 0.25–0.50% | 33,805 | 119 | 0.36% | 0.35% |
| 4 | 0.50–0.75% | 21,311 | 160 | 0.62% | 0.75% |
| 5 | 0.75–1.00% | 17,390 | 194 | 0.87% | 1.12% |
| 6 | 1.00–1.50% | 22,714 | 365 | 1.22% | 1.61% |
| 7 | 1.50–2.50% | 20,369 | 502 | 1.91% | 2.46% |
| 8 | 2.50–5.00% | 11,961 | 571 | 3.38% | 4.77% |
| 9 | 5–10% | 3,764 | 333 | 6.81% | 8.85% |
| 10 | 10–20% | 1,552 | 218 | 13.87% | 14.05% |
| 11 | 20–99% | 1,294 | 451 | 38.56% | 34.85% |

Read it this way: the ordering is right (DR rises with every grade). The safest 42% of firms
(grades 1–3) hold only 5.9% of defaults. But in 2010 grades 4–9 ran 20–40% **above** their PD,
because the level rule under-shot that stressed year. That's the cycle risk the deck must show.

### 3.7 Day 2: Moody's literature (literature-scout) → `docs/LIT_moodys_feature_cleaning_2026-10-30.md`

- **RiskCalc 4.0:** only the **France** edition was found and read (Buitrago, Makarov, Zhao &
  Dwyer 2015, R-33). No general 4.0 methodology and no Italy edition exist publicly. Always say
  "RiskCalc France 4.0", never "RiskCalc 4.0 says…" about Italy.
- **Confirms our choices:**
  - truncating extremes (R-05 p. 55);
  - ratio levels over trends (R-05 pp. 38–40);
  - macro left out of the model and handled separately (R-05 p. 41; R-10 p. 21);
  - VIF/correlation practice (R-33 p. 18);
  - Moody's Italy cycle adjustment uses public-firm data + unemployment (R-04), which we can't
    use (no external data), so our level must come from training data.
- **Pushes against our choices:**
  - mean/median imputation understates PDs (R-34). Our GLMs use median fill and have no
    missingness flag in the core;
  - Moody's drops implausible statements from training data (R-33 p. 6); we keep every row;
  - cash-flow-to-liabilities is the heaviest family in RiskCalc France (19.2%), but our core
    excludes `cfo_to_debt` (the GBM has it);
  - Moody's sets the PD level with external data; we can't.

---

## 4. What we learned (hypotheses from the PLAYBOOK)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: leverage, profitability, liquidity carry most of the signal | **Mostly confirmed.** Top drivers: ROA, equity ratio, tangible-asset share, short-term debt, interest burden. Asset structure (tangible ratio) matters more than expected | `reports/features/univariate.md`; GB-01 `explain.md` |
| H2: missingness is structural and informative | **Partly.** Structural: yes (block-missing, small firms, persistent). Informative: weak for the debt block (1.57% vs 1.38%), strong for blanks caused by negative equity | D-016, D-042 |
| H3: monotone GBM beats the probit by a small, *non-significant* margin | **Rejected.** The margin is +0.018 AUC and clearly significant | D-065 |
| H4: the level choice matters more than the algorithm | **Not confirmed in walk-forward** (best level gain 6.3e-4 vs 16–24e-4 from the algorithm), **but true if the level is missed by ≥30%** | D-074 |
| H5: new firms are ranked worse | **Confirmed.** GB-01 AUC 0.844 seen vs 0.787 new; RC 0.826 vs 0.765. New-firm PDs are also too spread out (slope 0.89) | D-047, D-078 |

**Five lessons to say in the deck:**
1. Power and level are separate problems with separate fixes. Recalibration fixed the level and
   didn't touch the ranking (R-07 pp. 89–91; verified on our data).
2. Face validity has a measurable price: 0.0065 AUC. We pay it because a model where more equity
   *raises* PD wouldn't be trusted or defensible.
3. The credit cycle shows up as a moving level. A model trained on calm years under-predicts in a
   downturn, and under-predicting is the more expensive error for pricing.
4. Missing data has accounting causes. Rebuild from identities before imputing.
5. A transparent RiskCalc-style model gets within 0.02 AUC of boosting. It's the right
   explainability companion and fallback.

---

## 5. Where we're weak (what Roger or the red-team will attack)

1. **D-062 selection leakage:** features chosen on 2007–09 data. Mitigation: quote 2010; state it.
2. **Calibration rule picked after seeing results** (7 rules tried). Mitigation: the holdout
   level uses a simple, pre-stated rule (latest cohort + hedge), and we say so.
3. **2010 mid-grades under-predicted by 20–40%.** If the holdout is a stressed year, we'll be low.
4. **New firms:** lower power and over-spread PDs. No fix yet.
5. **Label window unconfirmed** (D-002a). The calibration cost is small (0.57e-4), but the
   formulation story depends on it.
6. **No training-data filter, median imputation in GLMs:** contradicts Moody's practice (R-33,
   R-34); untested.
7. **Selection bias:** we can't verify the data is only booked loans. State it as an assumption.
8. **The red-team hasn't reviewed round 1 yet** (skipped to save time; planned for Day 3).
9. `reports/calibration/summary.md` was not written (the agent's file write failed). All numbers
   are in the CSVs and D-070..D-079; the report-writer should build the summary from them.
10. **Unread or unverified sources:** Ohlson, Shumway, XGBoost, Platt, isotonic, EBM, TreeSHAP,
    RiskCalc Italy. Don't describe their content.

---

## 6. Decisions we need to make together (with Claude's recommendation)

| # | Decision | Options | Recommendation and reason |
|---|---|---|---|
| 1 | Final model | GB-01 / RC-02 / ensemble | **GB-01**: best on both metrics, zero sign violations, SHAP reason codes. Keep RC-02 as benchmark + driver-curve figures + fallback |
| 2 | Holdout level | 1.49% (hedged) / 1.43% (latest cohort) | **1.49%**: under-prediction is ~1.5× costlier, and the cycle was rising; the difference is small either way |
| 3 | Include 2011 in the final fit | yes (11-month label) / no | **Yes**: the newest cohort carries the latest level and ranking info |
| 4 | Fix `src/common/labels.py` (D-068) | approve / leave | **Approve**: required if 2011 goes into the fit via shared code |
| 5 | Faster bootstrap in `metrics.py` (D-069) | approve / leave | **Approve**: identical results (verified to 1e-16), usable on 500k rows |
| 6 | D-062 caveat | accept + quote 2010 / redo feature selection per fold | **Accept** for now; redo only if we use the buffer (Section 9) |
| 7 | Email to Roger/TA | send / already sent? | **Send if not sent.** Label window and holdout period are the two answers that matter |
| 8 | Who checks what | — | Someone verifies `src/data/codes.py` tables; someone spot-checks R-33/R-34 paraphrases and the Stein/Basel page numbers |
| 9 | Keep the 3-day pace or use the buffer | finish Nov 1 / round 2 | See Section 9 |

---

## 7. What's still left on the table

**Must do (Day 3)**
- [ ] Harness: refit GB-01 on 2007–2011 (with the 11-month 2011 label), apply the 1.49% shift,
      floor/cap, freeze params; stress tests (NAs, blanks, odd codes, unseen categories, both
      column spellings, one-row input); computational complexity note (the GBM agent measured
      ~1 s per 100k rows, linear in rows).
- [ ] Deck (15–25 slides) + technical appendix: business problem, formulation, data, features,
      models, evaluation, calibration, explainability (driver curves, 3 example firms, grade
      table), deployment, limitations, what didn't work, bibliography, AI-use statement,
      contributions appendix. File name contains "Team Turquoise".
- [ ] Literature claim audit of the deck; red-team review (round 1 + deck).
- [ ] Moody's MUST: add the Moody's citations to D-019, D-020, D-004, D-025 (~20 min).
- [ ] Each member reads their assigned papers (`docs/STUDY_GUIDE_literature.md` Sec. 7) and can
      explain every slide.

**Should do if time allows (each is small)**
- [ ] Imputation check (R-34): are PDs too low on rows with median-filled core inputs? If yes, add an
      `n_core_imputed` flag.
- [ ] Training-only filter for gross balance-sheet violations (≤0.16% of rows); keep only if 2010
      improves.
- [ ] AUC by size band, by sector (real estate separately) and seen/new, for the deck (R-33 style).
- [ ] One sentence defending `cfo_to_debt` being GBM-only (collinear with ROA, D-022).
- [ ] Optional test submission of the harness to the TA (allowed one week before Nov 16).

**Nice / round 2 (only with the buffer)**
- [ ] Redo feature selection inside each fold (removes the D-062 caveat).
- [ ] Shape fix for new firms (slope 0.89), e.g. a separate calibration slope for unseen firms.
- [ ] Ensemble GB-01 + RC-02 (allowed only if it beats GB-01 by the promotion rule and still
      explains per firm).
- [ ] EBM/GAM challenger (cut in D-008).
- [ ] `legal_group` + `macro_area` in the RiskCalc model (+0.0013 AUC inner, not tested
      out-of-time).

**Deliberately not doing (and why)**
- NLP (no text), macro data (not cleared), growth/lag features (+0.002 AUC, only for seen
  firms, breaks the "current record only" simplicity), XGBoost (equivalent to LightGBM here),
  unconstrained GBM (fails face validity).

---

## 8. Strategy going forward

1. **Story first, model second.** Formulation + documentation are 25/30. The model is good
   enough; the remaining points are in explaining *why* each choice serves approval and pricing.
2. **Lead with Roger's own framework:** separate power and calibration, walk-forward, CIs with
   default counts, per-year results, ratio families, per-ratio driver curves. Then show where
   we go beyond it (monotone GBM + SHAP) and what it costs (0.0065 AUC for face validity).
3. **Be honest about the weak spots** (Section 5) in a limitations slide. Roger rewards foundation
   and penalises unsupported claims.
4. **Harness robustness beats the last 0.001 AUC.** A crash disqualifies us from the tournament.
5. **Every member must own a part and understand the whole.** Grades are individual.

**Suggested split for discussion** (just a starting point, please change it):

| Member | Owns | Must also be able to explain |
|---|---|---|
| Kiet | Harness + calibration (level, floor, grade table); final model spec | Data/label decisions |
| Desmond | Data + features (label window, missingness, ratio families, expected signs) | Model comparison |
| Chloe | Evaluation + literature (walk-forward, CIs, power vs calibration, Moody's lessons) | Harness and level |

---

## 9. Timeline

| When | What |
|---|---|
| Nov 1 (Day 3) | Team decisions (Section 6) → harness → deck draft → claim audit → red-team |
| Nov 2–8 (optional buffer) | Round 2 items if wanted; human reading and rehearsal; send the harness to the TA by ~Nov 9 for a test run |
| Nov 9–15 | Final polish, contributions appendix, AI-use statement, explain-back rehearsal |
| **Nov 16** | Submit deck + code |

---

## 10. File map

| What | Where |
|---|---|
| All decisions with evidence | `docs/DECISION_LOG.md` (D-001 … D-079) |
| Experiment registry | `docs/EXPERIMENTS.md` |
| References, claim map, claims we must NOT make | `docs/REFERENCES.md` |
| Plain-language literature guide (with runnable examples) | `docs/STUDY_GUIDE_literature.md` |
| Moody's feature/cleaning digest | `docs/LIT_moodys_feature_cleaning_2026-10-30.md` |
| Data audit | `reports/data/00_labels.md`, `01_profile.md`, `02_missingness.md` |
| Feature spec + expected signs | `docs/FEATURE_SPEC.md`, `src/features/build.py` |
| Tournament scorecard | `reports/tournament/scorecard_r1.md` (+ CSVs, figures) |
| Calibration evidence | `reports/calibration/*.csv`, `experiments/calibration/harness_level_params.json` |
| Explainability examples | `experiments/gbm/GB-01/explain.md`, `experiments/riskcalc/RC-03/explain.md`, `experiments/scorecard/SC-03/explain.md` |
| Email draft (not sent) | `docs/EMAIL_DRAFT_roger_ta.md` |
| AI-use log | `docs/AI_USE_LOG.md` |

*AI use: this briefing was compiled by Claude Code (orchestrator) from agent outputs. All headline
numbers were re-computed from saved files. The team must review it before relying on it.*
