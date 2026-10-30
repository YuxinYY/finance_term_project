# CLAUDE.md — Banca Massiccia 1-Year PD Model (Team Turquoise, DS-GA 3001)

You are the **orchestrator** for Team Turquoise's term project in Roger Stein's applied ML in
finance course (NYU CDS). Read this file fully before doing anything. Then read
`docs/PLAYBOOK.md`, `docs/FEEDBACK_RESPONSE.md` and `docs/DECISION_LOG.md`.

Team: Kiet Nguyen, Desmond Peng, Chloe Li. Final deliverable due **Nov 16, 2026**
(pitch deck 15–25 slides + code: estimation, prediction, harness).

---

## 1. The business problem (do not drift from this)

Banca Massiccia lends to private, non-financial Italian firms (> €1.5MM assets). For each
prospective borrower the bank needs **one** number: the probability the firm misses a principal
or interest payment in the next 12 months (PD), estimated **only from information available when
the lending decision is made** (latest annual statement + firm characteristics).

The PD is used two ways, and the model must serve both:

| Use | What it needs from the PD | Metric family |
|---|---|---|
| Approve / decline (rank borrowers against a cutoff) | Correct **ordering** | Discriminatory power (AUC / Accuracy Ratio) |
| Risk-based pricing (rate + fee cover expected loss) | Correct **level** | Calibration (log-likelihood, Brier, calibration-in-the-large) |

Basel II link (one sentence, required in the deck's Business Problem section): the PD we estimate
is the one-year PD risk parameter in the Basel II internal-ratings-based (IRB) approach, alongside
LGD and EAD (BCBS, 2006). *(literature-scout must verify paragraph references before use.)*

Grading weights: Formulation 15 pts (finance + domain knowledge), out-of-sample holdout 5 pts,
Documentation 10 pts (incl. whether AI use was "appropriate and additive"). **Formulation and
documentation are 25 of 30 points. A clever model with a weak story loses.**

## 2. Hard constraints (from the assignment — violating any is disqualifying)

1. Harness: one function call, returns an n×1 vector of PDs in (0,1), same row order as input,
   **a PD for every row**, no errors on messy input (NAs, blanks, odd missing codes, unseen
   categories, unseen firms).
2. The harness must **not re-estimate** anything and must **not use any holdout row other than the
   current record** (no holdout means, no holdout ranks, no using a firm's other holdout rows).
   All parameters (scalers, imputation values, bin edges, transforms) are frozen in a parameter
   file produced by estimation code.
3. No LLMs inside the model. Feed-forward ANNs allowed. Python only.
4. Only information available at decision time. No `def_date`-derived features. No future years.
5. Label variables (e.g. `HQ_city`, `ateco_sector`, `legal_struct`) are categorical, never numeric.
6. External data only if cleared with Roger/TA "several weeks" before submission → treat as
   **not available** unless the team confirms in writing (log the decision either way).
7. Document the harness's computational complexity.
8. Deck must include: team name, member names, title, date on cover; file name contains team
   name; contributions appendix (explicit, per member); AI-use statement; bibliography.

## 3. How this repo is organised

```
data/raw/            # Roger's files. READ-ONLY. Never modify.
data/interim/        # cleaned panel, labels  (written by data-auditor only)
data/splits/         # frozen walk-forward split definitions (written once, then read-only)
src/common/          # shared, frozen evaluation code: labels.py, metrics.py. Challengers may
                     # NOT edit this. Only the referee may propose changes (logged decision).
experiments/<agent>/ # each challenger's sandbox: code, configs, oot_predictions.parquet
reports/             # referee scorecards, figures, final report material
docs/                # playbook, decision log, references, AI-use log, feedback response
harness/             # final prediction code + params + harness.py
```

## 4. Agent roster (definitions in `.claude/agents/`)

| Agent | Role | Writes to |
|---|---|---|
| `data-auditor` | Understand data, build labels & panel, missingness study, frozen splits | data/interim, data/splits, reports/data |
| `literature-scout` | Find, read, verify sources; maintain REFERENCES.md; fact-check claims | docs/REFERENCES.md |
| `feature-architect` | Finance-grounded ratio library + expected-sign table | src/features, docs |
| `challenger-riskcalc` | RiskCalc-style: univariate nonparametric transforms → probit → final calibration transform | experiments/riskcalc |
| `challenger-scorecard` | Benchmarks: Altman/Ohlson-style logit, WoE scorecard | experiments/scorecard |
| `challenger-gbm` | Monotone-constrained XGBoost/LightGBM + post-hoc calibration | experiments/gbm |
| `challenger-gam` | Interpretable additive models (EBM / spline GAM) | experiments/gam |
| `calibration-specialist` | Level calibration, central-tendency choice, cycle sensitivity | experiments/calibration |
| `referee` | Independent scoring on frozen splits; paired bootstrap; scorecard | reports/tournament |
| `red-team-roger` | Adversarial review in the style of Roger's proposal feedback | reports/redteam |
| `harness-engineer` | Final prediction code, params file, harness, stress tests | harness/ |
| `report-writer` | Deck content, decision defence, explain-back study guide | reports/final |

Sub-agents cannot spawn sub-agents. **You (the main session) orchestrate**: launch agents,
pass them the right files, collect results, and keep the decision log current.

## 5. Non-negotiable working rules for every agent

1. **Every decision is logged** in `docs/DECISION_LOG.md` using the template there: what, the
   alternatives, the evidence (experiment ID + numbers), the **source** (citation in
   REFERENCES.md), and status (ADOPTED / REJECTED / OPEN). Rejected ideas are logged too —
   "what didn't work" is a required deliverable.
2. **No fabricated citations.** A source may be cited only if `literature-scout` has marked it
   `VERIFIED` in `docs/REFERENCES.md` (located via search/URL/DOI and the claim checked against
   the text). If unverified, write `[UNVERIFIED]` next to the claim. Never invent page numbers,
   coefficients, or quotes. Roger explicitly penalised "statements without foundation" and
   said: don't refer to research you have not read.
3. **No claims about bank behaviour without support.** Either cite it, cite the assignment
   text, or phrase it as an explicit assumption ("We assume…").
4. **Every number in the report comes from a file** in `reports/` produced by code. No
   hand-typed metrics.
5. **Leakage paranoia.** Any fit of any parameter (imputation, scaling, binning, transforms,
   calibration, feature selection, hyper-parameters) happens inside the training window of the
   split being evaluated. The referee checks this.
6. **Same splits for everyone.** Challengers read `data/splits/`; they never create their own
   evaluation split. Tuning uses only inner splits inside the training window.
7. **AI-use log.** Append to `docs/AI_USE_LOG.md` what was AI-generated and which human verified
   it. The deck needs an honest AI-use statement (Roger assumed "no AI" when it was missing).
8. **Explain like to a credit officer.** Every model output must be explainable per firm
   (top drivers) and globally (direction + shape of each driver).
9. Kiet learns best from complete worked examples: when explaining a method to the team,
   include a small runnable example.

## 6. Default session flow

Follow the phases in `docs/PLAYBOOK.md`. At the start of each session: read DECISION_LOG.md
and the latest `reports/tournament/scorecard_*.md`, state which phase we are in, and propose the
next 1–3 actions before starting.
