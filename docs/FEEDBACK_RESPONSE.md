# Response plan to Roger's proposal feedback

Every comment from the marked-up proposal, what it means, and where in the project it is fixed.
Agents: before finalising anything, check your output against the relevant rows.

## Overall comments

| # | Roger's point (paraphrased) | What it means for us | Fix / owner |
|---|---|---|---|
| G1 | Too much "data in → predictions out", not enough formulation; ideas divorced from banking reality | Every modelling choice must be motivated by how the bank uses the PD | CLAUDE.md §1; every decision log entry needs a business reason |
| G2 | No substantive missing-data strategy, though relationships useful for it were listed | Use accounting identities to reconstruct; study whether missingness is informative | data-auditor §3, feature-architect |
| G3 | NLP sentiment analysis promised without a plan | The data has no text. Drop it and say why | feature-architect logs REJECTED |
| G4 | Economic conditions promised without a method | Per-year evaluation incl. 2008–09 & 2011–12; calibration central-tendency study; external macro only if cleared | calibration-specialist, PLAYBOOK Q4 |
| G5 | Model requirements not connected to explaining the model to bankers | Per-firm reason codes, driver curves, rating-grade table | all challengers' `explain.md`, report-writer |
| G6 | Include team name | Cover + file name: Team Turquoise | report-writer |
| G7 | No AI-use statement → assumed no AI | Honest statement + AI_USE_LOG | all agents, report-writer |
| G8 | Standard DS proposal, not finance-oriented; review default-prediction literature | Literature map incl. RiskCalc, validation, banking | literature-scout |
| G9 | Statements without foundation about banks/firms; fuzzy understanding of methods | Claim audit; assumptions labelled as such | literature-scout claim audit, red-team-roger |
| G10 | Shallow literature review; no banking references | ≥ 15 verified refs across finance, banking, validation, ML | literature-scout |
| G11 | Think through practical implementation of every idea | No promise without code + result | red-team-roger "How, specifically?" |

## Margin notes

| # | Location in proposal | Roger's question | Answer we must be able to give |
|---|---|---|---|
| M1 | PD vs EL paragraph | "How do these two differ? Why are you telling me this?" | EL = PD×LGD×EAD only matters if we use it. Keep one sentence: we estimate PD only; LGD/EAD are outside the data. Use EL only in the business-case slide (pricing illustration). |
| M2 | "Only paid-off or defaulted loans are eligible" | "Why?" / "How does this relate to the business problem?" | The data is **firm-year financial statements**, not loans. Unit of analysis = firm-year (one statement → one PD), which is exactly what the bank scores at application. Rewrite the formulation around this. |
| M3 | "Dataset updated constantly" / rolling schema | "What does this mean?" | Remove. Describe the actual design: annual statements, walk-forward re-estimation. |
| M4 | NLP on notes/management report | "Are you promising to deliver this? How?" | Not deliverable — no text in data. (G3) |
| M5 | Unit of analysis = application | "How does this relate…?" / "What does this mean?" | See M2. Repeated firms → firm-clustered bootstrap and seen/new-firm split. |
| M6 | Late publication of reports | "How do you know this?" | Cite Italian Civil Code deadlines (approval & filing) once literature-scout verifies them, plus data evidence (stmt_date if present). Otherwise state as assumption + sensitivity test of label windows. |
| M7 | ATECO grouping into ~7 groups | "How?" | Document the mapping to ATECO 2007 sections, the minimum-default-count rule, and show default rates by group. |
| M8 | VIF collinearity check | "How?" | Show VIF table and the action rule (drop/combine/regularise) with results. |
| M9 | Raw amounts (financial expenses, cash flow) | "How do you plan to use this?" | Either convert into a ratio with a stated role or drop. |
| M10 | Walk-forward training | "How, specifically, will you implement this?" | `data/splits/walkforward.json` + README; shared code. |
| M11 | Hyper-parameter tuning | "How, specifically, will tuning work?" | Inner time-ordered validation inside each training window; search space logged per fold. |
| M12 | Per-round training | "Does each year's model have different variables?" | Feature set fixed by the spec (or selected inside each window) — report selection stability across folds; final model uses one fixed set re-estimated on all data. |
| M13 | ROC + CAP/AR | "How do these differ? What additional info?" | AR = 2·AUC − 1: same information (Engelmann, Hayden & Tasche 2003 — verify). Report one, mention the identity. |
| M14 | Metrics table | "Are there more robust metrics?" | Add log likelihood / log loss and Brier for probability accuracy; firm-clustered bootstrap CIs; per-year stability (Stein 2007 — verify). |
| M15 | Approval cutoff | "How do you know the bank uses a cut-off?" | Assignment's transaction flow (loan made only if the rate is within guidelines) + Stein (2005) cutoff/pricing framework; otherwise phrase as assumption. |
| M16 | Selection bias / RiskCalc as industry standard | "How do you know this?" | Source each statement (literature-scout) or reword as assumption. Selection bias: only borrowers the bank lent to — we must check whether the data actually is only booked loans (the assignment calls them "previous borrowers"). |
| M17 | Credit officers' trust | implicit | Face-validity matrix + reason codes. |
