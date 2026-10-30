# REPORT TEMPLATE — deck (15–25 slides) + technical report

Order follows the assignment ("in approximately the order given"). Each slide title states the
takeaway. Grading weights in brackets.

## Cover
Team Turquoise · Kiet Nguyen, Desmond Peng, Chloe Li · project title · date. File name contains
"Turquoise".

## 1. Business understanding [Formulation 15]
1. The problem: one-year PD for private Italian borrowers; two uses (cutoff ranking, pricing level).
   One sentence linking to Basel II IRB PD (verified source).
2. Why it is worth money: mispricing and adverse selection (Stein 2005 — verified claims only);
   simple illustrative business case (e.g. EL-based pricing under incumbent vs our model).
3. What has been done: accounting-ratio models (Altman, Ohlson), private-firm practitioner models
   (RiskCalc), ML benchmarks — what we borrow, what we do differently, why ours is better and where.

## 2. Formulation [Formulation]
4. Unit of analysis, label window, timing of information (diagram: FY end → filing → window).
5. Why a finance-informed formulation: ratio families with expected signs; monotonicity; cycle-aware
   calibration; out-of-time validation.

## 3. Data understanding & preparation [Documentation 10]
6. Data facts that shaped decisions (default rate by year, missing blocks, repeated firms).
7. Missing data strategy with evidence (identity reconstruction, indicators, informativeness).
8. Biases: selection (booked borrowers), survivorship, censoring at 2012-12-31, label definition.

## 4. Modelling
9. The tournament: who competed, under what identical rules.
10. Final model full specification (variables, transforms/constraints, calibration, clipping).
11. Drivers: global shapes vs expected signs (face validity).
12. Per-firm explanations: three worked borrowers in loan-officer language.

## 5. Evaluation
13. Power: pooled and per-year AUC (AR noted) with firm-clustered CIs vs benchmarks.
14. Probability accuracy: log loss, Brier, calibration plot, mean PD vs realised by year.
15. Robustness: crises years, new vs existing firms, size/industry slices.
16. What didn't work (and why) — from the decision log.

## 6. Deployment
17. How the bank uses it: PD → rating grades → pricing bands / cutoff; monitoring & recalibration.
18. Risks, ethics/legal (EU data protection, explainability obligations — sourced), where NOT to use.
19. Limitations & next steps.

## Appendices
A. Feature specification table · B. Decision log summary · C. Validation details & extra tables ·
D. Harness usage & computational complexity · E. AI-use statement · F. Contributions (explicit per
member) · G. Bibliography (verified references only).
