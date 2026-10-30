# REFERENCES — verified bibliography and claim map

Status meanings (see `.claude/agents/literature-scout.md`):
- **VERIFIED**: located (URL/DOI recorded), bibliographic details match, and each claim we use
  was checked against the text, with section/page noted.
- **LOCATED**: existence and bibliographic details confirmed from a reliable page, but the
  claims we want to use have NOT yet been checked against the full text. Do not cite claims yet.
- **TO-VERIFY**: believed to exist; not yet located.

Status as of Oct 29, 2026 (set when this kit was written; the literature-scout must upgrade).

| ID | Reference | Status | Where located | Intended use / claim to check |
|---|---|---|---|---|
| R-01 | Dwyer, D., Kocagil, A. E., & Stein, R. M. (2004). *The Moody's KMV EDF RiskCalc v3.1 Model: Next-Generation Technology for Predicting Private Firm Credit Risk.* Moody's KMV. | LOCATED | PDF on rogermstein.com (`/wp-content/uploads/RiskCalc-v3-1-Model.pdf`) and moodys.com product attachment | Architecture: per-ratio nonparametric transforms → probit with industry variables → final transform; FSO vs credit-cycle-adjusted modes; power & calibration evaluated separately |
| R-02 | Dwyer, D. W., & Stein, R. M. (2004). *Technical Document on RiskCalc v3.1 Methodology.* Moody's KMV. | LOCATED (cited by R-01 and Moody's France doc) | find a copy | Detailed methodology for transforms and calibration |
| R-03 | Moody's KMV. *RiskCalc v3.1 Italy* (model documentation). | LOCATED (secondary: ResearchGate listing shows the probit-on-transforms equation for the Italy model) | ResearchGate (Dwyer profile) | Italy-specific ratios and results — obtain the actual document before citing specifics |
| R-04 | Moody's Analytics (2020). *Tracking the Impact of COVID-19 on Credit Risk — Europe* (webinar slides). | LOCATED | ma.moodys.com PDF | States that Italy/Spain/Portugal RiskCalc models' credit-cycle adjustment combines public-firm data with the national unemployment rate |
| R-05 | Falkenstein, E., Boral, A., & Carty, L. V. (2000). *RiskCalc for Private Companies: Moody's Default Model.* Moody's Investors Service. | TO-VERIFY | — | Ratio families for private firms; private firms lack equity prices |
| R-06 | Sobehart, J. R., Keenan, S. C., & Stein, R. M. (2000/2001). *Benchmarking Quantitative Default Risk Models: A Validation Methodology.* Moody's; also Algo Research Quarterly 4(1/2). | LOCATED (secondary citation with URL) | czaz.akademiazamojska.edu.pl reference list | Walk-forward validation; CAP/AR |
| R-07 | Stein, R. M. (2007). Benchmarking default prediction models: Pitfalls and remedies in model validation. *Journal of Risk Model Validation*, 1(1), 77–113. (earlier Moody's KMV tech report 2002) | TO-VERIFY (in proposal; check text) | — | Validation pitfalls; sample-size effects on metrics; resampling |
| R-08 | Stein, R. M. (2005). The relationship between default prediction and lending profits: Integrating ROC analysis and loan pricing. *Journal of Banking & Finance*, 29(5), 1213–1236. | TO-VERIFY | — | Power → lending profit; cutoff vs pricing |
| R-09 | Bohn, J. R., & Stein, R. M. (2009). *Active Credit Portfolio Management in Practice.* Wiley. ISBN 978-0-470-08018-4. | LOCATED | Wiley product page | Econometric private-firm models; PD use in pricing; validation chapters |
| R-10 | Stein, R. M., Kocagil, A. E., Bohn, J., & Akhavein, J. (2003). *Systematic and Idiosyncratic Risk in Middle-Market Default Prediction: A Study of the Performance of the RiskCalc and PFM Models.* Moody's KMV. | LOCATED (secondary) | cited in R-01 and Bohn (2006) slides | Value of cycle information for private firms |
| R-11 | Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance*, 23(4), 589–609. | TO-VERIFY | — | Ratio-based default prediction origin |
| R-12 | Ohlson, J. A. (1980). Financial ratios and the probabilistic prediction of bankruptcy. *Journal of Accounting Research*, 18(1), 109–131. | TO-VERIFY | — | Logit on ratios; size effect |
| R-13 | Shumway, T. (2001). Forecasting bankruptcy more accurately: A simple hazard model. *Journal of Business*, 74(1), 101–124. | TO-VERIFY | — | Multi-period panel, firm-years as observations |
| R-14 | Engelmann, B., Hayden, E., & Tasche, D. (2003). Testing rating accuracy. *Risk*, 16(1). | TO-VERIFY | — | AR = 2·AUC − 1; CIs for AUC |
| R-15 | Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards: A Revised Framework (Comprehensive Version).* BIS. | TO-VERIFY | bis.org | IRB parameters PD/LGD/EAD; one-year PD; 0.03% corporate PD floor (find paragraph); default definition (90 days past due) |
| R-16 | BCBS (2005). *Studies on the Validation of Internal Rating Systems.* Working Paper No. 14. | TO-VERIFY | bis.org | Discrimination vs calibration tests; Hosmer–Lemeshow, binomial tests |
| R-17 | Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124–136. | TO-VERIFY | — | ML benchmark evidence (ensembles/boosting vs logit) |
| R-18 | Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*. | TO-VERIFY | — | GBM method; monotone constraints (check docs) |
| R-19 | Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML '05*. | TO-VERIFY | — | Boosted trees distort probabilities; Platt vs isotonic |
| R-20 | Platt, J. (1999). Probabilistic outputs for support vector machines… *Advances in Large Margin Classifiers.* | TO-VERIFY | — | Platt scaling |
| R-21 | Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD '02*. | TO-VERIFY | — | Isotonic calibration |
| R-22 | Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities. *Neural Computation*, 14(1), 21–41. | TO-VERIFY | — | Prior-shift (central tendency) adjustment |
| R-23 | Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*. | TO-VERIFY | — | SHAP reason codes |
| R-24 | Lou, Y., Caruana, R., & Gehrke, J. (2012). Intelligible models for classification and regression. *KDD '12*. | TO-VERIFY | — | GAM-style interpretable models |
| R-25 | Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. arXiv:1909.09223. | TO-VERIFY | — | EBM |
| R-26 | Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: a review. *JRSS A*, 160(3). | TO-VERIFY | — | Reject inference / selection bias discussion |
| R-27 | Italian Civil Code, art. 2435-bis (*bilancio in forma abbreviata*). | LOCATED | e.g. miolegale.it; Rödl & Partner 2024 note on thresholds | Small firms may file abbreviated statements with fewer line items → candidate explanation of structural missingness. NB thresholds changed over time; use the 2007–2012 version |
| R-28 | Italian Civil Code, art. 2364 (shareholders' approval of annual accounts within 120 days, extendable to 180). | TO-VERIFY | — | Statement availability lag (label window) |
| R-29 | Italian Civil Code, art. 2435 (filing of approved accounts with the Companies Register within 30 days). | TO-VERIFY | — | Statement availability lag |
| R-30 | Assignment: *Term Project Instructions*, DS-GA 3001, Oct 11, 2026. | VERIFIED | course file | Business need, transaction flow, harness rules, evaluation criteria |
| R-31 | Team Turquoise proposal with RMS feedback. | VERIFIED | course file | Feedback items (FEEDBACK_RESPONSE.md) |

## Claim map (fill as claims are verified)

| Claim used in deck/report | Ref | Section/page | Verified by / date |
|---|---|---|---|
| | | | |
