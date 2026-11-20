### D-030 Benchmark family and pre-declared settings (SC-01..SC-04)
- Date / phase / agent: 2026-10-31 / Round 1 / challenger-scorecard
- Status: ADOPTED
- Decision: four benchmarks on the frozen walk-forward folds, each refit per fold on `fs_year <= t`, `label_status == ok`, with `B.fit/B.transform(impute=True)`: SC-01 single best core numeric ratio (chosen by in-window |AUC-0.5|; it was roa_net in all 3 folds) -> logit; SC-02 unpenalised logit on the 10 core features + 3 categoricals; SC-03 monotone-WoE scorecard; SC-04 Z'-style logit on wc_ratio, equity_ratio, roa_oper, asset_turnover.
- Tuning: none needed. GLMs are unpenalised (no hyper-parameter). SC-03 settings were declared before looking at results and are identical in all folds, including fold 1 (single training year): 20 initial quantile bins, pool-adjacent-violators to monotone in the expected-sign direction, minimum final bin 2% of training rows. No inner validation split was used; no fold uses a default instead of a tuned value because nothing was tuned.
- Evidence: `experiments/scorecard/SC-0*/metrics.md`. Pooled AUC 0.746 / 0.803 / 0.817 / 0.770; log loss 0.0713 / 0.0672 / 0.0666 / 0.0699.
- Source(s): Stein (benchmarking logic, course); Altman lineage only via R-32; Ohlson (R-12) cited only as existence of logit lineage (content unread).

### D-031 SC-03 (WoE scorecard) is the benchmark to beat; SC-02 is the transparent logit reference
- Status: ADOPTED (round-1 provisional; the referee decides)
- Evidence: pooled AUC 0.8166 vs 0.8027 (SC-02); AUC higher in all three folds (+0.014, +0.011, +0.017); log loss 0.0666 vs 0.0672. Brier 0.0141 vs 0.0140 (overall probability accuracy, essentially tied). Calibration-in-the-large is poor for all (see D-033), so level comparisons between SC-02 and SC-03 are mostly about shape, not level. Paired bootstrap is the referee's job.
- Interpretation (hypothesis, not tested): the AUC gain of SC-03 over SC-02 comes from non-linear shapes in the ratios (e.g. roa_net default rate 9.4% in the lowest bin, falling to 0.43% at the top; the logit is linear in a winsorised ratio). A GAM/GBM challenger should beat SC-03 if shape is the only story.

### D-032 Sign check (Roger face-validity)
- Status: ADOPTED (reported)
- Result: SC-03 betas (on risk-oriented WoE) positive in all 13 features in all 3 folds; monotone bins enforce the numeric direction by construction. SC-01, SC-04: no violations. SC-02: one violation, `log_assets` has a positive coefficient in fold 1 (+0.049) and fold 3 (+0.012, SE 0.017, not significant) against expected sign -; negative in fold 2. Cause: size is correlated with the leverage/profit ratios and carries almost no marginal signal once they are in (univariate AUC 0.4765). Not "fixed" by dropping or sign-constraining; reported. The SC-03 `log_assets` WoE is monotone decreasing by construction with beta 0.43.
- Source(s): FEATURE_SPEC expected-sign table.

### D-033 Level calibration: all benchmarks under-predict (rising default rate, no cycle information)
- Status: OPEN (hand-off to calibration-specialist)
- Evidence: pooled mean PD vs observed default rate: SC-01 1.21% vs 1.54%; SC-02 1.08%; SC-03 1.10%; SC-04 1.17%. Calibration intercepts +0.25 to +0.45. Train-window default rates (1.0%, 1.1%, 1.3%) trail the test-year rates (1.28%, 1.62%, 1.70%) in every fold. Calibration slopes are near 1 (0.95 to 1.10), so ordering and spread are fine and the miss is a level shift. The logit models also lose a little level because the GLM fits the training years' mean exactly, which by construction was lower than the next year's.
- Not fixed here: any level adjustment is the calibration-specialist's decision (central tendency / cycle). The fix is not tuned on test years.

### D-034 SC-04 Altman-lineage benchmark is a proxy, not Z'
- Status: ADOPTED with caveat
- Decision: Z'-style ratios are rebuilt from the feature library: working capital/TA (wc_ratio), equity/assets (stands in for both retained earnings/TA and book equity/liabilities; retained earnings is not in the data), operating ROA (EBIT/TA proxy), sales/TA (asset_turnover); re-estimated as a logit, no published coefficients used. It is lineage only (R-32: original Z needs market equity; Z' replaces it with book equity); we do not claim it reproduces Altman's model. Pooled AUC 0.770, below SC-02 (0.803) with 4 vs 13 inputs.

### D-035 REJECTED / not done
- Points scaling is illustrative (PDO 20, 600 points at 50:1 good:bad odds): conventional choices, not a bank policy. Reported, not tuned.
- Isotonic/other post-hoc recalibration of the benchmarks: not done (calibration-specialist's scope).
- Penalised (L1/L2) logit: not used; with 475k rows and 13 features, no overfit visible (calibration slope 0.95 to 1.03).
