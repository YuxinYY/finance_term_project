# Tournament scorecard, round 1 (referee, 2026-10-31)

Scope: 13 experiments (SC-01..SC-04, RC-01..RC-03, GB-01..GB-06) on the frozen walk-forward folds
(`data/splits/walkforward.json` v1: test years 2008, 2009, 2010; 506,159 test rows; 7,784 defaults).
Label: panel `y` (start 6 months, 12-month horizon, D-002/D-009).

Every number below is produced by code and read from a file:
`reports/tournament/score_r1.py` (metrics, bootstrap, face validity, figures) writes `r1_*.csv`;
`reports/tournament/build_scorecard_r1.py` renders the tables; `reports/tournament/repro_check_r1.py`
refits fold 3 of SC-03, RC-03 and GB-01. All metrics come from `src/common/metrics.py`.

Conventions. AR = 2·AUC − 1 carries the same information as AUC (it is the CAP-curve view of the
same ranking; we report AUC as the headline). Log loss is the primary probability-accuracy metric
(D-006). Brier is reported as *overall probability accuracy*, not as a calibration measure (R-16
p. 32). Calibration = mean PD vs observed default rate (obs/pred), and the logistic recalibration
intercept/slope (perfect: 0 / 1). Decile tables with Clopper-Pearson bands are in
`r1_deciles_pooled.csv` and `fig/r1_calibration_deciles.png` (diagnostic only).

## 1. Decision rule (published with every scorecard)

A challenger replaces the incumbent only if:
1. its leakage / process verdict is CLEAN;
2. the 95% CI of Δlog-loss excludes zero in its favour, OR (the ΔAUC CI excludes zero in its favour
   AND log loss is not worse);
3. it has no more face-validity violations than the incumbent (or each violation is justified);
4. the per-year results show no year in which it is clearly worse (no hidden fragility).

Otherwise the simpler / more explainable model stays (statistical ties go to transparency).
When several challengers pass, the one that passes the same rule against every other passing
challenger is promoted; a passing challenger that is a statistical tie with a simpler one loses.

Round-1 specifics: incumbent at the start = **SC-03** (WoE scorecard, D-031). Bootstrap = paired,
firm-clustered (whole firms resampled with replacement, so a firm's 2008-2010 rows stay together),
**200 draws** (D-007), seed 0, the same draws for every model; percentile CIs. Because of D-062
(below), condition 4 is also checked on fold 3 (test 2010) alone, the only test year no design
choice has seen.

## 2. Result in one paragraph ("what changed and why")

**GB-01 (monotone-constrained LightGBM, 26 features, native NaN) replaces SC-03 as incumbent.**
Against SC-03 it is better on both power and probability accuracy with CIs far from zero (T4),
in every test year and for both seen and new firms (T5), including the clean 2010 fold; it has
zero face-validity violations (T8), the same as SC-03. It also beats the best transparent challenger
RC-03 in every year (T6, T7), so this is not a tie that should go to transparency. The unconstrained
twin GB-02 is better again but contradicts the expected sign on `wc_ratio` and `interest_coverage`
(and is borderline on `bank_debt_ratio`), so it fails condition 3. The RiskCalc models beat SC-03 on
log loss, but RC-01/RC-02 carry wrong-signed transforms (neg_equity; bank_debt_ratio) and RC-03's
fix was designed after seeing outer results (CONCERN, D-063); RC-03 is kept as the transparent
fallback and goes to calibration with GB-01. Every model under-predicts the level by about a third
(obs/pred 1.34-1.42 pooled, rising each year); a common, leakage-free logit shift (L1) and an oracle
per-year shift (L2, diagnostic) leave the ranking unchanged, so the comparison is not an artefact of
shared level error. The level itself is the calibration-specialist's decision.

## 3. Figures

- `fig/r1_roc_cap.png`: ROC and CAP overlays (pooled) for SC-02, SC-03, RC-01, RC-03, GB-01, GB-02.
- `fig/r1_calibration_deciles.png`: observed vs predicted default rate by PD decile and test year.
- `fig/r1_by_year.png`: AUC and log loss by test year, all 13 models.

## 4. Contract and reproduction checks

All 13 files pass the contract (one PD per test row and fold, all in (0,1), no NaN, no duplicates,
row ids identical to `walkforward_test_rows.parquet`; T11). The referee refitted fold 3 of SC-03,
RC-03 and GB-01 from the challengers' current code on the fold-3 training rows only, and predicted
on a test frame with every label column (`y*`, `status*`, `def_date`, `label_status`, `win_*`)
removed: the submitted fold-3 PDs are reproduced exactly (max |diff| = 0; T15).

### T1 Leaderboard (pooled 2008-2010, 506,159 rows; CIs = firm-clustered bootstrap percentiles, 200 draws)

| exp   | AUC [95% CI]            |    AR | log loss [95% CI]          |   log loss L1 (leak-free level shift) |   log loss L2 (oracle, diagnostic) |   Brier |   mean PD |   obs DR |   obs/pred |   cal. intercept |   cal. slope |
|:------|:------------------------|------:|:---------------------------|--------------------------------------:|-----------------------------------:|--------:|----------:|---------:|-----------:|-----------------:|-------------:|
| GB-02 | 0.8440 [0.8385, 0.8474] | 0.688 | 0.06332 [0.06232, 0.06453] |                               0.06287 |                            0.06258 | 0.01358 |    0.0114 |   0.0154 |       1.35 |            0.338 |        0.979 |
| GB-06 | 0.8380 [0.8330, 0.8412] | 0.676 | 0.06405 [0.06298, 0.06524] |                               0.0636  |                            0.06333 | 0.01366 |    0.0115 |   0.0154 |       1.34 |            0.331 |        0.976 |
| GB-05 | 0.8377 [0.8323, 0.8411] | 0.675 | 0.06414 [0.06309, 0.06534] |                               0.06369 |                            0.06341 | 0.01368 |    0.0115 |   0.0154 |       1.34 |            0.334 |        0.977 |
| GB-04 | 0.8376 [0.8323, 0.8409] | 0.675 | 0.06415 [0.06311, 0.06538] |                               0.0637  |                            0.06342 | 0.01368 |    0.0115 |   0.0154 |       1.34 |            0.333 |        0.974 |
| GB-01 | 0.8375 [0.8321, 0.8408] | 0.675 | 0.06415 [0.06309, 0.06536] |                               0.0637  |                            0.06343 | 0.01368 |    0.0115 |   0.0154 |       1.34 |            0.335 |        0.976 |
| GB-03 | 0.8298 [0.8250, 0.8335] | 0.66  | 0.06492 [0.06388, 0.06616] |                               0.06446 |                            0.06416 | 0.01378 |    0.0113 |   0.0154 |       1.36 |            0.342 |        0.981 |
| RC-02 | 0.8194 [0.8136, 0.8239] | 0.639 | 0.06580 [0.06463, 0.06706] |                               0.06533 |                            0.06503 | 0.01393 |    0.0113 |   0.0154 |       1.36 |            0.338 |        1.022 |
| RC-03 | 0.8192 [0.8135, 0.8237] | 0.638 | 0.06577 [0.06460, 0.06702] |                               0.06531 |                            0.06502 | 0.01392 |    0.0113 |   0.0154 |       1.36 |            0.334 |        1.024 |
| RC-01 | 0.8175 [0.8118, 0.8221] | 0.635 | 0.06598 [0.06482, 0.06726] |                               0.06551 |                            0.06522 | 0.01395 |    0.0113 |   0.0154 |       1.36 |            0.338 |        1.023 |
| SC-03 | 0.8166 [0.8115, 0.8208] | 0.633 | 0.06656 [0.06544, 0.06791] |                               0.06605 |                            0.06569 | 0.01408 |    0.011  |   0.0154 |       1.39 |            0.359 |        0.988 |
| SC-02 | 0.8027 [0.7967, 0.8070] | 0.605 | 0.06716 [0.06602, 0.06839] |                               0.06663 |                            0.06619 | 0.01396 |    0.0108 |   0.0154 |       1.42 |            0.385 |        0.983 |
| SC-04 | 0.7704 [0.7637, 0.7760] | 0.541 | 0.06986 [0.06878, 0.07114] |                               0.06954 |                            0.06932 | 0.01432 |    0.0117 |   0.0154 |       1.31 |            0.286 |        1.01  |
| SC-01 | 0.7455 [0.7386, 0.7507] | 0.491 | 0.07132 [0.07019, 0.07258] |                               0.071   |                            0.07088 | 0.01449 |    0.0121 |   0.0154 |       1.27 |            0.252 |        1     |

### T2 Per test year (AUC, log loss, obs/pred level ratio, calibration slope)

| exp   |   AUC 2008 |   LL 2008 |   obs/pred 2008 |   slope 2008 |   AUC 2009 |   LL 2009 |   obs/pred 2009 |   slope 2009 |   AUC 2010 |   LL 2010 |   obs/pred 2010 |   slope 2010 |
|:------|-----------:|----------:|----------------:|-------------:|-----------:|----------:|----------------:|-------------:|-----------:|----------:|----------------:|-------------:|
| SC-01 |     0.7564 |   0.06046 |            1.2  |         1.1  |     0.7406 |   0.07404 |            1.27 |         0.97 |     0.7361 |   0.0787  |            1.32 |         0.95 |
| SC-02 |     0.8136 |   0.05677 |            1.27 |         1.02 |     0.8009 |   0.06939 |            1.47 |         0.98 |     0.7944 |   0.07459 |            1.5  |         0.95 |
| SC-03 |     0.8278 |   0.05631 |            1.24 |         1.03 |     0.8117 |   0.06913 |            1.42 |         0.97 |     0.8115 |   0.07353 |            1.49 |         0.97 |
| SC-04 |     0.7818 |   0.05938 |            1.33 |         1.13 |     0.7693 |   0.07242 |            1.31 |         0.99 |     0.7584 |   0.07704 |            1.3  |         0.95 |
| RC-01 |     0.8311 |   0.05572 |            1.23 |         1.11 |     0.8144 |   0.06833 |            1.39 |         1    |     0.8102 |   0.07317 |            1.45 |         0.97 |
| RC-02 |     0.8328 |   0.05561 |            1.22 |         1.1  |     0.8174 |   0.06804 |            1.4  |         1.01 |     0.8115 |   0.07303 |            1.45 |         0.97 |
| RC-03 |     0.832  |   0.05561 |            1.22 |         1.1  |     0.817  |   0.06799 |            1.39 |         1.01 |     0.8117 |   0.07299 |            1.44 |         0.97 |
| GB-01 |     0.8441 |   0.05423 |            1.21 |         0.97 |     0.8359 |   0.06617 |            1.36 |         1.01 |     0.8307 |   0.07136 |            1.43 |         0.96 |
| GB-02 |     0.8485 |   0.05374 |            1.21 |         0.98 |     0.8419 |   0.0654  |            1.37 |         1    |     0.8402 |   0.07015 |            1.43 |         0.95 |
| GB-03 |     0.8381 |   0.05488 |            1.22 |         0.98 |     0.8274 |   0.06708 |            1.38 |         1.01 |     0.8226 |   0.07209 |            1.44 |         0.96 |
| GB-04 |     0.8448 |   0.05418 |            1.2  |         0.97 |     0.8364 |   0.06617 |            1.36 |         1.01 |     0.8302 |   0.07139 |            1.43 |         0.95 |
| GB-05 |     0.8447 |   0.05423 |            1.21 |         0.97 |     0.8358 |   0.06617 |            1.36 |         1.01 |     0.8309 |   0.07131 |            1.43 |         0.96 |
| GB-06 |     0.8456 |   0.05411 |            1.2  |         0.97 |     0.8349 |   0.06618 |            1.36 |         1    |     0.8321 |   0.07115 |            1.43 |         0.96 |

Test-set sizes (identical for every model):

| group   |      n |   defaults |   observed_dr |
|:--------|-------:|-----------:|--------------:|
| 2008    | 161261 |       2064 |     0.0127991 |
| 2009    | 170386 |       2752 |     0.0161516 |
| 2010    | 174512 |       2968 |     0.0170074 |
| seen    | 442415 |       6862 |     0.0155103 |
| new     |  63744 |        922 |     0.0144641 |

### T3 Seen-firm vs new-firm rows (pooled)

| exp   |   AUC seen |   AUC new | gap seen-new [95% CI]   |   LL seen |   LL new |   slope seen |   slope new |
|:------|-----------:|----------:|:------------------------|----------:|---------:|-------------:|------------:|
| SC-01 |     0.7579 |    0.6529 | 0.1050 [0.0858, 0.1246] |   0.07097 |  0.07375 |         1.02 |        0.75 |
| SC-02 |     0.8097 |    0.75   | 0.0597 [0.0439, 0.0767] |   0.06674 |  0.0701  |         1    |        0.81 |
| SC-03 |     0.823  |    0.7648 | 0.0582 [0.0438, 0.0753] |   0.06626 |  0.06864 |         1    |        0.84 |
| SC-04 |     0.7791 |    0.6981 | 0.0811 [0.0659, 0.0991] |   0.06952 |  0.0722  |         1.04 |        0.74 |
| RC-01 |     0.8241 |    0.7643 | 0.0598 [0.0454, 0.0750] |   0.06565 |  0.0683  |         1.03 |        0.89 |
| RC-02 |     0.8261 |    0.7654 | 0.0608 [0.0468, 0.0761] |   0.06545 |  0.06819 |         1.03 |        0.89 |
| RC-03 |     0.8259 |    0.7651 | 0.0608 [0.0467, 0.0758] |   0.06543 |  0.06817 |         1.04 |        0.89 |
| GB-01 |     0.8438 |    0.7865 | 0.0573 [0.0448, 0.0727] |   0.06375 |  0.06695 |         0.98 |        0.88 |
| GB-02 |     0.8505 |    0.792  | 0.0585 [0.0455, 0.0727] |   0.06287 |  0.06648 |         0.99 |        0.91 |
| GB-03 |     0.8364 |    0.7766 | 0.0598 [0.0467, 0.0753] |   0.06455 |  0.06751 |         0.99 |        0.88 |
| GB-04 |     0.8441 |    0.7856 | 0.0584 [0.0459, 0.0727] |   0.06374 |  0.06696 |         0.98 |        0.88 |
| GB-05 |     0.844  |    0.7872 | 0.0567 [0.0438, 0.0715] |   0.06374 |  0.0669  |         0.98 |        0.88 |
| GB-06 |     0.8445 |    0.7857 | 0.0587 [0.0457, 0.0734] |   0.06362 |  0.06697 |         0.98 |        0.88 |

### T4 Paired firm-clustered bootstrap vs incumbent SC-03 (pooled; negative dLogLoss favours new)

| new vs old     | subset   | dAUC [95% CI]              | dLogLoss [95% CI]             | dLogLoss L1 [95% CI]          | AUC CI excl. 0   | LL CI excl. 0   |
|:---------------|:---------|:---------------------------|:------------------------------|:------------------------------|:-----------------|:----------------|
| SC-01 vs SC-03 | ALL      | -0.0711 [-0.0755, -0.0666] | +0.00476 [+0.00441, +0.00502] | +0.00495 [+0.00460, +0.00521] | new worse        | new worse       |
| SC-02 vs SC-03 | ALL      | -0.0139 [-0.0154, -0.0118] | +0.00060 [+0.00044, +0.00074] | +0.00058 [+0.00041, +0.00072] | new worse        | new worse       |
| SC-04 vs SC-03 | ALL      | -0.0462 [-0.0507, -0.0416] | +0.00330 [+0.00304, +0.00357] | +0.00349 [+0.00323, +0.00375] | new worse        | new worse       |
| RC-01 vs SC-03 | ALL      | +0.0009 [-0.0001, +0.0020] | -0.00058 [-0.00069, -0.00049] | -0.00054 [-0.00065, -0.00045] | tie              | new better      |
| RC-02 vs SC-03 | ALL      | +0.0028 [+0.0016, +0.0040] | -0.00076 [-0.00087, -0.00067] | -0.00072 [-0.00083, -0.00063] | new better       | new better      |
| RC-03 vs SC-03 | ALL      | +0.0027 [+0.0016, +0.0039] | -0.00079 [-0.00090, -0.00070] | -0.00075 [-0.00085, -0.00066] | new better       | new better      |
| GB-01 vs SC-03 | ALL      | +0.0209 [+0.0188, +0.0231] | -0.00241 [-0.00260, -0.00223] | -0.00235 [-0.00254, -0.00217] | new better       | new better      |
| GB-02 vs SC-03 | ALL      | +0.0274 [+0.0249, +0.0303] | -0.00324 [-0.00350, -0.00298] | -0.00319 [-0.00344, -0.00293] | new better       | new better      |
| GB-03 vs SC-03 | ALL      | +0.0132 [+0.0113, +0.0150] | -0.00164 [-0.00179, -0.00149] | -0.00159 [-0.00175, -0.00145] | new better       | new better      |
| GB-04 vs SC-03 | ALL      | +0.0210 [+0.0188, +0.0234] | -0.00241 [-0.00259, -0.00221] | -0.00236 [-0.00254, -0.00216] | new better       | new better      |
| GB-05 vs SC-03 | ALL      | +0.0211 [+0.0191, +0.0233] | -0.00242 [-0.00262, -0.00225] | -0.00237 [-0.00256, -0.00219] | new better       | new better      |
| GB-06 vs SC-03 | ALL      | +0.0214 [+0.0192, +0.0237] | -0.00252 [-0.00273, -0.00235] | -0.00245 [-0.00267, -0.00229] | new better       | new better      |

### T5 Same, by test year and seen/new firms (fragility check)

| new vs old     | subset   | dAUC [95% CI]              | dLogLoss [95% CI]             | dLogLoss L1 [95% CI]          | AUC CI excl. 0   | LL CI excl. 0   |
|:---------------|:---------|:---------------------------|:------------------------------|:------------------------------|:-----------------|:----------------|
| RC-01 vs SC-03 | 2008     | +0.0034 [+0.0014, +0.0055] | -0.00059 [-0.00076, -0.00045] | -0.00059 [-0.00076, -0.00045] | new better       | new better      |
| RC-01 vs SC-03 | 2009     | +0.0027 [+0.0005, +0.0045] | -0.00080 [-0.00098, -0.00062] | -0.00076 [-0.00094, -0.00057] | new better       | new better      |
| RC-01 vs SC-03 | 2010     | -0.0014 [-0.0028, +0.0005] | -0.00035 [-0.00057, -0.00012] | -0.00027 [-0.00049, -0.00004] | tie              | new better      |
| RC-01 vs SC-03 | seen     | +0.0011 [-0.0001, +0.0022] | -0.00062 [-0.00075, -0.00052] | -0.00057 [-0.00070, -0.00048] | tie              | new better      |
| RC-01 vs SC-03 | new      | -0.0005 [-0.0039, +0.0032] | -0.00033 [-0.00062, -0.00007] | -0.00031 [-0.00060, -0.00005] | tie              | new better      |
| RC-03 vs SC-03 | 2008     | +0.0042 [+0.0018, +0.0066] | -0.00070 [-0.00087, -0.00056] | -0.00070 [-0.00087, -0.00056] | new better       | new better      |
| RC-03 vs SC-03 | 2009     | +0.0052 [+0.0027, +0.0073] | -0.00113 [-0.00132, -0.00095] | -0.00110 [-0.00128, -0.00091] | new better       | new better      |
| RC-03 vs SC-03 | 2010     | +0.0002 [-0.0017, +0.0025] | -0.00054 [-0.00074, -0.00034] | -0.00045 [-0.00065, -0.00024] | tie              | new better      |
| RC-03 vs SC-03 | seen     | +0.0029 [+0.0017, +0.0042] | -0.00084 [-0.00097, -0.00073] | -0.00079 [-0.00091, -0.00069] | new better       | new better      |
| RC-03 vs SC-03 | new      | +0.0003 [-0.0038, +0.0051] | -0.00047 [-0.00075, -0.00018] | -0.00043 [-0.00071, -0.00014] | tie              | new better      |
| GB-01 vs SC-03 | 2008     | +0.0164 [+0.0128, +0.0203] | -0.00208 [-0.00235, -0.00178] | -0.00208 [-0.00235, -0.00178] | new better       | new better      |
| GB-01 vs SC-03 | 2009     | +0.0242 [+0.0202, +0.0285] | -0.00296 [-0.00329, -0.00259] | -0.00290 [-0.00322, -0.00253] | new better       | new better      |
| GB-01 vs SC-03 | 2010     | +0.0191 [+0.0156, +0.0234] | -0.00217 [-0.00253, -0.00181] | -0.00206 [-0.00243, -0.00172] | new better       | new better      |
| GB-01 vs SC-03 | seen     | +0.0208 [+0.0188, +0.0235] | -0.00251 [-0.00271, -0.00232] | -0.00245 [-0.00265, -0.00226] | new better       | new better      |
| GB-01 vs SC-03 | new      | +0.0217 [+0.0150, +0.0284] | -0.00169 [-0.00210, -0.00122] | -0.00166 [-0.00207, -0.00118] | new better       | new better      |
| GB-03 vs SC-03 | 2008     | +0.0103 [+0.0071, +0.0139] | -0.00142 [-0.00165, -0.00118] | -0.00142 [-0.00165, -0.00118] | new better       | new better      |
| GB-03 vs SC-03 | 2009     | +0.0157 [+0.0125, +0.0186] | -0.00205 [-0.00232, -0.00177] | -0.00201 [-0.00228, -0.00174] | new better       | new better      |
| GB-03 vs SC-03 | 2010     | +0.0111 [+0.0082, +0.0145] | -0.00144 [-0.00169, -0.00118] | -0.00135 [-0.00160, -0.00108] | new better       | new better      |
| GB-03 vs SC-03 | seen     | +0.0134 [+0.0117, +0.0154] | -0.00171 [-0.00189, -0.00159] | -0.00166 [-0.00183, -0.00153] | new better       | new better      |
| GB-03 vs SC-03 | new      | +0.0118 [+0.0052, +0.0169] | -0.00113 [-0.00149, -0.00066] | -0.00112 [-0.00149, -0.00066] | new better       | new better      |

### T6 Key pairs (pooled)

| new vs old     | subset   | dAUC [95% CI]              | dLogLoss [95% CI]             | dLogLoss L1 [95% CI]          | AUC CI excl. 0   | LL CI excl. 0   |
|:---------------|:---------|:---------------------------|:------------------------------|:------------------------------|:-----------------|:----------------|
| SC-02 vs SC-03 | ALL      | -0.0139 [-0.0154, -0.0118] | +0.00060 [+0.00044, +0.00074] | +0.00058 [+0.00041, +0.00072] | new worse        | new worse       |
| RC-03 vs RC-01 | ALL      | +0.0017 [+0.0010, +0.0026] | -0.00021 [-0.00027, -0.00016] | -0.00021 [-0.00027, -0.00016] | new better       | new better      |
| RC-02 vs RC-01 | ALL      | +0.0019 [+0.0012, +0.0028] | -0.00018 [-0.00023, -0.00013] | -0.00018 [-0.00024, -0.00014] | new better       | new better      |
| RC-03 vs RC-02 | ALL      | -0.0002 [-0.0004, +0.0001] | -0.00003 [-0.00005, -0.00001] | -0.00002 [-0.00004, -0.00000] | tie              | new better      |
| GB-01 vs RC-01 | ALL      | +0.0200 [+0.0180, +0.0223] | -0.00183 [-0.00198, -0.00167] | -0.00181 [-0.00197, -0.00165] | new better       | new better      |
| GB-01 vs RC-03 | ALL      | +0.0183 [+0.0163, +0.0203] | -0.00162 [-0.00177, -0.00147] | -0.00160 [-0.00175, -0.00145] | new better       | new better      |
| GB-02 vs GB-01 | ALL      | +0.0065 [+0.0052, +0.0079] | -0.00083 [-0.00096, -0.00072] | -0.00084 [-0.00096, -0.00073] | new better       | new better      |
| GB-03 vs RC-01 | ALL      | +0.0123 [+0.0106, +0.0141] | -0.00106 [-0.00119, -0.00094] | -0.00106 [-0.00118, -0.00094] | new better       | new better      |
| SC-03 vs SC-02 | ALL      | +0.0139 [+0.0118, +0.0154] | -0.00060 [-0.00074, -0.00044] | -0.00058 [-0.00072, -0.00041] | new better       | new better      |
| GB-03 vs GB-01 | ALL      | -0.0077 [-0.0087, -0.0066] | +0.00077 [+0.00067, +0.00085] | +0.00075 [+0.00066, +0.00084] | new worse        | new worse       |
| GB-04 vs GB-01 | ALL      | +0.0001 [-0.0003, +0.0006] | -0.00001 [-0.00004, +0.00003] | -0.00001 [-0.00004, +0.00003] | tie              | tie             |
| GB-05 vs GB-01 | ALL      | +0.0002 [-0.0000, +0.0004] | -0.00002 [-0.00004, +0.00000] | -0.00002 [-0.00004, +0.00000] | tie              | tie             |
| GB-06 vs GB-01 | ALL      | +0.0005 [+0.0000, +0.0010] | -0.00011 [-0.00014, -0.00007] | -0.00011 [-0.00014, -0.00007] | new better       | new better      |

### T7 Key pairs by test year

| new vs old     |   subset | dAUC [95% CI]              | dLogLoss [95% CI]             | dLogLoss L1 [95% CI]          | AUC CI excl. 0   | LL CI excl. 0   |
|:---------------|---------:|:---------------------------|:------------------------------|:------------------------------|:-----------------|:----------------|
| RC-03 vs RC-01 |     2008 | +0.0009 [-0.0004, +0.0021] | -0.00011 [-0.00018, -0.00004] | -0.00011 [-0.00018, -0.00004] | tie              | new better      |
| RC-03 vs RC-01 |     2009 | +0.0025 [+0.0013, +0.0038] | -0.00033 [-0.00042, -0.00024] | -0.00033 [-0.00042, -0.00024] | new better       | new better      |
| RC-03 vs RC-01 |     2010 | +0.0016 [+0.0003, +0.0034] | -0.00019 [-0.00031, -0.00010] | -0.00018 [-0.00030, -0.00009] | new better       | new better      |
| GB-01 vs RC-01 |     2008 | +0.0130 [+0.0095, +0.0169] | -0.00149 [-0.00175, -0.00123] | -0.00149 [-0.00175, -0.00123] | new better       | new better      |
| GB-01 vs RC-01 |     2009 | +0.0215 [+0.0183, +0.0255] | -0.00216 [-0.00242, -0.00186] | -0.00213 [-0.00239, -0.00183] | new better       | new better      |
| GB-01 vs RC-01 |     2010 | +0.0205 [+0.0169, +0.0249] | -0.00181 [-0.00211, -0.00146] | -0.00179 [-0.00209, -0.00143] | new better       | new better      |
| GB-01 vs RC-03 |     2008 | +0.0121 [+0.0088, +0.0155] | -0.00138 [-0.00162, -0.00113] | -0.00138 [-0.00162, -0.00113] | new better       | new better      |
| GB-01 vs RC-03 |     2009 | +0.0190 [+0.0157, +0.0223] | -0.00183 [-0.00208, -0.00154] | -0.00180 [-0.00205, -0.00152] | new better       | new better      |
| GB-01 vs RC-03 |     2010 | +0.0189 [+0.0154, +0.0223] | -0.00162 [-0.00189, -0.00133] | -0.00161 [-0.00188, -0.00131] | new better       | new better      |
| GB-02 vs GB-01 |     2008 | +0.0043 [+0.0022, +0.0067] | -0.00049 [-0.00064, -0.00035] | -0.00049 [-0.00064, -0.00035] | new better       | new better      |
| GB-02 vs GB-01 |     2009 | +0.0060 [+0.0038, +0.0084] | -0.00077 [-0.00095, -0.00059] | -0.00078 [-0.00097, -0.00061] | new better       | new better      |
| GB-02 vs GB-01 |     2010 | +0.0095 [+0.0072, +0.0116] | -0.00121 [-0.00144, -0.00096] | -0.00122 [-0.00144, -0.00097] | new better       | new better      |


## 5. Level adjustment used for comparison (diagnostic, not a level choice)

- **L1, leakage-free**: one common logit shift per fold, δ_k = logit(default rate of the last training
  year) − logit(default rate of the whole training window), from training labels only, applied to
  every model (T14: fold 1 δ = 0 because it has one training year; fold 2 +0.129; fold 3 +0.219).
  It moves each model toward the most recent in-window year. It is a comparison device; it does
  not pick a central tendency.
- **L2, oracle (uses test labels)**: each model's own calibration-in-the-large intercept per test
  year, i.e. the log loss each model would have if its level were exactly right. This bounds what
  any level fix can achieve and isolates shape/ranking; it is not a model result.

Under both, the pooled order of the face-valid models is unchanged (T1) and every Δlog-loss CI
keeps its sign (T4, T6 column "dLogLoss L1"). The level gap is shared: obs/pred grows from
about 1.2 (2008) to about 1.4-1.5 (2010) for every serious candidate (SC-02/03, RC-x, GB-x; the weak SC-01/SC-04 benchmarks reach about 1.3; T2). Calibration slopes are close to 1
pooled (0.98-1.02 for the main candidates), so the miss is a level shift, not a shape problem; the
exception is new firms, where the main candidates (SC-03, RC-x, GB-x) have slopes 0.84-0.91 (PDs too spread out; T3).

## 6. Leakage and process verdicts

Verdict scale used by the caller: CLEAN / CONCERN / FAIL (= the agent definition's CLEAN /
SUSPECT / LEAKING). Code paths read: `experiments/scorecard/{run_all,scorecard_lib}.py`,
`experiments/riskcalc/{run,riskcalc,inner_validation}.py`, `experiments/gbm/{run_experiment,gbm_lib,compare}.py`,
`src/features/{build,core_selection,univariate}.py`, `src/common/{metrics,labels}.py`.

**Common to all 13 experiments (shared CONCERN, D-062).** Every fold fit is correctly confined to
the fold's training rows: `B.fit` (winsor bounds, sector medians, debt-share imputation, category
pooling), WoE bins, RiskCalc T_i / probit / F, GBM inner split, early stopping, grid choice,
Platt slope and level anchor are all computed from the frame `fs_year <= t, label_status == ok`
(asserted in each runner, confirmed by the fold-3 refits). No `def_date`, label or future-year field
is a feature; no firm-history feature exists (D-020). **However, the feature set itself
(`docs/FEATURE_SPEC.md`: the core set, which ratios exist, the `sign = none` decisions for
asset_turnover and extraord_to_assets) was chosen by the feature-architect on fs_year 2007-2009,
including an inner split that scored 2009 (`src/features/core_selection.py`).** 2008 and 2009 are
the test years of folds 1 and 2. That is feature selection using test-fold labels for folds 1-2. It
is common to incumbent and challengers (all use the same library), so it biases the paired
comparisons much less than the absolute levels, but fold-1/2 AUCs are somewhat optimistic for
everyone. Fold 3 (test 2010) is clean of it. The promotion decision below holds on 2010 alone.

| exp | verdict | reasons |
|---|---|---|
| SC-01 | CLEAN | Feature choice (max in-window |AUC−0.5|, roa_net every fold) inside the fold; unpenalised logit; nothing tuned. |
| SC-02 | CLEAN | Unpenalised logit on the fixed core set; no tuning. |
| SC-03 | CLEAN | 20 quantile bins, PAV, min bin 2%: fixed settings, identical in every fold, no inner or outer selection; bins/WoE/betas from training rows (refit reproduced exactly). |
| SC-04 | CLEAN | Fixed 4-ratio logit; no tuning. |
| RC-01 | CLEAN | Bin rule, prior, PAV, F settings are code defaults used unchanged in every fold. The inner validation (train 2007-08, validate 2009) was run *after* RC-01..03 outer results existed (file times 00:00-00:02 vs 23:57-23:59) and changed no setting, so it did not select anything; it must not be quoted as tuning evidence for folds 1-2 (2009 is fold 2's test year). |
| RC-02 | CLEAN | As RC-01; the two added ratios were declared as an H2 test. |
| RC-03 | CONCERN | The sign-drop rule runs inside each fold on training coefficients only, but the *rule itself* was added after RC-01/RC-02 outer metrics and coefficients were seen (`run.py` modified 23:59:03, after RC-02 ran at 23:58). It is a face-validity rule, not a performance search, and its measured effect vs RC-02 is nil (T6: dAUC −0.0002 [−0.0004, +0.0001]), so the optimism is small; but RC-03's numbers are post hoc and the RC-03 vs RC-01/RC-02 differences are not evidence. Report it as "RC-02 with a post-hoc sign rule". |
| GB-01 | CLEAN | Folds 2-3: time-ordered inner split (inner-train = earlier years; last training year split by firm into A = early stopping + grid, B = Platt slope); fold 1: random 70/15/15 firm split inside 2007 and the pre-declared DEFAULT config (a deviation from "time-ordered" in PLAYBOOK, unavoidable with one training year, no test data involved). Final trees = round(1.2 × best_iter): an untested heuristic, but fixed in advance and applied in-window. Platt slope from B, level anchored to the training-window rate: in-window. Variant results (GB-02..06) did not change GB-01, which was the first-declared spec. |
| GB-02 | CLEAN | Same pipeline without constraints (diagnostic). |
| GB-03 | CLEAN | Same pipeline, core features. |
| GB-04 | CLEAN | Reuses GB-01's per-fold configs, which were themselves chosen in-window. |
| GB-05 | CLEAN | Same pipeline, 'advanced' monotone method. |
| GB-06 | CLEAN | Same pipeline; `gbm_lib.py` was edited (00:06) after GB-01..05 ran to add this variant; the shared code path is unchanged (GB-01 fold 3 reproduced exactly with the current file). |

## 7. Face validity

### T8 Face-validity matrix (features with an expected sign; '-' = not in the model)

Linear models: coefficient sign per fold (all 3 folds). SC-03: beta on risk-oriented WoE > 0 and WoE monotone. RiskCalc: probit coefficient on Phi^-1(T_i) > 0 per fold and T_i monotone (all folds). GBM: partial dependence of the saved fold-3 model on 20,000 fold-3 training rows, 25 quantile grid points; VIOL = more than half of the curve's movement is against the expected sign; partial = 10-50%. 'violations' counts features.

|       |   violations | log_assets   | equity_ratio   | st_debt_ratio   | bank_debt_ratio   | st_debt_share   | wc_ratio   | current_ratio   | cash_ratio   | roa_net   | roa_oper   | ebitda_margin   | interest_coverage   | cfo_to_debt   | interest_to_assets   | ar_days   | ap_days   | tangible_ratio   | intangible_ratio   | neg_equity   | debt_split_missing   |
|:------|-------------:|:-------------|:---------------|:----------------|:------------------|:----------------|:-----------|:----------------|:-------------|:----------|:-----------|:----------------|:--------------------|:--------------|:---------------------|:----------|:----------|:-----------------|:-------------------|:-------------|:---------------------|
| SC-01 |            0 | -            | -              | -               | -                 | -               | -          | -               | -            | ok        | -          | -               | -                   | -             | -                    | -         | -         | -                | -                  | -            | -                    |
| SC-02 |            1 | VIOL(f1,3)   | ok             | ok              | -                 | -               | -          | -               | ok           | ok        | -          | -               | ok                  | -             | ok                   | ok        | -         | ok               | -                  | ok           | -                    |
| SC-03 |            0 | ok           | ok             | ok              | -                 | -               | -          | -               | ok           | ok        | -          | -               | ok                  | -             | ok                   | ok        | -         | ok               | -                  | ok           | -                    |
| SC-04 |            0 | -            | ok             | -               | -                 | -               | ok         | -               | -            | -         | ok         | -               | -                   | -             | -                    | -         | -         | -                | -                  | -            | -                    |
| RC-01 |            1 | ok           | ok             | ok              | -                 | -               | -          | -               | ok           | ok        | -          | -               | ok                  | -             | ok                   | ok        | -         | ok               | -                  | VIOL(f1,2,3) | -                    |
| RC-02 |            2 | ok           | ok             | ok              | VIOL(f1,2,3)      | -               | -          | -               | ok           | ok        | -          | -               | ok                  | -             | ok                   | ok        | ok        | ok               | -                  | VIOL(f1,2,3) | -                    |
| RC-03 |            0 | ok           | ok             | ok              | -                 | -               | -          | -               | ok           | ok        | -          | -               | ok                  | -             | ok                   | ok        | ok        | ok               | -                  | -            | -                    |
| GB-01 |            0 | ok           | ok             | ok              | ok                | ok              | ok         | ok              | ok           | ok        | ok         | ok              | ok                  | ok            | ok                   | ok        | ok        | ok               | ok                 | ok           | ok                   |
| GB-02 |            2 | partial      | ok             | ok              | partial           | partial         | VIOL(f3)   | partial         | partial      | partial   | partial    | partial         | VIOL(f3)            | partial       | ok                   | partial   | partial   | ok               | partial            | ok           | ok                   |
| GB-03 |            0 | ok           | ok             | ok              | -                 | -               | -          | -               | ok           | ok        | -          | -               | ok                  | -             | ok                   | ok        | -         | ok               | -                  | ok           | -                    |
| GB-04 |            0 | ok           | ok             | ok              | ok                | ok              | ok         | ok              | ok           | ok        | ok         | ok              | ok                  | ok            | ok                   | ok        | ok        | ok               | ok                 | ok           | ok                   |
| GB-05 |            0 | ok           | ok             | ok              | ok                | ok              | ok         | ok              | ok           | ok        | ok         | ok              | ok                  | ok            | ok                   | ok        | ok        | ok               | ok                 | ok           | ok                   |
| GB-06 |            1 | ok           | ok             | ok              | ok                | partial         | ok         | VIOL(f3)        | ok           | ok        | ok         | ok              | ok                  | ok            | ok                   | ok        | ok        | ok               | ok                 | ok           | ok                   |

### T9 GBM partial-dependence violation shares > 0 (fold-3 models)

| exp   | feature            |   expected | constrained   |   range |   violation_share |
|:------|:-------------------|-----------:|:--------------|--------:|------------------:|
| GB-02 | log_assets         |         -1 | False         |   0.404 |             0.191 |
| GB-02 | equity_ratio       |         -1 | False         |   1.761 |             0.054 |
| GB-02 | st_debt_ratio      |          1 | False         |   0.798 |             0.02  |
| GB-02 | bank_debt_ratio    |          1 | False         |   0.133 |             0.452 |
| GB-02 | st_debt_share      |          1 | False         |   0.259 |             0.115 |
| GB-02 | wc_ratio           |         -1 | False         |   0.329 |             0.83  |
| GB-02 | current_ratio      |         -1 | False         |   0.258 |             0.155 |
| GB-02 | cash_ratio         |         -1 | False         |   0.293 |             0.455 |
| GB-02 | roa_net            |         -1 | False         |   1.636 |             0.102 |
| GB-02 | roa_oper           |         -1 | False         |   0.844 |             0.29  |
| GB-02 | ebitda_margin      |         -1 | False         |   0.269 |             0.284 |
| GB-02 | interest_coverage  |         -1 | False         |   0.338 |             0.567 |
| GB-02 | cfo_to_debt        |         -1 | False         |   0.787 |             0.136 |
| GB-02 | interest_to_assets |          1 | False         |   0.817 |             0.045 |
| GB-02 | ar_days            |          1 | False         |   0.914 |             0.127 |
| GB-02 | ap_days            |          1 | False         |   0.524 |             0.288 |
| GB-02 | tangible_ratio     |         -1 | False         |   1.375 |             0.047 |
| GB-02 | intangible_ratio   |          1 | False         |   0.362 |             0.369 |
| GB-06 | st_debt_share      |          1 | False         |   0.245 |             0.106 |
| GB-06 | current_ratio      |         -1 | False         |   0.215 |             0.872 |
| GB-06 | tangible_ratio     |         -1 | False         |   1.357 |             0.034 |

### T10 Coefficient sign violations (linear / probit models)

| exp   |   fold | feature         | evidence                                 |   value |      z |
|:------|-------:|:----------------|:-----------------------------------------|--------:|-------:|
| SC-02 |      1 | log_assets      | logit coef                               |  0.0486 | nan    |
| SC-02 |      3 | log_assets      | logit coef                               |  0.0124 | nan    |
| RC-01 |      1 | neg_equity      | probit coef on Phi^-1(T_i) (must be > 0) | -0.0797 |  -1.14 |
| RC-01 |      2 | neg_equity      | probit coef on Phi^-1(T_i) (must be > 0) | -0.0772 |  -1.94 |
| RC-01 |      3 | neg_equity      | probit coef on Phi^-1(T_i) (must be > 0) | -0.076  |  -2.67 |
| RC-02 |      1 | neg_equity      | probit coef on Phi^-1(T_i) (must be > 0) | -0.064  |  -0.91 |
| RC-02 |      1 | bank_debt_ratio | probit coef on Phi^-1(T_i) (must be > 0) | -0.1814 |  -1.24 |
| RC-02 |      2 | neg_equity      | probit coef on Phi^-1(T_i) (must be > 0) | -0.0648 |  -1.62 |
| RC-02 |      2 | bank_debt_ratio | probit coef on Phi^-1(T_i) (must be > 0) | -0.2573 |  -3.17 |
| RC-02 |      3 | neg_equity      | probit coef on Phi^-1(T_i) (must be > 0) | -0.0551 |  -1.92 |
| RC-02 |      3 | bank_debt_ratio | probit coef on Phi^-1(T_i) (must be > 0) | -0.1496 |  -2.56 |

### T16 tangible_ratio: univariate AUC (PD falls with ratio) within ATECO group, fold-3 training rows

| ateco_group   |      n |   defaults |   median_tangible |   auc_tangible_within |
|:--------------|-------:|-----------:|------------------:|----------------------:|
| A             |   7222 |         54 |         0.720426  |              0.642227 |
| C             | 120810 |       1940 |         0.209914  |              0.561169 |
| E             |   4369 |         48 |         0.300139  |              0.64296  |
| F             |  83850 |       1082 |         0.028554  |              0.579863 |
| G             |  85682 |       1251 |         0.0730679 |              0.624193 |
| H             |  12927 |        170 |         0.226107  |              0.677167 |
| I             |  14031 |         97 |         0.800772  |              0.726973 |
| J             |  10419 |        169 |         0.0504537 |              0.625716 |
| L             |  91747 |        764 |         0.437937  |              0.691099 |
| M             |  20234 |        288 |         0.0236815 |              0.588894 |
| N             |   8999 |        160 |         0.0871891 |              0.621075 |
| R             |   3540 |         64 |         0.405208  |              0.671178 |
| OTHER         |  11930 |        107 |         0.315178  |              0.6362   |
| ALL           | 475760 |       6194 |         0.144154  |              0.616117 |

Reading:
- **SC-03, RC-03, GB-01, GB-03, GB-04, GB-05: zero violations.** SC-03 and RC-x enforce the shape
  by PAV (all 111 WoE/T_i monotonicity checks pass, `r1_monotone_shape_checks.json`); the GBMs by
  monotone constraints. For the GBMs only the fold-3 model is saved, so folds 1-2 are covered by
  construction (constraint vector in the code), not by measurement.
- **SC-02**: `log_assets` positive in folds 1 and 3 (+0.049, +0.012; not significant, D-032).
  Justified as a near-zero conditional effect, but it is a reason-code risk.
- **RC-01**: T_neg_equity negative in all folds (z −1.1, −1.9, −2.7). **RC-02**: neg_equity and
  bank_debt_ratio negative in all folds. Explained in D-041 (redundancy with T_equity_ratio;
  collinearity within leverage) — a fair explanation, but a credit officer would see a wrong-signed
  driver, so we count them.
- **GB-02** contradicts `wc_ratio` (83% of PD movement against the sign) and `interest_coverage`
  (57%); `bank_debt_ratio` is borderline (45% here on a 25-point grid; 56% in the GBM agent's
  check): two to three violations plus partial non-monotonicity on most ratios (T9). Confirms the
  challenger's own finding.
- **GB-06** (constraints dropped on three features): `current_ratio` 87% against the sign. This is
  the cost of the "decile break" relaxation: the model uses current_ratio upward once wc_ratio is
  constrained.
- **tangible_ratio as GB-01's #1 driver: plausible.** The expected sign (−, pledgeable collateral)
  holds univariately (fold-3 training rows: AUC 0.616 with PD falling as the ratio rises, T16) and,
  importantly, **within every ATECO group with >= 30 defaults** (within-group AUC 0.56 to 0.73, T16),
  so it is not just a proxy for asset-heavy, low-default sectors such as real estate (L). Its large
  mean |SHAP| (0.41, twice the next driver) is partly an attribution artefact: GBM splits shared
  leverage / asset-structure signal across correlated ratios and can load it on one of them. Explain
  it to a credit officer as "asset structure / collateral" and report SHAP by family (D-058). A human
  should still read the GB-01 dependence plot (`experiments/gbm/GB-01/dep_tangible_ratio.png`).

## 8. Promotion decision (rule applied)

| candidate | cond. 1 process | cond. 2 vs SC-03 | cond. 3 face-validity violations, features (SC-03: 0) | cond. 4 per year (incl. 2010) | outcome |
|---|---|---|---|---|---|
| GB-01 | CLEAN | LL and AUC both better, CIs exclude 0 | 0 | better in 2008, 2009, 2010, seen, new | **PROMOTED** (new incumbent) |
| GB-04 | CLEAN | passes | 0 | passes | tie with GB-01 (T6) → GB-01 kept (no imputation step) |
| GB-05 | CLEAN | passes | 0 | passes | tie with GB-01 → GB-01 kept (simpler default method) |
| GB-03 | CLEAN | passes | 0 | passes | dominated by GB-01 (T6) |
| RC-03 | CONCERN | LL better; AUC better pooled, tie in 2010 | 0 | no year worse | fails cond. 1; dominated by GB-01 in every year; **transparent fallback** |
| RC-01 | CLEAN | LL better; AUC tie | 1 feature (neg_equity, all 3 folds) | 2010 AUC tie | fails cond. 3 |
| RC-02 | CLEAN | passes | 2 features (neg_equity, bank_debt_ratio; all folds) | passes | fails cond. 3 |
| GB-02 | CLEAN | passes | 2 features (+1 borderline) | passes | fails cond. 3 (not for submission) |
| GB-06 | CLEAN | passes | 1 feature (current_ratio) | passes | fails cond. 3 |
| SC-01, SC-02, SC-04 | CLEAN | worse | 0 / 1 (log_assets) / 0 | - | not promoted |

**Incumbent after round 1: GB-01.** Hand-off to the calibration-specialist: **GB-01** (champion) and
**RC-03** (transparent fallback; if the humans decide explainability outweighs ~0.018 AUC and
0.0016 log loss, this is the model). Inputs available in-window: GB-01 `oot_predictions_raw.parquet`
and `inner_B_predictions.parquet`; RC-03 `params_fold*.json` (final transform F).

## 9. Notes for the next steps

- Calibration: all models under-predict by ~25-45% depending on the year (T2); the level gap grows
  with the default-rate trend. New-firm slopes ≈ 0.88 suggest a new-firm adjustment may help (H5).
- The new-firm AUC gap (seen − new ≈ 0.06 for every model with CIs ≈ [0.04, 0.08], T3) is a property
  of the population, not of a model; GB-01 has the same gap as SC-03.
- Round-2 ideas (if time): partial monotonicity only where the economic sign is weak; re-run the
  core-set selection per fold to remove the shared D-062 caveat; test the 1.2× tree heuristic on the
  inner split.

## Appendix: checks

### T11 Contract check

| exp   |   n_rows |   expected_rows |   dup_row_fold |   nan_pd |   pd_outside_open_01 |      min_pd |   max_pd |   missing_test_rows |   extra_rows |   rows_fold1 |   rows_fold2 |   rows_fold3 | PASS   |
|:------|---------:|----------------:|---------------:|---------:|---------------------:|------------:|---------:|--------------------:|-------------:|-------------:|-------------:|-------------:|:-------|
| SC-01 |   506159 |          506159 |              0 |        0 |                    0 | 0.000185558 | 0.207522 |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| SC-02 |   506159 |          506159 |              0 |        0 |                    0 | 1e-06       | 0.640652 |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| SC-03 |   506159 |          506159 |              0 |        0 |                    0 | 0.000201084 | 0.567384 |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| SC-04 |   506159 |          506159 |              0 |        0 |                    0 | 2.02879e-05 | 0.35838  |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| RC-01 |   506159 |          506159 |              0 |        0 |                    0 | 0.00027929  | 0.652211 |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| RC-02 |   506159 |          506159 |              0 |        0 |                    0 | 0.000261852 | 0.630876 |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| RC-03 |   506159 |          506159 |              0 |        0 |                    0 | 0.000258639 | 0.640214 |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| GB-01 |   506159 |          506159 |              0 |        0 |                    0 | 0.000216387 | 0.846686 |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| GB-02 |   506159 |          506159 |              0 |        0 |                    0 | 0.000116651 | 0.906562 |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| GB-03 |   506159 |          506159 |              0 |        0 |                    0 | 0.000220626 | 0.75299  |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| GB-04 |   506159 |          506159 |              0 |        0 |                    0 | 0.000202916 | 0.854083 |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| GB-05 |   506159 |          506159 |              0 |        0 |                    0 | 0.000239996 | 0.854403 |                   0 |            0 |       161261 |       170386 |       174512 | True   |
| GB-06 |   506159 |          506159 |              0 |        0 |                    0 | 0.000193758 | 0.833059 |                   0 |            0 |       161261 |       170386 |       174512 | True   |

### T12 D-011 label check (panel y vs make_labels, start 6, data_end 2013-05-31)

|   fs_year |   rows |   panel_ok |   make_labels_ok |   status_mismatch_rows |   y_mismatch_on_common_ok |   make_labels_ok_with_incomplete_window |   panel_defaults_ok |   make_labels_defaults_ok |
|----------:|-------:|-----------:|-----------------:|-----------------------:|--------------------------:|----------------------------------------:|--------------------:|--------------------------:|
|      2007 | 144528 |     144214 |           144214 |                      0 |                         0 |                                       0 |                1379 |                      1379 |
|      2008 | 161863 |     161261 |           161261 |                      0 |                         0 |                                       0 |                2064 |                      2064 |
|      2009 | 171150 |     170386 |           170386 |                      0 |                         0 |                                       0 |                2752 |                      2752 |
|      2010 | 175399 |     174512 |           174512 |                      0 |                         0 |                                       0 |                2968 |                      2968 |
|      2011 | 184973 |          0 |             2357 |                   2357 |                         0 |                                    2357 |                   0 |                      2357 |
|      2012 | 185639 |          0 |                0 |                      0 |                         0 |                                       0 |                   0 |                         0 |

### T13 Fast bootstrap vs metrics.paired_bootstrap (GB-01 vs SC-03, 15,000-firm subsample, 40 draws, seed 0)

| metric     |   shared_ci_low |   fast_ci_low |   shared_ci_high |   fast_ci_high |   shared_share |   fast_share |   n_rows |   n_firms |   n_boot |   max_abs_diff |
|:-----------|----------------:|--------------:|-----------------:|---------------:|---------------:|-------------:|---------:|----------:|---------:|---------------:|
| d_auc      |      0.0147621  |    0.0147621  |       0.0332037  |     0.0332037  |              1 |            1 |    37844 |     15000 |       40 |    1.04083e-16 |
| d_log_loss |     -0.00365502 |   -0.00365502 |      -0.00220921 |    -0.00220921 |              1 |            1 |    37844 |     15000 |       40 |    2.68882e-17 |

### T14 L1 level shift per fold (logit; training labels only)

|   fold |   delta_logit |
|-------:|--------------:|
|      1 |      0        |
|      2 |      0.128693 |
|      3 |      0.218824 |

### T15 Reproduction check: fold 3 refitted by the referee from current code

| exp   |      n |   max_abs_diff |   mean_abs_diff |
|:------|-------:|---------------:|----------------:|
| SC-03 | 174512 |              0 |               0 |
| RC-03 | 174512 |              0 |               0 |
| GB-01 | 174512 |              0 |               0 |

