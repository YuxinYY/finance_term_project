# Calibration round summary (2026-10-31, rendered 2026-11-03)

Rendered by the orchestrator from `docs/DECISION_LOG.md` D-070..D-079 and the CSVs in this folder
(the calibration-specialist could not write this file, see the log header). Every number below
appears in a decision-log entry that cites the named CSV; spot checks against `ct_targets.csv`,
`methods_by_year.csv`, `scenario_regret.csv` and `grade_table_2010.csv` matched. Models: GB-01 (champion)
and RC-02 (fallback). Pooled = test years 2008-2010. "e-4" = 0.0001 of log loss. Confidence intervals are
firm-clustered paired bootstrap, 200 draws, seed 0.

## 1. The problem

Ranking was fine, level was not. GB-01 calibration slope 0.976 pooled, but mean PD 1.15% against
observed 1.54%; observed/predicted 1.21 / 1.36 / 1.43 for 2008 / 2009 / 2010
(`methods_by_year.csv`, row M0). Default rates rose every year while the models were fit on earlier,
lower years, so the miss is a level shift (D-033, D-046, D-067).

## 2. Design: three separate questions

| Question | Answer | Decision |
|---|---|---|
| Shape (is the score-to-PD curve right?) | Each model's own calibration: Platt slope from a held-out slice, intercept anchored to the training rate | D-053 |
| Method (how to move the level?) | Prior shift: `PD' = expit(logit(PD) + delta)`, one frozen number | D-072 |
| Target (what level, for an unknown holdout year?) | tau = 1.493% (start-6 label) | D-071, D-073 |

Leakage rule (D-070): only training rows (and GB-01's in-window slice B) feed any level parameter. The
test cohort's early defaults are not used (they are other holdout rows, constraint 2).

## 3. Method comparison (GB-01, rule e, pooled; D-072)

| Method | Pooled log loss | Delta vs submitted (e-4) | Slope | Verdict |
|---|---|---|---|---|
| Submitted (M0) | 0.06415 | 0 | 0.976 | baseline |
| M1 prior shift to rule e | 0.06352 | -6.31 [-7.09, -5.60] | 0.98 | ADOPTED |
| M2 shift matched to last training year mean PD | n/a | -6.16 | n/a | slightly worse |
| M3raw Platt as fitted on slice B | 0.06353 | -6.21 | 0.975 | rejected: an implicit trend rule |
| M4 isotonic + level | n/a | -4.82 | 0.92 | rejected: slope worse, within-year AUC -0.0008 |
| Saerens EM | n/a | n/a | n/a | rejected: estimates 2.18 / 1.79 / 0.45% vs observed 1.28 / 1.62 / 1.70%; needs the holdout batch |

Prior shift leaves within-year AUC unchanged (`auc_invariance_check.csv`, |dAUC| = 0): it changes the level only.

## 4. Target rules (D-071)

Targets vs observed for 2008 / 2009 / 2010 (`ct_targets.csv`): e 1.17 / 1.50 / 1.61%, b 0.96 / 1.28 / 1.62%,
a 0.96 / 1.13 / 1.30%, observed 1.28 / 1.62 / 1.70%.

Delta log loss vs submitted, pooled (e-4): e last 6 months -6.31; c2 half-trend -6.28; c trend -6.16;
b last year -4.50; f mean of last 2 -1.57; a pooled 0; d TTC proxy +0.37. Rule e recovers 88% of the
attainable gain (oracle per-year level 0.06343).

Caveat: e was selected among 7 rules after seeing 2008-2010, so its gain is optimistic. On the 2011
pseudo-test (level-only extra log loss, e-4): a 0.33, d 0.48, f 0.48, e 0.74, b 0.80, c2 3.85, c 9.42.
Trend extrapolation fails at the 2011 turn. A true through-the-cycle level is not estimable (no post-turn
benign years in 2007-2010).

## 5. Holdout level (D-073, D-074, D-076)

- tau = 1.493%: expected rate two years after the fs2011 anchor (1.431%) under a driftless logit random
  walk, sigma 0.21/yr. delta = +0.059 (fit 2007-2010, r_w 1.409%) or +0.079 (fit incl. fs2011 with an
  11-month label, r_w 1.381%). Stored in `experiments/calibration/harness_level_params.json`.
- This is a judgment (assumed random-walk persistence and volatility). Cost of the hedge if truth is 1.43%:
  0.17e-4. Saving if truth is 1.70%: 1.1e-4 (`scenario_regret.csv`, row tau 1.50%).
- Asymmetry: 30% too low costs 8.17 / 10.35 / 10.93e-4 (2008 / 09 / 10); 30% too high 5.57 / 7.02 / 7.40e-4.
  Too low costs about 1.5x. Shipping ~1.5%: truth 1.28% costs 1.7e-4, 1.43% 0.2e-4, 1.62% 0.4e-4, 1.70%
  1.3e-4, 2.13% 11.8e-4.
- Label window (D-002a open): tau s0 1.37%, s3 1.39%, s6 1.49%. Switching costs at most 0.59e-4. Ranking is
  window-robust (fold-4 AUC on 2011: 0.885 s0, 0.869 s3, 0.850 s6 with an 11-month label).

## 6. Related decisions

- D-075 floor 0.03%, cap 99%: 31 PDs below the floor had 0 defaults; a 50% cap raises log loss
  0.063524 to 0.063621 (855 PDs above 50% have 65% observed). Floor analogy: R-15 para. 285, not a claim of IRB status.
- D-077 include fs2011 with an 11-month label: fold-3 analogue AUC / log loss 0.8301 / 0.07063 vs 0.8263 /
  0.07086 when dropped (point estimates, no CI). Build from the panel, not `make_labels` (D-011/D-068 bug).
- D-078 grade table (2010, CAL-01): observed default rate rises monotonically over all 11 grades; grades 4-9
  run 20-40% above grade PD in 2010 because rule e under-shot 2010 by 18%. New firms: slope 0.89
  (over-dispersed, a shape issue, D-047). Group-level corrections rejected (would be fit on test years).

## 7. H4 verdict

Level drift with the cycle was NOT confirmed as a first-order effect in the walk-forward: best leak-free
level gain 6.3e-4 vs algorithm gains of 16e-4 (GB-01 vs RC-02) and 24e-4 (vs SC-03). A 30% level miss
(8-11e-4) is half the GB/RC gap, so level is second-order only if set carefully.

## 8. Handed to the harness (D-079)

GB-01 final fit on 2007-2011 -> own calibrator anchored to r_w -> M1 shift with delta from
`harness_level_params.json` -> clip [0.0003, 0.99]. Referee contract files: `experiments/calibration/CAL-01`
(GB-01 + M1 + e), `CAL-02` (RC-02 + M1 + e), `CAL-03` (GB-01 + M1 + b). Recompute r_w and delta if the fitted rows change.

## 9. Open items

1. Human sign-off: D-073 (hedge 1.49% vs anchor 1.43%), D-077 (include fs2011), D-079.
2. D-002a / email Q3: label window and holdout period. They set the label window and the horizon h.
3. Referee has not independently scored CAL-01..03 (these are the specialist's own numbers).
4. New-firm over-dispersion (slope 0.89) needs a shape fix, not a level fix.
5. Citations: Platt formula and isotonic-vs-Platt behaviour are supported by R-19 (VERIFIED, full text, Secs. 2.1, 2.2, 5)
   and R-44. R-20 (Platt original) stays SECONDARY; R-21 (Zadrozny & Elkan 2002) not obtained, do not cite.
   Prior shift rests on R-22 eq. (4); the invariance assumption is doubtful in 2008-12 and used only as a level tool.
   The "500 defaults" rule for Platt vs isotonic is our own heuristic, not a published threshold.

## Figures

`fig/binned_calibration_by_year.png`, `fig/level_by_year.png`, `fig/monthly_hazard.png`.
