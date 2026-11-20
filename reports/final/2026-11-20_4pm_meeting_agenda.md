# Team Turquoise: 4 PM meeting, 2026-11-20 (deadline: tomorrow night)

Branch: `final-consolidation` on the shared repo (Chloe/Desmond `project-update` + Kiet's agent kit, merged).
Read before the meeting:
- `reports/features/2026-11-20_feature_lineage_raw_to_model.md`: raw data → 81 features, stage by stage, who did what
- `reports/features/2026-11-20_feature_consistency_report.md`: train vs harness parity and mismatches
- `experiments/gbm/v4/comparison.md`: LightGBM monotone vs unconstrained on the new split

## Where we are
| Done | Evidence |
|---|---|
| All work merged in one branch | `final-consolidation` |
| New split (D-156): train → validation year → test year; FY2011 lockbox closed | `reports/data/2026-11-20_v4_checks.md` |
| Feature files rebuilt on schema 1.3.0 (Desmond's fixes, Method G gone), fit on train only | same |
| Harness features = training features (0 differences on 20,003 rows) | consistency report §7 |
| LightGBM freshly tuned (13 configs, validation years only), monotone and free | `experiments/gbm/v4/` |

| Test year | Model | AUC | AR | Log-loss (shifted) | Obs / pred |
|---|---|---|---|---|---|
| FY2009 | mono | 0.8470 | 0.694 | 0.06262 | 1.23 |
| FY2009 | free | 0.8492 | 0.698 | 0.06236 | 1.22 |
| FY2010 | mono | 0.8392 | 0.678 | 0.06727 | 1.20 |
| FY2010 | free | 0.8407 | 0.681 | 0.06702 | 1.20 |

## Decisions needed (proposal in bold)
1. **Feature version: ship schema 1.3.0** and rebuild the harness from it (never mix with the 1.2.1 bundle). Owner: Kiet.
2. **Final model: LGB-v4-mono.** Free is +0.002 AUC (FY2009 CI [+0.0005, +0.0039]) but has wrong-signed drivers
   (e.g. higher interest coverage → higher PD). Sign-off: all.
3. **PD level: prior shift to tau = 1.464% (D-073), no Platt.** The provisional per-round rule (shift to the
   validation-year rate) left FY2009/10 about 1.2x under-predicted, so it is not used. Owner: Kiet (recheck on this model).
4. **`roe`**: keep as is (uncapped, unconstrained) and rename `roe_negative` → `loss_and_neg_equity` in docs/deck.
   Owner: Desmond.
5. **D-154 (1)-(3) denominator rules: known limitations in the deck**, not fixed tonight (each fix needs a retrain).
6. **Constraint changes**: drop `lt_debt_to_capital`, add 7 (lineage §9). Sign-off: Chloe.
7. **Method G off at prediction (D-152)**: sign-off Chloe. Her original bundle is not submitted.
8. Open the FY2011 lockbox **once**, after items 1-3 are signed off.

## Questions for Desmond / Chloe (lineage doc, last section)
`operating_cash_flow_to_debt` in or out? What did the "accruals bug" fix change? Docs 04/05 rewrite. Commit the ATECO lookup CSV.

## Remaining work and owners (proposal)
| # | Task | Owner | When |
|---|---|---|---|
| 1 | Lockbox score FY2011, final fit FY2007-11 (tree count from round 3: 727, scaled), tau 1.464% | Kiet (+Claude) | tonight |
| 2 | Rebuild harness on 1.3.0 + D-152 fixes; stress tests; complexity note | Kiet (+Claude), Chloe review | tonight |
| 3 | Docs 04/05 rewritten from code; `roe_negative` rename | Desmond | tonight |
| 4 | Final breakdown report (every decision, evidence, what didn't work) | Kiet (+Claude) | tomorrow AM |
| 5 | Deck 15-25 slides: cover (team, names, title, date), problem, data/label, features, split, model, calibration, results, limitations, harness, AI-use, contributions, bibliography; file name contains "Turquoise" | all; split slides | tomorrow |
| 6 | Red-team review + citation check, then submit | Kiet | tomorrow PM |

Limitations already known for the deck: new firms AUC about 0.76-0.78; only 2 selection years; D-154 edge cases;
"split type barely changes AUC" is only a non-controlled comparison (different settings), so we say "similar".
