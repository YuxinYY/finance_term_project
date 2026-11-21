# Counter-intuitive reason codes, and dropping goodwill_ratio

Team Turquoise, 2026-11-21. Decisions: **D-164** (drop goodwill_ratio). Related: D-157, D-163, D-154.

Every number comes from a file produced by code (paths in each section). Reason-code counts are in-sample: all
835,016 FY2007-2011 training rows, exact TreeSHAP.

---

## 0. Summary

1. **This is not the ablation, and not mainly about literature.**
   - The ablation measured how much *accuracy* each constraint costs.
   - This report measures how often an *explanation* (a reason code shown to a credit officer) points the wrong
     way for an individual firm.
   - It only happens on features with **no constraint**. Those were left free because the data contradicted the
     expected sign (size, goodwill_ratio, lt_debt_to_capital), or because no ratio sign applies (raw amounts:
     EBITDA, profit), or because there was no source.
2. **On the 20 constrained features there are zero counter-intuitive reason codes**, in both monotone models. In the
   unconstrained model the same features produce them for 47,768 firms (5.7%), almost all from the quick ratio.
3. **In the submission model (74 features) about 0.7% of firms see a genuine counter-intuitive code** among their
   top 5 drivers (5,769 firms). That is down from 1.2% in mono20 and 6.8% in the unconstrained model.
4. **Another 3.0% of firms see a code that only looks wrong** (24,934 firms whose only visible conflicts are artefacts) because of how debt_to_eqty is defined. With
   negative equity, debt / equity becomes negative, i.e. looks "low". The model correctly raises their PD.
   Fix this by displaying the reason as "negative equity" rather than "debt-to-equity".
5. **Dropping goodwill_ratio costs -0.00054 validation AUC** (within noise in every round), passes every harness
   test, and **halves the genuine counter-intuitive codes** (9,755 → 5,769 firms).

---

## 1. Why goodwill_ratio is dropped (D-164)

**The field is not goodwill.** The raw column `goodwill` behaves like **value added** (EBITDA + personnel cost):
- EBITDA ≤ `goodwill` on 99.99% of rows, and the median is €420,706, far too large and too common for accounting
  goodwill (D-010, `reports/data/01_profile.md` §5).
- `goodwill − ebitda` behaves like personnel cost on four checks (D-091):
  1. It is negative on only 0.012% of rows.
  2. It is zero on 26.2% of rows, and those are mostly firms with no staff fund (52.5% of them real estate).
  3. Its Spearman correlation with the staff severance fund is 0.93.
  4. Its median is 9.3% of sales.
- Our own feature library had already rejected goodwill / assets for this reason (FEATURE_SPEC,
  "value_added_to_assets: cannot be explained to a credit officer under its file name").

**What this did to the model.** The candidate sign was "+" because more goodwill (intangibles) means more risk.
FY2007 data showed the opposite (rho −0.89, D-157). That is exactly what value added would do: more value added per
euro of assets means a more productive firm and lower risk. So the feature was left unconstrained, and its reason
codes conflicted with its *name* on 2,150 firms in mono20 (Section 2).

**Cost** (walk-forward, same protocol as the constraint ablation; `experiments/gbm/2026-11-20_grid75/drop_goodwill/`):

| | Validation AUC (FY2008-10) | Validation log loss | Test AUC (FY2009-11, report only) | Best iteration |
|---|---|---|---|---|
| mono20 (75 features) | 0.85142 | 0.06011 | 0.84674 | 228 |
| mono20 without goodwill_ratio (74) | 0.85087 | 0.06015 | 0.84675 | 212.5 |
| **difference** | **−0.00054** | +0.00004 | **+0.00001** | |

Paired bootstrap, 74 minus 75 features (500 draws, validation, seed 1; `bootstrap.csv`):

| Round | ΔAUC | 95% interval | Δlog loss | 95% interval |
|---|---|---|---|---|
| 1 (FY2008) | −0.00053 | [−0.00190, +0.00075] | +0.00001 | [−0.00010, +0.00011] |
| 2 (FY2009) | +0.00002 | [−0.00123, +0.00122] | +0.00011 | [−0.00002, +0.00024] |
| 3 (FY2010) | −0.00047 | [−0.00152, +0.00045] | +0.00007 | [−0.00004, +0.00020] |

Every interval includes zero, and the test-year AUC is unchanged. The "mono" rows reproduce the earlier ablation
exactly (0.85142), so the comparison is like-for-like.

**The new final model and bundle** (`experiments/final/2026-11-21_final_fit/2026-11-21_10_train_mono20_nogw.py`):
- Model: 74 features, 20 constraints, 230 trees (mean round-3 best iteration 243 and 216), same setting and level
  recipe.
- In-sample (sanity only): AUC 0.8778 (mono20 0.8784), mean PD 1.4507%, PD 0.03%-89.0%.
- Bundle `harness/2026-11-21_variants/mono20_nogw/`, zip `2026-11-21_Turquoise_harness_mono20_nogw.zip`.
- Tests in the clean pinned virtual environment:
  - 1,023,552 PDs, 0 fallbacks, 29.8 s;
  - harness vs training pipeline max difference 1.7e-16 (`variants/mono20_nogw/parity_harness_vs_training.json`);
  - **25/25 stress tests passed** (`variants/mono20_nogw/stress_test_output.txt`).
- The team feature code still computes goodwill_ratio. The harness simply no longer selects it, so no shared code
  changed.

---

## 2. How often do reason codes point the wrong way?

Script: `experiments/final/2026-11-21_final_fit/2026-11-21_09_reason_code_conflicts.py`. Outputs:
`reports/final_model/2026-11-21_reason_code_conflicts_{summary.json, by_feature.csv, diagnostics.csv, examples.csv}`.

### 2.1 Definition
A firm has a **conflict** on feature f when either:
- it is in the **worst 10%** of firms on f (e.g. lowest equity ratio), but f **lowers** its PD; or
- it is in the **best 10%** on f, but f **raises** its PD.

For a flag: the flag is on (e.g. loss_flag = 1), but it lowers the PD.

A conflict is **visible** when f is among the firm's **top 5 drivers**, which is what a reason-code report would
show. "Worst" and "best" use each feature's intended direction:
- the sign table (D-157) for the 26 features with a candidate sign;
- for the raw amounts EBITDA and profit (higher = safer) and debt_to_eqty (higher = riskier): our assumption,
  marked as such, since no verified source gives a sign for a raw amount.

**Construction artefact.** debt_to_eqty and lt_debt_to_capital turn *negative* when equity (or long-term capital) is
negative. A negative value then means insolvency, not low leverage. When the model raises PD for such a firm it is
**right**, but the reason code reads "low debt-to-equity raised your PD". These are counted separately as
artefacts (the D-154 #1 construction issue).

### 2.2 Results: firms with at least one wrong-way code in their top 5

| Model | Any visible conflict (top 5) | **Genuine (excluding the artefact)** | In top 3 | On the 20 constrained features |
|---|---|---|---|---|
| Unconstrained (D-160) | 62,666 (7.5%) | **57,141 (6.8%)** | 18,853 (2.3%) | **47,768 (5.7%)** |
| mono20 (75 features) | 12,837 (1.5%) | **9,755 (1.2%)** | 913 (0.1%) | **0** |
| **mono20_nogw (74, submission)** | 30,703 (3.7%) | **5,769 (0.7%)** | 10,681 (1.3%) | **0** |

Notes:
- The constraints remove every wrong-way code on the 20 constrained features. That is the guarantee from D-163,
  now counted.
- The 74-feature model has the fewest genuine conflicts.
- Its *total* is higher because debt_to_eqty takes over part of what goodwill_ratio did, and debt_to_eqty is
  where the negative-equity artefact lives (25,187 of its 28,869 visible conflicts; next table). The "top 3"
  count rises for the same reason: for a negative-equity firm, debt_to_eqty is often a top driver.

### 2.3 By feature, submission model (mono20_nogw)

| Feature | Why it is unconstrained | Visible conflicts (top 5) | Of which artefact | Genuine | Mean \|SHAP\| when visible | What is going on |
|---|---|---|---|---|---|---|
| debt_to_eqty | no sign in the table; ratio breaks with negative equity | 28,869 | **25,187** | 3,682 | 0.29 | 87% have negative equity; the model correctly raises PD (default rate across all 28,869 is 9.6% vs 1.36% overall). Display as "negative equity" |
| ebitda (amount) | raw amount, no ratio sign | 1,066 | 0 | 1,066 | 0.19 | 84% have negative EBITDA, yet EBITDA lowers their PD. The PD is already pushed up by the cash-flow and profitability *ratios*; the amount acts as a conditional correction (an offset, as with the quick ratio) |
| size | data contradicted the sign (rho −0.37, D-157) | 620 | 0 | 620 | 0.14 | Small firms (median log assets 14.2 vs 15.1) where small size lowers PD; default rate 0.6% |
| profit (amount) | raw amount | 374 | 0 | 374 | 0.22 | Large positive profit that raises PD: a conditional offset against the margin ratios |
| lt_debt_to_capital | data contradicted the sign (+0.35) | 134 | 112 | 22 | 0.15 | Mostly negative capital (artefact) |
| ateco_sector_te | just under the bar (+0.68) | 9 | 0 | 9 | 0.10 | negligible |
| **all 20 constrained features** | — | **0** | — | **0** | — | the constraint guarantee |

### 2.4 For comparison: mono20 (75 features)

| Feature | Visible conflicts | Of which artefact |
|---|---|---|
| ebitda | 6,348 | 0 |
| debt_to_eqty | 2,933 | 2,896 |
| **goodwill_ratio** | **2,150** | 0 |
| profit | 695 | 0 |
| size | 487 | 0 |
| lt_debt_to_capital | 357 | 356 |
| ateco_sector_te | 45 | 0 |

Dropping goodwill_ratio removes its 2,150 conflicts, and EBITDA conflicts fall from 6,348 to 1,066. In mono20,
goodwill (value added = EBITDA + staff cost) and EBITDA were near-copies, and the model used EBITDA as an offset
against it.

### 2.5 For comparison: the unconstrained model, top rows

| Feature | Visible conflicts | Note |
|---|---|---|
| **quick_ratio** | **42,890** | constrained in mono20 → 0 |
| goodwill_ratio | 8,501 | dropped in the submission model |
| debt_to_eqty | 5,409 | all artefact |
| cash_interest_coverage | 3,354 | constrained in mono20 → 0 |
| cash_ratio | 1,428 | constrained in mono20 → 0 |
| size | 899 | |
| int_coverage | 266 | constrained in mono20 → 0 |

### 2.6 Why does an unconstrained feature point the wrong way?

SHAP gives each feature its share of *this model's* score, holding the other features at what the firm actually
has. When two features carry overlapping information, the trees often use one as the main effect and the other as
a **correction**, and a correction can have the opposite sign.

Example: a loss-making firm's cash-flow and profitability ratios already add a lot of risk. Within that group,
the trees found that the larger (more negative) the EBITDA amount, the slightly *lower* the extra risk, plausibly
because a large absolute EBITDA means a large firm. That is a statement about the remaining risk *given the
ratios*. It is not "losing money is good". But a credit officer reading "EBITDA lowered your PD" would hear the
latter.

A constraint forbids exactly this, which is why the 20 constrained features never do it. For raw amounts there is
no defensible sign to impose: an amount mixes size with performance, and the literature gives signs for ratios,
not amounts.

---

## 3. What to do with reason codes in the deck and the harness documentation

1. **Show reason codes from the 20 constrained ratios** (plus industry and legal form, which read naturally). These
   never point the wrong way.
2. **Translate negative-denominator ratios:** when equity < 0, report "negative equity (insolvent on the balance
   sheet)" instead of "debt-to-equity". This covers 25,187 + 112 of the visible artefact cases.
3. **Do not show raw-amount features (ebitda, profit, COGS, exp_financing) as reason codes.** They are inputs that
   sharpen the score, but their individual signs are conditional corrections.
4. **Disclose the remaining rate:** "for about 0.7% of firms, one of the top-5 drivers is an unconstrained input
   whose direction is a conditional correction; we do not display those."
5. **Not done, possible later:** constraining the amounts or debt_to_eqty would need a defensible sign; the
   construction fix for negative denominators is open as D-154.

---

## 4. Files

| What | Path |
|---|---|
| Goodwill drop, walk-forward | `experiments/gbm/2026-11-20_grid75/2026-11-21_11_drop_goodwill.py` → `drop_goodwill/{results,summary,bootstrap}.csv` |
| 74-feature final model and bundle | `experiments/final/2026-11-21_final_fit/2026-11-21_10_train_mono20_nogw.py` → `variants/mono20_nogw/`, `harness/2026-11-21_variants/mono20_nogw/`, zip |
| Bundle tests | `variants/mono20_nogw/parity_harness_vs_training.json`, `stress_test_output.txt` |
| Profile of the 74-feature model | `2026-11-21_08_mono20_profile.py mono20_nogw` → `reports/final_model/2026-11-21_mono20_nogw_*` |
| Reason-code conflicts | `2026-11-21_09_reason_code_conflicts.py` → `reports/final_model/2026-11-21_reason_code_conflicts_*` |
| Decision | `docs/DECISION_LOG.md` D-164 |
