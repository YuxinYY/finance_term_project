# Monotone constraints, the quick ratio, and the final-model choice: full record

Team Turquoise, 2026-11-21. Decision entries: D-157, D-158, D-159, D-160, D-161, **D-162** (this audit),
**D-163** (model choice). Written for the slide deck and its appendix.

Every number comes from a file produced by code; the path is given in each section. Quotes were checked against
the PDF text on the cited page.

**Contents**
- 0 Slide-ready summary and slide / appendix plan
- 1 The question and how the investigation went, including the corrections
- 2 Quick ratio: what is in the data
- 3 Shape before fitting
- 4 What the models do with the quick ratio
- 5 Operating cash flow ratio: a smaller issue
- 6 Literature check
- 7 How a monotone constraint works
- 8 Why 20 constraints when only 2 features "disagree"
- 9 Model choice: unconstrained (D-160) vs mono20 (D-161)
- 10 What did not hold up (appendix "what didn't work")
- 11 Limits and possible future fixes
- 12 File index

---

## 0. Slide-ready summary

**Five sentences:**
1. Before fitting, the quick ratio is **not linear**. The default rate rises from 1.35% to 2.13% over the lowest
   three deciles, then falls to 0.59%. The overall slope is down (decile rank correlation -0.81), in each of
   FY2007-2011.
2. The **unconstrained model reverses it**: a higher quick ratio *raises* the PD (value-vs-SHAP rank correlation
   +0.38). Three RiskCalc documents state the opposite directly (R-01 p. 13; R-33 p. 11; R-05 p. 54).
3. A monotone constraint **does not require linearity**, only "never goes the wrong way". It acts during
   training. Where the data are non-monotone, the model flattens instead of reversing.
4. We constrain **every ratio with an established direction (20)**, not just the two that disagree overall.
   Features that agree "on average" still go the wrong way over 10-75% of their curve. The constraint removes
   these local reversals at almost no cost, and on some features it improves accuracy.
5. **Cost: about 0.001 AUC** (0.00075 in the constraint ablation; 0.00134 in the grid bootstrap), almost all of
   it from the quick ratio. **We submit the monotone model (mono20)**: every constrained reason code follows
   credit logic.

**Suggested slides (main deck, 2-3 slides)**

| Slide | Content | Source section |
|---|---|---|
| "Monotone ≠ linear" | Quick-ratio decile default-rate chart (hump, then decline) next to the two SHAP curves (free reversed, mono20 two-step) | 3, 4 |
| "Why we constrain 20 ratios" | Coarse vs fine sign test table (top 8 rows of Section 8 table), cost 0.001 AUC | 8, 9 |
| "Model choice" | free vs mono20 table, the trade-off sentence, literature quotes | 6, 9 |

**Suggested appendix pages**

| Page | Content |
|---|---|
| A1 | Quick-ratio distribution and value states (Section 2 tables) |
| A2 | Bottom-decile sector mix and the short-term-debt ≈ 0 tail (Section 2) |
| A3 | Operating cash flow ratio deciles, and the "one correlation number misleads" lesson (Section 5) |
| A4 | Full literature table with quotes (Section 6) |
| A5 | Full 26-feature constraint table (Section 8) |
| A6 | Constraint ablation, all configurations (Section 8) |
| A7 | Grid bootstrap by round, and the deviation from the pre-set rule (Section 9) |
| A8 | What did not hold up (Section 10) |

---

## 1. The question and how the investigation went

1. **Start.** KN observed that the quick ratio does not have a linear relationship with PD before fitting, and
   that the unconstrained final model (D-160) seemed to give it the wrong sign. The question: should a
   relationship that is not linear still get a monotone constraint?
2. **Univariate check** (Section 3): a hump at the bottom, then a decline. The decline is stable in all five years.
3. **Model check** (Section 4): confirmed. The free model reverses the sign; mono20 does not.
4. **Data deep-dive** (Section 2): the extremes come from short-term debt ≈ 0 (top), and from real estate,
   construction and data errors (bottom).
5. **Second feature.** An earlier review flagged operating_cash_flow_ratio as "no direction" (+0.01). The decile
   view shows a correct cliff with a slight drift, so it was downgraded to low severity (Section 5).
6. **Literature check, triggered by KN.** No source states the direction for cash flow / total assets. Our claim
   map had marked it SUPPORTED, and that was corrected. The quick ratio has three direct statements (Section 6).
7. **Why 20 constraints?** The coarse test (overall trend) understates the conflicts; the fine test (share of
   the curve moving against the sign) shows many (Section 8).
8. **Decision:** submit mono20 (Section 9, D-163).

---

## 2. Quick ratio: what is in the data

**Definition in our model** (`feature_rules.py:74`): (cash + accounts receivable) / short-term debt.
- When short-term debt is exactly 0, the code divides by 1 instead (`_safe_divide`).
- The feature enters LightGBM as this raw value: no capping. Only 21 rows with a missing component are imputed.
- The literature's quick ratio is (current assets - inventory) / current liabilities (R-05 p. 36). Ours is close
  but not identical; say so on the slide.

Source: `experiments/gbm/2026-11-20_grid75/quick_ratio_audit/qr_distribution.csv`, `qr_states.csv`,
`qr_extremes.csv`, `qr_tails.csv`, `summary.json`. Data: the 835,016 FY2007-2011 rows the final model was fit on.

| Percentile | min | 1% | 10% | 25% | **median** | 75% | 90% | 99% | 99.9% | max |
|---|---|---|---|---|---|---|---|---|---|---|
| Quick ratio | -986,328 | 0.009 | 0.12 | 0.37 | **0.79** | 1.29 | 2.72 | 54 | 150,204 | 70,111,651 |

The median firm holds €0.79 of cash and receivables per €1 of short-term debt. The mean (1,729) is meaningless
because the extremes drive it.

| Type of value | Rows | Share | Default rate |
|---|---|---|---|
| 0-1 (tight liquidity) | 524,851 | 62.9% | **1.70%** |
| 1-3 | 234,285 | 28.1% | 0.78% |
| 3-10 | 48,615 | 5.8% | 0.71% |
| 10-100 | 20,639 | 2.5% | 0.84% |
| ≥ 100 | 4,296 | 0.5% | 1.21% |
| short-term debt = 0 | 1,383 | 0.17% | 2.17% |
| short-term debt < 0 | 319 | 0.04% | 0.94% |
| negative ratio (negative cash or receivables) | 430 | 0.05% | 0.70% |
| exactly 0 | 177 | 0.02% | 1.13% |
| missing component | 21 | 0.003% | 0% |

### Why values get so high
The cause is the **denominator going to zero**. These firms are not super-liquid.
- In the top 1%, median short-term debt is **0.1% of assets**, against 46.5% for all firms. Their cash plus
  receivables is a normal 26% of assets.
- All five largest values have short-term debt = 0, so the ratio becomes a euro amount. Example: €2.8M cash plus
  €67M receivables over zero debt gives 70,111,651.
- The default rate climbs back above a ratio of 10 (0.84%, then 1.21%). Our untested reading: a firm with no
  short-term debt is often a holding, shell or group-financed firm, not a strong one.

### Why values get so low
- **Negative values are data errors or recording conventions.** 1,766 rows have negative cash, 449 negative
  receivables and 319 negative short-term debt. The minimum (-986,328) is cash of -€1.29M divided by short-term
  debt of 0, which is replaced by 1.
- **Low positive values are mostly two sectors.** The bottom decile (ratio < 0.12) is 43% real estate (ATECO 68,
  19.5% of all rows) and 31.5% building construction (ATECO 41, 14% of all rows).
  - Construction: the median firm holds 81% of current assets as something other than cash or receivables
    (work in progress / inventory). The quick ratio excludes these by design, so it is low for them by
    construction.
  - Real estate has the lowest default rate of the three groups (0.88%, against 1.49% for other sectors). This
    pulls the bottom decile down. Why real estate defaults less is an untested assumption (e.g. asset-backed
    lending).
  - Source: `qr_deciles_by_sector.csv`, `summary.json`.

---

## 3. Shape before fitting

Source: `quick_ratio_audit/qr_deciles.csv`, `qr_deciles_by_sector.csv`.

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Upper bound | 0.12 | 0.28 | 0.46 | 0.63 | 0.79 | 0.95 | 1.14 | 1.52 | 2.72 | max |
| Default rate, all | 1.35% | 1.77% | **2.13%** | 2.09% | 1.57% | 1.33% | 1.24% | 0.73% | **0.59%** | 0.77% |
| Excluding ATECO 68/41 | 2.01% | 2.68% | 2.76% | 2.32% | 1.58% | 1.34% | 1.21% | 0.71% | 0.54% | 0.92% |
| FY2007 | 0.79% | 1.08% | 1.31% | 1.50% | 1.01% | 0.82% | 0.76% | 0.48% | 0.41% | 0.42% |
| FY2009 | 1.53% | 2.23% | 2.38% | 2.39% | 1.85% | 1.41% | 1.51% | 0.99% | 0.67% | 0.83% |
| FY2011 | 1.61% | 1.81% | 2.28% | 2.14% | 1.67% | 1.41% | 1.22% | 0.62% | 0.55% | 0.73% |

- **Shape:** a hump at the bottom, then a decline. Decile rank correlation -0.81 pooled; by year -0.76, -0.67,
  -0.81, -0.67, -0.81 (FY2007-2011).
- **Part of the hump is sector mix** (real estate sits at the bottom). Part survives without those sectors.
  Decile 1 also holds the negative-value data errors.
- **The decile-10 uptick** is the short-term debt ≈ 0 group.

---

## 4. What the models do with the quick ratio

Exact TreeSHAP on a 100,000-row sample (log-odds; positive = raises the PD). Source:
`quick_ratio_audit/qr_shap_deciles.csv`, `summary.json`.

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | value-vs-SHAP rank correlation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Unconstrained (D-160) | **-0.136** | -0.043 | +0.037 | +0.060 | -0.005 | -0.018 | -0.004 | -0.006 | +0.013 | **+0.083** | **+0.38 (reversed)** |
| Monotone mono20 (D-161) | +0.015 | +0.016 | +0.019 | +0.019 | -0.013 | -0.015 | -0.015 | -0.014 | -0.013 | -0.009 | -0.66 |

- **Unconstrained:** the least liquid firms get the biggest PD *reduction*, and the most liquid get the biggest
  *increase*. A credit officer cannot be given that explanation.
- **Monotone:** a two-step shape. Low liquidity (below about 0.63) adds risk; high liquidity removes it. The hump
  is flattened, not reversed.

### Why the unconstrained model reverses the sign
- **Overlap with a near-copy.** cur_debt_ratio (current assets / short-term debt) has the same denominator, with
  rank correlation +0.62. Once the model knows cur_debt_ratio, the quick ratio only carries a small conditional
  correction.
- **Stronger firms can run lean (sourced).** R-05 p. 54: "lower rated and unrated firms have significantly higher
  quick ratios than their investment-grade counterparts... A low quick ratio is therefore an effect of a firm with
  a high credit rating, not the cause of a high credit rating."
- **The tails mix different kinds of firms** (Section 2).

---

## 5. Operating cash flow ratio: a smaller issue

Definition: (profit + depreciation and amortisation) / total assets (`feature_rules.py:84`). Source:
`quick_ratio_audit/ocf_shap_deciles.csv`.

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Upper bound | -1.5% | -0.1% | 0.6% | 1.4% | 2.2% | 3.2% | 4.4% | 6.4% | 10.2% | max |
| Default rate | **6.37%** | 1.33% | 1.34% | 1.13% | 0.86% | 0.71% | 0.54% | 0.45% | 0.36% | 0.49% |
| Unconstrained SHAP | **+0.353** | -0.044 | -0.050 | **-0.062** | -0.057 | -0.052 | -0.055 | -0.049 | -0.036 | **-0.008** |
| mono20 SHAP | +0.399 | -0.027 | -0.030 | -0.039 | -0.046 | -0.049 | -0.053 | -0.056 | -0.057 | -0.058 |
| Twin ocf_to_debt SHAP, unconstrained | +0.319 | +0.112 | +0.087 | +0.057 | -0.015 | -0.068 | -0.124 | -0.153 | -0.171 | -0.170 |

- **The unconstrained model gets the important part right:** cash-burning firms get +0.35.
- **Above that it is flat, with a slight wrong-way drift** (-0.062 to -0.008). That is about 0.05 log-odds,
  roughly a 5% relative change in PD (e.g. 1.36% to 1.43%).
- **The gradual "more cash is better" slope sits on its near-copy** operating_cash_flow_to_debt (rank
  correlation 0.94).
- **Severity: low / cosmetic.** An earlier review called it the bigger problem; that was wrong (Section 10).

**Lesson for the deck:** a single value-vs-SHAP correlation (+0.01 here) misreads a cliff-shaped effect. 90% of
rows sit on a nearly flat stretch and swamp the large, correct jump at decile 1. Show decile tables or dependence
plots, not one number.

---

## 6. Literature check (verified against the PDF text)

| Feature | Direct statement of the PD direction? | Exact text |
|---|---|---|
| **quick_ratio** | **Yes** | R-01 (RiskCalc v3.1) p. 13: "Liquidity variables include cash and marketable securities to assets, the current ratio, and the quick ratio. → High liquidity reduces the probability of default." |
| | **Yes** | R-33 (RiskCalc 4.0 France) p. 11, Table 2: "Liquidity: Includes cash and marketable securities to assets, the current ratio, and the quick ratio. High liquidity reduces the probability of default." |
| | **Yes** | R-05 (RiskCalc 2000) p. 54: "the quick ratio was found to be strongly negatively related to default: the higher the quick ratio, the lower the default probability." |
| | Caveat | R-32 (Altman) pdf p. 10: current and quick ratios are "subject to perverse trends for some failing firms". |
| **operating_cash_flow_ratio** (cash flow / total assets) | **No direct statement** | Closest is its twin, R-33 p. 11: "High debt coverage reduces the probability of default." R-33 p. 12: "(Profit for Period + Depreciation and Amortization)/(Liabilities – Cash and Marketable Securities) is downward-sloping, indicating that large values of this ratio lower the probability of default." Same numerator as ours, different denominator. |

These sources mention cash flow but do **not** state the direction for cash flow / assets, so do not cite them for
the sign:
- R-01 p. 13: a cash-flow ratio reduced the size effect.
- R-01 p. 34: lists "Cash Flow to Current Liabilities".
- R-05 p. 15: Beaver found cash flow / assets differs between failed and viable firms; the sentence gives no direction.
- R-05 p. 33: "'Cash flow' is less valuable as an univariate predictor than simple net income".

The earlier SUPPORTED label in `docs/2026-11-16_feature_literature_claims.md` was corrected on 2026-11-21 (D-162).

**Slide phrasing for cash flow:** "sign by analogy to RiskCalc France's debt-coverage ratio (R-33 p. 12),
confirmed in our data (decile rank correlation -0.94, -0.94, -0.91 in FY2007-09)."

---

## 7. How a monotone constraint works

**The constraint acts during training, not as an after-the-fact check.**
- At every split on a constrained feature, LightGBM rejects any split that would make the prediction move the
  wrong way as that feature increases, holding all other features fixed.
- No tree in the model ever contains a wrong-way step for that feature.
- It does not force a straight line. Any shape is allowed as long as it never reverses: steep, flat, curved,
  stepped.

**Toy example** (synthetic data). The true PD falls with liquidity but has a bump near 0.5. Source:
`experiments/gbm/2026-11-20_grid75/2026-11-21_10_constraint_table_and_demo.py` →
`quick_ratio_audit/monotone_demo.txt`.

| Liquidity | 0.1 | 0.5 | 1.0 | 2.0 | 2.9 |
|---|---|---|---|---|---|
| Free PD | 5.05% | 9.35% | 0.76% | 0.20% | **1.11%** |
| Monotone (-1) PD | 6.85% | 6.85% | 1.42% | 1.04% | 0.18% |

- The free model follows the bump, and also invents an upward step at 2.9 out of pure noise.
- The monotone model flattens the bump (6.85% at both 0.1 and 0.5) and never goes back up.
- The noise step at 2.9 is exactly what an "overall trend agrees" check cannot see.

---

## 8. Why 20 constraints when only 2 features "disagree"

### Coarse test vs fine test
- **Coarse test:** the overall trend across all firms (value-vs-SHAP rank correlation). In the final
  unconstrained model only quick_ratio (+0.38) and operating_cash_flow_ratio (+0.01) fail it among the
  constrained features.
- **Fine test:** the share of the feature's curve that moves against the expected sign
  (`sign_violation_check.csv`, development fits). It catches local reversals.

**A feature can pass the coarse test and still give some firms a wrong-way reason code.**

Source: `quick_ratio_audit/constraint_table.csv`. "Wrong-way share" is from the development free model;
"value-vs-SHAP" is on the 100,000-row sample of the final models.

| Feature | Expected sign | Constrained in mono20 | Free model: wrong-way share | Final free: value-vs-SHAP | mono20: value-vs-SHAP |
|---|---|---|---|---|---|
| quick_ratio | − | yes | **83%** | **+0.38 ✗** | -0.66 ✓ |
| int_coverage | − | yes | **75%** | -0.64 ✓ | -0.79 ✓ |
| operating_cash_flow_margin | − | yes | **71%** | -0.38 ✓ | -0.80 ✓ |
| net_prof_margin | − | yes | 37% | -0.84 ✓ | -0.86 ✓ |
| wc_ratio | − | yes | 36% | -0.52 ✓ | -0.90 ✓ |
| operating_cash_flow_ratio | − | yes | 26% | **+0.01 ✗** | -0.89 ✓ |
| cash_interest_coverage | − | yes | 22% | -0.60 ✓ | -0.81 ✓ |
| op_margin | − | yes | 21% | -0.85 ✓ | -0.94 ✓ |
| roa | − | yes | 19% | -0.44 ✓ | -0.94 ✓ |
| cur_debt_ratio | − | yes | 16% | -0.85 ✓ | -0.82 ✓ |
| payables_ratio | + | yes | 14% | +0.81 ✓ | +0.91 ✓ |
| cash_ratio | − | yes | 13% | -0.90 ✓ | -0.93 ✓ |
| ebitda_to_debt | − | yes | 12% | -0.87 ✓ | -0.92 ✓ |
| cash_to_financing_exp_ratio | − | yes | 10% | -0.87 ✓ | -0.92 ✓ |
| operating_cash_flow_to_debt | − | yes | 10% | -0.92 ✓ | -0.94 ✓ |
| financing_expense_ratio | + | yes | 10% | +0.85 ✓ | +0.87 ✓ |
| tangible_asset_ratio | − | yes | 2% | -0.94 ✓ | -0.94 ✓ |
| equity_ratio | − | yes | 1% | -0.95 ✓ | -0.94 ✓ |
| AR_ratio | + | yes | 0% | +0.87 ✓ | +0.94 ✓ |
| loss_flag | + | yes | 0% | +0.80 ✓ | +0.82 ✓ |
| lt_debt_to_capital | + | **no** | 46% | -0.14 ✗ | -0.06 ✗ |
| goodwill_ratio | + | **no** | 37% | -0.55 ✗ | +0.05 ≈ 0 |
| ateco_sector_te | + | no | 19% | +0.81 ✓ | +0.80 ✓ |
| st_debt_share | + | no | 9% | +0.83 ✓ | +0.77 ✓ |
| size | − | no | 2% | -0.88 ✓ | -0.89 ✓ |
| debt_detail_missing | + | no | 0% | +0.60 ✓ | +0.60 ✓ |
| neg_bs_flag | + | no | — | unused | unused |

**Reading the table:**
1. Ten constrained features run against their sign over at least 16% of their curve in the free model. Three run
   against it over more than 70%. "Only 2 disagree" counts the coarse test only.
2. Under mono20, every constrained feature has the expected direction (✓ in the last column), by construction.
3. Value-vs-SHAP for mono20 is not exactly -1/+1 because of interactions and flat stretches. The constraint
   guarantees "never the wrong way", not "always moving".

### Which features are constrained, and why (D-157)
- A feature is constrained only if it has an expected sign (from the literature, the feature spec or the old
  team model) **and** the FY2007 training data agree (decile default-rate rank correlation of the right sign with
  |rho| ≥ 0.7; flags |z| ≥ 2).
- That gives 20 of the 75 features.
- Candidates with a sign but failing the data check stay **unconstrained**: size (-0.37), st_debt_share (+0.13),
  lt_debt_to_capital (+0.35), goodwill_ratio (expected +, data -0.89), neg_bs_flag, debt_detail_missing,
  ateco_sector_te (+0.68).
- So mono20 still shows goodwill_ratio and lt_debt_to_capital off their *candidate* sign. That is by design: we
  do not force a sign the data contradict.

### Constraint ablation: what each constraint costs or buys
Source: `experiments/gbm/2026-11-20_grid75/ablation_summary.csv`. Mean validation AUC over 6 fits (3 rounds × 2
seeds), D-158 winning setting. Seed noise is about 0.0005 AUC per round; differences below about 0.0003 are noise.

| Configuration | Constraints | AUC | vs full monotone | Log loss |
|---|---|---|---|---|
| mono_m3 (quick_ratio, int_coverage, ocf_margin released; **chosen on validation, information only**) | 17 | 0.85289 | +0.00147 | 0.05995 |
| lit13 (withdrawn, frees quick_ratio) | 13 | 0.85235 | +0.00093 | 0.06003 |
| Unconstrained | 0 | 0.85217 | +0.00075 | 0.05998 |
| release quick_ratio only | 19 | 0.85215 | **+0.00074** | 0.06002 |
| release operating_cash_flow_ratio only | 19 | 0.85197 | +0.00056 | 0.06005 |
| release cash_ratio only | 19 | 0.85192 | +0.00051 | 0.06007 |
| release payables_ratio only | 19 | 0.85188 | +0.00046 | 0.06007 |
| release ocf_to_debt only | 19 | 0.85170 | +0.00028 | 0.06008 |
| release ocf_margin only | 19 | 0.85162 | +0.00020 | 0.06006 |
| release cash_interest_coverage only | 19 | 0.85162 | +0.00020 | 0.06008 |
| release op_margin only | 19 | 0.85156 | +0.00014 | 0.06010 |
| release ebitda_to_debt only | 19 | 0.85156 | +0.00014 | 0.06011 |
| release cash_to_financing_exp_ratio only | 19 | 0.85153 | +0.00011 | 0.06010 |
| release cur_debt_ratio only | 19 | 0.85152 | +0.00010 | 0.06011 |
| release loss_flag only | 19 | 0.85150 | +0.00008 | 0.06010 |
| release net_prof_margin only | 19 | 0.85147 | +0.00005 | 0.06010 |
| release tangible_asset_ratio only | 19 | 0.85146 | +0.00004 | 0.06009 |
| release int_coverage only | 19 | 0.85144 | +0.00003 | 0.06010 |
| **Full monotone (mono20)** | 20 | **0.85142** | — | 0.06011 |
| release financing_expense_ratio only | 19 | 0.85136 | -0.00006 | 0.06011 |
| release roa only | 19 | 0.85131 | **-0.00011** | 0.06012 |
| release wc_ratio only | 19 | 0.85129 | **-0.00013** | 0.06012 |
| release AR_ratio only | 19 | 0.85127 | **-0.00015** | 0.06011 |
| release equity_ratio only | 19 | 0.85115 | **-0.00027** | 0.06011 |

**Reading the ablation:**
1. **Releasing quick_ratio alone recovers essentially the whole free-vs-monotone gap** (+0.00074 of +0.00075).
2. Most other single releases are within noise.
3. **Releasing equity_ratio, AR_ratio, wc_ratio, roa or financing_expense_ratio *lowers* AUC.** On those features
   the constraint acts like regularisation: it stops the trees chasing noise (the 2.9 step in Section 7).
4. mono_m3 looks best, but its three released features were chosen by looking at validation results, so its score
   is optimistically biased. It is not a fair candidate, and it frees the quick ratio, which the literature backs
   constraining.

---

## 9. Model choice: unconstrained (D-160) vs mono20 (D-161)

### Evidence

| | Unconstrained (D-160) | mono20 (D-161) | Source |
|---|---|---|---|
| Constraints | 0 | 20 | D-157 |
| Trees (final fit) | 210 | 240 | D-160, `variants_summary.csv` |
| Grid validation AUC, mean of 3 rounds | 0.8529 | 0.8519 | D-158, `grid75` report |
| Grid validation log loss | 0.05994 | 0.06002 | D-158 |
| Ablation validation AUC, 6 fits | 0.85217 | 0.85142 | `ablation_summary.csv` |
| Quick ratio reason code | **reversed** | correct, literature-backed | Section 4 |
| Constrained features with wrong-way curve segments | 10 with ≥ 16% | 0 | Section 8 |
| In-sample AUC (FY2007-11, sanity only) | 0.8836 | 0.8784 | `final_insample_summary.json`, `variants_summary.csv` |
| Mean final PD in-sample (same prior-shift level recipe) | 1.451% | 1.451% | same |
| Harness: 1,023,552 rows, parity, stress tests | PASS | PASS (0 fallbacks, parity 2.2e-16, 25/25) | `reports/harness/2026-11-21_harness_test_report.md` |

**Paired bootstrap, free minus monotone** (`experiments/gbm/2026-11-20_grid75/bootstrap_free_minus_mono.csv`):

| Round | ΔAUC | 95% interval | Δlog loss | 95% interval |
|---|---|---|---|---|
| 1 | +0.00037 | [-0.00165, +0.00235] | +0.00002 | [-0.00012, +0.00015] |
| 2 | +0.00074 | [-0.00065, +0.00228] | -0.00008 | [-0.00022, +0.00007] |
| 3 | **+0.00291** | [+0.00148, +0.00440] | -0.00029 | [-0.00044, -0.00015] |
| Mean of 3 | **+0.00134** | [+0.00044, +0.00232] | -0.00012 | [-0.00020, -0.00004] |

### Honest reading
- **The free model is measurably, but slightly, more accurate.** Its advantage is significant only in round 3
  (validation FY2010); rounds 1 and 2 are within noise.
- Both models use the same preprocessing and the same level recipe, so the mean PD level is the same.

### Deviation from the pre-set rule (must be disclosed)
- D-158 fixed this rule before results: "monotone is the default; unconstrained only if its bootstrap ΔAUC interval
  excludes 0 and none of its top-10 drivers moves > 50% against the expected sign."
- The free model passed that rule, which is why D-160 chose it.
- The rule only looked at the top 10 drivers. The quick ratio (gain rank 14; SHAP rank 18) sits outside them.
- We now override the rule on the grounds of explainability, using evidence found after the rule was set
  (Sections 4-6).
- This is a judgment call and is logged as such (D-163). It is not a data-driven selection: no accuracy number
  favours mono20.

### Decision (D-163): submit mono20
- **Approve / decline (ordering):** the cost is about 0.001 AUC, which is not visible in practice. Ordering is
  essentially unchanged.
- **Pricing (level):** the level recipe and mean PD are identical.
- **Explaining decisions (CLAUDE.md rule 8):** every constrained reason code follows credit logic. No borrower is
  told "your high liquidity raised your PD".
- **Literature:** the quick-ratio sign is stated directly by three RiskCalc documents.
- **Grading:** formulation and documentation are 25 of the 30 points; the holdout is 5.

**One-line version for the deck:** "We constrain every ratio whose direction is established, so each firm's reason
codes follow credit logic. In practice this changes one driver (the quick ratio), stops local wiggles in the
others, and costs about 0.001 AUC."

---

## 10. What did not hold up (appendix: "what didn't work")

| Earlier claim or idea | What we found | Where corrected |
|---|---|---|
| "Only 2 features disagree, so only 2 constraints matter" | The coarse test hides local reversals; 10 constrained features have ≥ 16% wrong-way movement | Section 8 |
| operating_cash_flow_ratio is "the bigger issue" / "no direction" / "as likely to say raises as lowers" | The decile view shows a correct cliff plus a slight drift; low severity | Section 5; correction note in `2026-11-21_sign_table_monotone_decision.md` |
| Cash-flow ratio sign "SUPPORTED" by R-01 p. 13 and p. 34 | Neither page states a direction; the sign rests on analogy to R-33 p. 12 plus our data | Section 6; `2026-11-16_feature_literature_claims.md` |
| lit13 (free the quick ratio, citing non-monotone caveats) | Three direct sources state "-"; the caveats are mostly about the current ratio | D-159 correction |
| "Non-linear before fitting, so don't constrain" | A monotone constraint does not need linearity; the decline is stable over 5 years | Sections 3, 7 |
| The quick-ratio sign reversal "is a receivables effect" (earlier guess) | Unsourced. The sourced explanation is R-05 p. 54 (strong firms run lean), plus overlap with cur_debt_ratio | Section 4 |
| mono_m3 as a candidate (best AUC) | Its released features were chosen on validation, so the score is optimistic; it also frees the quick ratio | Section 8 |

---

## 11. Limits and possible future fixes

- **Untested readings:** why real estate defaults less; what firms with short-term debt = 0 are.
- **SHAP is in-sample**, on training rows. It describes the model, not out-of-sample performance.
- **The ablation is a single setting** (the D-158 winner), 6 fits, with seed noise of about 0.0005 per round.
- **D-157's sign check uses one year** (FY2007, 1,260 defaults).
- **Possible fixes, not in the frozen model:**
  - cap the quick ratio near its 99th percentile;
  - set it to missing when short-term debt ≤ 0 or a component is negative. This would remove the mixed tails and
    probably most of the hump.

---

## 12. File index

| What | Path |
|---|---|
| Quick ratio / cash-flow audit script | `experiments/gbm/2026-11-20_grid75/2026-11-21_09_quick_ratio_ocf_audit.py` |
| Constraint table + toy demo script | `experiments/gbm/2026-11-20_grid75/2026-11-21_10_constraint_table_and_demo.py` |
| All audit outputs | `experiments/gbm/2026-11-20_grid75/quick_ratio_audit/` (`qr_*.csv`, `ocf_shap_deciles.csv`, `constraint_table.csv`, `monotone_demo.txt`, `summary.json`) |
| Sign table and D-157 rule | `experiments/gbm/2026-11-20_grid75/sign_table.csv`, `sign_table.md` |
| Fine sign test (development) | `experiments/gbm/2026-11-20_grid75/sign_violation_check.csv` |
| Ablation | `experiments/gbm/2026-11-20_grid75/ablation_summary.csv`, `ablation_results.csv` |
| Grid bootstrap | `experiments/gbm/2026-11-20_grid75/bootstrap_free_minus_mono.csv` |
| Final models | `experiments/final/2026-11-21_final_fit/final_lightgbm.txt` (free), `variants/mono20/lightgbm.txt` |
| mono20 harness bundle | `harness/2026-11-21_variants/mono20/`, zip `2026-11-21_Turquoise_harness_mono20.zip` |
| Earlier sign-conflict report | `reports/features/2026-11-21_sign_table_monotone_decision.md` |
| Literature claim map | `docs/2026-11-16_feature_literature_claims.md` |
| Decisions | `docs/DECISION_LOG.md` D-157 to D-163 |
