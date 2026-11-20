# Study guide: the 80 features, what they mean, what the literature says, what our data shows

For KN, 2026-11-16. Covers Chloe and Desmond's 11.15 feature set (the one in the harness).

**Where every claim comes from**
- **Literature**: only sources marked VERIFIED in `docs/REFERENCES.md`, cited as R-id + page. The full claim
  table with quotes and statuses is `docs/2026-11-16_feature_literature_claims.md` (literature-scout). Anything
  without a verified source is written as "we assume".
- **Our numbers**: `reports/data/2026-11-16_audit/r6_feature_cards.md` (generated from `r6_feature_profile.csv`
  and `r6_decile_default_rates.csv` by `src/audit/2026-11-16_feature_pipeline_audit.py`).
  - Population: 835,016 labelled firm-years, FY2007-2011, 12-month s5 label (D-151), average default rate 1.36%.
  - Features are those built by the frozen round-03 pipeline.
- **Audit caveats**: `reports/data/2026-11-16_audit/2026-11-16_feature_pipeline_audit_report.md` (§ refs below).

**How to read the numbers**
- **"AUC (as risk score)"** is the univariate AUC of the feature used on its own. 0.50 = useless; 0.76 = the best
  single feature we have. The guide says which direction is risky.
- **"Decile default rates"** sort firms into 10 equal groups by the feature (low → high) and give each group's
  12-month default rate in %. Fewer than 10 groups means many tied values (e.g. lots of exact zeros). This is
  the empirical shape: compare it with the shape the literature predicts.
- **Caveat on sources**: none of them studies our population (Italian private firms > €1.5M, all sectors,
  missed-payment label):
  - R-05 exhibits are US *public* firms at a 5-year horizon;
  - R-33 is France;
  - R-61 is UK SMEs, mostly tiny;
  - R-60 is Italian *manufacturing* (the closest);
  - R-66 is 115 Colombian retail bankruptcies (weak).

---

## 1. The big picture in one table

A credit officer asks four questions. Our strongest features answer them in this order:

| Question | Best feature in our data | AUC | What the literature calls it |
|---|---|---|---|
| Does the business generate cash relative to what it owes? | `operating_cash_flow_to_debt`, `ebitda_to_debt` | 0.764, 0.755 | Debt coverage: France's largest-weight input, 19.2% (R-33 Table 4 p. 14); Beaver's best single ratio (SECONDARY via R-32 pdf p. 9) |
| Is it profitable? | `profit` (sign), `roa`, `op_margin` | 0.752, 0.733, 0.702 | Profitability, "steeply convex" (R-01 p. 14) |
| How big is the owners' cushion? | `equity_ratio` | 0.740 | Leverage (R-01 p. 13; R-33 p. 12) |
| Can it pay its interest? | `cash_interest_coverage`, `int_coverage`, `financing_expense_ratio` | 0.736, 0.728, 0.627 | Coverage (R-01 App. p. 34); interest burden is the sharpest separator in Italy (R-60 pp. 385-386) |

Liquidity (cash, current ratio) comes next (AUC 0.56-0.61). Industry, legal form and region are weak on their own
(≤ 0.56). This ranking matches RiskCalc's message that profitability, leverage, coverage and liquidity are the core
families (R-01 p. 13), with one difference: our best liquidity measure is weaker than R-05 found for US private
firms (R-05 p. 36 calls cash/assets the single most important variable there; here `cash_ratio` has AUC 0.565).

**The shape lesson.** Almost every strong feature has the same curve. Risk is very high in the worst decile and
drops sharply, then the curve is flat. Example: `roa` deciles are 6.0%, 1.5%, 1.1%, 0.9% … 0.5%, so the worst
tenth of firms default at 4× the second tenth. This is the "convex, diminishing sensitivity" shape RiskCalc
documents (R-01 p. 14, Fig. 2). It is why RiskCalc transforms each ratio non-parametrically, and why a tree model
or a binned scorecard beats a linear logit on raw ratios.

---

## 2. Size

| | |
|---|---|
| **Formula** | `size = ln(asst_tot)` |
| **Meaning** | A bigger balance sheet means more diversification, deeper management and better funding access. |
| **Our data** | Median 15.1 (≈ €3.6M); p99 18.5 (≈ €110M). Deciles: 1.6 1.5 1.5 1.4 1.3 1.3 1.3 1.3 1.3 1.1. Gently safer with size; AUC only 0.527. |
| **Literature** | Larger firms default less (R-01 p. 13; R-33 p. 12, Fig. 5), but "adding a cash-flow ratio reduced the size advantage" (R-01 p. 13). **Italian manufacturing: size has a *positive* effect once financials are controlled, in every year** (R-60 p. 393, Table 3 p. 392). UK SMEs: hump-shaped, peaking near £350k of assets (R-61 pp. 19-23). Our firms are all > €1.5M, i.e. on the falling side of the hump. |
| **Takeaway** | Weak on its own. Do not force a monotone "bigger = safer" constraint: the Italian evidence and our own multivariate sign (D-022) say it can flip once profitability is in. |

**Raw euro amounts** (`profit`, `ebitda`, `prof_operations`, `rev_operating`, `COGS`, `exp_financing`) mix size with the
underlying quantity. `profit` is strong (AUC 0.752) mainly through its **sign**: 34.6% of firm-years make a loss.
`rev_operating` (0.513) and `COGS` (0.531) are mostly size. R-05 p. 35 shows sales and assets are nearly
interchangeable as size proxies. We have no verified source for feeding raw amounts to a model; Roger's feedback
asked for ratios (FEATURE_SPEC, M9). Expect questions on why raw amounts are in the model.

---

## 3. Leverage and capital structure

### `equity_ratio` = eqty_tot / asst_tot (AUC 0.740, higher = safer): the single best leverage measure
- **Meaning**: how much of the asset value can be lost before creditors take a loss.
- **Our data**: median 0.187. 3.7% of firm-years have negative equity. Deciles: **5.4** 1.8 1.5 1.2 1.0 0.7 0.6 0.4 0.4 0.5.
  The worst decile is 3× the next.
- **Literature**: equity/assets is RiskCalc France's leverage input, slopes down and is steepest where equity is thin
  (R-33 p. 12, Fig. 5). Negative-net-worth private firms default at about 3× the average (R-05 p. 47); here negative
  equity defaults at 11.1% vs 1.36% (`r7_segment_calibration.csv`), about 8×. It is the book-equity version of
  Altman's X4 (R-32 pdf pp. 11-12).
- **Verdict**: clean, monotone, constraint −1 agrees. Lead with it in explanations.

### `debt_to_eqty` = (debt_st + debt_lt) / eqty_tot (AUC 0.531)
- **Deciles**: **4.5** 0.4 0.4 0.5 0.7 0.9 1.1 1.4 1.7 2.0. The first decile is the negative-equity firms (ratio < 0).
  After that, more debt per euro of equity means more risk. A textbook **sign flip**.
- **Literature**: RiskCalc dropped debt/net worth for exactly this reason: net worth can be negative, so the
  relationship is "knife-edged" (R-05 p. 34).
- **Verdict**: correctly left unconstrained. Explain leverage with `equity_ratio` instead.

### `lt_debt_to_capital` = debt_lt / (debt_lt + eqty_tot) (AUC 0.516): **audit §B1**
- 27.7% of rows are exactly 0 (no long-term debt). Deciles (9 groups): **2.0** 0.4 0.5 0.6 0.8 1.0 1.1 1.4 2.5. U-shaped.
- **Literature**: a RiskCalc v3.1 leverage input, "LTD to (LTD plus net worth)" (R-01 App. p. 34), with + sign.
- **Problem**: when equity is very negative, the denominator goes below 0 and the ratio turns negative. 11,636 rows
  default at 18.3%, but the model's +1 constraint treats low values as safe. The model's level for these firms is
  fine (PD/DR 1.12) because other features compensate, but per-firm explanations get the wrong sign. Fix proposed in D-154.

### `st_debt_share` = debt_st / total debt (AUC 0.565, higher = riskier)
- **Deciles**: 0.6 0.8 1.1 1.4 1.6 1.7 1.7 1.6 1.8. Rises, then plateaus (a quarter of firms have only short-term debt).
- **Literature**: weak support only. The ST/LT mix is "of little use" (R-05 p. 36). The refinancing-risk story is
  our assumption.

### `bank_debt_conc` = bank debt / total debt (AUC 0.506)
- Flat (1.1-1.7% in every decile). 14% of rows have the debt breakdown imputed by sector medians (audit §R2), which
  blurs it further.
- **Literature**: no verified sign. Context only: Italian firms typically borrow from 6-8 banks (R-60 p. 378,
  citing others).
- **Verdict**: a candidate to drop.

### `cur_asst_ratio` = asst_current / asst_tot (AUC 0.593, higher = riskier) and `tangible_asset_ratio` = asst_tang_fixed / asst_tot (AUC 0.616, higher = safer)
- **Deciles**:
  - `cur_asst_ratio`: 0.6 … 2.0, rising;
  - `tangible_asset_ratio`: 2.0 2.2 1.9 … 0.7 0.4, falling.
- These are near-complements: both share total assets, and current plus fixed assets is almost all of it. So "more
  tangible" and "less current" are the same signal.
- **Literature**:
  - Tangibility raises what creditors can recover and so supports borrowing (R-48 Sec. 1 p. 1). That is a
    recovery/credit-access mechanism, so "tangible assets lower PD" is our assumption.
  - Altman saw current assets shrink relative to total assets in loss-making firms (R-32 pdf p. 10). That
    predicts the *opposite* sign to our data for `cur_asst_ratio`.
  - We read our sign as a sector effect: trading and construction firms are current-asset heavy, real-estate and
    manufacturing firms are fixed-asset heavy. This is an assumption.
- **Verdict**: fine for prediction. Do not claim both directions from the literature (claims file, Contradiction 11).

### `goodwill_ratio` = goodwill / asst_tot (AUC 0.638, higher = safer): **audit §B5**
- **Deciles**: **4.1** 1.3 1.0 1.2 1.3 1.3 1.0 0.9 0.8 0.8.
- The field is almost certainly **not** accounting goodwill:
  - 10.1% of values are negative;
  - it exceeds intangible assets in 84% of rows;
  - it is negative only when operating profit is negative (team doc 01).
- The team's working hypothesis is **value added** (D-010, D-091). Under that reading the closest verified evidence
  is labour productivity: defaulters have lower value added per employee, and productivity lowers PD in Italian
  manufacturing (R-60 p. 384, p. 393).
- **Verdict**: present it as "value added / assets (provider field labelled goodwill)". Never say goodwill.

---

## 4. Liquidity and activity

### `cash_ratio` = cash / asst_tot (AUC 0.565, higher = safer)
- **Deciles**: 1.7 1.7 1.7 1.7 1.5 1.4 1.2 1.0 0.8 0.9. Flat for the bottom half (half of firms hold < 1.2% of
  assets in cash), then falling.
- **Literature**: the most important single variable for private firms, falling across the whole range (R-05 p. 36,
  Exh. 5.4 p. 51). Slopes down in France (R-33 p. 12) and for UK SMEs (failed mean 0.125 vs 0.265; R-61 Table 7 p. 24).
  Only one weak source shows a U-shape, i.e. very high cash being risky (R-66, contradicted by R-05).
- **Our data vs the literature**: same direction, much weaker. The bottom half is flat, perhaps because Italian
  firms run on bank credit lines rather than cash balances. That reading is an assumption.

### `wc_ratio` = (current assets − short-term debt) / assets (AUC 0.614, higher = safer)
- **Deciles**: **3.3** 1.8 1.4 1.3 1.1 1.0 0.9 0.7 1.0 1.1. Steep at the negative end (30% of firms have negative
  working capital); slight uptick at the top.
- **Literature**: Altman's X1, the best of three liquidity ratios in 1968 (bankrupt mean −6.1% vs 41.4%; R-32 pdf
  p. 10, Table 1 p. 13).

### `cur_debt_ratio` = current assets / short-term debt (current ratio, AUC 0.613) and `quick_ratio` = (cash + AR) / short-term debt (AUC 0.587)
- **Deciles**:
  - current ratio: 1.5 **2.7** 2.2 1.8 1.2 1.0 0.8 0.8 0.8 0.8;
  - quick ratio: 1.3 1.8 **2.1 2.1** 1.6 1.3 1.2 0.7 0.6 0.8.
- **Hump-shaped**: the riskiest firms are not the ones with the lowest ratio.
- **Literature agrees this can happen**:
  - current and quick ratios are "subject to perverse trends for some failing firms" (R-32 pdf p. 10);
  - in failing UK small firms the current ratio *improves* because receivables build up (R-61 pp. 26-27);
  - R-66 finds a concave hump.
- **Two causes in our data**:
  - real: a distressed firm can't collect, so receivables pile up in current assets;
  - mechanical (audit §B3): when short-term debt is 0 the pipeline divides by 1, so the ratio becomes current
    assets in euros.
- **Formula note**: our quick ratio is (cash + receivables) / short-term debt. R-05 defines it as (current assets −
  inventory) / current liabilities. State this on the slide.

### `AR_ratio` = AR / revenue (receivable days / 365; AUC 0.582, higher = riskier)
- **Deciles**: 1.1 1.0 1.0 1.0 1.0 1.2 1.5 2.0 1.9 2.1. Flat until the median (≈ 152 days of sales outstanding),
  then rising.
- **Literature**: + for UK SMEs. Distressed small firms extend credit to keep sales, or are paid late (R-61 Table 4
  p. 13, p. 27). R-05 shows a non-monotone curve for US public firms (Exh. 4.7; our chart reading, flagged).
- **Audit**: zero revenue gives AR in euros (§B3); negative revenue flips it to the "safe" end (§B2).

### `payables_ratio` = AP_st / asst_tot (AUC 0.595, higher = riskier)
- **Deciles**: 0.8 0.8 1.0 1.1 1.3 1.3 1.5 1.6 1.6 2.5. Clean and monotone.
- **Literature**: stretching suppliers is a classic stress signal. Payables/sales slopes up in France (R-33 p. 12);
  trade creditors raise risk for UK SMEs (R-61 Table 4 p. 13); it is a UK RiskCalc input (R-01 App. p. 34). We
  scale by assets, not sales.

### `cash_to_financing_exp_ratio` = cash / interest bill (AUC 0.592, higher = safer)
- **Deciles**: 2.0 1.9 1.9 1.7 1.4 1.2 1.0 0.8 0.8 1.0. No verified source; "we assume" wording.

### `asset_turnover` = revenue / assets (AUC 0.506)
- **Deciles**: 1.8 **0.8** 1.0 1.5 1.8 1.7 1.5 1.3 1.1 1.2. Non-monotone. The first decile is mostly zero-revenue
  firms (3.1% of rows), likely holding and real-estate companies.
- **Literature**: "non-monotonic and very flat" (R-05 p. 37). Dropped from Altman's Z'' because turnover varies so
  much by industry (R-32 pdf pp. 12-13, 26).
- **Verdict**: the data match the literature exactly. Keep it unconstrained, or drop it.

---

## 5. Profitability

### `roa` (= 100 × operating profit / assets; AUC 0.733, higher = safer)
- **Deciles**: **6.0** 1.5 1.1 0.9 0.8 0.8 0.7 0.7 0.5 0.5. The classic convex shape: almost all the information
  separates loss-makers from the rest.
- **Literature**:
  - PD falls with return on assets with diminishing sensitivity (R-01 p. 14, Fig. 2; R-01 uses net income).
  - Altman's X3 (EBIT/TA), "the true productivity of the assets", is our closest match: bankrupt mean −31.8% vs 15.4%
    (R-32 pdf p. 11, Table 1 p. 13).
  - Return on assets dominates other profit measures (R-05 pp. 32-33).

### `op_margin` = operating profit / revenue (AUC 0.702) and `net_prof_margin` = profit / revenue (AUC 0.713)
- **Deciles**:
  - `op_margin`: 4.2 3.0 1.4 0.9 0.9 0.8 0.6 0.5 0.5 **0.7**;
  - `net_prof_margin`: 4.3 2.7 1.6 1.4 0.9 0.7 0.5 0.4 0.3 **0.8**.
- The top-decile uptick is mostly zero-revenue firms whose "margin" becomes profit in euros (audit §B3).
- **Literature**: the operating-margin transform slopes down and flattens at high margins in France (R-33 p. 12).
  Defaulters have clearly lower return on sales in Italian manufacturing (R-60 pp. 383-384, p. 393). Margins are
  weaker than return on assets because they vary by industry (R-05 p. 33).

### `roe` = profit / equity × 100 (provider field; AUC 0.551): **audit §B4**
- **Deciles**: 2.2 1.1 0.8 1.2 0.7 **3.7** 1.9 0.5 0.5 0.9. Erratic. The 3.7% spike is the decile containing the
  imputed median value.
- **Why it is erratic**:
  - the provider blanks ROE whenever |ROE| > 150, so blanks hold the extreme and negative-equity firms (default rate
    8.2% vs 0.86%), and we fill them with a median;
  - a high ROE can mean tiny or negative equity, not high profit.
- **Literature**: return on equity turns "perverse" above about its 60th percentile, i.e. higher ROE, higher default,
  driven by low or negative book equity (R-05 p. 33).
- **Verdict**: add a missing flag or drop it. It should not appear in an explanation.

### `loss_flag` = profit < 0 (AUC 0.675)
- 34.6% of firm-years are loss-making. They default at **2.71% vs 0.64%**, 4.2× the rate of profitable firms.
- **Literature**: the direction follows from the profitability family (R-01 p. 13). Ohlson's loss dummy is
  [UNVERIFIED] (R-12 not read), so don't cite Ohlson.

---

## 6. Coverage and cash flow (the strongest family)

`cf_operations` = net profit + depreciation (an accounting proxy for operating cash flow; identity holds for 99.86%
of rows, audit R1).

| Feature | AUC | Deciles (low → high) |
|---|---|---|
| `operating_cash_flow_to_debt` = cf / total debt | **0.764** | **6.2** 1.5 1.5 1.3 0.9 0.7 0.5 0.4 0.3 0.4 |
| `operating_cash_flow_ratio` = cf / assets | 0.759 | **6.4** 1.3 1.3 1.1 0.9 0.7 0.5 0.5 0.4 0.5 |
| `ebitda_to_debt` = EBITDA / total debt | 0.755 | **5.8** 1.8 1.4 1.2 1.0 0.8 0.6 0.4 0.3 0.3 |
| `cash_interest_coverage` = cf / interest | 0.736 | **5.5** 2.2 1.4 1.1 0.8 0.6 0.4 0.4 0.4 0.6 |
| `int_coverage` = EBITDA / interest | 0.728 | **4.9** 2.8 1.3 1.2 0.9 0.6 0.5 0.4 0.5 0.6 |
| `operating_cash_flow_margin` = cf / revenue | 0.726 | **4.7** 2.8 1.6 1.1 0.8 0.6 0.5 0.4 0.4 0.7 |
| `financing_expense_ratio` = interest / assets | 0.627 | 1.1 0.9 0.8 0.7 0.9 1.0 1.2 1.4 1.9 **3.5** (higher = riskier) |

- **Meaning**: can the business service its debt from what it earns? This is the bank's core question.
- **Literature**:
  - **Debt coverage** ((profit + D&A) / net liabilities) is RiskCalc France's largest-weight input at 19.2%; it slopes
    down and flattens (R-33 Table 4 p. 14, p. 12). RiskCalc v3.1 added cash-flow coverage ratios in every region
    (R-01 App. p. 34), and "adding a cash-flow ratio improved the model" (R-01 p. 13).
  - **Interest burden** is the financial variable that most sharply separates Italian defaulters, significant in
    every year (R-60 pp. 385-386, 391). Interest expense/sales slopes up in France (R-33 p. 12). Our
    `financing_expense_ratio` (scaled by assets) shows the same rising curve, steepest in the top decile (3.5%).
  - **Interest coverage** is non-monotone at the very bottom, because tiny interest bills with slightly negative EBIT
    give extreme values, and it is weaker for private firms (R-05 p. 34). Our `int_coverage` top decile (0.6%, above
    0.4%) is the mirror of that effect: firms with near-zero interest get huge coverage values.
- **Sources disagree on cash flow vs earnings**: RiskCalc v1 found cash flow no better than net income (R-05 p. 33),
  and Altman found EBIT/assets beat cash flow (R-32 pdf p. 11). In *our* data the cash-flow versions rank first
  (0.764 vs `roa` 0.733).
- **Stability**: these AUCs hold in every year 2007-2011 (e.g. `operating_cash_flow_to_debt` 0.741-0.776 as a risk
  score; `audit_log.md`). They are not a crisis-year artefact.

---

## 7. Flags

| Flag | Share = 1 | Default rate if 1 vs 0 | What the literature says |
|---|---|---|---|
| `loss_flag` | 34.6% | 2.71% vs 0.64% | see §5 |
| `debt_detail_missing` | 14.1% | 1.47% vs 1.34% | Legal basis: an abbreviated balance sheet need not split payables into banks/suppliers (R-27 art. 2435-bis; R-67 art. 2424), so a missing split may mark a short-form filer. That this causes our gaps is a hypothesis. UK abbreviated-accounts firms remain predictable (AUC 0.75; R-61 Table 9 p. 28). |
| `neg_pl_flag` | 1.0% | 2.54% vs 1.35% | No source for the sign. |
| `neg_bs_flag` | 0.5% | 1.54% vs 1.36% | Moody's keeps flagged statements; the model was better with them (R-01 pp. 21-22). Our signal is tiny. |
| `zero_*` (16 flags) | mostly < 3.5% | e.g. zero EBITDA 3.43%, zero revenue 1.89%, zero total debt **0.00%** (233 rows), zero cash 1.55% | Zeros mean different things by sector, e.g. > 40% of service firms carry no inventory (R-01 p. 18). Near-zero denominators create misleading extremes (R-05 pp. 33-34). |

Correction to team docs: `neg_bs_flag` "5.51%" and `debt_detail_missing` "5.29%" in docs 01/04 are *ever-defaulted*
rates. The 12-month label rates are 1.54% and 1.47% (audit §B6).

## 8. Categoricals

| Feature | Our data | Literature |
|---|---|---|
| Industry (ATECO section, target-encoded) | 19 sections in the frozen mapping (values 0.6%-5.1%, section U highest); ties leave 6 decile groups with default rates 0.8-1.8%; AUC 0.562 | Industry controls add little ranking power (accuracy ratio 54.4% → 55.1%) but much calibration (R-01 pp. 17-18). < 5% of sector dummies significant in Italian manufacturing (R-60 p. 391). Target encoding needs shrinkage plus cross-fitting (R-47). **Audit §A1: the validation/test files leaked the test year's industry default rates into this feature.** |
| Legal form | SRL 72%, SRU 14.7% (1.97% vs 1.25%), SPA 10.5% (0.94%), SAU 2.5% (0.90%) | Definitions from Roger's handout (unofficial): SRS = simplified SRL, SRU probably single-shareholder SRL (R-70). Sign = assumption. Single-owner SRU is riskier and SPA/SAU (larger, more capital) safer, plausibly via size. |
| Region (20 dummies) | All AUC within 0.483-0.510. Campania 1.76%, Toscana 1.73% vs Lombardia 1.18%, Trentino-Alto Adige 0.82% | South vs Centre-North gap in bad-loan rates peaked at 4.8 points at end-2012, fell to about 1 by 2018 (R-69). The gap moves with the cycle, so region is not a stable predictor. Moody's Italy model uses *national*, not regional, cycle data (R-04 slide 13). |

---

## 9. Where our data and the literature disagree (deck-ready)

1. **Size**: RiskCalc says bigger is safer. Italy (R-60) and our multivariate model say + once financials are in.
   Present it as "the size benefit is mostly better financials" (R-01 p. 13).
2. **Cash** is far weaker here (AUC 0.565) than R-05 found for US private firms. The bottom half is flat.
3. **Current and quick ratios are humped**, not monotone. Supported by R-32, R-61 and R-66, so a forced −1
   constraint is debatable.
4. **Equity sign flips** in `debt_to_eqty`, `roe` and `lt_debt_to_capital` (R-05 pp. 33-34). Use `equity_ratio` plus a
   negative-equity flag in explanations.
5. **"Goodwill"** is not goodwill. Treat it as value added (D-091).
6. **Cash flow beats earnings** in our data. The literature is split.
7. **Growth is missing**. RiskCalc treats growth as U-shaped (R-01 p. 13; R-33 p. 12); our features are levels only (D-020).

---

## 10. Worked example (runnable)

A made-up firm, "Beta S.r.l.". The script computes its key ratios with the pipeline's formulas, then looks up each
ratio's decile default rate in our data. This is the univariate view a credit officer can follow; the model then
combines them. Run from the repo root with any Python that has pandas:

```python
import pandas as pd

# Beta S.r.l., FY2010 statement (EUR); made-up numbers
firm = dict(asst_tot=4_000_000, eqty_tot=200_000, debt_st=1_900_000, debt_lt=1_200_000,
            asst_current=2_300_000, AR=1_300_000, cash_and_equiv=30_000, AP_st=900_000,
            rev_operating=3_100_000, prof_operations=-60_000, profit=-150_000,
            ebitda=90_000, exp_financing=140_000, cf_operations=-10_000)
f = firm
debt = f["debt_st"] + f["debt_lt"]
ratios = {
    "equity_ratio": f["eqty_tot"] / f["asst_tot"],
    "operating_cash_flow_to_debt": f["cf_operations"] / debt,
    "ebitda_to_debt": f["ebitda"] / debt,
    "roa": 100 * f["prof_operations"] / f["asst_tot"],
    "int_coverage": f["ebitda"] / f["exp_financing"],
    "financing_expense_ratio": f["exp_financing"] / f["asst_tot"],
    "cash_ratio": f["cash_and_equiv"] / f["asst_tot"],
    "AR_ratio": f["AR"] / f["rev_operating"],
    "payables_ratio": f["AP_st"] / f["asst_tot"],
}
dec = pd.read_csv("reports/data/2026-11-16_audit/r6_decile_default_rates.csv")
avg = 0.0136  # 12-month default rate, all labelled firm-years
print(f"{'feature':30s}{'value':>10s}  decile  default rate in that decile  vs average")
for name, x in ratios.items():
    d = dec[dec.feature == name].reset_index(drop=True)
    i = int(d.x_max.searchsorted(x)) if x <= d.x_max.iloc[-1] else len(d) - 1
    print(f"{name:30s}{x:10.3f}  {i + 1:>4d}    {d.DR[i]:.2%}                      {d.DR[i] / avg:4.1f}x")
```

Output (run 2026-11-16):

```
feature                            value  decile  default rate in that decile  vs average
equity_ratio                       0.050     3    1.50%                       1.1x
operating_cash_flow_to_debt       -0.003     2    1.52%                       1.1x
ebitda_to_debt                     0.029     4    1.16%                       0.9x
roa                               -1.500     2    1.49%                       1.1x
int_coverage                       0.643     2    2.81%                       2.1x
financing_expense_ratio            0.035    10    3.55%                       2.6x
cash_ratio                         0.007     5    1.48%                       1.1x
AR_ratio                           0.419     6    1.18%                       0.9x
payables_ratio                     0.225     7    1.51%                       1.1x
```

**What this teaches.** Beta looks weak: 5% equity, a small operating loss, roughly zero cash flow. Yet it lands in
deciles 2-4, not decile 1. The 5-6% "worst decile" holds far more extreme firms: negative equity, deep losses
(`equity_ratio` decile 1 is below 1.2%; `roa` decile 1 is below −2.0). Many Italian private firms run thin, so "thin" is normal in this
population. Beta's real red flags are the **interest burden** (top decile, 3.55%, 2.6× average) and
**EBITDA that covers interest only 0.6 times** (2.1×). That matches the Italian evidence that interest burden
separates defaulters most sharply (R-60 pp. 385-386). The model combines these lines, so Beta's PD will sit
well above average. Run the harness on a one-row CSV with Beta's raw fields to see the actual number.

## 11. Self-test

1. Why does `debt_to_eqty` have a 4.5% default rate in its *lowest* decile? (Sign flip on negative equity, R-05 p. 34.)
2. Name the two families our data ranks first, and the verified source that weights the first one most. (Debt and
   cash-flow coverage, profitability; R-33 Table 4 p. 14.)
3. Why could forcing "higher current ratio = safer" be wrong? (Receivables pile up in distress: R-32 pdf p. 10,
   R-61 pp. 26-27; plus our divide-by-1 rule.)
4. What is the `goodwill` field, probably, and what can we cite? (Value added; R-60 on productivity.)
5. Why is size weak, and what does the Italian evidence add? (R-01 p. 13: the size benefit is absorbed by cash flow;
   R-60 p. 393: + conditional effect.)

Human verification pending: KN to spot-check five citations against the PDFs (start with R-60 p. 393, R-33 p. 12,
R-05 p. 34) and run the example.
