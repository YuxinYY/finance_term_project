# Sign table and the monotone vs unconstrained decision (75-feature LightGBM)

Prepared 2026-11-21 by Claude Code for KN. Inputs: `experiments/gbm/2026-11-20_grid75/` (grid, sign table,
evidence and ablation scripts and CSVs), `docs/2026-11-16_feature_literature_claims.md` (literature-scout,
page-level claim map), `docs/MONOTONE_SIGN_AUDIT_2026-11-03.md`, `docs/REFERENCES.md`.
**Every source cited below is VERIFIED in REFERENCES.md; page numbers are copied from the claim map, which
checked them against the text. Open the page yourself before putting a citation on a slide.**

Data evidence uses training rows only: the sign table rule used FY2007 (round-1 train); the stability and
model-curve evidence below uses FY2007-2009 (round-3 train). No validation, test or lockbox year is used to
describe signs.

---

## 0. One-page summary

1. **What a monotone constraint says.** "Holding every other input fixed, a higher value of this ratio can
   never lower (or never raise) the PD." It is a statement about the *conditional* effect, not the
   univariate default-rate curve.
2. **The accuracy gap is small.** Unconstrained minus monotone, same winner setting: +0.0008 AUC
   (3 seeds, full binning; free ahead in 9/9 runs), log loss -0.00011 (about 0.2%). Bootstrap on one seed:
   +0.0013 [+0.0004, +0.0023], driven by round 3 (FY2010).
3. **Univariately every constrained ratio is cleanly monotone in every training year** (Section 2,
   columns rho 2007/2008/2009). The unconstrained model breaks the sign only **conditionally**, and mostly
   in the **top 10% tail**.
4. **The conflicts have three different causes** (Section 3):
   - **Type 1, twin features (conditional meaning changes).** quick_ratio given cash_ratio and the current
     ratio is a receivables measure; operating_cash_flow_margin given net_prof_margin is a D&A-intensity
     measure. Both rise with risk, which matches verified sources for receivables (R-61).
   - **Type 2, construction artefacts.** The zero-denominator rule (0 becomes 1, so the ratio equals the
     numerator in euros) and negative denominators (negative revenue, negative equity) put unusual firms in
     the extreme tails. Example: firms with negative revenue (0.2% of rows, default rate 6.6%) sit at the
     99th percentile of the margin ratios, where the economic sign says "safe".
   - **Type 3, non-monotone shapes documented in the literature.** Interest coverage at the bottom (R-05
     p. 34), current/quick ratios in distress (R-32 pdf p. 10; R-61 pp. 26-27), receivables (R-05 Exh. 4.7),
     size in Italy (R-60 p. 393).
5. **On real firms (SHAP), the unconstrained model's #2 and #3 drivers have unreliable reason codes**
   (operating_cash_flow_ratio: SHAP vs value +0.12, roa -0.46; monotone -0.90, -0.95). Summed over each family
   of near-twin features its economics are right (-0.83, -0.75): it ranks firms sensibly but splits the
   credit between twins (Section 3.5).
6. **The whole accuracy cost of the monotone model is one constraint, quick_ratio** (releasing it alone
   recovers +0.00074 of the +0.00075 gap). Constraints that are correct (equity_ratio, roa, wc_ratio,
   AR_ratio) slightly *help* (Section 4).
7. **Recommendation: full monotone (D-157, 20 constraints)** (revised; the earlier lit13 recommendation is
   withdrawn). The quick ratio, whose constraint is almost the whole accuracy cost, has direct verified
   support for "higher = lower PD" (R-01 p. 13; R-33 p. 11; R-05 p. 54; Section 3.1a). Cost vs
   unconstrained: about 0.0008 AUC (0.1%). Optional: free cur_debt_ratio (strongest documented caveats, no
   measurable cost). Section 5.

---

## 1. How to read the tables

- **Expected sign**: + means a higher value should raise PD; - means it should lower PD.
- **Status** (from the claim map): SUPPORTED = checked against a verified source's text;
  SUPPORTED (weak) = verified but small or distant sample; ASSUMPTION = team reasoning, must be worded
  "we assume".
- **Documented non-monotone?** = a verified source reports a shape that is not monotone for this ratio or a
  close variant. This is the main reason *not* to constrain, regardless of our data.
- **rho 2007 / 2008 / 2009** = Spearman correlation between decile index and decile default rate in each
  training year. Close to -1 or +1 = clean monotone univariate curve.
- **Free model, % against** = share of the unconstrained round-3 model's partial-dependence movement that goes
  against the expected sign (5%-95% of the training distribution). 0 = obeys the sign everywhere.
- **Gain rank** = importance rank in the unconstrained round-3 model (75 features).
- **D-157 constraint** = what the grid's monotone model used (constrain only if a sign exists AND FY2007 data
  agree with |rho| >= 0.7).

Population warning for every citation: none of the sources studied our population (Italian private firms
above EUR 1.5m assets, all sectors, missed-payment label). R-05 exhibits are US public firms at 5 years,
R-33 is France, R-61 is UK insolvency of mostly very small firms, R-60 is Italian manufacturing at one bank,
R-66 is 115 Colombian retail bankruptcies (claim map, "Who the sources studied").

---

## 2. The sign table: 27 features with a candidate sign

### 2a. Leverage and capital structure

| feature | formula | what it measures for a credit officer | exp. sign | verified source (page) | status | documented non-monotone? | rho 07/08/09 | free % against (gain rank) | D-157 |
|---|---|---|---|---|---|---|---|---|---|
| equity_ratio | eqty_tot / asst_tot | Owners' cushion: how much asset value can be lost before creditors are hit | - | R-33 Table 3 p. 11, p. 12, Fig. 5 p. 13 (equity/assets slopes down, convex); R-01 p. 13 (leverage group); R-05 p. 47 (negative net worth ~3x average default rate) | SUPPORTED | No | -0.98 / -0.96 / -0.95 | 1.4% (1) | **-** |
| lt_debt_to_capital | debt_lt / (debt_lt + eqty_tot) | Long-term gearing of permanent capital | + | R-01 Appendix p. 34 (LTD to LTD + net worth is a v3.1 input), p. 13 | SUPPORTED | **Yes, by construction**: denominator <= 0 when equity is negative (claim map; same issue as R-05 p. 34) | +0.35 / +0.25 / +0.25 | 46% (30) | 0 |
| st_debt_share | debt_st / (debt_st + debt_lt) | Maturity mix: how much must be repaid or rolled within a year | + | R-05 p. 36 + fn. 31: ST/LT mix "of little use" | SUPPORTED (weak) for "no reliable sign"; + is ASSUMPTION | Weak/unclear | +0.13 / +0.31 / +0.21 | 9% (11) | 0 |
| tangible_asset_ratio | asst_tang_fixed / asst_tot | Pledgeable, saleable collateral | - | R-48 Sec. 1 p. 1, fn. 2: tangibility raises what creditors recover (credit-access mechanism) | Mechanism SUPPORTED; "lowers PD" = ASSUMPTION (REFERENCES Contradiction 11) | No | -0.95 / -0.93 / -0.96 | 2.1% (5) | **-** |
| goodwill_ratio | goodwill / asst_tot | **Field probably holds value added, not goodwill** (D-010, D-091) | + if goodwill; - if value added | R-60 p. 384 Fig. 4, p. 393 (value added per employee lower for defaulters) under the D-091 reading; R-05 p. 34 (intangibles "appear really to be worth something") | SUPPORTED only under D-091 | Unclear | -0.89 / -0.39 / -0.79 (data say -) | 37% vs + (17) | 0 |

### 2b. Liquidity and activity

| feature | formula | what it measures | exp. sign | verified source (page) | status | documented non-monotone? | rho 07/08/09 | free % against (gain rank) | D-157 |
|---|---|---|---|---|---|---|---|---|---|
| cash_ratio | cash / asst_tot | Cash on hand for the next payments | - | R-05 p. 36, p. 51 Exh. 5.4, p. 61 (most important single variable for private firms); R-33 Table 3 p. 11, p. 12; R-61 Table 3 p. 12, Table 7 p. 24 | SUPPORTED | Only R-66 Fig. 3 p. 10 (U-shape, weak; R-05 private curve contradicts) | -0.96 / -0.90 / -0.83 | 13% (26) | **-** |
| cash_to_financing_exp_ratio | cash / exp_financing | Months of interest that cash could pay | - | none found | ASSUMPTION | Zero-denominator artefact (3.2% of rows have zero interest; ratio = cash in EUR, at the 98th pct) | -0.95 / -0.95 / -0.95 | 10% (25) | **-** |
| wc_ratio | wc_net / asst_tot | Current assets left after short-term liabilities | - | R-32 pdf p. 10, Table 1 pdf p. 13 (Altman X1; bankrupt mean -6.1% vs 41.4%); R-05 p. 36; R-66 Table 3 p. 9 | SUPPORTED | No | -0.71 / -0.84 / -0.79 | 36% (23) | **-** |
| cur_debt_ratio (current ratio) | asst_current / debt_st | Current assets per euro of short-term claims | - | R-01 p. 13, Appendix p. 34 | SUPPORTED | **Yes**: "subject to perverse trends for some failing firms" R-32 pdf p. 10; rises in failing small firms because receivables rise R-61 pp. 26-27; hump R-66 Fig. 3 p. 10 | -0.88 / -0.87 / -0.85 | 16% (27) | **-** |
| quick_ratio | (cash + AR) / debt_st | Liquid assets (ex inventory) per euro of short-term claims | - | R-05 p. 36, p. 50 Exh. 5.2, p. 61 (quick ratio a v1 input; curves fall) | SUPPORTED; **our formula differs** (R-05: current assets - inventory) | Caveat only: "perverse trends for some failing firms" (R-32 pdf p. 10); direct sources state "-" (R-01 p. 13; R-33 p. 11; R-05 p. 54); see 3.1a | -0.73 / -0.70 / -0.81 | **83%** (14) | **-** |
| AR_ratio | AR / rev_operating | Sales not yet collected: slow or doubtful customers | + | R-61 Table 4 p. 13, Table 8 p. 25, p. 27 (trade debtors positive; distressed small firms extend credit or are paid late) | SUPPORTED (UK) | **Yes**: AR/sales high at both ends for US public firms, R-05 p. 37 Exh. 4.7 (chart reading); "other activity ratios have different relationships", R-01 p. 13 | +0.83 / +0.95 / +0.81 | 0.2% (13) | **+** |
| payables_ratio | AP_st / asst_tot | Supplier credit used: stretching suppliers is a liquidity-stress symptom | + | R-33 Table 2 p. 11, p. 12, Fig. 5; R-61 Table 4 p. 13, p. 27; R-01 Appendix p. 34 | SUPPORTED (sources use sales/liabilities denominators) | No | +0.81 / +0.99 / +0.98 | 14% (6) | **+** |

### 2c. Profitability

| feature | formula | what it measures | exp. sign | verified source (page) | status | documented non-monotone? | rho 07/08/09 | free % against (gain rank) | D-157 |
|---|---|---|---|---|---|---|---|---|---|
| roa | 100 x prof_operations / asst_tot | Earning power of the assets before financing | - | R-01 p. 14, Fig. 2 + fn. 10 (convex, diminishing at the top); R-32 pdf p. 11, Table 1 (EBIT/TA; bankrupt mean -31.8% vs 15.4%); R-05 pp. 32-33 | SUPPORTED | No (convex, flattening) | -0.90 / -0.99 / -0.96 | 19% (3) | **-** |
| op_margin | prof_operations / rev_operating | Operating profit per euro of sales | - | R-33 p. 11-12, Fig. 5; R-60 pp. 383-384 Fig. 3, p. 393 (Italy) | SUPPORTED | No; but **negative-revenue firms land at the top** (Type 2) | -0.88 / -0.94 / -0.92 | 21% (33) | **-** |
| net_prof_margin | profit / rev_operating | Bottom-line profit per euro of sales | - | R-01 Appendix p. 34 (UK v3.1 input); R-66 Table 3 p. 9 | SUPPORTED | No; same negative-revenue issue | -0.88 / -0.92 / -0.87 | 37% (21) | **-** |
| loss_flag | profit < 0 | Firm is losing money this year | + | R-01 p. 13 (profitability lowers PD, so a loss raises it) | SUPPORTED for direction | No | DR 1.80% vs 0.48% (2007, z = +24.6) | 0% (63) | **+** |

### 2d. Coverage, interest burden, cash flow

`cf_operations` = net profit + D&A (accounting proxy, FEATURE_SPEC), not a cash-flow-statement figure.

| feature | formula | what it measures | exp. sign | verified source (page) | status | documented non-monotone? | rho 07/08/09 | free % against (gain rank) | D-157 |
|---|---|---|---|---|---|---|---|---|---|
| int_coverage | ebitda / exp_financing | Times operating cash earnings cover the interest bill | - | R-61 Table 3 p. 12 (EBITDA/interest negative); R-01 Appendix p. 34 | SUPPORTED | **Yes**: at very low coverage "default risk actually decreases" (tiny interest with slightly negative EBIT), R-05 p. 34 Exh. 4.4; weaker for private firms (unreliable interest figures), R-05 p. 34, p. 47 | -0.85 / -0.90 / -0.86 | **75%** (22) | **-** |
| cash_interest_coverage | cf_operations / exp_financing | Same, with cash earnings | - | R-01 Appendix p. 34, p. 13 (cash flow / interest is a v3.1 input) | SUPPORTED | Same structure as int_coverage (interest denominator); zero-interest artefact | -0.95 / -0.93 / -0.87 | 22% (31) | **-** |
| financing_expense_ratio | exp_financing / asst_tot | Interest burden = cost of debt x leverage | + | R-60 pp. 385-386, 391, Table 3 p. 392 (interest/sales separates Italian defaulters most sharply); R-33 p. 11-12, Fig. 5 | SUPPORTED (sales denominator in sources) | Flattening at top (R-33) | +0.93 / +0.70 / +0.66 | 10% (16) | **+** |
| ebitda_to_debt | ebitda / (debt_st + debt_lt) | Years of operating earnings to repay debt (inverted) | - | R-33 Table 3 p. 11, p. 12, Fig. 5, Table 4 p. 14 (debt coverage, largest weight 19.2%) | SUPPORTED (formula differs) | No (convex) | -1.00 / -1.00 / -1.00 | 12% (29) | **-** |
| operating_cash_flow_ratio | cf_operations / asst_tot | Cash earnings per euro of assets | - | R-01 Appendix p. 34, p. 13 (adding a cash-flow ratio improved v3.1); R-33 Table 3 p. 11 (profit + D&A numerator) | SUPPORTED | No | -0.94 / -0.94 / -0.91 | 26% (2) | **-** |
| operating_cash_flow_to_debt | cf_operations / (debt_st + debt_lt) | Cash earnings per euro of debt | - | R-33 Table 3 p. 11 (close variant); R-01 Appendix p. 34 | SUPPORTED | No | -0.98 / -0.93 / -0.95 | 10% (4) | **-** |
| operating_cash_flow_margin | cf_operations / rev_operating | Cash earnings per euro of sales | - | R-01 Appendix p. 34, p. 13 (family) | SUPPORTED (family) | No; negative-revenue issue | -0.87 / -0.93 / -0.88 | **71%** (32) | **-** |

Sources disagree on cash flow vs earnings: v3.1 adopted cash-flow coverage (R-01 Appendix p. 34); RiskCalc v1
found cash flow no better than net income (R-05 p. 33); Altman found EBIT/TA beat cash flow (R-32 pdf p. 11).

### 2e. Size, flags, industry

| feature | what it measures | exp. sign | verified source (page) | status | documented non-monotone? | data | free % against (gain rank) | D-157 |
|---|---|---|---|---|---|---|---|---|
| size = ln(asst_tot) | Diversification, management depth, funding access | - (univariate) | R-01 Sec. 3.1 p. 13 (large firms default less; effect weakens once cash flow is in); R-33 p. 12, Fig. 5 p. 13 | SUPPORTED univariate | **Yes**: **+ conditional on ratios in Italian manufacturing**, R-60 p. 393, Table 3 p. 392; hump in UK SMEs, R-61 pp. 19-20, 23 | rho -0.37 / -0.61 / -0.79 | 2% (28) | 0 |
| neg_bs_flag | Balance-sheet items reported negative | + | none for the sign | ASSUMPTION | n/a | DR 0.54% vs 0.88% (2007, z = -1.0) | n/a | 0 |
| debt_detail_missing | Bank/financial/trade debt split missing (possibly abbreviated statement) | + (weak) | R-27 art. 2435-bis + R-67 art. 2424 (abbreviated balance sheet need not show the split: legal possibility); R-61 p. 4, p. 23, Table 3 p. 12 (less disclosure predicts insolvency, UK) | SUPPORTED as legal possibility; link to our data = hypothesis | n/a | DR 0.85% vs 0.88% (2007, z = -0.3); D-114 obs/pred 1.04 | 0% (48) | 0 |
| ateco_sector_te | Sector default rate (target-encoded) | + by construction | R-01 Sec. 3.3 pp. 17-18, Table 2 (industry controls); R-47 Sec. 1.2, 2.7 (encoding needs shrinkage + cross-fitting) | Mechanical: the rising shape is imposed by how it is built (sign audit, tier M) | n/a | rho +0.59 / +0.94 / +0.64 | 19% (10) | 0 |

---

## 3. Anatomy of the conflicts

### 3.0 Why a ratio can be monotone univariately and reversed conditionally

A partial-dependence curve asks: take a firm, change ONLY this ratio, keep every other ratio fixed, how does
the PD move? When two features share a numerator or denominator, "changing only one" changes what the
change *means*. Runnable example (synthetic; the true risk driver is receivables, not liquidity):

```python
import numpy as np, lightgbm as lgb
rng = np.random.default_rng(0); n = 200_000
cash, ar, other_ca, debt_st = rng.lognormal(0, 1, (4, n))
quick   = (cash + ar) / debt_st                       # (cash + AR) / ST debt
current = (cash + ar + other_ca) / debt_st            # current ratio
cash_r  = cash / (cash + ar + other_ca)               # cash share
logit = -4.5 - 0.6 * np.log(current) - 0.8 * cash_r + 0.5 * np.log(ar / (cash + ar + other_ca))
y = rng.random(n) < 1 / (1 + np.exp(-logit))
print("univariate corr(quick, default):", np.corrcoef(quick, y)[0, 1].round(4))
X = np.column_stack([quick, current, cash_r])
m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, verbose=-1).fit(X, y)
base = X[:5000].copy()
for q in (10, 50, 90):
    Xq = base.copy(); Xq[:, 0] = np.percentile(quick, q)
    print(f"quick at p{q}: mean PD {m.predict_proba(Xq)[:, 1].mean():.4f}")
```

Output (run 2026-11-21, datascience env, LightGBM 4.7.0):

```
univariate corr(quick, default): -0.013
quick at p10: mean PD 0.0022
quick at p50: mean PD 0.0274
quick at p90: mean PD 0.1957
```

Univariately the quick ratio looks (slightly) safe; conditionally the PD rises with it. With the current
ratio and cash share held fixed, a higher quick ratio can only mean more receivables, and in this toy world
receivables are the risk. Our real data behave the same way (3.1), much more mildly.

Second lesson from the same output: the size of the rise (0.2% to 19.6%) is exaggerated because a partial-
dependence curve also evaluates **impossible firms** (a quick ratio above the firm's own current ratio).
Partial-dependence reversals on twin features partly measure behaviour where no real firm lives; SHAP on
real rows (planned for the final model) is the better per-firm check.

### 3.1a quick_ratio literature ledger (added 2026-11-21 after KN's challenge; supersedes the quick_ratio parts of 3.1, Section 5 and lit13)

Read directly in the local PDFs (R-05, R-01, R-33, R-66; R-05 printed page = pdf page) or from verified notes
with pages (R-32, R-61; PDFs not stored locally). R-60 (Italy) has no liquidity-ratio content.

**Direct statements of the sign (all say higher quick ratio = lower PD):**
- R-01 p. 13: "Liquidity variables include cash and marketable securities to assets, the current ratio, and the
  quick ratio ... High liquidity reduces the probability of default."
- R-33 p. 11, Table 2: same wording for France.
- R-05 p. 54: "the quick ratio was found to be strongly negatively related to default: the higher the quick
  ratio, the lower the default probability."
- R-05 p. 36 (Exh. 4.6): "Liquidity ratios help predict default; current and quick ratios dominate working
  capital"; quick ratio "slightly more powerful" than WC/assets. p. 61: quick ratio = RiskCalc v1 input, 7%.
- R-05 p. 50 Exh. 5.2: private-firm 5-year default curve by quick ratio (direction per verified notes; the
  chart line cannot be read from the text extraction - open the page).

**Caveats:**
- R-32 pdf p. 10: current and quick ratios "subject to perverse trends for some failing firms" (one sentence,
  1960s US manufacturers).
- R-05 p. 36: "a high current ratio doesn't necessarily imply health" (current ratio).
- R-61 pp. 26-27: the current ratio "can provide conflicting evidence", improving in distressed UK small firms
  because receivables rise; but quick assets / current assets is negative in their Table 4 p. 13.
- R-66 p. 11: quick assets/sales concave, quick assets/TA convex (different formulas; 115 retail bankruptcies).
- R-05 p. 54: rated firms show the opposite pattern, explained as an effect of credit access; supports the
  default-based negative sign.

**Inference, not a source:** that the model's conditional reversal is a receivables effect (combining R-05 p. 37,
"accounts payable and accounts receivable information is already 'contained in' the quick ratio", with R-61
pp. 26-27). No verified source tests the quick ratio's conditional sign with other liquidity ratios held
fixed, and none uses monotone constraints in boosted trees.

**Conclusion:** the literature backs constraining quick_ratio with sign "-" (direct statements in R-01, R-33,
R-05). Classifying it as "documented non-monotone" (lit13) overstated the caveats. A literature-faithful set
constrains it, which costs about 0.0007 AUC (ablation: releasing it alone gains +0.00074). The ratio with the
strongest documented caveats is the current ratio (cur_debt_ratio: R-32 p. 10, R-05 p. 36, R-61 pp. 26-27);
releasing its constraint changes nothing measurable (+0.00010, noise). Our formula differs from R-05's
((cash + AR) / debt_st vs (current assets - inventory) / current liabilities).

### 3.1 quick_ratio (83% against, gain rank 14) - Type 1 + Type 3

- Univariate: clean - in every year (rho -0.73 / -0.70 / -0.81; 2009 deciles 1.5 2.2 2.4 2.5 1.8 1.4 1.5 1.0
  0.6 0.9: a hump at deciles 2-4, a small uptick at decile 10).
- Free model curve: PD 1.08% at the 1st pct, rising to 1.52% at the 99th; monotone model: flat 1.24% to 1.18%.
- Why: its twins are cur_debt_ratio (Spearman 0.61) and cash_ratio. Holding the current ratio and the cash
  ratio fixed, a higher quick ratio = more receivables per euro of short-term debt. Receivables raise PD for
  small firms (R-61 Table 4 p. 13, p. 27), and the current ratio "improves in many failing small firms
  because trade receivables rise" (R-61 pp. 26-27); "perverse trends" for current and quick ratios (R-32
  pdf p. 10). So the reversed conditional direction has a verified economic reading - but it is a
  *receivables* reason code wearing a *liquidity* label, which a credit officer would misread.
- Top tail artefact: 759 rows (0.16%) have zero ST debt; their quick ratio = cash + AR in euros (median
  257,870), all at the 99.9th pct, default rate 1.98% (above the 1.24% average).

### 3.2 int_coverage (75% against, gain rank 22) - Type 2 + Type 3

- Univariate: clean - (rho -0.85 / -0.90 / -0.86), small uptick in decile 10 (2009: 0.6 to 0.7).
- Free curve: falls from 1.27% (30th pct) to 1.17% (60th pct), then **rises to 1.38% above the 90th pct**
  (values > 90x). Monotone: 1.26% to 1.15%, flat at the top.
- Why the top tail is not a "coverage" region: 15,145 rows (3.2%) have zero interest expense, so the ratio =
  EBITDA in euros (median 14,971; 99th pct of the feature = 160,000 "times"). The top of the feature mixes
  true coverage with euro amounts of EBITDA, i.e. with firm size. Twin: cash_interest_coverage (0.86),
  which takes over the main coverage effect.
- Bottom: R-05 p. 34 documents that very low coverage can show lower risk (tiny interest bills with slightly
  negative EBIT). Our bottom 10% (values < -1.8) is driven by 66,952 rows with negative EBITDA (default
  rate 4.29%), so our bottom is monotone, but the source warns the ratio is fragile at both ends and weaker
  for private firms (R-05 p. 34, p. 47).

### 3.3 operating_cash_flow_margin (71% against, gain rank 32) - Type 1 + Type 2

- Univariate: clean - (rho -0.87 / -0.93 / -0.88).
- Free curve: rises gently from 1.19% (10th pct) to 1.31% (99th). Monotone: 1.22% to 1.19%.
- Why: twin net_prof_margin (0.79). cf_operations = profit + D&A, so holding net margin fixed, a higher cash
  margin = more depreciation per euro of sales = capital-intensive, low-turnover firms. That is a business-model
  effect, not "more cash flow is bad". Assumption (no verified source for D&A intensity and PD).
- Tail artefacts: 14,454 rows (3.0%) with zero revenue get ratio = cf_operations in euros (extreme both ends);
  1,082 rows with negative revenue (default rate **6.56%**) sit high in the distribution (median at the 79th
  pct) where the sign says "safe".

### 3.4 Moderate conflicts among the top drivers

| feature (gain rank) | free % against | where | likely cause |
|---|---|---|---|
| operating_cash_flow_ratio (2) | 26% | PD 0.995% at the median rising to 1.12% at the 99th pct; monotone flat at 0.95% | Twin operating_cash_flow_to_debt (0.94): two near-copies split one effect; top tail "too good to be true" cash earnings |
| roa (3) | 19% | 0.99% at the 70th pct rising to 1.15% at the 99th; monotone flat 0.97% | Convex flattening documented (R-01 p. 14 Fig. 2); the uptick is top-tail only |
| ateco_sector_te (10) | 19% | rises to the 70th pct then dips at the top (1.49% to 1.31%) | Mechanical feature; top sectors are few and small, shrinkage pulls them; not a sign question (left free in both modes) |
| wc_ratio (23) | 36% | small wiggles 1.14-1.18% above the median | Twin cur_debt_ratio (0.91) |
| net_prof_margin (21) | 37% | wiggle at the 80-95th pct | Twins profit (0.87), operating_cash_flow_margin; negative-revenue rows |
| lt_debt_to_capital (30) | 46% | flat; free in both modes | Negative equity: 6,926 rows (1.45%) with denominator <= 0, default rate **17.5%**, land at value 0 next to firms with no LT debt; D-154 #1 |

### 3.5 Real-firm check with SHAP (the check that matters for reason codes)

Partial dependence can evaluate impossible firms (3.0), so the same question was asked on real rows: exact
TreeSHAP (LightGBM `pred_contrib`) on 20,000 FY2007-09 training rows, round-3 models
(`2026-11-21_07_shap_sign_check.py`, `shap_sign_check.csv`). Column = Spearman correlation between a firm's
feature value and that feature's SHAP contribution. A correctly signed reason code has the sign of the
expected direction (- for a "higher is safer" ratio).

| feature (free gain rank) | expected | SHAP vs value, unconstrained | SHAP vs value, monotone |
|---|---|---|---|
| equity_ratio (1) | - | -0.94 | -0.94 |
| **operating_cash_flow_ratio (2)** | - | **+0.12** | -0.90 |
| **roa (3)** | - | **-0.46** | -0.95 |
| operating_cash_flow_to_debt (4) | - | -0.92 | -0.95 |
| tangible_asset_ratio (5) | - | -0.96 | -0.96 |
| payables_ratio (6) | + | +0.75 | +0.94 |
| **quick_ratio (14)** | - | **+0.39** | -0.68 |
| int_coverage (22) | - | -0.57 | -0.82 |
| wc_ratio (23) | - | -0.43 | -0.83 |
| operating_cash_flow_margin (32) | - | -0.42 | -0.86 |
| size (28, unconstrained in both) | - | -0.86 | -0.91 |
| st_debt_share (11, unconstrained in both) | + | +0.86 | +0.78 |

**On real firms the conflict reaches the unconstrained model's #2 and #3 drivers**, which partial dependence
understated (26% and 19%). For a firm whose operating cash flow is strong, the unconstrained model's
operating_cash_flow_ratio reason code is as likely to say "raises your PD" as "lowers it".

**But the economics survive at family level.** Summing SHAP over each family of near-twins (`shap_family_check.csv`, same script):

| family | anchor ratio | combined SHAP vs anchor, unconstrained | monotone |
|---|---|---|---|
| cash flow (CF ratio, CF/debt, CF margin, cash interest coverage) | operating_cash_flow_ratio | -0.83 | -0.92 |
| profitability (roa, op_margin, net_prof_margin, ebitda_to_debt, int_coverage) | roa | -0.75 | -0.83 |
| liquidity (quick, current, cash, wc) | cur_debt_ratio | -0.48 | -0.77 |

So the unconstrained model's **ranking** is economically sensible; what breaks is the **attribution**: it
splits one cash-flow or profitability effect across near-copies and lets individual pieces carry opposite
signs. That is exactly what a per-firm reason code exposes (CLAUDE.md rule 8).

Note: size is unconstrained in both models but gets the economic sign on real firms (-0.86 / -0.91) even
though the FY2007 univariate curve was too flat to pass the D-157 rule (and R-60 p. 393 reports a +
conditional sign for Italian manufacturing). Leaving it free was the safe choice.

### 3.6 What the conflicts are NOT

- They are not unstable signs: every constrained ratio has the expected univariate sign in 2007, 2008 and
  2009 separately.
- They are not where the model's main risk ranking comes from: the top four drivers (equity_ratio,
  operating_cash_flow_ratio, roa, operating_cash_flow_to_debt; 56% of gain) obey or nearly obey the sign in
  the free model (1-26% against, all in the top tail).

---

## 4. Where does the unconstrained gain come from? (constraint ablation)

Winner setting, full-data binning, seeds 1-2 x rounds 1-3 = 6 fits per configuration, validation AUC and log
loss averaged (`2026-11-21_06_constraint_ablation.py`, `ablation_summary.csv`). Seed-to-seed spread is about
0.0005 AUC per round, so differences below about 0.0003 in the 6-fit mean are noise.

| configuration | constraints | mean val AUC | vs monotone | mean val log loss | vs monotone |
|---|---|---|---|---|---|
| mono_m3: monotone minus quick_ratio, int_coverage, operating_cash_flow_margin (data-driven, see caveat) | 17 | 0.85289 | +0.00147 | 0.05995 | -0.00016 |
| lit13 (withdrawn, see 3.1a: frees quick_ratio despite direct literature support) | 13 | **0.85235** | **+0.00093** | 0.06003 | -0.00007 |
| free | 0 | 0.85217 | +0.00075 | 0.05998 | -0.00012 |
| release quick_ratio only | 19 | 0.85215 | +0.00074 | 0.06002 | -0.00009 |
| release operating_cash_flow_ratio only | 19 | 0.85197 | +0.00056 | 0.06005 | -0.00006 |
| release cash_ratio only | 19 | 0.85192 | +0.00051 | 0.06007 | -0.00004 |
| release payables_ratio only | 19 | 0.85188 | +0.00046 | 0.06007 | -0.00003 |
| release any of 12 others | 19 | 0.85144-0.85170 | +0.00003 to +0.00028 (noise) | | |
| **monotone (D-157)** | 20 | 0.85142 | 0 | 0.06011 | 0 |
| release financing_expense_ratio / roa / wc_ratio / AR_ratio / equity_ratio | 19 | 0.85115-0.85136 | -0.00006 to -0.00027 | | |

Per round (AUC): monotone 0.85522 / 0.85253 / 0.84650; free 0.85545 / 0.85309 / 0.84797; lit13 0.85549 /
0.85367 / 0.84789.

What this says:

1. **The whole cost of the monotone model is essentially one constraint: quick_ratio.** Releasing it alone
   recovers the full free-vs-mono gap (+0.00074 vs +0.00075). That matches Section 3.1: quick_ratio's
   conditional effect is a receivables effect, and verified sources document perverse quick/current ratio
   trends in distress (R-32 pdf p. 10; R-61 pp. 26-27).
2. **Correct constraints help.** Releasing equity_ratio, AR_ratio, wc_ratio, roa or financing_expense_ratio
   makes the model slightly worse: there the constraint acts as regularisation against tail noise.
3. **lit13 matches the unconstrained model's AUC** (but see 3.1a: lit13 frees quick_ratio, which the literature supports constraining, so lit13 is withdrawn as a recommendation) while keeping the constraints on the top drivers
   (equity_ratio, operating_cash_flow_ratio, roa, operating_cash_flow_to_debt), i.e. correctly signed reason
   codes where it matters most. Its log loss sits between monotone and free.
4. mono_m3 scores best, which says the remaining 17 constraints regularise usefully. But it was chosen by
   looking at which features the free model violated on validation-era evidence, so its edge is partly
   selection; it also keeps wrong-signed reason codes for the three released features.

Honesty note on lit13: its 13 features were picked from the literature map before the ablation ran, but
after the free-vs-mono sign check had shown the quick_ratio / int_coverage conflicts. So it is not fully
blind to the data. The rule behind it (direct verified sign and no documented non-monotone shape) can be
stated and defended without reference to our results.

---

## 5. Decision framework

### Criteria (in the order the assignment weighs them)

1. **Face validity / explainability** (Formulation 15 pts; documentation 10 pts). The assignment says graders
   "examine model robustness and face validity (i.e., do the variables and parameters of the model make
   sense)" (R-30 p. 3). R-01 p. 14: "if an increase in leverage resulted in a decreased default risk ...
   most users would find the implication counter-intuitive." CLAUDE.md rule 8: per-firm top drivers must be
   explainable.
2. **Out-of-sample accuracy** (5 pts holdout). The measured gain is +0.0008 AUC / -0.2% log loss.
3. **Defensibility of each constraint**: a constraint is itself an assumption and needs a source. "Good
   variables ... tend to be those that have a conditionally monotone relationship"; "these are guidelines
   rather than rules" (R-05 p. 28). "Monotonicity ... implies a stable and real relationship, not just an
   accident of the sample" (R-05 p. 37).

### Options

| option | what | accuracy | explainability | defensibility | effort |
|---|---|---|---|---|---|
| A. Full monotone (D-157, 20 constraints) | current grid winner | -0.0008 AUC vs free | Every constrained reason code has the economic sign | quick_ratio and the other main constraints have direct verified signs (3.1a). Caveats a reviewer could raise: cur_debt_ratio (R-32 pdf p. 10, R-05 p. 36, R-61 pp. 26-27) and int_coverage (R-05 p. 34 bottom tail); cash_to_financing_exp_ratio and tangible_asset_ratio have no verified PD-sign source. Releasing any of these four changes AUC by noise only (+0.00003 to +0.00011) | none |
| B. Unconstrained | rule winner | best | 3 ratios with wrong-signed reason codes (quick 83%, int_cov 75%, CF margin 71%) and unreliable reason codes for the #2 and #3 drivers on real firms | No constraint to defend, but the deck must explain the reversals | none |
| C. lit13 (SUPERSEDED, see 3.1a) | constrain 13; free quick_ratio, cur_debt_ratio, int_coverage, cash_interest_coverage, cash_to_financing_exp_ratio, AR_ratio, tangible_asset_ratio | +0.0009 AUC vs A | quick_ratio reason code can reverse | **Not literature-faithful**: quick_ratio has direct verified support for "-" (R-01 p. 13, R-33 p. 11, R-05 p. 54); lit13 overstated the caveats. Its AUC edge comes almost entirely from freeing quick_ratio | one refit |
| D. Fix features, then constrain | D-154 fixes (explicit risky-end caps instead of "0 becomes 1", negative-revenue/negative-equity flags) | unknown | best | best | feature rebuild by Chloe/Desmond, new grid: likely too late for the TA test, possible for final |

### Recommendation

(Revised 2026-11-21 after the quick_ratio literature check, Section 3.1a. The earlier lit13 recommendation is
withdrawn.)

**Option A, full monotone (D-157, 20 constraints), for the final model; Option D (fix the zero/negative-
denominator construction) as a post-TA improvement if time allows.** Decision is KN's.

Why A:
- Literature: the constrained signs that carry the model are stated directly in verified sources (Section 2;
  quick_ratio in 3.1a: R-01 p. 13, R-33 p. 11, R-05 p. 54).
- Explainability: every constrained reason code has the economic sign, including the #2 and #3 drivers, which
  are unreliable in the unconstrained model on real firms (SHAP vs value +0.12 and -0.46; Section 3.5).
- Cost: about 0.0008 AUC (0.1%) and 0.2% log loss vs unconstrained; almost all of it is the quick_ratio
  constraint, which is the one with the strongest direct literature support.
- It matches the course's framing: face validity is graded (R-30 p. 3), and monotone relations "imply a stable
  and real relationship, not just an accident of the sample" (R-05 p. 37).

Optional tweak (no measurable cost either way): free cur_debt_ratio, the ratio with the strongest documented
caveats (R-32 pdf p. 10; R-05 p. 36; R-61 pp. 26-27). Releasing it changed AUC by +0.00010 (noise). Keeping
it constrained is also defensible; cite the caveats if asked.

What A leaves to manage:
- cash_to_financing_exp_ratio and tangible_asset_ratio are constrained on data + mechanism only (no verified
  PD-sign source); say "we assume" or free them (releasing either is noise: +0.00011, +0.00004).
- int_coverage: R-05 p. 34 documents a non-monotone bottom tail; our bottom tail is monotone (negative-EBITDA
  firms, default rate 4.3%), so the constraint fits our data.
- The level problem (obs/pred 1.22-1.46 in validation) is the same for every option and is solved by
  calibration, not by the constraint choice.

**Option B (unconstrained)** is not recommended: +0.0008 AUC, but wrong-signed or unreliable reason codes on
quick_ratio, operating_cash_flow_ratio (#2) and roa (#3).

### What to write in the deck either way

- The monotone/unconstrained comparison with the bootstrap interval and the three reversal plots, as a
  "trade-off / what didn't work" slide.
- One sentence on why reversals happen: twin features change the conditional meaning (quick ratio given
  current ratio = receivables), and zero/negative denominators put unusual firms in the tails.
- Formula differences from the literature (claim map caveat 9): quick ratio excludes non-cash,
  non-receivable current assets; payables and interest are over assets; roa is operating; cf_operations is
  net profit + D&A.

---

## 6. The 48 features left unconstrained in both modes (no candidate sign)

| group | features | why no sign | source |
|---|---|---|---|
| Sign flips with negative equity | debt_to_eqty, roe | Non-monotone by construction: equity <= 0 flips the sign | R-05 p. 34 (debt/net worth "knife-edged"); R-05 p. 33 (NI/equity "perverse" above ~60th pct) |
| No reliable sign | asset_turnover | Non-monotone and flat; industry-dependent | R-05 p. 37; R-32 pdf pp. 12-13, 26 |
| No verified source | bank_debt_conc | No verified sign; context only | claim map Sec. 2 (ASSUMPTION/SECONDARY) |
| Near-complement of tangible_asset_ratio | cur_asst_ratio | Expecting both to lower PD is inconsistent | REFERENCES Contradiction 11; R-32 pdf p. 10 (weak) |
| Raw euro levels | profit, ebitda, COGS, exp_financing | Mix firm size with the underlying quantity | R-05 p. 35 Exh. 4.5 (revenue ~ size proxy); otherwise ASSUMPTION |
| Data-quality flag | neg_pl_flag | Sign not sourced | ASSUMPTION |
| Zero flags (15) | zero_total_debt, zero_debt_st, zero_debt_lt, zero_bank_debt, zero_cash, zero_asst_current, zero_ar, zero_rev_operating, zero_AP_st, zero_wc_net, zero_ebitda, zero_prof_operations, zero_profit, zero_cf_operations, zero_exp_financing | A zero is structural in some sectors and a red flag in others | R-01 p. 18, Table 3 (zeros mean different things by sector) |
| Legal form dummies (4) | legal_SAU, legal_SPA, legal_SRS, legal_SRU | No verified sign | R-70 (definitions only, unofficial) |
| Region dummies (19) | city_* | South riskier than Centre-North but the gap varies over time | R-69 (landing page; bank-loan measure) |

---

## 7. Files

- Sign table rule and output: `experiments/gbm/2026-11-20_grid75/2026-11-20_01_sign_table.py`, `sign_table.csv`
- Stability, denominator states, model curves: `2026-11-21_05_sign_evidence.py`, `sign_evidence.csv`,
  `sign_evidence_states.csv`, `sign_evidence_curves.csv`
- SHAP (real rows): `2026-11-21_07_shap_sign_check.py`, `shap_sign_check.csv`, `shap_family_check.csv`
- Free vs mono: `2026-11-20_grid75_report.md`, `bootstrap_free_minus_mono.csv`, `seed_check.csv`,
  `sign_violation_check.csv`
- Ablation: `2026-11-21_06_constraint_ablation.py`, `ablation_results.csv`, `ablation_summary.csv`
