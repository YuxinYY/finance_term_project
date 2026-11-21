# Feature literature claims — 80-feature set (literature-scout, 2026-11-16)

Purpose: for each feature family of Chloe and Desmond's 80-feature set, say (1) what risk it
measures in credit-officer language, (2) the expected PD direction and any documented
non-monotone shape, and (3) which **verified** source says so, with page/section. Only sources
marked VERIFIED in `docs/REFERENCES.md` are cited as support. Everything else is labelled.

**Status column**
- **SUPPORTED** — the claim was checked against the text of a VERIFIED source (R-id + page).
- **SECONDARY** — we only have it through a verified source that reports someone else's result.
- **ASSUMPTION** — economic reasoning by the team; no source read. Word it "We assume / we expect".
- **[UNVERIFIED]** — a source is believed to say this but was not read.

**Who the sources studied (read before citing a shape)**

| ID | Source | Population | Outcome / horizon | Note |
|---|---|---|---|---|
| R-01 | RiskCalc v3.1 (Dwyer, Kocagil & Stein 2004) | North American / other private firms (CRD) | default, 1 and 5 yr | Directions stated in text p. 13 |
| R-05 | RiskCalc for Private Companies (Falkenstein et al. 2000) | Exhibits in Sec. IV = **US public** (Compustat) 1980–99; Exh. 5.2–5.5 add private CRD | **5-year** cumulative default | Shapes are univariate, smoothed |
| R-33 | RiskCalc 4.0 France (Buitrago et al. 2015) | French private firms 1993–2014 | 1-yr EDF transforms | **France, not Italy** |
| R-32 | Altman (2000) Z-score revisited | 66 US manufacturers (1946–65), matched sample | bankruptcy, 1 statement before | Group means, not PDs |
| R-60 | Bottazzi, Grazzi, Secchi & Tamagni (2011) | **19,628 Italian manufacturing** limited-liability firms, sales ≥ €1m, 1998–2003 | default (Basel-style) at one bank, 2003–04; 147 defaults | Only Italian firm-level source read |
| R-61 | Altman, Sabato & Wilson (2010) | 5.8m UK SME accounts 2000–07, mostly very small | insolvency | Abridged-accounts model included |
| R-66 | Valencia et al. (2019) | 2,922 Colombian retail SMEs, 115 bankruptcies | bankruptcy, 1 yr | GAM shapes; small sample |
| R-48 | Almeida & Campello (2007) | US public manufacturers | investment / borrowing capacity | Mechanism only (Contradiction 11) |

Not usable for claims (LOCATED/SECONDARY only): Ohlson 1980 (R-12), Beaver 1966 (R-63, only via
R-32), Altman & Sabato 2007 (R-62, ratio list only via R-61), Ciampi & Gordini 2013 (R-64),
Bonfim 2009 (R-65), RiskCalc Italy (R-03). **The local "Financial and Economic Determinants of
Firm Default.pdf" is Bottazzi et al. (R-60), not Bonfim.**

---

## 1. Size

| feature(s) | finance meaning | expected sign / shape | source (R-id + page/section) | quote or close paraphrase | status |
|---|---|---|---|---|---|
| `size` = ln(asst_tot) | Bigger balance sheet: more diversification, deeper management, better access to funding; small firms are more exposed to single shocks | **Univariate: −** (larger → lower PD), flattening | R-01 Sec. 3.1 p. 13 | Size group: "Large firms default less often"; size = real total assets in US/Canada v3.1 | SUPPORTED |
| | | − with flattening; steepest at the small end (France) | R-33 p. 12; Fig. 5 p. 13 | Real total assets transform slopes down | SUPPORTED |
| | | Size effect weakens once cash flow is in the model | R-01 Sec. 3.1 p. 13 | Adding a cash-flow ratio reduced the size advantage: a small firm with healthy statements is not much riskier than a larger one | SUPPORTED |
| | | Private firms: strong − below ~$40m assets, flat above; data-collection biases distort the curve | R-05 pp. 49–51, Exh. 5.3; p. 35 Exh. 4.5 (public) | Manually added defaults over-represent large firms; Moody's best estimate is monotone − with dampening | SUPPORTED |
| | | **Conditional on financial ratios, + in Italian manufacturing** | R-60 p. 393; Table 3 p. 392; Fig. 1 pp. 382–383 | Size (log sales) has a positive, significant marginal effect in every year; size densities of defaulters and survivors nearly identical | SUPPORTED — **contradicts "−"** |
| | | **Hump-shaped**: risk rises up to ~£350k assets then falls (UK) | R-61 pp. 19–20, 23; Fig. 1; Table 3 p. 12 | Tiny firms are rarely pushed into insolvency because creditors gain little; modelled with log assets and its square | SUPPORTED (UK; our firms > €1.5m sit on the falling side) |
| raw euro levels: `rev_operating`, `ebitda`, `profit`, `COGS`, `prof_operations`, `exp_financing` | Partly size proxies; partly the underlying quantity (profit, interest bill) | Mixed; revenue ≈ assets as a size proxy | R-05 p. 35, Exh. 4.5 | Sales and total assets are almost indistinguishable as size proxies (public firms) | SUPPORTED for revenue; ASSUMPTION for the others |
| | | Raw amounts mix scale with the ratio signal; ratios are preferred | none found | Roger's feedback M9 asked for ratios (FEATURE_SPEC) | ASSUMPTION |

Team data: `log_assets` is weakly − univariately but + in the multivariate core logit
(FEATURE_SPEC "Where the data disagree", D-022). R-60 and R-61 make that a known pattern, not a bug.

## 2. Leverage and capital structure

| feature(s) | finance meaning | expected sign / shape | source | quote or close paraphrase | status |
|---|---|---|---|---|---|
| `equity_ratio` = eqty_tot / asst_tot | Owners' cushion: how much asset value can be lost before creditors are hit | **−, convex**: steepest where equity is thin | R-33 Table 3 p. 11; p. 12; Fig. 5 p. 13 | Equity/assets is France's leverage input; large values lower PD; slope flattens as equity rises | SUPPORTED |
| | | Mirror of liabilities/assets (high leverage ↑PD) | R-01 p. 13; R-05 p. 34 | Leverage group (liabilities/assets, LTD/assets); equity/assets is the mathematical complement of L/A | SUPPORTED |
| | | Negative equity is bad but not fatal | R-05 p. 47 | L/A > 1 for 10% of private firms; negative-net-worth firms default at ~3× the average rate | SUPPORTED |
| | | Book-equity version of Altman's X4 | R-32 pdf pp. 11–12, 25 | Z' re-estimates X4 with book equity / total liabilities | SUPPORTED |
| `debt_to_eqty` = (debt_st + debt_lt) / eqty_tot | Gearing: euros of debt per euro of owners' money | **Non-monotone**: sign flips when equity ≤ 0 | R-05 p. 34 | Debt/net worth was non-monotone because net worth can be negative; excluded for "knife-edged" interpretation | SUPPORTED — use with a negative-equity flag or prefer `equity_ratio` |
| | | Short-term debt/equity raises risk (SMEs) | R-61 Table 3 p. 12 (Altman–Sabato ratio) | ST debt/equity positive in the UK re-estimate | SUPPORTED (same sign-flip risk: our inference) |
| `st_debt_share` = debt_st / (debt_st + debt_lt) | Maturity mix: how much must be repaid or rolled within a year | Weak / unclear | R-05 p. 36 + fn. 31 | ST/LT debt mix "of little use"; even less for private firms because multi-year loans sit in 364-day facilities (US rules) | SUPPORTED (weak) |
| | | + (refinancing risk) | none found | — | ASSUMPTION |
| `lt_debt_to_capital` = debt_lt / (debt_lt + eqty_tot) | Long-term gearing of permanent capital | + | R-01 Appendix p. 34; p. 13 | "LTD to (LTD plus Net worth)" is a US/Canada v3.1 leverage input; high leverage ↑PD | SUPPORTED; **denominator can be ≤ 0** when equity is negative (our inference, same issue as R-05 p. 34) |
| `bank_debt_conc` = bank debt / total debt | How much of the debt is owed to banks (vs suppliers, others) | **No verified sign** | none found | R-60 p. 378 reports (citing others) that Italian firms typically borrow from 6–8 banks; R-61 p. 11 reports (citing Hudson 1986) that trade creditors, not banks, often force small-firm bankruptcy | ASSUMPTION / SECONDARY context only |
| `cur_asst_ratio` = asst_current / asst_tot | Share of assets that turn into cash within a year | − (weak) | R-32 pdf p. 10 | A firm with persistent operating losses usually sees current assets shrink relative to total assets (Altman, on X1) | SUPPORTED (weak; 1960s US manufacturers) |
| `tangible_asset_ratio` = asst_tang_fixed / asst_tot | Pledgeable, saleable collateral | − (team data) | R-48 Sec. 1 p. 1, fn. 2 | Tangibility raises what creditors can recover and so supports borrowing — a **recovery/credit-access** mechanism, proxy ≠ balance-sheet share | SUPPORTED as mechanism only; "lowers PD" = ASSUMPTION (Contradiction 11) |
| `goodwill_ratio` = goodwill / asst_tot | **If** accounting goodwill: purchased intangibles that lose value in distress. **But** the field likely holds value added (D-010, D-091) | If value added/assets: − | R-60 p. 384 (Fig. 4), p. 393 | Defaulters have lower labour productivity (value added per employee); productivity lowers PD | SUPPORTED only under the D-091 hypothesis; goodwill reading not applicable (Contradiction 17) |
| | | (Intangibles not worthless) | R-05 p. 34 | Subtracting intangibles did not help leverage predict; intangibles "appear really to be worth something" | SUPPORTED (public firms) |

## 3. Liquidity and activity

| feature(s) | finance meaning | expected sign / shape | source | quote or close paraphrase | status |
|---|---|---|---|---|---|
| `cash_ratio` = cash / asst_tot | Cash on hand to meet the next payments | **−**, steepest at low cash | R-05 p. 36; p. 51 Exh. 5.4; p. 61 | Most important single variable for private firms; private default frequency falls across the whole range; 12% weight in v1 | SUPPORTED |
| | | − (France, near-linear on percentile scale) | R-33 Table 3 p. 11; p. 12; Fig. 5 | Cash and marketable securities/assets slopes down | SUPPORTED |
| | | − (UK SMEs) | R-61 Table 3 p. 12; Table 7 p. 24 | Negative coefficient; failed mean 0.125 vs non-failed 0.265 | SUPPORTED |
| | | **U-shape** (very high cash also riskier) | R-66 Fig. 3 p. 10; p. 11 | Convex cash/TA effect in Colombian retail | SUPPORTED but weak (115 defaults); **R-05 private curve contradicts** (Contradiction 15) |
| `AR_ratio` = AR / rev_operating | Days of sales not yet collected: slow or doubtful customers, possibly overstated receivables | + (SMEs) | R-61 Table 4 p. 13; Table 8 p. 25; p. 27 | Trade debtors/assets positive; distressed small firms extend credit to keep sales, or are paid late | SUPPORTED (UK, assets denominator) |
| | | **Non-monotone** (public firms) | R-05 p. 37, Exh. 4.7 | Activity ratios have less straightforward relations; AR/sales curve high at both ends (our reading of the chart); AR and AP individually stronger than inventories but left out because the quick ratio holds their information | SUPPORTED (chart reading flagged) |
| | | Activity ratios differ in direction | R-01 p. 13; Appendix p. 34 | "Other activity ratios have different relationships to default"; v3.1 uses *change* in AR turnover (US/Canada), change in trade debtors/turnover (UK), receivables/sales under liquidity (Japan) | SUPPORTED |
| `payables_ratio` = AP_st / asst_tot | Supplier credit used: stretching suppliers is a classic liquidity-stress symptom | **+** | R-33 Table 2 p. 11; p. 12; Fig. 5 | Large AP/sales raises PD (France) | SUPPORTED (our denominator is assets, not sales) |
| | | + | R-61 Table 4 p. 13; p. 27 | Trade creditors/total liabilities positive; credit-rationed small firms substitute supplier credit | SUPPORTED (UK) |
| | | + | R-01 Appendix p. 34 | UK v3.1 input "trade creditors to turnover" | SUPPORTED |
| `wc_ratio` = wc_net / asst_tot | Current assets left after paying short-term liabilities | **−** | R-32 pdf p. 10; Table 1 pdf p. 13 | Altman's X1; the best of three liquidity ratios in 1968; bankrupt mean −6.1% vs 41.4% | SUPPORTED |
| | | − (but quick ratio slightly better) | R-05 p. 36 | Quick ratio slightly more powerful than WC/assets (public firms) | SUPPORTED |
| | | − linear | R-66 Table 3 p. 9 | WC/TA linear negative | SUPPORTED (weak) |
| `cur_debt_ratio` = asst_current / debt_st (current ratio) | Current assets per euro of short-term claims | − | R-01 p. 13; Appendix p. 34 (UK) | Liquidity group includes the current ratio; high liquidity lowers PD | SUPPORTED |
| | | **Can be perverse in distress** | R-32 pdf p. 10 | Current and quick ratios "subject to perverse trends for some failing firms" | SUPPORTED |
| | | Rises in distressed small firms | R-61 pp. 26–27 | Current ratio improves in many failing small firms because trade receivables rise | SUPPORTED (UK) |
| | | **Hump** (concave) | R-66 Fig. 3 p. 10; p. 11 | Current ratio concave in Colombian retail; excess liquidity "not a good symptom" | SUPPORTED (weak) |
| | | A high current ratio need not mean health | R-05 p. 36 | Text only — **do not cite the Exh. 4.6 line** (labelled as rising, against the text) | SUPPORTED (text) |
| `quick_ratio` = (cash + AR) / debt_st | Liquid assets (excluding inventory) per euro of short-term claims | **−**, steep at low values | R-05 p. 36; p. 50 Exh. 5.2; p. 61 | Quick ratio a v1 input (7%); public and private curves both fall | SUPPORTED. **Our formula differs**: R-05 uses (current assets − inventory) / current liabilities |
| | | **−** (direct statement) | R-01 p. 13; R-33 p. 11 Table 2; R-05 p. 54 | R-01/R-33: liquidity incl. the quick ratio, "High liquidity reduces the probability of default"; R-05 p. 54: "strongly negatively related to default: the higher the quick ratio, the lower the default probability" (added 2026-11-21, D-162) | SUPPORTED (direct) |
| | | Rated-firm paradox | R-05 p. 54 | "lower rated and unrated firms have significantly higher quick ratios than their investment-grade counterparts"; a low quick ratio is an effect of a high rating, not the cause. Explains why a multivariate model can flip the sign (D-162) | SUPPORTED |
| `asset_turnover` = rev_operating / asst_tot | Sales generated per euro of assets | **No reliable sign; non-monotone; industry-dependent** | R-05 p. 37 | Sales/assets non-monotonic and very flat; Z variants without it performed better | SUPPORTED |
| | | | R-32 pdf pp. 12–13; p. 26 | X5 least significant alone (F = 2.84) but second in multivariate contribution; dropped in Z'' because turnover varies widely by industry | SUPPORTED |
| | | − nonlinear (retail) | R-66 Fig. 3 p. 10 | Sales/TA monotone decreasing in the GAM | SUPPORTED (weak) |

Team data: `asset_turnover` shows an inverted U (FEATURE_SPEC, D-022) — consistent with R-05.

## 4. Profitability

| feature(s) | finance meaning | expected sign / shape | source | quote or close paraphrase | status |
|---|---|---|---|---|---|
| `roa` (= 100 × operating profit / asst_tot) | Earning power of the assets before financing | **−, convex** (big gains at the loss end, little at the top) | R-01 p. 14, Fig. 2 + fn. 10 | PD falls as NI/assets rises, with diminishing sensitivity; quadratics capture this poorly | SUPPORTED (R-01 uses net income; ours is operating) |
| | | − | R-32 pdf p. 11; Table 1 pdf p. 13 | X3 EBIT/TA = true productivity of assets; outperforms other profitability measures incl. cash flow; bankrupt mean −31.8% vs 15.4% | SUPPORTED (closest match to our operating ROA) |
| | | − | R-05 pp. 32–33, Exh. 4.3; p. 50 Exh. 5.2 | NI/assets dominates alternatives; v1 uses (NI − extraordinary)/assets | SUPPORTED |
| `roe` = profit / equity | Return to owners | **Non-monotone**; high ROE can mean tiny or negative equity | R-05 p. 33 | NI/equity "perverse" above ~60th percentile — higher NI/equity, higher default — driven by low/negative book equity (12% of private firms) | SUPPORTED — expect a U or flip; pair with negative-equity flag |
| `op_margin` = prof_operations / rev | Operating profit per euro of sales | **−, flattening** | R-33 p. 11 (replaces EBITDA/sales), p. 12, Fig. 5 | Operating margin transform slopes down, flatter at high margins | SUPPORTED |
| | | − | R-60 pp. 383–384 Fig. 3; p. 393 | Defaulters have clearly lower ROS (gross operating income/sales); profitability lowers PD | SUPPORTED (Italy) |
| | | Weaker than ROA; margins vary by industry | R-05 p. 33 | Operating margin did worse than NI/assets | SUPPORTED |
| `net_prof_margin` = profit / rev | Bottom-line profit per euro of sales | − | R-01 Appendix p. 34 | UK v3.1: net P&L / turnover | SUPPORTED |
| | | − nonlinear | R-66 Table 3 p. 9; p. 11 | NI/sales non-linear, PD rises as it falls | SUPPORTED (weak) |
| `loss_flag` = profit < 0 | Firm is losing money this year | + | R-01 p. 13 (profitability ↓PD) | Direction follows from the profitability family | SUPPORTED for direction |
| | | Loss dummy as a separate input | Ohlson (1980) INTWO-type variable | not read | [UNVERIFIED] (R-12 LOCATED only) |
| raw `profit`, `ebitda`, `prof_operations`, `COGS` | Scale × profitability | Mixed | see Section 1 | — | ASSUMPTION |

## 5. Coverage and cash flow

| feature(s) | finance meaning | expected sign / shape | source | quote or close paraphrase | status |
|---|---|---|---|---|---|
| `int_coverage` = ebitda / exp_financing | Times operating cash earnings cover the interest bill | − overall | R-61 Table 3 p. 12; R-01 Appendix p. 34 (Japan: EBITDA/interest) | EBITDA/interest paid negative (UK SMEs) | SUPPORTED |
| | | **Non-monotone at the bottom**: very low coverage can show lower risk | R-05 p. 34, Exh. 4.4 | Extreme low values come from tiny interest expense with slightly negative EBIT | SUPPORTED |
| | | Less powerful for private firms | R-05 p. 34; p. 47 | Moves from most important (public) to minor (private), possibly because interest on unaudited statements is unreliable; many firms negative or > 10, so rules of thumb fail | SUPPORTED |
| `financing_expense_ratio` = exp_financing / asst_tot | Interest burden = cost of debt × leverage | **+** | R-60 pp. 385–386, 391; Table 3 p. 392 | Interest expense/sales is the financial variable that most sharply separates defaulters; significant in every year (Italy) | SUPPORTED (sales denominator) |
| | | + (flattening at top) | R-33 p. 11, p. 12, Fig. 5 | Interest expense/sales slopes up (France) | SUPPORTED (sales denominator) |
| `cash_to_financing_exp_ratio` = cash / exp_financing | Months of interest that cash on hand could pay | − | none found | — | ASSUMPTION |
| `ebitda_to_debt` = ebitda / (debt_st + debt_lt) | Years of operating cash earnings to repay debt (inverted) | **−, convex** | R-33 Table 3 p. 11; p. 12; Fig. 5; weight Table 4 p. 14 | Debt coverage (profit + D&A)/(liabilities − cash & securities) slopes down and flattens; largest weight (19.2%) | SUPPORTED (formula differs) |
| | | − | R-32 pdf p. 9 (reporting Beaver) | Beaver found cash flow / debt the best single ratio | SECONDARY (R-63 not read) |
| `operating_cash_flow_margin`, `cash_interest_coverage`, `operating_cash_flow_ratio`, `operating_cash_flow_to_debt` (cf_operations = net profit + D&A, FEATURE_SPEC) | Internally generated funds relative to sales, interest, assets, debt | − | R-01 Appendix p. 34; p. 13 | v3.1 uses cash flow / interest expense (US), cash flow / current liabilities (Canada), cash flow / interest charges (UK); adding a cash-flow ratio improved the model | **CORRECTED 2026-11-21 (D-162): input-use only.** Neither page states a PD direction; do not cite for the sign |
| | | − (direct, closest ratio) | R-33 p. 11; p. 12 | "High debt coverage reduces the probability of default"; (profit + D&A)/(liabilities − cash & securities) "is downward-sloping, indicating that large values of this ratio lower the probability of default" | SUPPORTED for `operating_cash_flow_to_debt` (formula differs); **ANALOGY only** for `operating_cash_flow_ratio` (/assets), `operating_cash_flow_margin`, `cash_interest_coverage` |
| | | − (numerator matches R-33) | R-33 Table 3 p. 11 | Profit + D&A is France's debt-coverage numerator | SUPPORTED |
| | | Cash flow not better than earnings (public data) | R-05 p. 33; R-32 pdf p. 11 | EBIT/assets less steep than NI/assets; EBIT/TA outperforms cash flow | SUPPORTED — sources disagree on cash flow vs earnings |
| | | CF/net worth − linear (retail) | R-66 Table 3 p. 9 | — | SUPPORTED (weak) |

## 6. Flags

| feature(s) | finance meaning | expected sign / shape | source | quote or close paraphrase | status |
|---|---|---|---|---|---|
| `neg_bs_flag`, `neg_pl_flag` (negative items that should be ≥ 0) | Data-quality warning; possible distress or reclassification | + (weak) | none for the sign | — | ASSUMPTION |
| | Keeping and flagging suspect rows | — | R-01 pp. 21–22 + fn. 20 | Rows flagged for excess rounding were kept; the model was better with them | SUPPORTED (design choice) |
| | Moody's drops implausible statements in development | — | R-33 Sec. 2.2 p. 6; R-10 App. A p. 30 | e.g. assets ≠ liabilities + net worth | SUPPORTED (we cannot drop at prediction time) |
| `debt_detail_missing` (bank / financial / trade debt split missing) | Possibly a short-form (abbreviated) statement: less disclosure, usually smaller firm | Weak + | R-27 art. 2435-bis; R-67 art. 2424 Passivo D | Abbreviated balance sheets keep only capital-letter and Roman-numeral items; Debiti (D) has only Arabic sub-items (4 banks, 7 suppliers), so the split need not be shown | SUPPORTED as *legal possibility*; link to our missingness = hypothesis |
| | Less information still predicts; providing more information goes with lower risk | | R-61 p. 4, p. 23; Table 3 p. 12; Table 9 p. 28 | 60% of UK accounts abridged; "no cash-flow statement" dummy positive; abridged-accounts model AUC 0.75 out-of-sample | SUPPORTED (UK) |
| | Statement quality mostly absorbed by size | | R-05 pp. 40–41, Exh. 4.15 | Audited private firms default ~30% less, but audit is highly correlated with size | SUPPORTED |
| | Filling missing inputs with averages biases PDs down | | R-34 (vendor statement); R-05 pp. 60–61 | Moody's says mean imputation understates PDs (no numbers) | SUPPORTED as Moody's statement |
| 16 `zero_*` flags (zero revenue, zero debt, zero cash, …) | A zero can be structural (no inventory in services, no debt) or a red flag (no revenue) | Sign depends on the item and the sector | R-01 p. 18, Table 3 | > 40% of service/construction/mining-transport firms carry no inventory; low inventory should not earn credit there | SUPPORTED (for "zeros mean different things by sector") |
| | Near-zero denominators create extreme ratios | | R-05 pp. 33–34, 47 | Near-zero equity or interest produces extreme ratios that mislead | SUPPORTED |
| | Zero cash → riskier | − cash, steep at bottom | R-05 p. 51 Exh. 5.4 | — | SUPPORTED (direction) |
| | Zero revenue / zero debt signs | | none found | — | ASSUMPTION |

## 7. Categoricals

| feature(s) | finance meaning | expected sign / shape | source | quote or close paraphrase | status |
|---|---|---|---|---|---|
| industry (ATECO section, target-encoded) | Sector risk level and how ratios should be read in that sector | Level shift, not monotone | R-01 Sec. 3.3 pp. 17–18, Table 2 | Industry controls: small AR gain (54.4% → 55.1%), large log-likelihood gain; fix spurious ratio effects | SUPPORTED |
| | | | R-61 p. 20; Table 3 p. 12 | Industry entered as prior-year sector failure log-odds (WoE); significant | SUPPORTED (UK) |
| | | Small effect once financials are in (Italy) | R-60 p. 391 | < 5% of 2-digit sector dummies significant in Italian manufacturing | SUPPORTED (manufacturing only) |
| | Encoding | | R-54 pp. 51–54; R-47 §1.2, §2.7 | ATECO section ranges; target encoding needs shrinkage + cross-fitting | SUPPORTED |
| legal form (SRL, SPA, SAU, SRU, SRS …) | Governance, capital requirements, disclosure duties | No verified sign | R-70 (course handout) | Roger's **unofficial** readings: SRS = simplified SRL (capital < €10k); SRU probably single-shareholder SRL; SAU meaning unclear | SUPPORTED for definitions (unofficial); sign = ASSUMPTION |
| | | Group membership lowers risk (UK) | R-61 Table 3 p. 12; p. 23 | Subsidiary dummy negative | SUPPORTED (UK; we have no group flag) |
| region (20 regions from HQ province) | Local economy, local credit conditions | South riskier than Centre-North; **gap varies over time** | R-69 (landing page) | South vs Centre-North gap in firms' credit deterioration rate peaked at 4.8 pp at end-2012, ~1 pp by early 2018 | SUPPORTED (bank-loan measure, not our label) |
| | | Model accuracy differs by region | R-64 abstract | Italian small-firm NN more accurate for Central Italy | [UNVERIFIED] (abstract only) |
| | Mapping | | R-53 | ISTAT 110-province code list | SUPPORTED |
| | Moody's Italy cycle input is national, not regional | | R-04 slide 13 | Italy, Spain, Portugal models add the national unemployment rate | SUPPORTED |

---

## Contradictions and caveats

1. **Size.** RiskCalc (R-01 p. 13; R-33 p. 12) says larger is safer; Italian manufacturing
   (R-60 p. 393) finds a *positive* conditional size effect, and UK SMEs (R-61 pp. 19, 23) a hump.
   Our multivariate sign is + (D-022). Present size as "lower risk on its own, but much of the
   benefit is explained by better financials" (R-01 p. 13). A monotone "−" constraint on size is
   not backed by the Italian evidence.
2. **High cash.** No verified source shows private firms with very high cash are riskier; R-05
   p. 51 shows the opposite. R-66 is the only U-shape (small Colombian retail sample).
3. **Current ratio** can be perverse: R-32 pdf p. 10, R-61 pp. 26–27, R-66 p. 11. Treat a hump
   as plausible; do not force monotone −.
4. **Equity sign flips.** `debt_to_eqty`, `roe`, `lt_debt_to_capital` are not interpretable when
   equity ≤ 0 (R-05 pp. 33–34). Prefer `equity_ratio` + a negative-equity flag in explanations.
5. **Interest coverage** is non-monotone at the bottom (R-05 p. 34): tiny interest bills produce
   extreme values. Also weaker for private firms.
6. **Asset turnover** has no reliable sign (R-05 p. 37; R-32 pdf pp. 12, 26). Our inverted U is
   consistent.
7. **Receivables.** R-61 (UK SMEs) linear +; R-05 Exh. 4.7 (US public, 5-yr) non-monotone.
8. **Cash flow vs earnings.** RiskCalc v3.1 adopted cash-flow coverage (R-01 Appendix p. 34);
   RiskCalc v1 found cash flow no better than net income (R-05 p. 33); Altman found EBIT/TA beat
   cash flow (R-32 pdf p. 11).
9. **Formula differences from the literature (say so on slides):** our quick ratio excludes
   non-cash, non-receivable current assets (R-05: current assets − inventory); payables and
   interest are over assets (R-33, R-60: over sales); `ebitda_to_debt` uses total financial
   debt (R-33: liabilities net of cash and securities); `roa` is operating (R-01/R-05: net income);
   `cf_operations` is net profit + D&A, an accounting proxy, not a cash-flow statement figure.
10. **`goodwill` is probably value added** (D-010, D-091): no goodwill literature applies.
11. **`tangible_asset_ratio` and `cur_asst_ratio` are near-complements** (both share total
    assets). Expecting both to lower PD is internally inconsistent unless intangible/financial
    fixed assets carry the difference; let the data decide and do not cite both directions.
12. **`bank_debt_conc`, `cash_to_financing_exp_ratio`, zero-revenue/zero-debt flags** have no
    verified source: wording must be "we assume".
13. **Population mismatch.** R-05 exhibits are US public firms at a 5-year horizon; R-33 is
    France; R-61 is UK insolvency of mostly tiny firms; R-66 is 115 Colombian retail
    bankruptcies; R-60 is Italian manufacturing only, with defaults from one bank. None is our
    population (Italian private firms > €1.5m assets, all sectors, missed-payment label).
14. **Sources not read**: Ohlson (R-12), Beaver (R-63, only via R-32, which dates it "1967"),
    Altman & Sabato 2007 (R-62, ratio list only via R-61), Ciampi & Gordini (R-64), Bonfim (R-65),
    RiskCalc Italy (R-03). Nothing about their variables, coefficients or signs may appear in the
    deck except as listed above.
15. **Growth**: RiskCalc treats growth as U-shaped (R-01 p. 13; R-33 p. 12; R-05 pp. 37–38).
    Our 80 features are levels only (D-020), so this is not tested; mention it as a known omission.
