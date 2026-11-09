# From zero: the data, the EDA, every feature, and how the model uses them (2026-11-08)

For Kiet (and then Chloe and Desmond). It assumes **no finance and no machine learning**. Each idea is built
from the one before, with one running example firm ("Alfa S.r.l."). Every number is from a file in this repo,
cited inline.

**Reading order (about 3 hours, can be split):**
- Parts 1–2 build the finance base.
- Part 3 is the EDA.
- Part 4 is ML from zero.
- Parts 5–6 explain every feature.
- Parts 7–8 cover what we dropped and how the model uses the rest.
- Part 9 covers Chloe's features and the harness.
- Part 10 is a quiz.

Companion: `docs/2026-11-08_STUDY_GUIDE_calibration.md` (calibration, timeline, holdout rules).

---

## Part 1. The business in one page

**What a bank does.** It lends money now and expects it back with interest. Sometimes the borrower cannot pay.
That is a **default**. In our data it is the firm's `def_date`, the date it missed a payment. When a firm
defaults, the bank loses part of the loan.

**What the bank needs from us.** One number per applicant: the **PD (probability of default)**, the chance the
firm defaults in the next 12 months. The bank uses it twice:
1. **Approve or decline:** is this firm riskier than a cutoff? This needs the right *order* of firms.
2. **Price the loan:** the interest rate must cover the expected loss, and expected loss = PD × loss if default
   × amount. This needs the right *level*.

**What we get to look at.** Only what the firm reports in its **annual financial statements**, plus its sector,
legal form and province. That is it: no credit history, no stock price, no news.

---

## Part 2. Reading a financial statement from zero (meet Alfa S.r.l.)

A financial statement has two main pages. We use a made-up firm, **Alfa S.r.l.**, a manufacturer in the
North-West. Its numbers are chosen so that its features match the worked example the GB-25 model was tested on
(`experiments/gbm/GB-25/doc_examples_output.txt`).

### 2.1 The balance sheet: a photo on 31 Dec

"What the firm owns" on the left, "who paid for it" on the right. **Both sides always add up to the same total.**

| Owns (assets) | € | Paid for by (liabilities + equity) | € |
|---|---|---|---|
| Tangible fixed assets (machines, buildings) `asst_tang_fixed` | 3.0M | Equity (owners' money and kept profits) `eqty_tot` | 2.0M |
| Intangible fixed assets (software, patents) `asst_intang_fixed` | 0.4M | Short-term debts, due within a year `debt_st` | 5.0M |
| Financial fixed assets (stakes in other firms) `asst_fixed_fin` | 0.6M | Long-term debts, due later `debt_lt` | 2.8M |
| Current assets (stock, receivables, cash) `asst_current` | 6.0M | Staff severance fund (TFR) `liab_lt_emp` | 0.2M |
| … of which receivables, what customers owe Alfa `AR` | 2.5M | | |
| … of which cash `cash_and_equiv` | 0.5M | | |
| **Total assets** `asst_tot` | **10.0M** | **Total** | **10.0M** |

(The right side here sums to 10.0M. In real files there is often a small unlisted item; see Part 3.4.)

Inside the debts, the file sometimes splits *who* is owed:
- **Bank debt** `debt_bank_st/lt`: 3.5M here.
- **Other financial lenders** `debt_fin_st/lt`.
- **Suppliers** `AP_st/lt` (accounts payable): 1.8M short-term here.

**Equity is the cushion.** If the assets lose value, the owners lose first and the lenders only after equity is
gone. Negative equity means the cushion is already used up.

### 2.2 The income statement: a video of the year

| Line | € | Plain meaning |
|---|---|---|
| Revenue `rev_operating` | 12.0M | sales |
| − Operating costs `COGS` | 11.4M | everything it cost to run the business (in this file COGS = *all* operating costs, `01_profile.md` §5) |
| = Operating profit `prof_operations` | 0.6M | profit from the business itself |
| EBITDA `ebitda` | 1.2M | operating profit before depreciation, a rough "cash the business throws off" |
| Interest paid `exp_financing` | 0.3M | the cost of the debt |
| Extraordinary items `inc_extraord` | −0.05M | one-offs (asset sales, write-downs) |
| Net profit `profit` | 0.2M | what is left after everything |
| Cash flow proxy `cf_operations` | 0.8M | net profit + depreciation (= 0.2 + (1.2 − 0.6)) |

### 2.3 Why we use ratios instead of raw euros

A €0.3M interest bill is heavy for a €3M firm and trivial for a €300M firm. Dividing by size makes firms
comparable. Total assets is the safest divisor: it is never zero or negative here, because every firm has more
than €1.5M (Appendix B; `01_profile.md` §6).

### 2.4 The five questions a credit officer asks (and the ratio families that answer them)

| Question | Family | Alfa's answer |
|---|---|---|
| How big is the cushion? | **Leverage** | equity / assets = 2.0 / 10 = **20%** |
| Can it pay what is due soon? | **Liquidity** | cash / assets = **5%**; (current assets − short-term debt) / assets = **10%** |
| Does it make money? | **Profitability** | net profit / assets = **2%** |
| Can profits cover interest? | **Coverage** | EBITDA / interest = 1.2 / 0.3 = **4×** |
| Is it collecting and paying on time? | **Activity** | receivables days = 2.5 / 12 × 365 = **76 days** |
| Plus: what it owns, size, industry, place | asset structure, size, categorical | tangible / assets = **30%** |

These families come from Moody's RiskCalc (R-01 Sec. 3.1 p. 13), the best-documented private-firm PD model.

---

## Part 3. The EDA: what is in the file, and what we found

EDA = "exploratory data analysis": look at the data before modelling. Ours is in `reports/data/01_profile.md`,
`02_missingness.md` and `00_labels.md`.

### 3.1 Shape
- **1,023,552 rows; one row = one firm's statement for one year** (a "firm-year"). There are 0 duplicate
  (firm, year) pairs.
- 237,711 firms; fiscal years 2007–2012; every statement ends 31 Dec.
- 101,169 firms appear in all 6 years; 29,963 appear only once (12.6%).
- Firms that appear only 1–3 years default far more often: about 15% of them ever default, against 0.39% of
  the 6-year firms (`01_profile.md` §2). A short history is itself a warning sign, partly because firms that
  get into trouble stop filing.

### 3.2 How often firms default (label, start-6 numbers, v1)
- 0.96% (FY2007) → 1.28% → 1.62% → 1.70% (FY2010). Defaults rose every year through the crisis.
- Under the new 5-month label (D-151) the numbers shift slightly; see `reports/data/2026-11-08_labels_s5.md`
  once it is built.
- **Rare event:** about 1–2 defaulters per 100 firms. This drives many choices: AUC and log loss instead of
  accuracy, and calibration matters a lot.

### 3.3 Who defaults more (simple averages, `01_profile.md` §3)

| Group | Lowest | Highest |
|---|---|---|
| Sector | Electricity 0.68%, Health 0.70%, Hotels/food 0.79%, Real estate 0.92% | Admin services 1.99%, Arts 1.82%, ICT 1.76%, Manufacturing 1.67% |
| Legal form | SAU 0.88%, SPA 0.96% | SRU (single-member S.r.l.) 2.04% |
| Region | Valle d'Aosta 0.69%, Trentino 0.79% | Marche 1.89%, Toscana 1.75%, Campania 1.71% |

Lesson: categories matter, but less than the financial ratios (Part 6).

### 3.4 The file is messy: what we discovered

| Finding | Evidence | What we did |
|---|---|---|
| `goodwill` is **not** goodwill. It behaves like *value added* (EBITDA + staff costs) | ≥ EBITDA on 99.99% of rows; correlation 0.83 with revenue | never called "goodwill"; used only to estimate staff costs (`personnel_to_sales`, D-091) |
| `inc_financing` is a **net** financial result (usually negative) | median −32,164; equals −interest on 39.9% of rows | use `exp_financing` (interest paid) instead |
| `taxes` equals profit *before* tax | identity holds on 99.7% | no "tax" feature (it would duplicate profit) |
| `roe` is cut at ±150% and **blanked when equity is negative** | 72,937 blanks, all extreme or negative equity | missing `roe` = "negative equity", not "unknown". Recompute from profit and equity instead |
| `days_rec` is 72% empty and doesn't match its own definition | matches 365×AR/revenue on 1.2% | rebuilt as `ar_days` |
| `eqty_corp_family_tot` is 100% empty | | dropped |
| `roa` = operating profit / assets (not net profit) | exact on 100% | we compute both versions ourselves |
| Assets ≠ sum of the four components on about 20% of rows | 79.3% within 1% | an unlisted item (accruals); `bs_gap` measures it |

### 3.5 Missing values: the bank/supplier debt split
- The six "who is owed" fields (`debt_bank_st`, `debt_fin_st`, `AP_st` and their `_lt` versions) are missing
  **together as a block** on about 12% of rows. The totals `debt_st`/`debt_lt` are almost always present.
- Small firms are missing the block more often: 15% at €1.5–2M of assets vs 1.3% above €50M. This fits Italy's
  "abbreviated statement" option for small firms (R-27), but that is a hypothesis.
- If the block was missing last year, it is missing this year 57% of the time (persistent).
- Block missing doubles in 2008–2010 and falls in 2011: partly a data-collection effect, so the holdout may differ.
- Firms missing it default slightly more: 1.57% vs 1.38%.
- Everything else is missing on fewer than 0.1% of rows (`02_missingness.md` §1).

**Lesson for the deck: "missing" is often information.** A blank `roe` means negative equity; a blank debt block
means a small firm that filed a short statement.

### 3.6 Sign oddities (things that "can't" be negative)
- Revenue is negative on 3,270 rows; those rows default at 7.0% (`01_profile.md` §6).
- Equity is negative on 40,748 rows (4%), with a **10.35% default rate**. That is real distress, not an error.
- |ROA| above 100% on 878 rows; 36.7% of them default.

### 3.7 Selection: who is in the file
- Only firms above €1.5M of assets, checked each year. A shrinking firm can drop out *before* it defaults, so
  some pre-default statements are missing. That is one reason 7,700 of 16,863 defaulting firms never get a
  "defaulted" row (`01_profile.md` §6).
- The file looks like a broad sample of Italian companies, not one bank's loan book (hypothesis, §7).

---

## Part 4. Machine learning from zero (just what this project needs)

### 4.1 Features, label, model
- **Label (y):** 1 if the firm defaulted in its 12-month window, else 0.
- **Features (X):** the numbers we feed in, e.g. Alfa's 20% equity ratio.
- **Model:** a recipe that turns X into a PD. **Training** means choosing the recipe's settings so its PDs match
  the labels on past data.

### 4.2 Two scores we judge a model by (tiny examples)

**AUC (ranking).** Pick one defaulter and one survivor at random. AUC is the chance the defaulter got the higher
PD. 0.5 is a coin flip; 1.0 is perfect.

Example: defaulters have PDs {8%, 3%}, survivors {5%, 1%, 0.5%}. That gives 6 pairs:
- 8% beats all 3 survivors.
- 3% beats 1% and 0.5%.
- 5 of 6 pairs are ranked correctly: **AUC = 0.83**.

Our models score about 0.83–0.85.

**Log loss (honesty of the probability).** The penalty is −ln(the probability you gave to what actually happened):

| Your PD | Firm survived | Firm defaulted |
|---|---|---|
| 2% | −ln(0.98) = 0.020 (tiny) | −ln(0.02) = 3.91 (big) |
| 0.03% | 0.0003 | 8.11 (huge) |

Average over all firms; lower is better. It punishes confident mistakes, so it rewards correct *levels*.
That is why it is our main probability metric (D-006).

### 4.3 Overfitting, and why we split by time
A model can memorise last year's firms instead of learning general patterns: great on training data, worse on
new data. So we always score on **later years the model never saw** (walk-forward, D-015). The calibration
guide explains the train / validation / test / lockbox / holdout roles (Part 1.3 there).

### 4.4 A decision tree in one picture

```
            equity_ratio < 0.05 ?
              /              \
           yes                no
   PD ≈ 6% (thin cushion)   roa_net < -0.02 ?
                              /        \
                           yes          no
                       PD ≈ 3%       PD ≈ 0.7%
```

A tree asks yes/no questions and gives each "leaf" the default rate of the training firms that landed there.
It only cares about the **order** of values (is it below 0.05?), so dividing a ratio by 2 or taking its log
changes nothing for a tree.

### 4.5 Gradient boosting (LightGBM): many small trees, each fixing the last one's mistakes
Work in log-odds (calibration guide, Part 11):
1. **Start:** everyone gets the average, e.g. 1.3% → log-odds −4.33.
2. **Tree 1** looks at who was under-predicted (defaulters with thin equity) and adds, say, +0.4 to their log-odds.
3. **Tree 2** looks at the remaining errors and adds a small push for low-profit firms, and so on.
4. After a few hundred tiny trees, each firm's log-odds = start + sum of its pushes. Convert to a PD.

GB-25 uses trees with at most 8 leaves and depth 3, learning rate 0.03, and 461–576 trees per fold
(`experiments/gbm/GB-25/train_log.md`). Small trees and many of them is the standard way to avoid memorising.

### 4.6 Monotone constraints: forcing common sense
Left alone, a model might learn "more equity → higher PD" in some odd corner of the data. A credit officer would
reject that. We **force** each feature's direction from finance logic: equity ratio −, short-term debt +, and
so on (the "expected sign" in Part 6). The cost, measured against the same model without constraints (GB-02): AUC −0.0016 pooled and −0.0054 on 2010
(`r2_scorecard.md` §1.4). The benefit is a model nobody can catch contradicting finance (D-051, D-126).

### 4.7 SHAP: why did this firm get this PD?
SHAP splits one firm's log-odds into "average + push from each feature". For Alfa (fold-3 model;
`doc_examples_output.txt`):

```
average log-odds                      −5.245
tangible_ratio (30% hard assets)      −0.304   lowers risk
roa_net (2% profit)                   −0.181   lowers risk
interest_coverage_sect_pct            −0.145   good coverage for its sector
province_te (province's history)      +0.133   riskier-than-average province
payables_to_sales (15%)               −0.120
asset_turnover (1.2)                  +0.119
… (all others)                         …
= Alfa's log-odds                     −5.884 → raw PD 0.28% → after calibration 0.37%
```

This is the "top drivers per firm" Roger asks for (CLAUDE.md rule 8).

---

## Part 5. The tools that turn raw columns into features

| Tool | What it does | Tiny example | Why |
|---|---|---|---|
| **Ratio to assets** | divide by `asst_tot` | equity 2.0M / 10M = 0.20 | comparable across sizes, never divides by zero |
| **Safe division rule** | decides what happens when the bottom of a ratio is 0 or negative | Alfa with 0 interest: coverage → "+infinity", clipped to the top bound | a firm with no debt is *safe*, not "missing" |
| **slog** (signed log) | sign(x) · ln(1 + \|x\|) | coverage 4 → ln 5 = 1.609; −4 → −1.609 | squeezes giant values; keeps order and sign (D-019) |
| **Winsorise** | clip to the 1st/99th percentile of the *training* years | equity ratio −3.0 → −0.081 (fold-3 bound) | one crazy firm cannot dominate a linear model |
| **Identity rebuild** | refill a blank from an exact accounting rule | blank `rev_operating` = `prof_operations` + `COGS` (88% recoverable) | use real numbers before guessing |
| **Missing flag** | 1 if the debt block is missing | `debt_split_missing` | lets the model price "blank" itself |
| **Native NaN (trees)** | LightGBM sends blanks down their own branch | | no guessing needed for trees (D-056) |
| **Sector percentile** | where the firm stands among its sector's firms (0–1) | Alfa's 20% equity is the **43rd** percentile in manufacturing | 20% equity means different things in real estate and in retail (R-01 Sec. 3.3) |
| **Shrunk default-rate encoding** | the category's own default history, pulled toward average when the category is small | below | turns 83 sectors / 110 provinces into one number each without overfitting |
| **Categories as labels** | ATECO sector, legal form, macro-area stay categories | code 15 = Milano, never "15 > 1" | Roger's rule: label variables are never numeric |

**The shrunk encoding, with numbers** (`ateco_div_te`, D-094). Formula:
`enc = logit((defaults + m·p0) / (firms + m)) − logit(p0)`, with p0 = 1.3% (training average) and m = 555
(fitted strength).
- **Big sector:** 50 defaults in 2,000 firms (raw 2.5%). (50 + 7.2) / 2,555 = 2.24% → enc = **+0.55**.
- **Tiny sector:** 3 defaults in 60 firms (raw 5%). (3 + 7.2) / 615 = 1.66% → enc = **+0.25**.

The tiny sector's scary 5% is mostly noise, so it is pulled toward average. The encoding is computed
**out-of-fold**, so a firm's own label never feeds its own value. The referee checked this by flipping labels
(`r2_scorecard.md` §2).

**Everything is fitted on training years only, then frozen** in a parameter file the harness reads (Roger's
harness rule; CLAUDE.md §2.2). That covers winsor bounds, medians, percentile grids and encodings.

---

## Part 6. Every feature in our model (GB-25 uses all 45 of `full2`)

How to read each row:
- **Sign:** what should happen to PD when the feature rises (− = safer, + = riskier, none = no forced direction).
- **AUC:** the feature alone, on training years 2007–09 (< 0.5 means a higher value is safer).
- **D1→D10:** default rate in the lowest vs highest 10% of firms (`reports/features/univariate.md` §1–2).

Alfa's value is in brackets.

### 6.1 Leverage: how big is the cushion?
| Feature | Plain words | Formula | Sign | AUC | D1→D10 |
|---|---|---|---|---|---|
| `equity_ratio` [0.20] | owners' share of assets | eqty_tot / asst_tot | − | 0.267 | 4.94% → 0.49% |
| `neg_equity` [0] | cushion already gone | 1 if equity < 0 | + | 0.624 | 9.74% if 1 vs 0.97% if 0 |
| `st_debt_ratio` [0.50] | bills due within a year, per € of assets | debt_st / asst_tot | + | 0.706 | 0.45% → 4.33% |
| `bank_debt_ratio` [0.35] | bank loans per € of assets | (bank st + lt) / assets | + | 0.579 | 0.98% → 2.20% |
| `st_debt_share` [0.64] | share of debt due soon | debt_st / (debt_st + debt_lt) | + | 0.570 | 0.51% → about 1.5% |
| `nonbank_fin_debt` [0] | leasing, factoring, shareholder loans | (fin st + lt) / assets | + | 0.523 | |
| `ap_lt_ratio` [0] | supplier debt pushed beyond a year (often renegotiated) | AP_lt / assets | + | 0.501 | |

### 6.2 Liquidity: can it pay what is due soon?
| Feature | Plain words | Formula | Sign | AUC | D1→D10 |
|---|---|---|---|---|---|
| `wc_ratio` [0.10] | current assets left after paying short-term debt | (asst_current − debt_st) / assets | − | 0.384 | 3.15% → 1.00% |
| `current_ratio` [slog 1.2 = 0.79] | current assets per € due soon | slog(asst_current / debt_st) | − | 0.383 | 1.54% → 0.72% |
| `cash_ratio` [0.05] | cash per € of assets | cash / assets | − | 0.441 | 1.55% → 0.83% |

### 6.3 Profitability: does it make money?
| Feature | Plain words | Formula | Sign | AUC | D1→D10 |
|---|---|---|---|---|---|
| `roa_net` [0.02] | net profit per € of assets | profit / assets | − | 0.256 | **5.90%** → 0.53% |
| `roa_oper` [0.06] | operating profit per € of assets | prof_operations / assets | − | 0.283 | 5.61% → 0.51% |
| `ebitda_margin` [slog 0.10] | cash-like profit per € of sales | slog(ebitda / revenue) | − | 0.279 | 5.08% → 0.47% |
| `operating_margin` [slog 0.05] | operating profit per € of sales (RiskCalc France input) | slog(prof_operations / revenue) | − | 0.291 | |
| `extraord_to_assets` [−0.005] | one-off gains or losses | inc_extraord / assets | none (both ends risky) | 0.462 | **2.80%** ← 1.10% mid → **2.01%** |

### 6.4 Coverage: can profits pay the interest?
| Feature | Plain words | Formula | Sign | AUC | D1→D10 |
|---|---|---|---|---|---|
| `interest_coverage` [slog 4 = 1.61] | how many times EBITDA covers interest | slog(ebitda / exp_financing) | − | 0.276 | 4.86% → 0.56% |
| `cfo_to_debt` [0.10] | cash flow per € of debt (≈ 1 / years to repay) | cf_operations / (debt_st + debt_lt) | − | **0.248 (strongest)** | 5.75% → 0.41% |
| `interest_to_assets` [0.03] | interest bill per € of assets | exp_financing / assets | + | 0.634 | 0.99% → 3.54% |
| `interest_to_sales` [slog 0.025] | share of each sales € spent on interest | slog(exp_financing / revenue) | + (restored in GB-25) | 0.580 | hump-shaped in 2007–09 (D-092, D-130) |

### 6.5 Activity: is cash coming in and going out on time?
| Feature | Plain words | Formula | Sign | AUC | D1→D10 |
|---|---|---|---|---|---|
| `ar_days` [slog 76 days] | days customers take to pay | slog(365 · AR / revenue) | + | 0.583 | 0.87% → 1.84% |
| `ap_days` [slog 58 days] | days Alfa takes to pay suppliers | slog(365 · AP_st / COGS) | + | 0.591 | 0.98% → 1.93% |
| `payables_to_sales` [slog 0.15] | supplier debt per € of sales (stretching suppliers = stress) | slog(AP_st / revenue) | + | 0.636 | |
| `personnel_to_sales` [slog 0.125] | staff costs per € of sales (hard to cut fast) | slog((goodwill − ebitda) / revenue), hypothesis-dependent | + | 0.570 | |
| `asset_turnover` [1.2] | sales per € of assets | revenue / assets | none (inverted U) | 0.504 | 1.48%, dips to 0.72%, peaks 1.71% |
| `rev_nonpositive` [0] | no (or negative) sales | 1 if revenue ≤ 0 | none | 0.507 | 1.86% vs 1.28% |

### 6.6 Asset structure and size: what does it own, how big is it?
| Feature | Plain words | Formula | Sign | AUC | D1→D10 |
|---|---|---|---|---|---|
| `tangible_ratio` [0.30] | hard assets a lender can take as collateral | asst_tang_fixed / assets | − | 0.384 | 1.75% → **0.37%** |
| `intangible_ratio` [0.04] | soft assets worth little in a crisis | asst_intang_fixed / assets | + | 0.550 | → 2.27% |
| `fin_fixed_ratio` [0.06] | stakes in other firms (holding structure) | asst_fixed_fin / assets | none | 0.503 | |
| `tfr_ratio` [0.02] | staff severance fund (labour intensity) | liab_lt_emp / assets | none | 0.510 | |
| `log_assets` [ln 10M = 16.12] | size | ln(asst_tot) | − | 0.477 | 1.57% → 1.08% |

### 6.7 Peer-relative versions ("compared with its own sector")
`equity_ratio_sect_pct` [0.43], `roa_net_sect_pct` [0.68], `interest_coverage_sect_pct` [0.48],
`cash_ratio_sect_pct` [0.65], `st_debt_ratio_sect_pct` [0.47], `tangible_ratio_sect_pct` [0.62].
- Each is the firm's percentile (0–1) within its ATECO section, with the same sign as its base ratio.
- They are highly correlated with their base (rho 0.88–0.98), which is fine for trees (D-093).

### 6.8 Category features
| Feature | Plain words | How it enters | Evidence |
|---|---|---|---|
| `ateco_group` [C, manufacturing] | sector (21 sections, small ones pooled to OTHER) | category | 0.68%–1.99% by section |
| `legal_group` [SRL] | legal form | category | SRU 2.04% vs SPA 0.96% |
| `macro_area` [North-West] | 5 macro-regions | category | 1.30%–1.62% |
| `ateco_div_te` [+0.025] | the 2-digit sector's shrunk default history | number (Part 5) | AUC 0.580 |
| `province_te` [+0.18] | the province's shrunk default history | number | AUC 0.547 |
| `region_te` [−0.01] | the region's shrunk default history | number | AUC 0.530 |

### 6.9 Data-quality features
| Feature | Plain words | Sign |
|---|---|---|
| `debt_split_missing` [0] | the bank/supplier block is blank | + (weak: AUC 0.505) |
| `n_core_imputed` [0] | how many core inputs were filled by a sector median | + |
| `bs_gap` [0] | share of assets not explained by the listed liabilities | none |

### 6.10 What actually drives GB-25 (mean |SHAP| on 2010, `shap_all_features_2010.csv`)
1. `tangible_ratio` (13.3% of all pushes)
2. `asset_turnover` (6.1%)
3. `st_debt_ratio` (5.4%)
4. `interest_coverage_sect_pct` (5.3%)
5. `equity_ratio_sect_pct` (5.2%)
6. `roa_net` (5.1%)
7. `ar_days` (4.3%)
8. `tfr_ratio` (4.1%)
9. `payables_to_sales` (4.1%)
10. `roa_net_sect_pct` (3.5%)

**Caution:** correlated features share credit. Equity appears three times (ratio, flag, sector percentile), so
report **families**, not single ratios (FEATURE_SPEC). Family view: leverage + profitability + coverage together
dominate, and asset structure is the single biggest one.

---

## Part 7. What we built and threw away, and why (Roger wants "what didn't work")

| Rejected | Why | Where |
|---|---|---|
| tax / assets | `taxes` is profit before tax, so it duplicates roa_net (rho 0.96) | FEATURE_SPEC |
| ROE (file or recomputed) | blank or explodes when equity ≤ 0; replaced by equity_ratio + neg_equity + roa_net | D-024 |
| days_rec (file) | 72% empty, wrong definition; replaced by ar_days | D-021 |
| net financial result / assets | field meaning only hypothesised; interest paid is unambiguous | D-024 |
| "goodwill" / assets | the field is not goodwill; a banker can't interpret it | D-024 |
| log revenue | a second size measure; size enters once | D-024 |
| net_debt_coverage | duplicate of cfo_to_debt (rho 0.99) | D-099 |
| n_missing_raw | duplicate of debt_split_missing (rho 0.994) | D-099 |
| rounding flag | 503 rows, 2 defaults: nothing to learn | D-099 |
| growth / last-year changes | need the prior statement (Part 9); +0.002 AUC only | D-020 |
| text / news, macro data | none in the file; external data not cleared | D-003, D-004 |
| raw province (110) / division (83) dummies | too thin (45 provinces with < 30 defaults); replaced by shrunk encodings | D-023, D-094 |

---

## Part 8. How the pieces fit (one picture)

```
raw row (44 columns, may be messy)
  → clean: unify blanks ("NA", "", "n.d." …), rebuild from identities          (D-014, D-021)
  → 45 features: ratios, slog, winsor bounds, percentiles, encodings — all FROZEN from training years
  → LightGBM GB-25: about 500 small trees, 8 leaves each, monotone directions forced  (D-050, D-130)
  → raw PD → calibration (slope + prior shift to τ) → floor 0.03%, cap 99%        (calibration guide)
  → one PD per row, same order as the input                                      (harness)
```

Headline (2010, cleanest year): GB-25 AUC 0.835 [0.827, 0.842]; with calibration, obs/pred 1.19
(`r2_scorecard.md` §0, §1.2).

---

## Part 9. Chloe's three "last-year" features and the harness, plainly

Her `annualized_sales_growth`, `annualized_change_in_roa` and `change_in_ar_turnover` compare **this year's
statement with the same firm's previous one**, e.g. "sales grew 10% since last year". To compute them you need
two rows of the same firm.

**In training: yes, fine.** All training rows are historical. When the bank scores a FY2009 statement, it already
holds the firm's FY2008 statement. Nothing comes from the future.

**In the harness: it depends where last year's row comes from.** The assignment (p. 5) says the prediction
code may use no test data "besides the current record" (calibration guide, Part 17):

| Situation in Roger's holdout | Can we build the growth feature? |
|---|---|
| Firm seen in training, and its previous-year statement is **in our training file** (e.g. holdout FY2013, previous FY2012 is in train) | **Probably yes**, via a frozen lookup file of training statements. Assignment p. 5 allows shipping a training data file ("e.g., for KNN"). Still email question 4; our current answer is "no" to be safe (D-020) |
| Firm seen in training, but its previous year is **another holdout row** (e.g. holdout has FY2013 and FY2014; scoring FY2014) | **No.** That would use another holdout row |
| **New firm** (not in training) | **No.** There is no previous statement we are allowed to use |

So you are right with one refinement: **the rule is "one *holdout* row at a time", not "one row of data at a time"**.
Rows from our own training file may be looked up; other holdout rows may not.

**Why it is still risky even when allowed: train/serve mismatch.**
- In training, the previous year exists for 81.5% of FY2008 rows, 87.0% of FY2009 and 88.0% of FY2010 (panel
  count, 2026-11-08). The model learns to lean on growth for nearly everyone.
- In the holdout, new firms and multi-year holdout firms would all get blanks.
- If many rows are blank, the model sees "growth missing" far more often than it ever did in training. Its PDs
  for those rows rest on a pattern it rarely saw.
- **How to make it safe if we want it:**
  1. Build the lookup only from the training file.
  2. In the walk-forward, compute test-year growth only from training-window rows. This already holds for her
     temporal splits.
  3. Report new-firm vs seen-firm results separately.
- **Is it worth it?** Our test found +0.002 AUC for seen firms and nothing for new firms (D-020).

**Her current code** builds these features from a `history.csv` that contains the very rows being scored. That is
fine inside her walk-forward, because last year is a training year there. In a production harness it would need
rewiring to the frozen training lookup.

---

## Part 10. Check yourself

1. Why divide by total assets, not by equity? *(Assets are always ≥ €1.5M; equity can be 0 or negative, which
   makes the ratio explode or flip sign.)*
2. A firm has no interest expense. Is its interest coverage "missing"? *(No: it owes no interest, so the
   coverage is effectively infinite, clipped to the top bound. Safe, not unknown.)*
3. Why is a blank `roe` a danger sign? *(The provider blanks it exactly when equity is negative or |ROE| ≥ 150%.)*
4. Why does a tree not care about slog, but a logit does? *(Trees use only the order of values; a linear model
   uses the distances.)*
5. Why shrink a small sector's default rate toward the average? *(3 defaults in 60 firms is mostly luck.)*
6. Why can't the harness use the firm's other holdout rows? *(Assignment p. 5: "besides the current record".)*
7. Name the strongest single feature and say why it makes sense. *(`cfo_to_debt`, AUC 0.248: cash generated per €
   of debt, i.e. how fast the firm could repay from its own earnings.)*
8. Alfa's biggest risk-reducing driver was `tangible_ratio`. Explain it to a banker. *(30% of assets are machines
   and buildings: collateral that keeps value, and a sign of a real productive business.)*
