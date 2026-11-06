# Monotone constraints: what they do and where each sign comes from

Audit of the 36 constrained features in GB-25 (`experiments/gbm/GB-25/config.json`; 45 features, 9 free).
Prepared by Claude Code for KN on 2026-11-03. **Open each page and confirm it yourself before citing.**
No sign, code or decision was changed by this audit.

## 1. Sources used

| ID | Document | Where to read it |
|---|---|---|
| R-01 | Dwyer, Kocagil & Stein (2004), *RiskCalc v3.1 Model* (Roger is a co-author) | rogermstein.com/wp-content/uploads/RiskCalc-v3-1-Model.pdf |
| R-05 | Falkenstein, Boral & Carty (2000), *RiskCalc for Private Companies* | efalken.com/papers/riskcalc.pdf |
| R-33 | Buitrago, Makarov, Zhao & Dwyer (2015), *RiskCalc 4.0 France* | `docs/RiskCalc_40_France_FINAL.pdf` |

Limits: R-05's exhibits are US public firms at a 5-year horizon. R-33 is French, and its curves show each
ratio "in isolation" (p. 12). None of the three is Italian.

## 2. Why constrain: the sourced argument

- Assignment (R-30, p. 3): graders "examine model robustness and face validity (i.e., do the variables and
  parameters of the model make sense)".
- R-01 p. 14: "if an increase in leverage resulted in a decreased default risk for a corporation, most users
  would find the implication counter-intuitive."
- R-05 p. 28: "Good variables for inclusion in a model tend to be those variables that have a conditionally
  monotone relationship with default probabilities"; "These are guidelines rather than rules."
- R-05 p. 37: "Monotonicity is a good thing in modeling, as it implies a stable and real relationship, not
  just an accident of the sample."

**Not in these sources:** a hard constraint on every ratio inside a boosted tree model. That is our design
choice, and D-051 records it as one. The sources describe choosing variables and checking curves.

**Our own result runs against the stability argument.** Out of time, the unconstrained twin GB-02 beat GB-01
by +0.0065 AUC [+0.0052, +0.0079] (D-051). So the case for constraints is explainability, not accuracy.

## 3. Sign table

Tiers:
- **A, direct:** a source names this ratio or a close variant and gives the direction.
- **B, family:** the source gives the direction for the family only; applying it to this ratio is our inference.
- **C, disputed or data only:** a source argues against it, or only our data supports it.
- **D, no real support:** assumption, with almost no signal in our data.
- **M, mechanical:** the direction follows from how the feature is built.

D1 and D10 are default rates in the lowest and highest deciles, 2007-09 training rows
(`reports/features/deciles.csv`, `round2_univariate_stats.csv`).

| Feature | Sign | Tier | Source for the direction | D1 to D10 | Note |
|---|---|---|---|---|---|
| equity_ratio | − | A | R-33 p. 12 (Equity/Assets) | 4.94 to 0.49 | |
| cash_ratio | − | A | R-33 p. 12; R-01 p. 13 | 1.55 to 0.83 | R-05 p. 36: most important single variable for private firms |
| log_assets | − | A | R-33 p. 12; R-01 p. 13 | 1.57 to 1.08 | R-01 p. 13: size effect declined next to cash flow; D-022: coefficient about 0 |
| current_ratio | − | A | R-01 p. 13 (named under liquidity) | 1.54, 2.57 ... 0.72 | D1-D2 break. Do not cite R-05 Exh. 4.6: its line is labelled as rising, against the text |
| wc_ratio | − | A | R-05 Exh. 4.6 p. 36 | 3.15 to 1.00 | Public firms |
| roa_net | − | A | R-01 p. 13 (net income group); p. 14 (ROA) | 5.90 to 0.53 | |
| roa_oper | − | A | R-01 p. 13 (operating profit / assets) | 5.61 to 0.51 | |
| ebitda_margin | − | A | R-01 p. 13 (EBITDA / sales) | 5.08 to 0.47 | GB-08b would free it (D-103, OPEN) |
| operating_margin | − | A | R-33 p. 12 | 4.53 to 0.55 | |
| cfo_to_debt | − | A | R-33 p. 12 (close variant) | 5.75 to 0.41 | R-33 denominator is liabilities minus cash |
| interest_to_sales | + | A | R-33 p. 12 | 0.93 to 1.43 | **Our data are hump-shaped** (peak 2.20 in D8). The sign is from R-33 alone; costs -0.0021 AUC on 2010 (D-130) |
| payables_to_sales | + | A | R-33 p. 11 Table 2; p. 12 | 0.79 to 2.43 | Code and D-092 say "p. 10": wrong, Table 2 is on p. 11 |
| personnel_to_sales | + | A | R-33 p. 12 | D2 0.96 to 2.58 | The field itself is a hypothesis (D-091) |
| equity_ratio_sect_pct | − | A, inherited | as equity_ratio | 5.32 to 0.52 | Peer-rank version: our inference |
| roa_net_sect_pct | − | A, inherited | as roa_net | 5.92 to 0.58 | Same |
| cash_ratio_sect_pct | − | A, inherited | as cash_ratio | 1.67 to 0.86 | Same |
| st_debt_ratio | + | B | R-01 p. 13 leverage family | 0.45 to 4.33 | Short-term liabilities are a subset of liabilities |
| bank_debt_ratio | + | B | R-01 p. 13 leverage family | 0.98 to 2.20 | D1 > D2. R-05 p. 34: debt/assets does worse than L/A for private firms |
| nonbank_fin_debt | + | B | R-01 p. 13 leverage family | D8 1.54 to 1.64 | Code says `src="assumption"` |
| neg_equity | + | B | Complement of equity/assets, R-33 p. 12 | 9.74 vs 0.97 | D-021; 0 splits in GB-25 (D-130) |
| interest_coverage | − | B | R-33 p. 11 Table 2 (debt coverage) | 4.86 to 0.56 | **R-05 p. 34: at very low coverage "default risk actually decreases"** |
| interest_to_assets | + | B | No named source; reads as interest burden x leverage | 0.99 to 3.54 | Weakest of tier B |
| st_debt_ratio_sect_pct | + | B, inherited | as st_debt_ratio | 0.57 to 4.42 | |
| interest_coverage_sect_pct | − | B, inherited | as interest_coverage | 5.10 to 0.58 | |
| ar_days | + | C | None for the direction | 0.87 to 1.84 | **R-05 Exh. 4.7 p. 37: AR/Sales is U-shaped; "others are ambiguous"** |
| ap_days | + | C | R-33 for AP/Sales only | 0.98 to 1.93 | **R-05 Exh. 4.7: AP/COGS dips, then rises** |
| st_debt_share | + | C | None | 0.51, 1.68 peak, 1.53 | **R-05 p. 36: "of little use in forecasting"**, less so for private firms |
| tangible_ratio | − | C | None for PD | 1.75 to 0.37 | REFERENCES Contradiction 11: R-48 supports recovery, not lower PD |
| intangible_ratio | + | C | None | D2 1.24 to 2.27 | **R-05 p. 34: "Intangibles appear really to be worth something"**, against the reason written in `build.py` |
| tangible_ratio_sect_pct | − | C, inherited | as tangible_ratio | 2.10 to 0.77 | |
| ap_lt_ratio | + | D | Assumption | AUC 0.501 | No signal |
| debt_split_missing | + | D | Our data (D-016) | AUC 0.505 | 0 splits in GB-25 (D-130) |
| n_core_imputed | + | D | R-34, a web article with no numbers | AUC 0.505 | 0 splits in GB-25 (D-130) |
| ateco_div_te | + | M | Built from default rates | 0.79 to 2.22 | The rising shape is imposed, not evidence (D-124) |
| province_te | + | M | Built from default rates | 1.01 to 1.73 | Same |
| region_te | + | M | Built from default rates | 1.05 to 1.65 | Same |

**Count:** A 16 (3 inherited), B 8 (2 inherited), C 6 (1 inherited), D 3, M 3. Total 36.

Support for leaving signs free: R-05 p. 37 calls sales/assets "non-monotonic and very flat", which backs
`asset_turnover` = none and could be added to D-022.

## 4. Three weak spots a reviewer could press

1. **Our signs are conditional; the sources' curves are not.** A LightGBM constraint means "holding every other
   feature fixed". R-33's curves are univariate. D-051 found the unconstrained model using wc_ratio,
   bank_debt_ratio and interest_coverage against sign once the other ratios were in, so the univariate
   sources do not settle those cases.
2. **Data-backed signs came from 2007-09.** Those years include the fold-1 and fold-2 test years, so tier C
   signs should be described as data-derived and not presented as prior knowledge.
3. **Family logic can hide a bad reason.** A sign can be fine while the written reason is unsupported, as
   with intangible_ratio.

## 5. Sources not yet usable

These were located only, or seen through secondary summaries. **Do not cite them until a team member has read them.**

| Source | Why it matters | Status |
|---|---|---|
| Beaver (1966), *Journal of Accounting Research* 4, 71-111 | Liquid-asset "reservoir" framework: the closest thing to a first-principles reason for the signs | Secondary summary only |
| Ohlson (1980), R-12 | Logit with size, leverage, working capital and income ratios | R-12 is LOCATED; signs seen only in secondary sources |
| Altman, Marco & Varetto (1994), *JBF* 18(3) | Italian firms, Centrale dei Bilanci data | Abstract level; ratios not seen |
| Altman & Sabato (2007), *Abacus* 43(3) | SME logit | Variables not seen |
| Diamond (1991), *QJE* | Maturity and liquidity risk, possible basis for st_debt_share | Abstract only |
| Finance textbooks | | **None read. The search returned only study notes and blogs, so no chapter is confirmed** |

## 6. Runnable example

`experiments/gbm/monotone_demo.py` (synthetic data; run with the `datascience` conda env, LightGBM 4.7.0).
Only equity matters; `wc` is a noisy twin of it.

```
defaults: 966 of 60000 (1.61%)
unconstrained      test AUC 0.6839 | wc curve: upward steps 10 of 20, PD at wc=-2 0.242%, at wc=+2 1.389%
constrained (-,-)  test AUC 0.7208 | wc curve: upward steps 0 of 20, PD at wc=-2 1.954%, at wc=+2 1.223%
one firm, constrained: any PD rise as wc rises? False
```

The constraint raises AUC here because this sample is small and the true signs are known. On our real data
it lowers AUC (section 2). The demo shows the mechanism, not the size of the benefit.
