# Feature consistency audit: training pipeline vs harness vs reference library

Date: 2026-11-20. Audited code: commit **478b058** (branch `final-consolidation`), read with `git show`, never
from the working tree. Provenance commit of the frozen bundle: **4e0ba3a** (= 6bf455a~1; proven below).
Dynamic test: `src/audit/2026-11-20_train_harness_parity.py` → `reports/features/2026-11-20_parity_by_feature.csv`
and `reports/features/2026-11-20_parity_summary.csv`. Every number in §7 comes from those two files; numbers
quoted from earlier work cite their file (`reports/data/2026-11-16_audit/...`).

Three implementations were compared:

| Label | Owner | Files (at 478b058) | Role |
|---|---|---|---|
| **Training path** | Chloe / Desmond | `src/finance_project/data_processing/preprocess.py`, `config/feature_schema.json` (1.3.0), `notebooks/features/01_flag_anomalies.ipynb`, `02_missing_values.ipynb` (export cells are `exec`'d by `preprocess.py`), docs `docs/project-update/features/01..05` | Builds model features for training |
| **Harness, Chloe** | Chloe | `tests/harness_submission/` (schema 1.2.1, Method G on) | Original submission bundle |
| **Harness, Kiet** | Kiet | `harness/2026-11-16_turquoise_submission/` (schema 1.2.1, Method G off; D-152) | Bundle sent for the TA test run |
| Reference | Kiet | `src/features/build.py` ("full2", rounds 1-2) | Reference only; not used by the submitted model |

---

## 1. Executive summary

**Bottom line for the 4 PM meeting.** The submitted harness reproduces the training arithmetic exactly. With the
same frozen parameters and no prior-year rows, all 80 features match on every row (§7). The frozen
`preprocessing.json` is exactly what the bundle-era code (4e0ba3a) fits on the 476,183 labelled FY2007-09 rows
(medians to 6e-16, target encoding and categories exactly). **The inconsistency is between the current training
code (schema 1.3.0, Desmond's 6bf455a) and the frozen bundle (schema 1.2.1).** The two now define `roe`, the
small-industry imputation and Method G differently, and the current code cannot load the frozen artifact. A model
refit with today's `preprocess.py` cannot be scored by today's harness, and the frozen model cannot be
reproduced from today's code. Most of the other findings are documentation and face-validity issues, not
arithmetic bugs.

| # | Mismatch / issue | Severity | Evidence | Recommended fix | Owner |
|---|---|---|---|---|---|
| M1 | **Schema split.** Training code is at schema 1.3.0 (`roe` recomputed, new `roe_negative`, `roe` no longer imputed). Both harnesses and the frozen model are at 1.2.1. `preprocess.transform_with_preprocessor` at 478b058 raises "artifact feature schema does not match" on the frozen artifact | **HIGH** | §2a, §7 (smoke test, T_pin vs H_kiet) | Pick ONE today. (a) Submit the 1.2.1 bundle and revert/branch 6bf455a for the final refit, or (b) refit on 1.3.0 and rebuild the harness from the refit (`missing_rules.py` = new notebook export, `config/feature_schema.json` 1.3.0, new `preprocessing.json`). Never mix the two | Kiet decides; Desmond + Chloe execute |
| M2 | **`roe` definition.** The frozen model saw `roe` only in [−150, 149.93] (provider field, blanks filled with the industry median). Under 1.3.0, `roe` equals the provider value wherever it exists (100% within 0.01). It differs only on the 6.83% of FY2011 rows where the provider left it blank, which are the distressed firms (median gap 214.7 points; 4.85% of rows above \|150\|, max 48,706). A 1.3.0 refit is internally consistent, but scoring 1.3.0 `roe` with the 1.2.1 model is not | **HIGH** (only if the versions are mixed) | §7 `T_pin:roe_*`; `lightgbm.txt` feature_infos | Same as M1. If 1.3.0 is kept, cap `roe` or replace it with `profit/asst_tot` + `neg_equity` (team doc 02 method I's own earlier advice) | Desmond |
| M3 | **`roe_negative` is misnamed.** It is 1 when profit < 0 AND equity < 0, i.e. when the computed ROE is *positive*. A credit officer reading "roe_negative = 0" for a loss-making insolvent firm will be misled | MED | `notebooks/features/02_missing_values.ipynb` (478b058) method I | Rename to `loss_and_neg_equity` (or `roe_sign_misleading`). Expected sign + | Desmond |
| M4 | **Method G (previous-year zero).** The frozen model was trained WITH G (training rows read the firm's previous statement from `history.csv`). Kiet's harness has G OFF (compliant). Chloe's harness has G ON and reads other rows of the test batch, which breaks Appendix A. The 1.3.0 training code removed G entirely | MED (compliance HIGH for Chloe's bundle; numerically tiny) | §7: G fired on 279 of 476,183 training anchors (0.059%) and would fire on 2 FY2011 rows; Chloe's harness changes those PDs (max 0.00069) when the FY2010 row is in the batch, Kiet's does not | Submit only Kiet's patched bundle (or Chloe's with G removed). After a 1.3.0 refit, G is gone on both sides and the skew disappears | Chloe (sign off D-152) |
| M5 | **"3 smallest industry groups use the overall median"** is implemented in the 1.3.0 training code (notebook export), not in the harness. It applies to sections O, P, U (13, 266, 57 firms) | LOW (137 training rows, 19 FY2011 rows) | §7 `small_group_fallback:*` | Comes for free with the M1 rebuild. Document that it affects imputation only, not target encoding | Desmond |
| M6 | **Unsafe ratio edge cases** (D-154 items 1-3 still open): `lt_debt_to_capital` with a negative denominator (11,636 rows, DR 18.3%) is pushed to low PD by its +1 constraint. A negative revenue sends `AR_ratio` and the margins to the "safe" end. A zero denominator is replaced by 1, which turns ratios into euro amounts | MED (explanations; accuracy barely affected) | `reports/data/2026-11-16_audit/r4_constraint_conflicts.csv`, `r4_denominators.csv` | For the final refit: denominator ≤ 0 → NaN-then-risky-end + flag, or drop the +1 constraint on `lt_debt_to_capital`. If no refit: state it as a known limitation in the deck | Kiet (decide); Chloe (code) |
| M7 | **Docs vs code.** Doc 04 says invalid denominators → NaN; the code uses /1 and keeps negatives. Doc 04 says `neg_pl_flag` includes `ebitda < prof_operations`; the code omits it. Doc 04 says `cash_interest_coverage` is NaN when exp_financing ≤ 0; the code divides. Doc 04 lists 7 industry groups; the code uses 21 A–U sections + target encoding. Doc 04 says "ROE dropped"; the model uses `roe`. Doc 05 says `operating_cash_flow_to_debt` was dropped; it is still in the schema and the model | MED (Roger grades documentation) | §3 table | Rewrite doc 04/05 from `config/feature_schema.json` + `preprocess.py` (code is the truth) | Desmond (docs), Kiet (review) |
| M8 | **Raw euro amounts as features** (`profit`, `rev_operating`, `ebitda`, `COGS`, `prof_operations`, `exp_financing`) next to `size`. Size enters 7 times, against the ratio-first principle | MED (defensibility) | schema `period_numeric_features` | Either drop them in a refit, or defend them: trees use only rank, and "profit < 0" carries most of the signal | Kiet |
| M9 | **Constant / near-constant / dead features**: 6 features LightGBM could not use in training (`zero_asst_tot`, `zero_total_debt`, `zero_asst_current`, `zero_wc_net`, `legal_SRS`, `city_Unknown`), and 19 features with zero splits, including `loss_flag` despite its +1 constraint (D-154 item 6) | LOW | §7 `model:*` | Drop in the refit; harmless for the frozen model | Chloe |
| M10 | **Reproducibility.** `data/interim/ateco_sector_english.csv` is git-ignored, so `preprocess.py` raises FileNotFoundError on import in a fresh clone. The round-03 preprocessing/model joblibs named in the manifest are not in the repo | MED (the TA may run estimation code) | §7 `repro:*`, `provenance:*` | Commit the 5 KB lookup (or un-ignore it). Note in the README that the joblibs are reproduced by `fit` | Chloe |
| M11 | **Default rates in team docs** are "ever defaulted" (4.37%), not the 12-month label (D-154 item 7) | MED (any number in the deck) | doc 01, 03, 04 still contain "4.37%" | Recompute with `label` before anything goes in the deck | Desmond |
| M12 | **Legal form: missing or unseen value is treated as SAA** (the all-zero reference category; 248 training rows) | LOW | §5, §7 `unseen:*` | Acceptable; document it. In a refit, use "Unknown/other" as the reference and pool SRS | Chloe |

What is **not** a problem (verified dynamically, §7): arithmetic of all 80 features (training = harness);
province → region map (team = Kiet's `src/data/codes.py`); target-encoding mapping and medians (frozen = refit);
row-order and batch invariance of Kiet's harness; unseen ATECO / city / legal handling.

---

## 2. The four already-known findings: confirmed or refuted

**(a) Desmond's 6bf455a: does `preprocess.py` compute `roe_negative` and recompute `roe`? Does the schema
validation fail?**
*Partly refuted.* 6bf455a changed only `config/feature_schema.json`, `docs/.../02-missing-value.md` and
`notebooks/features/02_missing_values.ipynb` (`git show --stat 6bf455a`). `preprocess.py` is byte-identical
at 4e0ba3a and 478b058 (its sha256 matches the bundle manifest at both commits, §7). **But `preprocess.py`
does not hold the imputation code**: at import it `exec`s the notebook cells tagged `export`
(`_load_notebook_functions("02_missing_values")`, preprocess.py lines 209-222). The new notebook cell sets
`df["roe"] = profit / eqty_tot.mask(eqty_tot == 0, 1) * 100` and `df["roe_negative"] = (profit < 0) & (eqty_tot < 0)`
inside `fill_missing`. `make_period_features` copies the frame, so both columns reach the schema check at
line 353 (`_validate_period_feature_contract`). **The check passes.** The dynamic smoke test runs
`fit_preprocessor` at 478b058 end to end, and `roe_negative` is in the feature list (§7). What does fail is
using the *frozen* 1.2.1 artifact with the 1.3.0 code (`_assert_artifact_schema_compatible`, line 364): it
raises (§7). That is finding M1.

**(b) Method G: still in `tests/harness_submission/missing_rules.py`? Where is imputation done in training?**
*Confirmed.* Method G is still active in Chloe's `tests/harness_submission/missing_rules.py` (lines 84-87).
It is gated behind `use_firm_history=False` in Kiet's `harness/2026-11-16_turquoise_submission/missing_rules.py`.
In the training path, all missing-value imputation lives in **the notebook** (`02_missing_values.ipynb`,
export cells), executed by `preprocess.py`. `preprocess.py` contains no imputation code of its own. At 4e0ba3a
(the model's provenance) G was on for training and transform. At 478b058 G was deleted from the notebook.
Note on the frozen harness copy: Chloe's `missing_rules.py` is byte-identical to the 4e0ba3a notebook export,
apart from the import header (diff checked).

**(c) "3 smallest industry groups fall back to the overall median": code or docs?**
*Implemented in training code at 478b058*: `sector_medians` drops
`train["id"].groupby(industry_group).nunique().nsmallest(3).index` (notebook export, line 42 of the cell). On
the round-03 training window these are O, P, U (§7). **Not in either harness**: the frozen `preprocessing.json`
has its own medians for all 19 observed sections (no K, T). It does not affect the target encoding.

**(d) D-154 status**: see §6.

---

## 3. Per-feature table (the 80 features of the frozen model, plus the 1.3.0 addition)

Conventions. "Denominator rule" is `_safe_divide`: **a zero denominator is replaced by 1** (so the ratio equals
the numerator, in euros if the numerator is an amount). A **negative denominator is divided as is** (sign flips).
NaN propagates. "Missing" lists the imputation methods (§4) that feed the inputs. All medians are fitted in
`preprocess.fit_preprocessor` on the round's training anchors and frozen in `artifacts/preprocessing.json`.
**No winsorisation or caps anywhere** in the training path or the harness. Doc 03 decided "no handling" because
trees use only the order. Kiet's `build.py` winsorises at train p1/p99. "Constraint" is the LightGBM monotone
constraint in the frozen model (`lightgbm.txt`, equal to `MONOTONE_DIRECTIONS` in `lightgbm_model.py`, §7).
"AUC" is the univariate AUC as a risk score and its data direction, from
`reports/data/2026-11-16_audit/r6_feature_cards.md` (835,016 labelled rows). Row counts for zero / negative
denominators come from `r4_denominators.csv`, `r4_constraint_conflicts.csv` and §B3 of the 11-16 audit report. "Same path" = same code in
training and harness. "Y*" means identical arithmetic; the only difference is the Method G input skew (M4).

| Feature | Formula | Zero / negative denominator | Missing (inputs) | Flags used with it | Expected sign (reason) | Constraint | AUC, data direction | Same path | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `size` | ln(asst_tot) | asst_tot ≤ 0 → NaN (never happens: min €1.5M) | asst_tot missing → sector median **amount** | – | − (bigger firms: diversified, better funding access) | 0 | 0.527, higher safer | Y | |
| `equity_ratio` | eqty_tot / asst_tot | asst_tot > 0 | B | – | − (loss-absorbing cushion) | −1 | 0.740, safer | Y | strongest clean leverage driver |
| `debt_to_eqty` | (debt_st+debt_lt) / eqty_tot | eqty = 0 → /1 (11 rows); eqty < 0 → negative (31,269 rows, DR 11.1%) | B | – | + (more leverage), but the sign flips with negative equity | 0 | 0.531; U-shaped | Y | correctly unconstrained; explain via `equity_ratio` |
| `st_debt_share` | debt_st / (debt_st+debt_lt) | total 0 → numerator (233 rows); negative (81) | B, G (debt_lt) | `zero_total_debt` | + (refinancing / rollover risk) | +1 | 0.565, riskier | Y* | |
| `lt_debt_to_capital` | debt_lt / (debt_lt+eqty_tot) | denominator < 0: 11,636 rows, DR 18.3%, at the 14th percentile | B, G | `zero_debt_lt` | + | **+1 (conflicts)** | 0.516; U-shaped | Y* | **D-154 (1) open**; wrong-signed explanations for insolvent firms |
| `bank_debt_conc` | (bank_st+bank_lt) / total debt | 0 → /1 | **H then B: ~sector constant when details are missing** | `debt_detail_missing` | ambiguous (bank monitoring vs dependence) | 0 | 0.506 | Y* | weak; imputed rows carry the sector median |
| `tangible_asset_ratio` | asst_tang_fixed / asst_tot | – | F | – | − (collateral) | 0 | 0.616, safer | Y | could be constrained −1 |
| `goodwill_ratio` | goodwill / asst_tot | – | B | – | ? field behaves like a profit measure (doc 01) | 0 | 0.638, safer | Y | do not call it "goodwill" without a caveat |
| `cash_ratio` | cash / asst_tot | – | B | `zero_cash` | − (liquidity buffer) | −1 | 0.565, safer | Y | |
| `cash_to_financing_exp_ratio` | cash / exp_financing | exp = 0 → cash in **euros**; exp < 0 (557 rows) → negative | B | `zero_exp_financing` | − | −1 | 0.592, safer | Y | D-154 (3) |
| `cur_asst_ratio` | asst_current / asst_tot | – | B | `zero_asst_current` | + in data (asset-light, trade firms) | 0 | 0.593, riskier | Y | |
| `AR_ratio` | AR / rev_operating | rev = 0 → AR in euros (25,613 rows); rev < 0 (2,341 rows, DR 7.6%) → negative → low PD | B, D | `zero_ar`, `zero_rev_operating` | + (slow collection) | +1 | 0.582, riskier | Y | **D-154 (2)(3) open** |
| `payables_ratio` | AP_st / asst_tot | – | H, B | `zero_AP_st`, `debt_detail_missing` | + (stretching suppliers) | +1 | 0.595, riskier | Y | |
| `wc_ratio` | wc_net / asst_tot | – | E | `zero_wc_net` | − | −1 | 0.614, safer | Y | |
| `cur_debt_ratio` | asst_current / debt_st | debt_st = 0 → euros (1,384 rows) | B | `zero_debt_st` | − (current ratio) | −1 | 0.613, safer; hump | Y | D-154 (3) |
| `quick_ratio` | (cash+AR) / debt_st | same | B | `zero_debt_st` | − | −1 | 0.587, safer | Y | |
| `asset_turnover` | rev / asst_tot | – | B, D | `zero_rev_operating` | ambiguous | 0 | 0.506 | Y | |
| `roa` | provider field (= prof_operations/asst_tot×100) | – | E: identity if blank | – | − | −1 | 0.733, safer | Y | |
| `roe` | **1.2.1 (frozen model, harness):** provider field, blank when \|ROE\|>150, filled with the plain industry median. **1.3.0 (training code now):** profit/eqty×100 for all rows, eqty = 0 → /1 | eqty < 0 flips the sign | 1.2.1: industry median (no flag); 1.3.0: recomputed | 1.3.0: `roe_negative` | − only when eqty > 0 | 0 | 0.551; non-monotone | **N** | **M1/M2**; D-154 (4) addressed differently in each version |
| `op_margin` | prof_operations / rev | rev 0 → euros; rev < 0 → sign flip to "safe" | D, B | `zero_rev_operating` | − | −1 | 0.702, safer | Y | D-154 (2)(3) |
| `net_prof_margin` | profit / rev | same | B, D | same | − | −1 | 0.713, safer | Y | same |
| `profit` | raw € | – | B | `loss_flag`, `zero_profit` | − | 0 | 0.752, safer | Y | raw amount (M8) |
| `rev_operating` | raw € | – | D, B | `zero_rev_operating` | ambiguous | 0 | 0.513 | Y | M8 |
| `ebitda` | raw € | – | B | `zero_ebitda` | − | 0 | 0.735, safer | Y | M8 |
| `COGS` | raw € | – | D, B | – | ambiguous | 0 | 0.531 | Y | M8 |
| `prof_operations` | raw € | – | D, B | `zero_prof_operations` | − | 0 | 0.732, safer | Y | M8 |
| `exp_financing` | raw € | – | B | `zero_exp_financing` | + | 0 | 0.584, riskier | Y | M8 |
| `int_coverage` | ebitda / exp_financing | exp = 0 → EBITDA in euros (28,790 rows with exp = 0) | B | `zero_exp_financing` | − | −1 | 0.728, safer | Y | D-154 (3) |
| `financing_expense_ratio` | exp_financing / asst_tot | – | B | – | + (interest burden) | +1 | 0.627, riskier | Y | |
| `ebitda_to_debt` | ebitda / total debt | total 0 → euros | B, G | `zero_total_debt` | − (inverse of Debt/EBITDA) | −1 | 0.755, safer | Y* | top driver |
| `operating_cash_flow_margin` | cf_operations / rev | rev 0 / < 0 as above | B | – | − | 0 | 0.726, safer | Y | doc says "per unit of profit" (wrong) |
| `cash_interest_coverage` | cf_operations / exp_financing | doc: NaN if ≤ 0; **code: /1 or divide** | B | – | − | 0 | 0.736, safer | Y | doc/code mismatch (D-154 (5)) |
| `operating_cash_flow_ratio` | cf_operations / asst_tot | – | B | `zero_cf_operations` | − | −1 | 0.759, safer | Y | |
| `operating_cash_flow_to_debt` | cf_operations / total debt | total 0 → euros | B, G | – | − | 0 | 0.764, safer | Y* | doc 05 says dropped; it is not |
| `neg_bs_flag` | any of 11 BS fields < 0 (reported values, before imputation) | – | missing never triggers | – | + | 0 | 0.500 | Y | |
| `neg_pl_flag` | rev<0 \| COGS<0 \| exp_fin<0 | – | evaluated **after** imputation | – | + | 0 | 0.504 | Y | doc includes `ebitda < prof_operations` (D-154 (5)) |
| `debt_detail_missing` | any of 6 debt details missing (method C, before fill) | – | – | – | + | 0 | 0.506 | Y | |
| `loss_flag` | profit < 0 | – | after B | – | + | +1 | 0.675 | Y | |
| `zero_*` (16 flags) | input == 0 after imputation: asst_tot, total_debt, debt_st, debt_lt, bank_debt, cash, asst_current, ar, rev_operating, AP_st, wc_net, ebitda, prof_operations, profit, cf_operations, exp_financing | – | after imputation | – | marks the /1 cases | 0 | all ≈ 0.50 except `zero_debt_lt` 0.523 | Y* (`zero_debt_lt` via G) | `zero_asst_tot` constant; `zero_wc_net` 18 rows |
| `ateco_sector_te` | smoothed industry default rate (A–U section; m = 20; 5-fold group OOF on train rows; full-train map at transform) | – | missing / unmapped / unseen section → train global mean | – | + by construction | +1 | 0.562 | Y (train rows use OOF values, by design) | §5 |
| `legal_{SAU,SPA,SRL,SRS,SRU}` | one-hot, reference SAA | – | missing → "Unknown" → all zeros (= SAA) | – | none | 0 | ≤ 0.53 | Y | `legal_SRS` 8 rows |
| `city_{20 regions incl. Unknown}` | HQ_city (ISTAT province) → region, one-hot, reference Abruzzo | – | missing / unknown code → `city_Unknown` | – | none | 0 | ≤ 0.52 | Y | 21 regions = 20 + Unknown |
| *`roe_negative`* (1.3.0 only) | profit < 0 AND eqty_tot < 0 | – | after B | – | + | – (not in frozen model) | n/a | **N (absent in harness)** | M3 misnamed |

Kiet's `build.py` (reference) differs by design. It uses only ratios on total assets plus slog-transformed
coverage/margins. A denominator ≤ 0 → NaN or ±inf by an explicit rule, then winsorised at train p1/p99. It
has `neg_equity`, `rev_nonpositive` and `debt_split_missing` flags, ATECO sections pooled below 30 train
defaults, and macro-area (5) instead of region (20). These are the rules D-154 proposes to port.

---

## 4. Imputation methods (A–I): training vs harness

Source: `docs/project-update/features/02-missing-value.md` and the notebook export at each commit.

| Method | What | Variables | Training path @4e0ba3a (frozen model) | Training path @478b058 (now) | Harness, Chloe | Harness, Kiet | Consistent? |
|---|---|---|---|---|---|---|---|
| A | missing HQ_city → "Unknown" | HQ_city | yes | yes | yes | yes | Y |
| B | sector (A–U) median of x/asst_tot × asst_tot; unseen or missing section → overall median | 26 amount columns (asset items, equity, liab_lt(_emp), 6 debt details, debt_st/lt, P&L items, cf_operations); asst_tot safeguard = median amount | all 19 observed sections own median | **O, P, U → overall median** | 19 sections (frozen) | 19 sections (frozen) | N vs 478b058 (M5) |
| C | flag `debt_detail_missing` before filling | 6 debt-detail columns | yes | yes | yes | yes | Y |
| D | prof_operations = rev − COGS identity | rev_operating, COGS, prof_operations | yes | yes | yes | yes | Y |
| E | identities after B/D | margin_fin, wc_net, roa | yes | yes | yes | yes | Y |
| F | fixed-asset reconstruction from eqty_tot − margin_fin, else 0 | 3 fixed-asset columns | yes | yes | yes | yes | Y |
| G | prior-year 0 → 0 (reads the firm's previous row) | liab_lt, liab_lt_emp, debt_lt | **on** (reads history.csv) | **removed** | **on (reads other test rows; non-compliant)** | off | N (M4) |
| H | detail = 0 when its total is 0 | debt_bank_*, debt_fin_*, AP_* | yes | yes | yes | yes | Y |
| I | `roe` | roe | provider value + plain industry median fill | recompute profit/eqty×100 + `roe_negative` | as 4e0ba3a | as 4e0ba3a | N (M1/M2) |
| – | drop `eqty_corp_family_tot` (100% missing), `days_rec` (72% missing) | | yes | yes | yes | yes | Y |
| – | rows with any NaN/inf feature after all of this | | **dropped from training** | dropped | **fallback PD = train DR 1.237%** | same | by design (§7 shows how many) |

Where the parameters are fitted: `sector_medians(train)` on the round's **training anchors only**
(`fit_preprocessor`, line 741). For round 03 this is all 476,183 labelled FY2007-09 rows. Their default rate
equals the frozen fallback PD to 18 digits (§7). Transform never refits.

---

## 5. Categorical handling

| Field | Training path | Harness | Unseen / missing at scoring | Reference `build.py` |
|---|---|---|---|---|
| `ateco_sector` | 2-digit division → A–U section via `data/interim/ateco_sector_english.csv`. Non-integer or unknown code → NaN | identical function (`missing_rules._ateco_industry_group`) | NaN / unseen section → TE = training global mean (0.012373); imputation → overall median | section, pooled to OTHER if < 30 train defaults |
| Industry target encoding | `ateco_sector_te`: (defaults + 20·mean) / (n + 20). Training rows get 5-fold **StratifiedGroupKFold by firm** out-of-fold values; transform uses the full-train mapping | frozen mapping (19 sections; K, T absent → global mean) | global mean | logit-scale TE (full2) or one-hot |
| Smallest-groups rule | 1.3.0: O, P, U use the overall median **for imputation only** | not present | – | MIN_DEFAULTS_PER_LEVEL = 30 |
| `HQ_city` | ISTAT province code 1–110 → 20 regions (`REGION_BY_PROVINCE`) + "Unknown"; one-hot, reference = Abruzzo | identical dict (`feature_rules._region`) | unknown code / missing → `city_Unknown` = 1 | 5 NUTS-1 macro-areas |
| Province map check | – | – | §7: team map = Kiet's `src/data/codes.py` map | ISTAT 2009-16 province set [UNVERIFIED: ISTAT source not yet verified by literature-scout] |
| `legal_struct` | one-hot of sorted training levels, first level (SAA, 248 rows) is the reference; SRS (10 rows) kept | identical | missing → "Unknown" → all zeros → **treated as SAA**; unseen → all zeros | rare levels pooled into OTHER |
| Codes as numbers? | never: all three are mapped to labels before encoding | never | – | never |

---

## 6. D-154 status (at 478b058)

| Item | Proposal (D-154) | Status | Evidence |
|---|---|---|---|
| (1) | `lt_debt_to_capital`: denominator ≤ 0 → risky end, or drop the +1 constraint | **OPEN**. `_safe_divide` unchanged; constraint +1 still in `MONOTONE_DIRECTIONS` and in `lightgbm.txt` | preprocess.py l.279; §7 `model:monotone_constraints_match_code` |
| (2) | rev ≤ 0 → AR_ratio and margins to the risky end + flag | **OPEN** | preprocess.py l.288-300 |
| (3) | replace zero-denominator → 1 with an explicit capped value | **OPEN** (the new 1.3.0 `roe` even copies the /1 rule) | notebook method I |
| (4) | `roe_missing` flag or drop `roe` | **PARTLY, in a different way**: 1.3.0 recomputes `roe` (unbounded) + `roe_negative`; not in the harness or frozen model | 6bf455a; M1-M3 |
| (5) | align schema text with code (neg_pl_flag, cash_interest_coverage, OCF margin, debt_to_eqty) | **OPEN**: doc 04 unchanged since 11-16 | doc 04 table |
| (6) | remove `zero_asst_tot`, `zero_wc_net`, `legal_SRS` | **OPEN**: all still in schema 1.3.0 | config/feature_schema.json |
| (7) | recompute doc default rates with the 12-month label | **OPEN**: "4.37%" still in docs 01, 03, 04 | grep |

---

## 7. Dynamic parity test

Script: `src/audit/2026-11-20_train_harness_parity.py`. Runtime about 2 minutes. All code is pulled from git
at 4e0ba3a / 478b058 into a temp folder, so concurrent edits to the working tree cannot affect it.
Results: `2026-11-20_parity_by_feature.csv` (561 rows = comparison × feature: rows compared, cells differing,
cells differing materially at relative tolerance 1e-9, max abs diff, NaN mismatches) and
`2026-11-20_parity_summary.csv` (every other number below).

**Design: is it apples-to-apples?** Yes, for the bundle. The round-03 preprocessing joblib is not in the repo,
so the script rebuilds the training-path artifact from the frozen `preprocessing.json`. It first proves that
this is legitimate: re-fitting with the 4e0ba3a code on the labelled FY2007-09 rows (476,183; default rate
0.012373394262289918 = the frozen fallback PD exactly) reproduces the frozen medians (max abs diff 5.6e-16,
floating point), the target-encoding mapping and global mean (diff 0.0), and the legal / region category lists
(identical; references SAA and Abruzzo). The manifest's `source_sha256` matches 4e0ba3a for `preprocess.py`,
both notebooks, the schema and the ATECO lookup. At 478b058 the schema and the 02 notebook differ. Anchors:
20,000 random FY2011 rows (seed 20261120) from `train-2025.csv` (= `data/raw/harness_test_data_2011.csv`),
plus every FY2011 row with `liab_lt`/`liab_lt_emp`/`debt_lt` missing (only 3 exist in FY2011; Method G would
fire on 2), giving 20,003 rows. The training path receives the firms' FY2007-2011 history, exactly as
`make_walk_forward_splits` builds `history.csv`.

| Comparison | Rows | Features differing | Cells differing (material) | PD effect | Reading |
|---|---|---|---|---|---|
| Training @4e0ba3a, no prior years vs Kiet's harness | 20,003 | **0 of 80** | 0 | max \|ΔPD\| = 0 | **Arithmetic parity: identical** |
| Training @4e0ba3a with FY2007-11 history vs Kiet's harness | 20,003 | 7 | 13 (2 rows) | 2 rows, max \|ΔPD\| = 0.00069 | Method G skew only (training reads FY2010; harness does not) |
| Training @4e0ba3a with history vs Chloe's harness (FY2011-only batch) | 20,003 | 7 | 13 (2 rows) | same 2 rows | Chloe's G finds no FY2010 row in a one-year batch |
| Chloe's harness vs Kiet's harness, same batch | 20,003 | 0 | 0 | 0 | identical when the batch has one year |
| Training @478b058 (schema 1.3.0) vs Kiet's harness (1.2.1) | 20,003 | `roe` (+ `roe_negative` absent in harness) | 19,963 `roe` cells | not scorable (81 features) | **Version split (M1/M2)** |

Affected features in the G rows: `debt_to_eqty`, `st_debt_share`, `lt_debt_to_capital`, `bank_debt_conc`,
`ebitda_to_debt`, `operating_cash_flow_to_debt`, `zero_debt_lt` (max abs diff 0.31 in `st_debt_share`).
Other features in the 1.3.0 comparison differ only at the 1e-13 floating-point level (median refit order).

**`roe` under 1.3.0 vs the frozen bundle.** Where the provider value exists, the recomputed `roe` agrees within
0.01 in 100% of rows (share with |diff| > 0.01 = 0.0). The whole difference sits in the **6.83%** of rows whose
provider `roe` is blank. There the harness uses the industry median and 1.3.0 uses the true ratio; the median
gap is 214.7 points. In 1.3.0, 4.85% of rows have |roe| > 150 (max 48,706), but the frozen model saw `roe`
only in [−150, 149.93]. `roe_negative` = 1 in 3.65% of rows.

**Batch / single-row invariance (Appendix A "current record only").**

| Test | Kiet's harness | Chloe's harness |
|---|---|---|
| Full batch vs the same rows shuffled into 10 chunks | 0 cells differ | 0 cells differ |
| 503 rows scored alone vs in the 20,003-row batch | 0 cells differ | 0 cells differ |
| The 3 G-candidate FY2011 rows, with vs without the firm's FY2010 row in the batch | **0 cells; max \|ΔPD\| = 0** | **13 cells in 2 rows; max \|ΔPD\| = 0.00069** |

So Kiet's harness is row-independent. Chloe's harness gives a firm a different PD depending on whether its
prior-year statement is in the same file, which Appendix A forbids (D-152).

**Size of the Method G effect in training.** G fired on 279 of 476,183 round-03 training anchors (0.0586%).
The frozen model's training features and the harness's features differ on those few rows. This is immaterial
for accuracy but is a definitional mismatch, gone after a 1.3.0 refit.

**Small-industry fallback (1.3.0).** O, P, U (13, 266, 57 firms) fall back to the overall median. Affected rows
(a Method-B field missing in those sections): 137 of 476,183 training rows and 19 of 184,973 FY2011 rows. None
were in the 20,000-row sample, so no feature diff shows in the CSV.

**Other checks (all from the summary CSV).**
- Province → region: the team map (`preprocess.py` = `feature_rules.py`) and Kiet's `src/data/codes.py` agree on
  all codes 0-111 plus "Unknown"/"nan" (0 mismatches).
- Unseen categories, on a messy row (legal "XYZ", HQ_city 999, ATECO blank, equity "n.a."): legal dummies all 0
  (= SAA), `city_Unknown` = 1, `ateco_sector_te` = training global mean. The row still gets a model PD.
- Frozen model monotone constraints (by name) equal `MONOTONE_DIRECTIONS` in `lightgbm_model.py` (19
  constrained features, incl. `lt_debt_to_capital` +1 and `AR_ratio` +1).
- Features LightGBM could not use in round-03 training (`feature_infos = none`: constant, or too rare to split):
  `zero_asst_tot`, `zero_total_debt`, `zero_asst_current`, `zero_wc_net`, `legal_SRS`, `city_Unknown`.
  Features with zero splits in the 424 trees (19): `neg_bs_flag`, `neg_pl_flag`, `loss_flag`, 11 `zero_*`
  flags, `legal_SRS`, `city_Liguria`, `city_Sardegna`, `city_Umbria`, `city_Unknown`. They are safe to drop
  in a refit and should not appear in the deck as drivers.
- Training ranges seen by the model (`lightgbm.txt`): `AR_ratio` [−12.7M, 181.7M], `cur_debt_ratio`
  [−1.87M, 18.7M], `int_coverage` [−102M, 89M], `lt_debt_to_capital` [−294.5, 3,752]. These are the
  zero-denominator → 1 and negative-denominator cases (D-154 (3)).
- Row drops: no FY2011 sample row had a non-finite feature in any path (20,003 kept everywhere), so the
  "training drops vs harness fallback PD" difference did not trigger on this sample.
- Smoke tests at 478b058: `fit_preprocessor` runs end to end (3,000-row demo; `roe_negative` is in the
  feature list, so the schema check at line 353 passes). `transform_with_preprocessor` with the frozen 1.2.1
  artifact **raises** `ValueError: Preprocessing artifact feature schema does not match...`.
- `data/interim/ateco_sector_english.csv` is git-ignored (`git check-ignore` true). `preprocess.py` cannot be
  imported from a fresh clone without it.

**Limits of this test.** (1) It runs on pandas 2.3.3 / LightGBM 4.7.0; the bundle pins pandas 3.0.1 /
LightGBM 4.6.0. Both paths use the same versions here, so parity holds, but absolute PDs were not compared
with the TA environment. (2) The 1.3.0 path uses frozen TE and categories: the TE code is unchanged and the
training window is the same, so a refit would give identical values. (3) Only FY2011 anchors were tested;
G-type history effects for other years are bounded by the 279 training anchors above.

---

## 8. Decisions the team must make today

1. **Which feature version ships?** (M1) Option A: submit the frozen 1.2.1 bundle (Kiet's patched harness;
   proven consistent, §7) and freeze 6bf455a out of the submission. Option B: refit on 1.3.0 and rebuild the
   harness from that refit, which needs about one pipeline run plus a new `preprocessing.json`, manifest and
   stress test. Do not submit a mix. *Recommendation: A for the holdout submission unless the 1.3.0 refit is
   finished and re-stress-tested before tonight.*
2. **Method G**: confirm Kiet's D-152 patch (G off at inference) as the submitted behaviour, and sign off D-152
   (Chloe). If the refit is 1.3.0, G is gone everywhere.
3. **D-154 (1)-(3)**: fix in the refit, or list as a known limitation in the deck (with the segment PD/DR
   evidence from `r7_segment_calibration.csv`). Kiet must be able to answer "why does insolvency lower PD
   through `lt_debt_to_capital`?"
4. **`roe`**: keep (1.2.1 median fill, no flag), recompute (1.3.0, unbounded, rename the flag), or drop.
5. **Raw amounts (M8)**: keep and defend, or drop in the refit.
6. **Docs 04/05 rewrite owner and deadline** (M7, M11): the deck's feature slide must be built from the
   schema, not from doc 04.
7. **Commit the ATECO lookup** (M10), so the estimation code runs from a clean clone.
