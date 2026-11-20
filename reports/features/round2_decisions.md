# Round-2 feature decisions (feature-architect, 2026-11-02): D-090 .. D-099

For the orchestrator to merge into `docs/DECISION_LOG.md`. Every number comes from
`python -m src.features.univariate_v2` (`reports/features/round2_univariate.md` and `round2_*.csv`) or
`python -m src.features.ablation_v2` (`round2_ablation_seeds.csv`). The data are training rows fs_year
2007-2009 with label_status ok (475,861 rows, 6,195 defaults, base rate 0.0130). The panel is read with a
parquet filter fs_year <= 2009, so nothing from 2010 or later is loaded (LOCKBOX). Inner check: fit on
2007-08 and score 2009. The D-062 caveat applies to all of round 2 as well: 2008 and 2009 are the outer
test years of folds 1-2.

---

### D-090 Feature set `full2` (opt-in; v1 unchanged)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED (for the GBM challenger; NOT for the probit/logit core)
- Decision: `FULL2_FEATURES` = the 26 v1 features (unchanged) + 19 new features = 45. It is opt-in:
  `B.fit(train, feature_set="full2")` adds `params["v2"]`, and `B.transform(rows, params, feature_set="full2")`.
  The default `fit`/`transform` calls give byte-identical v1 params and outputs (test `test_v1_unchanged`).
  `CORE_FEATURES` and `ALL_FEATURES` are unchanged. Two candidates were evaluated and rejected
  (D-092, D-096). They can still be computed with `feature_set="full2_candidates"`, which is used only by
  the evidence scripts.
- Alternatives considered: changing the full set in place. Rejected because round-1 experiments depend on it.
- Evidence (inner LightGBM, fixed settings, no tuning, `round2_lgb_ablation.csv`, `round2_ablation_seeds.csv`):
  v1 full AUC 0.8313 (3-seed mean, s.d. 0.0006). All 21 candidates: 0.8415 (+0.0102), log loss −0.00063.
  **Final 45-feature set** (interest_to_sales unconstrained; `round2_lgb_ablation.csv`, seeds 0/1/2):
  AUC 0.8403 / 0.8415 / 0.8412, mean 0.8410, vs v1 0.8314 / 0.8307 / 0.8318, mean 0.8313 → **+0.0096 AUC**.
  Log loss mean 0.06664 vs 0.06735 → **−0.00071**. Dropping the two duplicates costs about 0.0005 AUC against all
  21 candidates (within 1 s.d.).
  **No single new feature explains the gain.** Adding any one of them to v1 adds ≤ +0.0025, and dropping any
  one from the full candidate set loses ≤ 0.0033 (tfr_ratio). Part of the +0.010 may therefore come from
  the fixed, untuned LightGBM gaining capacity with more columns rather than from new information. The GBM
  challenger must compare v1 and full2 under its own tuned settings on the frozen folds before claiming the gain.
- Source(s): see D-091..D-097 per feature.
- Risks / what would change our mind: the gain disappears under the challenger's tuned model, or fold-3
  (2010) does not confirm it (D-062 rule).
- Human reviewer (initials) and date:

### D-091 Test of the D-010 hypothesis "`goodwill` = value added"; personnel cost / sales is adopted as hypothesis-dependent
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED (hypothesis supported by four checks; the feature is labelled HYPOTHESIS-DEPENDENT)
- Test (`round2_data_checks.csv`): if goodwill = value added = EBITDA + personnel cost, then
  goodwill − ebitda = personnel cost, and it should be (1) never negative, (2) zero only for firms with
  no staff, (3) linked to the staff-severance fund (TFR, `liab_lt_emp`), which only exists when there are
  employees, and (4) plausible in size. Results:
  (1) negative on 0.012% of rows. (2) zero on 26.2% of rows; of these only 0.94% have TFR > 0, versus
  93.2% of rows with a positive value; 52.5% of the zero rows are real estate (section L, 19.3% of all
  rows), i.e. property companies without employees. (3) Spearman(goodwill − ebitda, liab_lt_emp) = 0.93.
  (4) median personnel cost / sales = 9.3%. Every check is consistent with the hypothesis, and none
  contradicts it. So the feature is kept.
- Decision: `personnel_to_sales` = slog((goodwill − ebitda) / rev_operating); a negative value → NaN (data
  error); the revenue rule is the same as for ar_days. Expected sign + (R-33 p. 12: upward-sloping transform).
  Data: AUC 0.570 (2009 inner 0.562); deciles flat at about 1.1-1.3%, then D10 2.58% → + confirmed.
- Risks: if the bank's field definition says otherwise, drop the feature. It is not in the core.
  Spearman with tfr_ratio is 0.75 (both measure labour intensity).
- Source(s): R-33 Table 3 p. 11, p. 12; field meaning: D-010 (data-auditor), tested here.

### D-092 RiskCalc 4.0 France ratios (F1): what was adopted, and what was rejected because our data already contain it
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: operating_margin, payables_to_sales, personnel_to_sales, interest_to_sales ADOPTED;
  net_debt_coverage and gross_margin REJECTED
- Decisions and evidence (`round2_univariate_stats.csv`, `round2_rho_pairs.csv`):
  - **net_debt_coverage** = cf_operations / (total liabilities − cash), total liabilities = asst_tot − eqty_tot.
    Rule for a denominator ≤ 0 (cash covers every liability, 2.9% of rows, default rate 0.70% vs 1.30%):
    +inf if the firm generates cash (clipped to p99 = the safest value), 0 if not. The ratio works
    (AUC 0.247, monotone deciles 5.75% → 0.40%), but its Spearman with v1 `cfo_to_debt` is **0.990**. In our data,
    netting cash and widening the denominator from debt to all liabilities hardly changes the ranking.
    Adding it to v1 gives +0.0003 AUC (noise; s.d. 0.0006). → REJECTED as a duplicate. Our `cfo_to_debt` already
    *is* RiskCalc France's debt-coverage family, which answers L7(i).
  - **gross_margin** = (rev − COGS)/rev is **identical** to operating_margin on 99.993% of rows, because
    prof_operations = rev_operating − COGS exactly (01_profile.md §5; COGS = all operating costs). → REJECTED
    (not implemented).
  - **operating_margin** = slog(prof_operations / rev), rev ≤ 0 → NaN. AUC 0.291, deciles 4.53% → 0.55%. Spearman
    0.87 with ebitda_margin (the measure RiskCalc 4.0 France replaced). ADOPTED for full2 (trees tolerate the overlap, D-025).
  - **interest_to_sales** = slog(exp_financing / rev); rev = 0 → +inf if interest > 0, else 0; rev < 0 → NaN. AUC 0.580,
    but the deciles are hump-shaped (0.93 … 2.14, 2.20, then 1.27, 1.43 in D9-D10). The top deciles are firms
    with small sales relative to debt (holding / real-estate type). R-33 p. 12 reports an upward slope.
    **Our data contradict a monotone '+' at the top, so expected sign = none** (no monotone constraint), the same
    handling as asset_turnover in D-022. ADOPTED.
  - **payables_to_sales** = slog(AP_st / rev); same revenue rule; NaN if the debt-split block is missing (impute:
    sector share × debt_st, D-021). AUC 0.636, monotone deciles 0.79% → 2.43%. Spearman 0.93 with ap_days (COGS ≈
    revenue here). Added alone it adds nothing measurable (−0.0003). ADOPTED for full2 because it is the R-33
    definition and its sales denominator is defined for more firms (COGS ≤ 0 → NaN in ap_days). A GLM user should take one of the two.
  - Zero / negative revenue: the same rule as v1 `ar_days` / `rev_nonpositive` (D-019).
- Source(s): R-33 Table 3 p. 11 (inputs), Table 4 p. 14 (weights), p. 12 (directions of the transforms), p. 10 Table 2 (AP/sales example).
- Human reviewer (initials) and date:

### D-093 Sector-relative percentiles (F2)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED for full2 (GBM); not for GLMs next to their base ratio
- Decision: for equity_ratio, roa_net, interest_coverage, cash_ratio, st_debt_ratio and tangible_ratio, add
  `<ratio>_sect_pct`, the firm's percentile within its ATECO section. Each section has a 101-point quantile grid
  of the training-window (winsorised) feature, stored in params. Linear interpolation is used, ties in the grid
  get their mid-percentile, values outside the grid → 0 / 1. A section with < 2,000 training rows, or a
  missing / unseen section, → pooled grid. NaN base → NaN (impute view: computed from the imputed base, so ≈ 0.5).
- Evidence: Spearman with the base ratio is 0.88-0.98 (tangible 0.88, st_debt 0.91, the rest ≥ 0.96), so the
  correction is modest. The univariate AUC is close to the base. In the 3-seed ablation, adding single
  percentiles to v1 gives between −0.0002 and +0.0008. Dropping the whole block from the 21-candidate set
  loses 0.0009 AUC and 0.00013 log loss (about 1.5 s.d.). This is small but consistent.
- Alternatives: sector-demeaned ratios (z-scores) need a scale per sector and are more fragile with tails;
  percentiles are the RiskCalc-style "percentile score" (R-33 p. 12).
- Source(s): R-01 Sec. 3.3 pp. 17-18 (industry corrects ratios whose meaning differs by sector).
- Risks: VIF up to 11.7 (st_debt_ratio_sect_pct) in a GLM. Do not use a percentile and its base together in a GLM.

### D-094 Shrunk default-rate encodings for ATECO division, province and region (F3)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED for full2
- Decision: `ateco_div_te` (83 divisions), `province_te` (110 HQ_city codes) and `region_te` (20 regions).
  Each value = logit((d_l + m·p0)/(n_l + m)) − logit(p0): the level's default rate pulled toward the
  training base rate p0, on the log-odds scale, centred so that 0 = average. **Shrinkage strength m by
  empirical Bayes** (method of moments: between-level variance of default rates minus binomial noise
  gives tau²; m = p0(1−p0)/tau²), fitted in-window, with guard rails [50, 50,000]. Fitted values on 2007-09:
  m = 555 (division), 1,834 (province), 3,879 (region) rows. A level with n rows keeps the share n/(n+m)
  of its own rate; none hit a guard rail. Missing / unseen level → 0 (the prior).
  **Out-of-fold:** training rows get encodings from 5 folds grouped by firm id (fold = crc32(id) mod 5,
  deterministic, stored in params as per-fold tables): `transform(..., oof=True)`, training only.
  Scoring / harness: the full-window table (`oof=False`, the default).
- Evidence (`round2_target_encoding.csv`, `round2_lgb_ablation.csv`): training AUC in-sample vs OOF: division
  0.592 vs 0.580, province 0.564 vs 0.547, region 0.536 vs 0.530. The in-sample values are inflated by the
  rows' own labels. 2009 inner AUC: 0.572 / 0.537 / 0.529. LightGBM on 2009: v1 + OOF encodings 0.8356, but
  v1 + IN-SAMPLE (leaky) encodings 0.8330, and log loss even worse than v1 (+0.00007). Leakage hurts out of
  time, which confirms the OOF design. 3-seed single additions: province +0.0017, region +0.0017, division +0.0010.
- Alternatives: one-hot of 83 / 110 levels (many with < 30 defaults, D-023); fixed m chosen by hand (arbitrary).
- Source(s): assumption (standard empirical-Bayes / m-estimate encoding; no VERIFIED reference in REFERENCES.md;
  literature-scout could add one). R-01 Sec. 3.1 p. 14 supports industry information in general.
- Risks: geography encodes local conditions *in 2007-09*, which may not persist. Province and region are nested (Spearman 0.45).

### D-095 Missingness / data-quality counts (F4, part 1)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: n_core_imputed ADOPTED; n_missing_raw REJECTED
- Evidence (`round2_count_rates.csv`):
  - **n_missing_raw** (count of 32 missing raw amounts; roe and margin_fin excluded because they are blank
    by construction when equity ≤ 0): values 0 / 3 / 6 cover 99.7% of rows (the debt-split block). Spearman with
    v1 `debt_split_missing` = **0.994**, AUC 0.505, single addition −0.0001. → REJECTED as a duplicate.
  - **n_core_imputed** (core numeric features still NaN after identity reconstruction, i.e. those a GLM fills
    with a sector median): 0 on 474,379 rows (1.29% default), **1 on 1,169 rows (6.33% default)**, 2 on 278 rows
    (2.16%). The "1" group is almost all `ar_days` undefined because **revenue is negative** (1,078 rows with
    rev < 0, default rate 6.8%). The v1 flag `rev_nonpositive` mixes zero and negative revenue and has no
    sign (D-022). This is exactly the L1 / R-34 effect: a median fill would make these firms look average when
    they are 5× riskier. Sign +. AUC 0.505 only because of its low prevalence. In the GBM ablation its single
    addition is within noise (−0.0004), so the case rests on the per-group default rates and on harness safety
    (messy holdout rows), not on AUC. ADOPTED.
- Source(s): R-34 (median fill understates PDs), R-05 pp. 60-61 (less real data, worse estimate), via L1.
- Risks: on the holdout the count may also pick up formatting junk that a training row never had; it is
  never extrapolated beyond the training values by trees.

### D-096 Balance-sheet gap and rounding flag (F4, part 2)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: bs_gap ADOPTED (sign none); rounding flag REJECTED
- **Units check first:** all amounts are whole euros (100% integer-valued; median assets EUR 3.44MM), so
  "rounded to thousands" is a meaningful test. Only 0.26% of asst_tot values are multiples of 1,000 (chance
  would be 0.1%).
- **Rounding flag** (all 5 key amounts non-zero multiples of 1,000): 503 rows (0.11%) with **2 defaults**
  (0.40% vs 1.30%). A looser version (≥ 3 of 5) has 702 rows and 7 defaults. Far too few defaults to estimate
  anything, so → REJECTED. R-01 p. 22 found rounded rows informative in a much larger sample. Here the flag
  would be noise.
- **bs_gap** = (asst_tot − eqty_tot − debt_st − debt_lt − liab_lt − liab_lt_emp) / asst_tot: the liability-side
  item the file does not classify (data-auditor: residual ≥ −€2 on 99.84% of rows; most likely accruals /
  deferred income [hypothesis]). Distribution: median 0.1%, p90 2.5%, p99 24%. AUC 0.482 (weakly −); deciles
  are non-monotone (1.48 … 1.11 … 1.41, 1.27). Sign none. In GBM it is one of the stronger single additions
  (+0.0018). ADOPTED for full2 without a monotone constraint. In the deck, describe it as "unclassified
  liabilities", not as a data error.
- Source(s): R-01 Sec. 3.4.1 p. 22 (rounding); R-33 Sec. 2.2 p. 6 (balance-sheet plausibility); assumption for the meaning.

### D-097 Unused fields (F5) and their signs
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED for full2
- nonbank_fin_debt = (debt_fin_st + debt_fin_lt)/asst_tot, sign + (more fixed claims). Data: AUC 0.523, mass
  at 0 then 1.54% / 1.64% in the top deciles → + confirmed. NaN when the block is missing (15.7%). Impute: sector share.
- ap_lt_ratio = AP_lt/asst_tot, sign + (suppliers turned into long-term creditors). Mostly zero; AUC 0.501, top
  decile 1.41% vs 1.30% → weak + (not contradicted). Single GBM addition +0.0006 (about 1 s.d.).
- fin_fixed_ratio = asst_fixed_fin/asst_tot (holding / group structure), sign **none**: group support vs.
  illiquid stakes. Deciles non-monotone (1.21-1.52%); AUC 0.503; GBM single addition +0.0014, log loss −0.00014.
- tfr_ratio = liab_lt_emp/asst_tot (staff-severance fund). In round 1 it was rejected for having "no expected
  sign" (D-024). Now it is **adopted with sign none**. Two opposite stories (labour-intensive firms with rigid
  costs vs. a cheap internal source of funding owed to staff); deciles hump-shaped (1.15, 1.29, 1.65, 1.53 …
  1.09, 1.37). Univariate AUC 0.510, **but it is the largest single GBM contribution**: +0.0025 added alone,
  −0.0033 when dropped from the candidate set (5 s.d.). The tree uses it in interaction (labour intensity is
  a sector / business-model descriptor). It is unconstrained, so face validity rests on the shape plot.
- Source(s): assumption (no VERIFIED source for these fields).
- Risks: tfr_ratio is Italy-specific; the Spearman with personnel_to_sales is 0.75. Explaining it to a
  credit officer: "how labour-intensive the firm is", not "more TFR = more risk".

### D-098 Collinearity of full2 (Roger M8)
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: ADOPTED
- Decision: full2 is a tree-model set. D-025 applies: in GBM, high VIF is tolerated and SHAP is reported by
  family. Pairs with |rho| ≥ 0.70 that involve an adopted new feature (`round2_rho_pairs.csv`): the 6 sector
  percentiles with their bases (0.88-0.98); operating_margin–ebitda_margin 0.87; payables_to_sales–ap_days 0.93;
  personnel_to_sales–tfr_ratio 0.75; roa_oper / cfo_to_debt–roa_net_sect_pct 0.73 / 0.78; cfo_to_debt–
  interest_coverage_sect_pct 0.72. The two duplicates at ≥ 0.99 were dropped (D-092, D-095). VIF (imputed, all
  v1 + candidates): st_debt_ratio 48, tangible_ratio 37, wc_ratio 36, ebitda_margin / operating_margin 15.5,
  st_debt_ratio_sect_pct 11.7. Any GLM using full2 must use L2 regularisation or drop one member of each listed pair.
- Source(s): R-01 Sec. 4.2 p. 26 (VIFs).

### D-099 Rejected / not built in round 2, and open caveats
- Date / phase / agent: 2026-11-02 / round 2 / feature-architect
- Status: REJECTED (items 1-4), OPEN (5-7)
1. gross_margin: identical to operating_margin (D-092).
2. net_debt_coverage: duplicate of cfo_to_debt, rho 0.990 (D-092).
3. n_missing_raw: duplicate of debt_split_missing, rho 0.994 (D-095).
4. rounding flag: 503 rows / 2 defaults (D-096).
5. NLP features: still REJECTED (D-003). The data have no text field. Macro / cycle features: still not
   features (D-004); this is left to calibration.
6. OPEN: the sign of interest_to_sales was changed from + to none **after** seeing the 2007-09 deciles. That is
   the same post-hoc pattern the referee flagged in D-063. Report it that way.
7. OPEN: the round-2 choices were made on 2007-09 (D-062): quote fold 3 (2010) separately. `docs/FEATURE_SPEC.md`
   is regenerated by `src.features.univariate` (v1), which **overwrites** the round-2 section. Rerun
   `src.features.univariate_v2` afterwards to restore it.
