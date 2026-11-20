"""Six-round audit of Chloe/Desmond's 11.15 feature pipeline (KN request, 2026-11-16).

Run:  <venv with harness requirements + pyarrow>/bin/python src/audit/2026-11-16_feature_pipeline_audit.py
Writes CSV/MD outputs to reports/data/2026-11-16_audit/. Every number in the audit report comes from here.

Inputs (read-only):
  data/raw/train-2025.csv                                   Roger's raw file
  data/interim/2026-11-08_panel_s5.parquet                  s5 labels (D-151) = Chloe's label rule
  ~/Downloads/non-aggregated feature data sets (80 features version 11.15)/round 1/*.csv   Chloe's features
  harness/2026-11-16_turquoise_submission/                  frozen round-03 pipeline (Chloe's code)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/data/2026-11-16_audit"
OUT.mkdir(parents=True, exist_ok=True)
CHLOE = Path.home() / "Downloads/non-aggregated feature data sets (80 features version 11.15)/round 1"
sys.path.insert(0, str(ROOT / "harness/2026-11-16_turquoise_submission"))
import inference  # noqa: E402  (Chloe's frozen pipeline)
import missing_rules  # noqa: E402

LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    LOG.append(msg)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


raw = pd.read_csv(ROOT / "data/raw/train-2025.csv", low_memory=False)
panel = pd.read_parquet(ROOT / "data/interim/2026-11-08_panel_s5.parquet", columns=["id", "fs_year", "y_s5", "status_s5"])
raw = raw.merge(panel, on=["id", "fs_year"], how="left", validate="one_to_one")
lab = raw["y_s5"]
log(f"# Audit run\nraw rows {len(raw):,}; labelled (s5) rows {lab.notna().sum():,}; default rate {lab.mean():.4%}")

# ---------------------------------------------------------------- ROUND 1: raw-data claims
log("\n## Round 1: re-verify documented raw-data claims (01-financial-anomalies.md, 02-missing-value.md)")
R = raw
tol = 1.0
checks = {
    "D: prof_operations = rev_operating - COGS": (R.prof_operations - (R.rev_operating - R.COGS)).abs() <= tol,
    "E: wc_net = asst_current - debt_st": (R.wc_net - (R.asst_current - R.debt_st)).abs() <= tol,
    "E: margin_fin = eqty_tot - fixed assets": (R.margin_fin - (R.eqty_tot - R.asst_intang_fixed - R.asst_tang_fixed - R.asst_fixed_fin)).abs() <= tol,
    "E: roa = prof_operations/asst_tot*100 (+-0.01)": (R.roa - R.prof_operations / R.asst_tot * 100).abs() <= 0.01,
    "I: roe = profit/eqty_tot*100 (+-0.01)": (R.roe - R.profit / R.eqty_tot * 100).abs() <= 0.01,
    "cf_operations = profit + (ebitda - prof_operations)": (R.cf_operations - (R.profit + R.ebitda - R.prof_operations)).abs() <= tol,
    "balance: eqty+debt_st+debt_lt+liab_lt+liab_lt_emp = asst_tot (within 1%)": ((R.eqty_tot + R.debt_st + R.debt_lt + R.liab_lt + R.liab_lt_emp) - R.asst_tot).abs() <= 0.01 * R.asst_tot,
    "assets: fixed+current = asst_tot (within 1%)": ((R.asst_intang_fixed + R.asst_tang_fixed + R.asst_fixed_fin + R.asst_current) - R.asst_tot).abs() <= 0.01 * R.asst_tot,
    "bank_st+fin_st+AP_st = debt_st (+-1)": ((R.debt_bank_st + R.debt_fin_st + R.AP_st) - R.debt_st).abs() <= tol,
    "bank_lt+fin_lt+AP_lt = debt_lt (+-1)": ((R.debt_bank_lt + R.debt_fin_lt + R.AP_lt) - R.debt_lt).abs() <= tol,
}
# match rate is computed only over rows where every input is present
def rate(expr_ok, *cols):
    m = R[list(cols)].notna().all(axis=1)
    return expr_ok[m].mean(), int(m.sum())
spec = {
    "D: prof_operations = rev_operating - COGS": ["prof_operations", "rev_operating", "COGS"],
    "E: wc_net = asst_current - debt_st": ["wc_net", "asst_current", "debt_st"],
    "E: margin_fin = eqty_tot - fixed assets": ["margin_fin", "eqty_tot", "asst_intang_fixed", "asst_tang_fixed", "asst_fixed_fin"],
    "E: roa = prof_operations/asst_tot*100 (+-0.01)": ["roa", "prof_operations", "asst_tot"],
    "I: roe = profit/eqty_tot*100 (+-0.01)": ["roe", "profit", "eqty_tot"],
    "cf_operations = profit + (ebitda - prof_operations)": ["cf_operations", "profit", "ebitda", "prof_operations"],
    "balance: eqty+debt_st+debt_lt+liab_lt+liab_lt_emp = asst_tot (within 1%)": ["eqty_tot", "debt_st", "debt_lt", "liab_lt", "liab_lt_emp"],
    "assets: fixed+current = asst_tot (within 1%)": ["asst_intang_fixed", "asst_tang_fixed", "asst_fixed_fin", "asst_current"],
    "bank_st+fin_st+AP_st = debt_st (+-1)": ["debt_bank_st", "debt_fin_st", "AP_st", "debt_st"],
    "bank_lt+fin_lt+AP_lt = debt_lt (+-1)": ["debt_bank_lt", "debt_fin_lt", "AP_lt", "debt_lt"],
}
r1 = pd.DataFrame([{"identity": k, "match_rate": rate(checks[k], *v)[0], "rows_computable": rate(checks[k], *v)[1]} for k, v in spec.items()])
r1.to_csv(OUT / "r1_identities.csv", index=False)
log(r1.to_string(index=False, float_format=lambda x: f"{x:.4%}"))

miss = R.drop(columns=["y_s5", "status_s5"]).isna().mean().sort_values(ascending=False)
miss = miss[miss > 0]
miss.rename("missing_share").to_csv(OUT / "r1_missing_rates.csv")
log("\nMissing shares (>0): " + ", ".join(f"{k} {v:.3%}" for k, v in miss.items()))
neg_cols = ["asst_intang_fixed", "asst_tang_fixed", "asst_fixed_fin", "liab_lt", "liab_lt_emp", "debt_bank_st", "debt_bank_lt", "debt_fin_st", "debt_fin_lt", "debt_st", "debt_lt"]
negbs = (R[neg_cols] < 0).any(axis=1)
log(f"neg_bs rows {negbs.sum():,} (doc: 5,117); default rate among labelled {lab[negbs].mean():.2%}")
log(f"asst_tot min {R.asst_tot.min():,.0f}; rows asst_tot<1.5M: {(R.asst_tot < 1.5e6).sum():,}")
log(f"eqty_tot<0 share {(R.eqty_tot < 0).mean():.2%}; goodwill<0 share {(R.goodwill < 0).mean():.2%}; rev_operating==0 share {(R.rev_operating == 0).mean():.2%}")
log(f"def_date years: {pd.to_datetime(R.def_date).dt.year.value_counts().sort_index().to_dict()}")

# ---------------------------------------------------------------- ROUND 2: imputation accuracy by masking
log("\n## Round 2: imputation accuracy (mask known values in FY2010 rows, impute with frozen round-3 medians, compare to truth)")
P, B = inference.load_parameters()
medians = pd.DataFrame(**P["medians"])
rng = np.random.default_rng(0)
base = R[R.fs_year == 2010].drop(columns=["y_s5", "status_s5"]).reset_index(drop=True)
mask_cols = ["asst_current", "AR", "cash_and_equiv", "eqty_tot", "debt_st", "debt_lt", "debt_bank_st", "debt_bank_lt", "AP_st",
             "rev_operating", "COGS", "ebitda", "profit", "exp_financing", "cf_operations", "liab_lt", "asst_tang_fixed", "roe"]
r2 = []
for c in mask_cols:
    known = base.index[base[c].notna()]
    idx = rng.choice(known, size=min(3000, len(known)), replace=False)
    masked = base.copy()
    truth = masked.loc[idx, c].to_numpy(float)
    masked.loc[idx, c] = np.nan
    filled = missing_rules.fill_missing(masked, medians)  # Method G off (harness setting)
    est = filled.loc[idx, c].to_numpy(float)
    scale = base.loc[idx, "asst_tot"].to_numpy(float) if c != "roe" else np.ones(len(idx))
    err = np.abs(est - truth) / scale
    rho = spearmanr(est / scale, truth / scale).statistic if np.std(est / scale) > 0 else np.nan
    r2.append({"column": c, "n_masked": len(idx), "median_abs_err_(x asst_tot; roe in pts)": np.median(err),
               "p90_abs_err": np.quantile(err, 0.9), "spearman_est_vs_truth": rho,
               "share_truth_exactly_0": float((truth == 0).mean()), "share_est_0": float((est == 0).mean())})
r2 = pd.DataFrame(r2)
r2.to_csv(OUT / "r2_imputation_masking.csv", index=False)
log(r2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------------------------------------------------------------- features for all rows (frozen round-3 pipeline)
feat = inference.transform(R.drop(columns=["y_s5", "status_s5"]), P)
feat = feat.reindex(R.index)
F = feat.copy()
F["id"], F["fs_year"], F["y"] = R.id.values, R.fs_year.values, lab.values

# ---------------------------------------------------------------- ROUND 3: independent recompute + schema/code consistency
log("\n## Round 3: independent recompute of features from raw (my own formulas, plain division) on complete-input rows")
c1 = pd.read_csv(CHLOE / "features_test.csv")  # FY2009, round-1 preprocessor
rr = R[R.fs_year == 2009].set_index("id")
td = rr.debt_st + rr.debt_lt
mine = pd.DataFrame({
    "size": np.log(rr.asst_tot), "equity_ratio": rr.eqty_tot / rr.asst_tot, "debt_to_eqty": td / rr.eqty_tot,
    "st_debt_share": rr.debt_st / td, "lt_debt_to_capital": rr.debt_lt / (rr.debt_lt + rr.eqty_tot),
    "bank_debt_conc": (rr.debt_bank_st + rr.debt_bank_lt) / td, "tangible_asset_ratio": rr.asst_tang_fixed / rr.asst_tot,
    "goodwill_ratio": rr.goodwill / rr.asst_tot, "cash_ratio": rr.cash_and_equiv / rr.asst_tot,
    "cash_to_financing_exp_ratio": rr.cash_and_equiv / rr.exp_financing, "cur_asst_ratio": rr.asst_current / rr.asst_tot,
    "AR_ratio": rr.AR / rr.rev_operating, "payables_ratio": rr.AP_st / rr.asst_tot, "wc_ratio": rr.wc_net / rr.asst_tot,
    "cur_debt_ratio": rr.asst_current / rr.debt_st, "quick_ratio": (rr.cash_and_equiv + rr.AR) / rr.debt_st,
    "asset_turnover": rr.rev_operating / rr.asst_tot, "roa": rr.roa, "roe": rr.roe, "op_margin": rr.prof_operations / rr.rev_operating,
    "net_prof_margin": rr.profit / rr.rev_operating, "int_coverage": rr.ebitda / rr.exp_financing,
    "financing_expense_ratio": rr.exp_financing / rr.asst_tot, "ebitda_to_debt": rr.ebitda / td,
    "operating_cash_flow_margin": rr.cf_operations / rr.rev_operating, "cash_interest_coverage": rr.cf_operations / rr.exp_financing,
    "operating_cash_flow_ratio": rr.cf_operations / rr.asst_tot, "operating_cash_flow_to_debt": rr.cf_operations / td,
    "loss_flag": (rr.profit < 0).astype(float),
})
mine = mine.replace([np.inf, -np.inf], np.nan)
cj = c1.set_index("id").loc[mine.index.intersection(c1.id)]
mm = mine.loc[cj.index]
r3 = []
for col in mm.columns:
    both = mm[col].notna() & cj[col].notna()
    eq = np.isclose(mm.loc[both, col], cj.loc[both, col], rtol=1e-9, atol=1e-12)
    r3.append({"feature": col, "rows_compared": int(both.sum()), "exact_match_rate": eq.mean(),
               "rows_mine_undefined(denominator 0/NaN)": int(mm[col].isna().sum())})
r3 = pd.DataFrame(r3)
r3.to_csv(OUT / "r3_independent_recompute.csv", index=False)
log(r3.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
# schema csv vs code
negpl_schema = (rr.rev_operating < 0) | (rr.COGS < 0) | (rr.exp_financing < 0) | (rr.ebitda < rr.prof_operations)
negpl_code = (rr.rev_operating < 0) | (rr.COGS < 0) | (rr.exp_financing < 0)
log(f"neg_pl_flag: schema definition (incl. ebitda<prof_operations) flags {int(negpl_schema.sum())} FY2009 rows; code flags {int(negpl_code.sum())}")
cic_nonpos = (rr.exp_financing <= 0).sum()
log(f"cash_interest_coverage: schema says NaN when exp_financing<=0; FY2009 rows with exp_financing<=0: {cic_nonpos:,}; code instead divides by (0+1) or by the negative value")

# ---------------------------------------------------------------- ROUND 4: zero/negative denominators vs monotone constraints
log("\n## Round 4: zero/negative denominators and monotone constraints (pooled labelled FY2007-2011)")
MONO = {"equity_ratio": -1, "st_debt_share": 1, "lt_debt_to_capital": 1, "cash_ratio": -1, "cash_to_financing_exp_ratio": -1,
        "AR_ratio": 1, "payables_ratio": 1, "wc_ratio": -1, "cur_debt_ratio": -1, "quick_ratio": -1, "roa": -1, "op_margin": -1,
        "net_prof_margin": -1, "int_coverage": -1, "financing_expense_ratio": 1, "operating_cash_flow_ratio": -1,
        "ebitda_to_debt": -1, "loss_flag": 1, "ateco_sector_te": 1}
L = R[lab.notna()]
yL = lab[lab.notna()]
tdL = L.debt_st + L.debt_lt
DEN = {"debt_to_eqty": L.eqty_tot, "st_debt_share": tdL, "lt_debt_to_capital": L.debt_lt + L.eqty_tot, "bank_debt_conc": tdL,
       "cash_to_financing_exp_ratio": L.exp_financing, "AR_ratio": L.rev_operating, "cur_debt_ratio": L.debt_st,
       "quick_ratio": L.debt_st, "op_margin": L.rev_operating, "net_prof_margin": L.rev_operating, "int_coverage": L.exp_financing,
       "ebitda_to_debt": tdL, "operating_cash_flow_margin": L.rev_operating, "cash_interest_coverage": L.exp_financing,
       "operating_cash_flow_to_debt": tdL}
r4 = []
for f, d in DEN.items():
    z, n = d.eq(0), d.lt(0)
    r4.append({"feature": f, "constraint": MONO.get(f, 0), "rows_denom_0": int(z.sum()), "DR_denom_0": yL[z].mean() if z.any() else np.nan,
               "rows_denom_neg": int(n.sum()), "DR_denom_neg": yL[n].mean() if n.any() else np.nan, "DR_all": yL.mean()})
r4 = pd.DataFrame(r4)
r4.to_csv(OUT / "r4_denominators.csv", index=False)
log(r4.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
# Constraint conflict: where does the 'broken' region sit in the ranking, and what is its default rate vs the rest of that end?
FL = F[F.y.notna()]
r4b = []
for f, s in MONO.items():
    if f not in DEN:
        continue
    x = FL[f].to_numpy(float)
    y = FL.y.to_numpy(float)
    d = DEN[f].reindex(FL.index).to_numpy(float)
    bad = d < 0
    if bad.sum() == 0:
        continue
    pct = (rankdata(x)[bad] / len(x))  # where negative-denominator rows land in the feature ranking
    r4b.append({"feature": f, "constraint": s, "rows_neg_denom": int(bad.sum()), "DR_neg_denom": y[bad].mean(),
                "median_percentile_of_those_rows": float(np.median(pct)),
                "constraint_implies": ("low PD" if (s == 1 and np.median(pct) < 0.5) or (s == -1 and np.median(pct) > 0.5) else "high PD")})
r4b = pd.DataFrame(r4b)
r4b.to_csv(OUT / "r4_constraint_conflicts.csv", index=False)
log("\nNegative-denominator rows vs monotone constraint:")
log(r4b.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------------------------------------------------------------- ROUND 5: dropped rows, labels, target-encoding
log("\n## Round 5: rows dropped by Chloe's pipeline, labels, target encoding")
for fname, yr in [("features_train.csv", 2007), ("features_validation.csv", 2008), ("features_test.csv", 2009)]:
    ch = pd.read_csv(CHLOE / fname, usecols=["id", "fs_year", "label", "ateco_sector_te", "pred_start_dt"])
    ry = R[(R.fs_year == yr) & R.y_s5.notna()]
    dropped = ry[~ry.id.isin(ch.id)]
    log(f"FY{yr} {fname}: labelled raw {len(ry):,}, Chloe rows {len(ch):,}, dropped {len(dropped):,}; "
        f"DR dropped {dropped.y_s5.mean():.2%} vs kept {ch.label.mean():.2%}")
    j = ch.merge(R[["id", "fs_year", "y_s5"]], on=["id", "fs_year"])
    log(f"   label agreement with our s5 label: {(j.label == j.y_s5).mean():.4%}; pred_start_dt values: {sorted(ch.pred_start_dt.unique())[:3]}")
    if len(dropped):
        why = feat.loc[dropped.index].isna().sum()
        log("   NaN features in dropped rows (top): " + ", ".join(f"{k}={v}" for k, v in why[why > 0].sort_values(ascending=False).head(6).items()))
    ind = missing_rules._ateco_industry_group(R.loc[R.fs_year == yr, "ateco_sector"], missing_rules._ateco_industry_lookup())
    ch2 = ch.merge(pd.DataFrame({"id": R.loc[R.fs_year == yr, "id"].values, "ind": ind.values}), on="id")
    nuniq = ch2.groupby("ind").ateco_sector_te.nunique()
    log(f"   ateco_sector_te distinct values per industry: median {nuniq.median():.0f}, max {nuniq.max()} "
        f"({'OOF (expected on train)' if fname == 'features_train.csv' else 'should be 1 on val/test'})")
dup_ids_years = R.duplicated(["id", "fs_year"]).sum()
log(f"duplicate (id, fs_year): {dup_ids_years}")

# ---------------------------------------------------------------- ROUND 6: distributions and univariate default relationships
log("\n## Round 6: distributions and empirical default rates (frozen round-3 features, labelled FY2007-2011)")
feats = P["feature_columns"]
r6 = []
dec_rows = []
for f in feats:
    x = FL[f].to_numpy(float)
    y = FL.y.to_numpy(float)
    ok = np.isfinite(x)
    x, yy = x[ok], y[ok]
    a = auc(yy, x)
    q = np.quantile(x, [0.01, 0.25, 0.5, 0.75, 0.99])
    r6.append({"feature": f, "n": int(ok.sum()), "p01": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p99": q[4],
               "share_zero": float((x == 0).mean()), "share_neg": float((x < 0).mean()), "univariate_AUC": a,
               "direction": "higher->riskier" if a > 0.5 else "higher->safer", "power_|AUC-0.5|": abs(a - 0.5),
               "spearman_with_default": spearmanr(x, yy).statistic, "constraint": MONO.get(f, 0)})
    if len(np.unique(x)) > 10:
        # bin on values, keeping tied values (e.g. many exact 0s or 1s) in the same bin
        bins = pd.qcut(pd.Series(x), 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"b": bins, "x": x, "y": yy}).groupby("b").agg(x_min=("x", "min"), x_max=("x", "max"), DR=("y", "mean"), n=("y", "size"))
        g.insert(0, "feature", f)
        dec_rows.append(g.reset_index())
    else:
        g = pd.DataFrame({"x": x, "y": yy}).groupby("x").agg(DR=("y", "mean"), n=("y", "size")).reset_index()
        g.insert(0, "feature", f)
        g["b"], g["x_min"], g["x_max"] = g.x, g.x, g.x
        dec_rows.append(g[["feature", "b", "x_min", "x_max", "DR", "n"]])
r6 = pd.DataFrame(r6)
r6["constraint_vs_data"] = np.where(r6.constraint == 0, "",
                                    np.where(((r6.constraint == 1) & (r6.univariate_AUC > 0.5)) | ((r6.constraint == -1) & (r6.univariate_AUC < 0.5)), "agrees", "CONFLICT"))
r6.sort_values("power_|AUC-0.5|", ascending=False).to_csv(OUT / "r6_feature_profile.csv", index=False)
pd.concat(dec_rows).to_csv(OUT / "r6_decile_default_rates.csv", index=False)
show = r6.sort_values("power_|AUC-0.5|", ascending=False)[["feature", "median", "share_zero", "share_neg", "univariate_AUC", "constraint", "constraint_vs_data"]]
log(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
# non-monotone check: decile DR pattern for constrained features
log("\nDecile default-rate pattern for constrained features (D1 = lowest values):")
dec = pd.concat(dec_rows)
for f in MONO:
    d = dec[dec.feature == f]
    if len(d) > 2:
        log(f"  {f:28s} ({len(d)} bins) " + " ".join(f"{v:.3f}" for v in d.DR))
# by-year stability of univariate AUC for the 15 strongest features
log("\nUnivariate AUC by year (top 15 features):")
top = r6.sort_values("power_|AUC-0.5|", ascending=False).feature.head(15)
yrs = sorted(FL.fs_year.unique())
log("  feature                      " + " ".join(f"{y}" for y in yrs))
for f in top:
    vals = []
    for yr in yrs:
        s = FL[FL.fs_year == yr]
        ok = np.isfinite(s[f])
        vals.append(auc(s.y[ok].to_numpy(float), s[f][ok].to_numpy(float)))
    log(f"  {f:28s} " + " ".join(f"{v:.3f}" for v in vals))

(OUT / "audit_log.md").write_text("\n".join(LOG) + "\n")
print(f"\nWrote outputs to {OUT}")
