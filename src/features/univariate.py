"""Univariate evidence for the feature layer (feature-architect, 2026-10-30).

Run:  python -m src.features.univariate
Uses ONLY training rows of fold 3 (fs_year 2007-2009, label_status == 'ok'); fs_year 2010 (a test
year) is never read into any statistic. Writes:
  reports/features/univariate.md, univariate_stats.csv, deciles.csv, deciles.png,
  correlation_spearman.csv, vif.csv, categorical_rates.csv, missing_treatment.csv,
  auc_by_year.csv, params_fold3_example.json
  docs/FEATURE_SPEC.md (spec table from build.FEATURES + the evidence above)
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.common.metrics import auc  # noqa: E402
from src.features import build as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "reports", "features")
SPEC = os.path.join(ROOT, "docs", "FEATURE_SPEC.md")
TRAIN_YEARS = [2007, 2008, 2009]
SIGN = {1: "+", -1: "−", 0: "none"}
CLUSTER_RHO = 0.70
VIF_LIMIT = 5.0


def md(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    return d.to_markdown(index=False)


def vif_table(Z: pd.DataFrame) -> pd.Series:
    R = np.corrcoef(Z.values, rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(R)), index=Z.columns)


def decile_rates(x: pd.Series, y: pd.Series) -> list[float]:
    """Ties stay together (a mass point such as 'no bank debt' fills one or more deciles, the
    rate is shown in the decile holding the tie's mid-rank; deciles with no mid-rank are NaN)."""
    m = x.notna()
    pct = x[m].rank(method="average", pct=True)
    q = np.minimum((pct * 10).astype(int), 9)
    return y[m].groupby(q.values).mean().reindex(range(10)).tolist()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    panel = pd.read_parquet(os.path.join(ROOT, "data", "interim", "panel.parquet"))
    tr = panel[(panel["label_status"] == "ok") & (panel["fs_year"].isin(TRAIN_YEARS))].copy()
    assert tr["fs_year"].max() <= 2009
    del panel
    y = tr["y"].astype(float).reset_index(drop=True)
    tr = tr.reset_index(drop=True)
    base = float(y.mean())

    params = B.fit(tr)
    B.save_params(params, os.path.join(OUT, "params_fold3_example.json"))
    X = B.transform(tr, params)                 # NaN kept (tree view)
    Xi = B.transform(tr, params, impute=True)   # imputed (GLM view)

    feats = B.NUM_FEATURES + B.FLAG_FEATURES
    rows, dec = [], {}
    for c in feats:
        m = X[c].notna()
        a = auc(y[m], X[c][m])
        ai = auc(y, Xi[c])
        emp = 0 if abs(a - 0.5) < 0.01 else (1 if a > 0.5 else -1)
        exp = B.META[c]["sign"]
        dec[c] = decile_rates(X[c], y) if c in B.NUM_FEATURES else None
        d = dec[c]
        rows.append(dict(
            feature=c, family=B.META[c]["family"], expected=SIGN[exp],
            auc=a, auc_imputed=ai, power=abs(a - 0.5) * 2,
            empirical=SIGN[emp],
            agrees=("n/a" if exp == 0 else ("yes" if emp == exp else ("weak" if emp == 0 else "NO"))),
            missing_rate=float(1 - m.mean()),
            dr_missing=float(y[~m].mean()) if (~m).sum() else np.nan,
            dr_present=float(y[m].mean()),
            dr_d1=d[0] if d else np.nan, dr_d10=d[9] if d else np.nan,
            winsor_lo=params["winsor"][c][0] if c in params["winsor"] else np.nan,
            winsor_hi=params["winsor"][c][1] if c in params["winsor"] else np.nan,
            core=B.META[c]["core"]))
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(OUT, "univariate_stats.csv"), index=False)
    dect = pd.DataFrame({c: v for c, v in dec.items() if v}, index=[f"D{i+1}" for i in range(10)]).T
    dect.to_csv(os.path.join(OUT, "deciles.csv"))

    # flags: default rate when 1 vs 0
    flag_rows = []
    for c in B.FLAG_FEATURES:
        on = Xi[c] == 1
        flag_rows.append(dict(flag=c, share=float(on.mean()), dr_flag1=float(y[on].mean()),
                              dr_flag0=float(y[~on].mean()), defaults_flag1=int(y[on].sum())))
    flagt = pd.DataFrame(flag_rows)

    # AUC by training year (stability, Roger M12)
    by_year = []
    for yr in TRAIN_YEARS:
        s = tr["fs_year"] == yr
        by_year.append({"feature": "n_defaults", "year": yr, "auc": float(y[s].sum())})
        for c in B.NUM_FEATURES:
            m = s & X[c].notna()
            by_year.append({"feature": c, "year": yr, "auc": auc(y[m], X[c][m])})
    aby = pd.DataFrame(by_year).pivot(index="feature", columns="year", values="auc")
    aby.to_csv(os.path.join(OUT, "auc_by_year.csv"))

    # decile plot
    nf = len(B.NUM_FEATURES)
    ncol = 4
    fig, axes = plt.subplots(int(np.ceil(nf / ncol)), ncol, figsize=(16, 3 * np.ceil(nf / ncol)))
    for ax, c in zip(axes.ravel(), B.NUM_FEATURES):
        ax.bar(range(1, 11), np.array(dec[c]) * 100, color="steelblue")
        ax.axhline(base * 100, color="k", lw=0.8, ls="--")
        a = st.set_index("feature").loc[c, "auc"]
        ax.set_title(f"{c} (exp {SIGN[B.META[c]['sign']]}, AUC {a:.3f})", fontsize=9)
        ax.set_xticks(range(1, 11))
        ax.tick_params(labelsize=7)
        ax.set_ylabel("default rate %", fontsize=7)
    for ax in axes.ravel()[nf:]:
        ax.axis("off")
    fig.suptitle("Default rate by feature decile, train fs_year 2007-2009 (dashed = base rate)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "deciles.png"), dpi=110)
    plt.close(fig)

    # categoricals
    cat_rows = []
    for c in B.CAT_FEATURES:
        g = pd.DataFrame({"lvl": Xi[c].astype(str), "y": y}).groupby("lvl")["y"]
        for lvl, n in g.size().items():
            cat_rows.append(dict(feature=c, level=lvl, rows=int(n), defaults=int(g.sum()[lvl]),
                                 default_rate=float(g.mean()[lvl]),
                                 reference=(lvl == params["reference_level"][c])))
    catt = pd.DataFrame(cat_rows)
    catt.to_csv(os.path.join(OUT, "categorical_rates.csv"), index=False)
    # which raw levels were pooled into OTHER
    raw_cats = B._cats_raw(B._reconstruct(B.coerce_input(tr)))
    pooled = {}
    for c in B.CAT_FEATURES:
        lv = raw_cats[c].fillna("missing")
        d = y.groupby(lv.values).agg(["size", "sum"])
        pooled[c] = {k: (int(v["size"]), int(v["sum"])) for k, v in d.iterrows()
                     if k not in params["categories"][c]}

    # correlation clusters + VIF (imputed, winsorised values)
    num = B.NUM_FEATURES
    rho = Xi[num].rank().corr()
    rho.to_csv(os.path.join(OUT, "correlation_spearman.csv"))
    pairs = [(a, b, rho.loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:]
             if abs(rho.loc[a, b]) >= CLUSTER_RHO]
    pairs = sorted(pairs, key=lambda t: -abs(t[2]))
    core_num = [c for c in B.CORE_FEATURES if c in num or c in B.FLAG_FEATURES]
    v_full = vif_table(Xi[num + B.FLAG_FEATURES])
    v_core = vif_table(Xi[core_num])
    vif = pd.DataFrame({"vif_full_set": v_full, "vif_core_set": v_core}).reset_index(names="feature")
    vif.to_csv(os.path.join(OUT, "vif.csv"), index=False)

    # missing-data treatment comparison for the debt-split block (G2)
    blk = X["debt_split_missing"] == 1
    mt = []
    for c in B.BLOCK_FEATURES:
        med = params["impute_median"][c]
        grp = Xi["ateco_group"].astype(str)
        naive = X[c].fillna(grp.map(med).astype(float).fillna(med["__global__"]))
        # naive median must use medians of the observed (non-imputed) values: recompute
        obs_med = X[c].groupby(grp).median()
        naive = X[c].fillna(grp.map(obs_med).astype(float).fillna(float(X[c].median())))
        for name, v in [("share_x_total (adopted)", Xi[c]), ("sector_median", naive)]:
            Z = np.column_stack([v.values, X["debt_split_missing"].values])
            Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
            lr = LogisticRegression(max_iter=500).fit(Z, y)
            mt.append(dict(feature=c, treatment=name, auc_feature_alone=auc(y, v),
                           auc_feature_plus_flag=auc(y, lr.predict_proba(Z)[:, 1]),
                           auc_on_block_missing_rows=auc(y[blk], v[blk])))
        mt.append(dict(feature=c, treatment="NaN rows dropped (tree view)",
                       auc_feature_alone=auc(y[~X[c].isna()], X[c].dropna()),
                       auc_feature_plus_flag=np.nan, auc_on_block_missing_rows=np.nan))
    mtt = pd.DataFrame(mt)
    mtt.to_csv(os.path.join(OUT, "missing_treatment.csv"), index=False)
    blk_rate = dict(rows_missing=int(blk.sum()), share=float(blk.mean()),
                    dr_missing=float(y[blk].mean()), dr_present=float(y[~blk].mean()))

    # identity reconstruction count on train
    raw0 = B.coerce_input(tr)
    rec = B._reconstruct(raw0)
    recon = {c: int((raw0[c].isna() & rec[c].notna()).sum()) for c in B.RAW_NUM
             if (raw0[c].isna() & rec[c].notna()).sum() > 0}

    # core-set multivariate sanity: logit on core design matrix, coefficient signs
    Zc = B.design_matrix(Xi, params, B.CORE_FEATURES)
    Zs = (Zc - Zc.mean()) / Zc.std().replace(0, 1)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Zs, y)
    coef = pd.Series(lr.coef_[0], index=Zc.columns)
    core_auc_insample = auc(y, lr.predict_proba(Zs)[:, 1])
    coef_rows = []
    for c in core_num:
        e = B.META[c]["sign"]
        s_ = int(np.sign(coef[c]))
        coef_rows.append(dict(feature=c, expected=SIGN[e], coef_std=float(coef[c]),
                              multivariate_sign=SIGN[s_],
                              agrees=("n/a" if e == 0 else ("yes" if s_ == e else "NO"))))
    coeft = pd.DataFrame(coef_rows)
    coeft.to_csv(os.path.join(OUT, "core_logit_signs.csv"), index=False)

    # rejected candidates: evidence computed on the same training rows
    r = B._reconstruct(B.coerce_input(tr))
    ta = r["asst_tot"]
    def _a(v):
        v = v.replace([np.inf, -np.inf], np.nan); m = v.notna()
        return auc(y[m], v[m]), float(1 - m.mean())
    def _rho(a, b):
        return float(pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna().rank().corr().iloc[0, 1])
    ebt_ok = float(((r["taxes"] - (r["prof_operations"] + r["inc_financing"] + r["inc_extraord"]
                     + r["prof_financing"])).abs() <= np.maximum(2, 0.01 * r["taxes"].abs())).mean())
    cand = {
        "tax_to_assets (taxes / asst_tot)": (r["taxes"] / ta, X["roa_net"],
            f"`taxes` equals profit before tax (= prof_operations + inc_financing + inc_extraord + prof_financing on {ebt_ok:.1%} of rows), so this is pre-tax ROA: duplicate of roa_net"),
        "roe_recomputed (profit / eqty_tot, equity > 0)": (r["profit"] / r["eqty_tot"].where(r["eqty_tot"] > 0), X["roa_net"],
            "undefined for negative equity (the riskiest rows); explodes as equity → 0; replaced by roa_net + equity_ratio + neg_equity"),
        "roe (file field)": (r["roe"], X["roa_net"], "truncated at ±150% and blanked when equity ≤ 0 (01_profile §5): missingness = negative equity"),
        "days_rec (file field)": (pd.to_numeric(tr["days_rec"], errors="coerce"), X["ar_days"], "72% missing, negative on 7% of rows, inconsistent with AR/revenue; replaced by ar_days"),
        "margin_fin / asst_tot": ((r["eqty_tot"] - r["asst_intang_fixed"] - r["asst_tang_fixed"] - r["asst_fixed_fin"]) / ta, X["equity_ratio"],
            "equity minus fixed assets: mixes leverage and asset structure already covered by equity_ratio and tangible_ratio"),
        "net_fin_result_to_assets (inc_financing / asst_tot)": (r["inc_financing"] / ta, X["interest_to_assets"],
            "field meaning is a data-derived hypothesis (net financial result); exp_financing is unambiguous and used instead (M9)"),
        "value_added_to_assets (goodwill / asst_tot)": (r["goodwill"] / ta, X["roa_oper"],
            "`goodwill` behaves like value added (D-010 hypothesis); cannot be explained to a credit officer under its file name"),
        "log_revenue": (np.log(r["rev_operating"].where(r["rev_operating"] > 0)), X["log_assets"],
            "second size measure; size enters once (log_assets)"),
        "employee_fund_to_assets (liab_lt_emp / asst_tot)": (r["liab_lt_emp"] / ta, X["log_assets"],
            "no clear expected sign (labour intensity), negligible power"),
        "lt_debt_ratio (debt_lt / asst_tot)": (r["debt_lt"] / ta, X["equity_ratio"],
            "sign ambiguous (long-term debt is safer than short-term but still leverage); weak power; maturity captured by st_debt_ratio / st_debt_share"),
    }
    rej = []
    for k, (v, ref, why) in cand.items():
        a_, miss_ = _a(v)
        rej.append(dict(candidate=k, auc=a_, missing_rate=miss_, closest_kept=ref.name,
                        spearman_with_kept=_rho(v, ref), reason=why))
    rejt = pd.DataFrame(rej)
    rejt.to_csv(os.path.join(OUT, "rejected_candidates.csv"), index=False)
    core_pairs = [abs(rho.loc[a_, b_]) for i, a_ in enumerate(core_num) for b_ in core_num[i + 1:]
                  if a_ in num and b_ in num]
    core_stats = dict(max_abs_rho=float(max(core_pairs)), max_vif=float(v_core.max()))

    write_univariate_md(st, dect, flagt, aby, catt, pooled, rho, pairs, vif, mtt, blk_rate,
                        recon, coeft, core_auc_insample, params, base, len(tr), int(y.sum()), rejt)
    write_spec(st, vif, pairs, params, base, len(tr), int(y.sum()), mtt, blk_rate, coeft, pooled,
               rejt, core_stats)
    print("done", len(tr), int(y.sum()))


# ---------------------------------------------------------------------------------------------
def write_univariate_md(st, dect, flagt, aby, catt, pooled, rho, pairs, vif, mtt, blk_rate,
                        recon, coeft, core_auc, params, base, n, nd, rejt):
    L = ["# Feature univariate evidence (train fs_year 2007-2009 only)", "",
         "Generated by `python -m src.features.univariate` (feature-architect, 2026-10-30). "
         f"Rows: {n:,} usable training rows of fold 3 (fs_year 2007-2009, primary label), "
         f"{nd:,} defaults, base rate {base:.4f}. fs_year 2010 (test year of fold 3) is never "
         "read. Parameters fitted on the same rows: `params_fold3_example.json` (challengers "
         "refit per fold with `build.fit`).", "",
         "AUC is computed on rows where the feature is defined (`auc`) and after imputation "
         "(`auc_imputed`). AUC < 0.5 means PD falls as the feature rises (expected sign −). "
         "`empirical` is the sign implied by the AUC (|AUC−0.5| < 0.01 → none). "
         "`dr_d1` / `dr_d10` = default rate in the lowest / highest decile.", "",
         "## 1. Univariate power, direction and missingness", "",
         md(st.sort_values("power", ascending=False)), "",
         "Decile plot: `deciles.png`; decile table: `deciles.csv`.", "",
         "![deciles](deciles.png)", "",
         "## 2. Default rate by decile (%)", "",
         md((dect * 100).reset_index(names="feature"), "{:.2f}"), "",
         "## 3. Flags", "", md(flagt), "",
         "## 4. Stability: univariate AUC by training year (Roger M12)", "",
         md(aby.reset_index()), "",
         "## 5. Categoricals (levels after pooling; reference = largest level)", "",
         f"Pooling rule: {params['min_count_rule']} per level, learned in `fit` on the training "
         "rows; unseen or missing codes → OTHER.", "",
         md(catt), "",
         "Raw levels pooled into OTHER in this window (rows, defaults): "
         + "; ".join(f"`{k}`: {v}" for k, v in pooled.items()), "",
         "## 6. Correlation clusters and VIF (Roger M8)", "",
         f"Spearman |rho| ≥ {CLUSTER_RHO} pairs (imputed, winsorised values; full matrix in "
         "`correlation_spearman.csv`):", "",
         md(pd.DataFrame(pairs, columns=["feature_a", "feature_b", "spearman"])), "",
         "VIF (= diagonal of the inverse Pearson correlation matrix of the winsorised, imputed "
         "features). Rule: VIF > 5 inside the core set → drop or replace the feature; in the "
         "full (GBM) set high VIF is tolerated because trees and regularised fits do not need "
         "invertible designs, but redundancy is reported.", "",
         md(vif), "",
         "## 7. Missing-data treatment evidence for the debt-split block (Roger G2)", "",
         f"Block missing (any of debt_bank_st, debt_bank_lt, AP_st): {blk_rate['rows_missing']:,} "
         f"rows ({blk_rate['share']:.2%}); default rate {blk_rate['dr_missing']:.4f} missing vs "
         f"{blk_rate['dr_present']:.4f} present.", "",
         md(mtt), "",
         "Raw fields filled by an exact accounting identity in `_reconstruct` (train rows): "
         + (", ".join(f"{k}: {v}" for k, v in recon.items()) or "none"), "",
         "## 8. Core set: multivariate sign check (in-sample logit, standardised, train rows)", "",
         f"In-sample AUC of the core logit (diagnostic only, not a tournament result): {core_auc:.4f}.", "",
         md(coeft), "",
         "## 9. Rejected candidates (same training rows)", "", md(rejt), ""]
    with open(os.path.join(OUT, "univariate.md"), "w") as fh:
        fh.write("\n".join(L))


def write_spec(st, vif, pairs, params, base, n, nd, mtt, blk_rate, coeft, pooled, rejt, core_stats):
    s = st.set_index("feature")
    v = vif.set_index("feature")
    rows = []
    for f in B.FEATURES:
        c = f["name"]
        kind = f.get("kind", "num")
        if kind == "cat":
            miss = "missing/unseen → OTHER"
            ev = "see univariate.md §5"
        else:
            if c in B.BLOCK_FEATURES:
                miss = ("(b') NaN for trees; impute: sector-median share × own debt total "
                        "+ flag `debt_split_missing`")
            elif kind == "flag":
                miss = "never missing (0/1)"
            else:
                miss = "(a) identity rebuild of inputs; then NaN for trees; impute: train sector median"
            r = s.loc[c]
            ev = (f"AUC {r['auc']:.3f}; D1 {r['dr_d1']:.2%} → D10 {r['dr_d10']:.2%}"
                  if kind == "num" else f"AUC {r['auc']:.3f}")
        wb = (f"[{params['winsor'][c][0]:.4g}, {params['winsor'][c][1]:.4g}]"
              if c in params["winsor"] else "—")
        agree = s.loc[c, "agrees"] if c in s.index else "n/a"
        rows.append({"feature": f"`{c}`", "core": "**core**" if f["core"] else "full",
                     "family": f["family"], "formula": f["formula"],
                     "exp. sign vs PD": SIGN[f["sign"]], "economic reason": f["reason"],
                     "zero/negative-denominator rule": f["zero_rule"],
                     "missing treatment": miss, "winsor [p1,p99] (fold-3 train)": wb,
                     "train evidence": ev, "data agrees": agree,
                     "VIF core": (f"{v.loc[c, 'vif_core_set']:.2f}" if c in v.index and
                                  pd.notna(v.loc[c, "vif_core_set"]) else "")})
    table = pd.DataFrame(rows).to_markdown(index=False)
    signs = json.dumps(B.expected_signs(), indent=None)
    contra = st[st["agrees"].isin(["NO", "weak"])]
    contra_txt = ("; ".join(f"`{r.feature}` expected {r.expected}, univariate {r.empirical} "
                            f"(AUC {r.auc:.3f})" for r in contra.itertuples()) or "none")
    mcontra = coeft[coeft["agrees"] == "NO"]
    mcontra_txt = ("; ".join(f"`{r.feature}` expected {r.expected}, multivariate {r.multivariate_sign}"
                             for r in mcontra.itertuples()) or "none")
    rk = st.sort_values("power", ascending=False).reset_index(drop=True)
    ranked = "\n".join(
        f"{i+1}. `{r.feature}` ({r.family}, exp {r.expected}, AUC {r.auc:.3f}"
        f"{', core' if r.core else ''}) — {B.META[r.feature]['reason'].split(':')[0].split(';')[0].rstrip('.')}."
        for i, r in enumerate(rk.itertuples()))
    rejected = "\n".join(
        f"- **{r.candidate}** — AUC {r.auc:.3f}, Spearman {r.spearman_with_kept:+.2f} with kept "
        f"`{r.closest_kept}`: {r.reason}." for r in rejt.itertuples())
    text = f"""# FEATURE SPEC — Banca Massiccia 1-year PD (feature-architect, 2026-10-30)

Code: `src/features/build.py` (`fit`, `transform`, `design_matrix`, `expected_signs`).
Evidence: `reports/features/univariate.md` (all numbers below are generated by
`python -m src.features.univariate` from training rows fs_year 2007-2009 only: {n:,} rows,
{nd:,} defaults, base rate {base:.4f}; fs_year 2010 is never read).
This file is regenerated by that script; edit the metadata in `build.FEATURES`, not this file.

## How to use (challengers)

```python
from src.features import build as B
params = B.fit(train_rows)                        # train rows of the fold ONLY (label col 'y')
X      = B.transform(rows, params)                # NaN kept: GBM view
Xi     = B.transform(rows, params, impute=True)   # NaN filled: GLM / probit view
Z      = B.design_matrix(Xi, params, B.CORE_FEATURES)   # one-hot, reference levels dropped
signs  = B.expected_signs()                       # {{feature: +1/-1/0}} for monotone constraints
```

Refit `params` inside every fold (bounds, medians, category levels all come from training rows).
The feature *set* is fixed by this spec for every fold (Roger M12); only parameters change.

## Design principles (each defended in DECISION_LOG D-017..D-024)

1. **Ratios, size once.** Every feature is scale-free except `log_assets`. Raw euro amounts
   (exp_financing, inc_financing, cash flow) are never used directly (Roger M9).
2. **Seven families** following RiskCalc v3.1 (R-01 Sec. 3.1, p. 13: profitability, leverage,
   debt coverage, growth, liquidity, activity, size; one ratio per family at least, too many ratios
   overfit). Growth is not available (D-019). We add "asset structure" (collateral) as an eighth.
3. **Expected sign for every feature** (table below); `sign = none` means no monotone constraint.
4. **Ratio pathologies handled explicitly** (column "zero/negative-denominator rule"): total
   assets is ≥ EUR 1.5MM by construction, so most ratios divide by it; where the natural ratio has
   a fragile denominator we either use a bounded alternative (equity/assets instead of
   debt/equity or ROE) or assign the economically correct limit (+inf → clipped to the upper
   winsor bound) and document it.
5. **Winsorisation** at the training-window 1st/99th percentiles (stored in params; ±inf from a
   documented rule lands on the bound).
6. **Missing data**: (a) exact accounting identities rebuild missing inputs first; (b) for the
   debt-split block a missing flag + identity-bounded share imputation; (c) NaN left for trees.
   `roe`, `margin_fin`, `days_rec` from the file are not used (recomputed or replaced).
7. **Categoricals as labels**: ATECO section, legal form, macro-area; pooled when < 30 training
   defaults; unseen → OTHER. Never numeric codes.

## Feature table

{table}

Machine-readable expected signs (`build.expected_signs()`): `{signs}`

## Where the data disagree with the expected sign

- Univariate (train 2007-2009): {contra_txt}.
- Multivariate core logit (in-sample diagnostic): {mcontra_txt}.
See the decision log (D-021) for how each is handled.

## Core set (probit / logit) vs full set (GBM)

- **Core ({len(B.CORE_FEATURES)} incl. 3 categoricals)**: {", ".join(f"`{c}`" for c in B.CORE_FEATURES)}.
  One or two per family. Computed: max |Spearman rho| between core numeric features
  {core_stats['max_abs_rho']:.2f} (rule < {CLUSTER_RHO}), max core VIF {core_stats['max_vif']:.2f} (rule < {VIF_LIMIT:g}); every
  core coefficient sign checked in univariate.md §8 and on an inner 2007-08 → 2009 split in
  `reports/features/core_selection.csv` (set E = adopted core).
- **Full ({len(B.ALL_FEATURES)})**: all features in the table. Redundant pairs are listed in
  univariate.md §6; GBM handles them, but SHAP attributions are split across correlated
  features — report driver *families*, not single ratios, when explaining.

## Ranked candidate list (by univariate power |2·AUC − 1|, train 2007-2009)

{ranked}

## Rejected features (evidence: `reports/features/rejected_candidates.csv`)

{rejected}

Also rejected without a data test: **growth / lagged ratios** (D-019: the current record has no
prior-year fields and the harness may not use a firm's other holdout rows; a frozen training
lookup is OPEN pending email Q4), **NLP features** (D-003: no text fields), **external macro
variables** (D-004: not cleared; cycle handled in calibration), raw euro amounts (M9),
raw `HQ_city` / division one-hot (too thin: 45 of 110 provinces and 36 of 83 divisions have
< 30 defaults, 01_profile.md §3).

## Missing-data treatment (Roger G2)

- Debt-split block (bank / financial / trade debt): missing in {blk_rate['share']:.2%} of training
  rows, default rate {blk_rate['dr_missing']:.4f} vs {blk_rate['dr_present']:.4f}. Totals
  `debt_st`/`debt_lt` are present, so each missing component is rebuilt as
  (train sector-median share of the total) × (the firm's own total) — the identity
  `component ≤ total` bounds it — and `debt_split_missing` = 1 lets the model price the
  missingness. Evidence in univariate.md §7.
- `margin_fin`, `roe`: blanked by the provider exactly when equity ≤ 0 (or |ROE| ≥ 150%);
  not used. Equity enters through `equity_ratio` + `neg_equity`, profitability through `roa_net`.
- `days_rec`: 72% missing and inconsistent; replaced by `ar_days` = 365·AR/revenue.
- `eqty_corp_family_tot`: 100% empty; dropped.
- All other raw fields are missing on < 0.1% of rows: identity rebuild where possible, else
  NaN (trees) or train sector median (GLM). Pooled categorical levels: {pooled}.
"""
    with open(SPEC, "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
