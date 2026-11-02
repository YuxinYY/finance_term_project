"""GB-22: face validity and TreeSHAP for the full2 GBM (fold-3 model, trained on 2007-2009).

usage: python experiments/gbm/explain_r2b.py GB-20 [GB-01]
Writes experiments/gbm/GB-22/{face_validity_<exp>.csv, shap_top15_<exp>.csv, shap_by_family_<exp>.csv,
dependence_bins.csv, tfr_by_segment.csv, ice_check.csv, reason_codes.csv, dep_*.png}.

Explanations are computed on a random sample of 2010 rows (the fold-3 test year; nothing is fitted on
them, as in GB-01/explain.md). TreeSHAP = LightGBM pred_contrib (R-41), log-odds scale.
"""
from __future__ import annotations

import os
import pickle
import sys

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gbm_lib as G  # noqa: E402
from src.features import build as B  # noqa: E402

ROOT = G.ROOT
OUT = os.path.join(HERE, "GB-22")
N_SHAP, N_PD, N_ICE = 20000, 3000, 500
GRID_Q = np.linspace(0.01, 0.99, 41)
FREE_NEW = ["interest_to_sales", "bs_gap", "fin_fixed_ratio", "tfr_ratio"]
ENC = ["ateco_div_te", "province_te", "region_te"]
PLAIN = {
    "tangible_ratio": "few fixed tangible assets to back the debt", "asset_turnover": "sales relative to assets",
    "st_debt_ratio": "much debt falling due within a year", "ar_days": "customers take long to pay, or almost no revenue",
    "ap_days": "the firm takes long to pay suppliers", "payables_to_sales": "large trade payables relative to sales",
    "roa_net": "weak net profit on assets", "roa_net_sect_pct": "net profitability low against sector peers",
    "equity_ratio": "thin equity", "equity_ratio_sect_pct": "equity thin against sector peers",
    "interest_coverage_sect_pct": "interest cover low against sector peers", "interest_coverage": "earnings barely cover interest",
    "tfr_ratio": "little staff-severance fund: few or short-tenured staff, or a holding/property structure",
    "province_te": "province with a historically high default rate", "region_te": "region with a historically high default rate",
    "ateco_div_te": "industry division with a historically high default rate",
    "extraord_to_assets": "extraordinary items (one-off gains or losses)", "ateco_group": "industry group",
    "interest_to_sales": "interest cost relative to sales", "bs_gap": "unclassified liabilities on the balance sheet",
}


def load_art(exp):
    with open(os.path.join(HERE, f".{exp}_fold3_art.pkl"), "rb") as fh:
        d = pickle.load(fh)
    d["booster"] = lgb.Booster(model_str=d.pop("booster_str"))
    d["feature_set"] = "full2" if len(d["features"]) > len(B.ALL_FEATURES) else "full"
    return d


def X_of(art, df):
    return B.transform(df, art["feature_params"], feature_set=art["feature_set"])[art["features"]]


def pdp(bst, X, feat, grid):
    out = []
    for g in grid:
        Xg = X.copy()
        Xg[feat] = g
        out.append(np.mean(bst.predict(Xg, raw_score=True)))
    return np.array(out)


def main(exps):
    os.makedirs(OUT, exist_ok=True)
    panel = pd.read_parquet(os.path.join(ROOT, "data/interim/panel.parquet"), filters=[("fs_year", "<", 2011)])
    assert panel["fs_year"].max() <= 2010
    te = panel[(panel["fs_year"] == 2010) & (panel["label_status"] == "ok")].reset_index(drop=True)
    sub = te.iloc[np.random.default_rng(0).choice(len(te), N_SHAP, replace=False)].reset_index(drop=True)
    for exp in exps:
        art = load_art(exp)
        bst, feats = art["booster"], art["features"]
        signs = B.expected_signs(feats)
        X = X_of(art, sub)
        sv = pd.DataFrame(bst.predict(X, pred_contrib=True), columns=feats + ["_bias"])
        imp = sv[feats].abs().mean().sort_values(ascending=False)
        imp.head(15).rename("mean_abs_shap").to_frame().assign(
            share=lambda d: d["mean_abs_shap"] / imp.sum(),
            new_in_full2=lambda d: [f in B.NEW_FEATURES_V2 for f in d.index]).to_csv(
            os.path.join(OUT, f"shap_top15_{exp}.csv"), index_label="feature")
        fam = {f: B.META[f].get("family", "categorical") if f in B.META else "categorical" for f in feats}
        (sv[feats].abs().mean().groupby(pd.Series(fam)).sum().sort_values(ascending=False)
         .rename("mean_abs_shap_sum").to_csv(os.path.join(OUT, f"shap_by_family_{exp}.csv"), index_label="family"))
        # ---- monotone check: PD curve (3,000 rows) + per-row ICE (500 rows), every constrained feature
        rows, ice_rows = [], []
        splits = dict(zip(bst.feature_name(), bst.feature_importance("split")))
        Xpd, Xice = X.iloc[:N_PD], X.iloc[:N_ICE]
        for f in feats:
            if f in B.CAT_FEATURES:
                continue
            s = signs.get(f, 0)
            v = X[f].dropna()
            grid = np.unique(np.r_[v.min(), np.quantile(v, GRID_Q), v.max()])  # rare values must be tested too
            curve = pdp(bst, Xpd, f, grid)
            d = np.diff(curve)
            m = X[f].notna()
            rho = spearmanr(X.loc[m, f], sv.loc[m, f]).correlation if v.nunique() > 1 else np.nan
            viol = 0.0 if s != 0 else np.nan
            if s != 0 and len(d):
                viol = float(np.abs(d[(d * s) < -1e-9]).sum() / max(np.abs(d).sum(), 1e-12))
                # per-row ICE: worst step against the sign for any of 500 rows
                ice = np.stack([bst.predict(Xice.assign(**{f: g}), raw_score=True) for g in grid], axis=1)
                worst = float(np.max(-(np.diff(ice, axis=1) * s))) if ice.shape[1] > 1 else 0.0
                ice_rows.append({"feature": f, "sign": s, "n_rows": N_ICE, "grid_points": len(grid),
                                 "max_step_against_sign_logodds": max(worst, 0.0),
                                 "rows_with_any_violation": int(((np.diff(ice, axis=1) * s) < -1e-9).any(axis=1).sum())})
            rows.append({"feature": f, "new_in_full2": f in B.NEW_FEATURES_V2,
                         "expected": {1: "+", -1: "-", 0: "none"}[s], "constrained": s != 0,
                         "mean_abs_shap": imp[f], "rank": int(list(imp.index).index(f)) + 1,
                         "n_splits": int(splits[f]), "grid_points": len(grid),
                         "x_min": float(v.min()), "x_max": float(v.max()),
                         "shap_spearman": rho, "pd_start": curve[0], "pd_end": curve[-1],
                         "pd_range": curve.max() - curve.min(), "pd_violation_share": viol,
                         "verdict": ("no expectation" if s == 0 else
                                     "unused (0 splits)" if splits[f] == 0 else
                                     "no movement" if np.ptp(curve) == 0 else
                                     "agrees" if (viol < 1e-9 and (curve[-1] - curve[0]) * s >= 0) else "VIOLATION")})
            if exp == exps[0] and (f in FREE_NEW + ENC):
                _dep_plot(X, sv, sub, f, grid, curve, exp)
        fv = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
        fv.to_csv(os.path.join(OUT, f"face_validity_{exp}.csv"), index=False)
        pd.DataFrame(ice_rows).to_csv(os.path.join(OUT, f"ice_check_{exp}.csv"), index=False)
        print(exp, "violations:", (fv["verdict"] == "VIOLATION").sum(), "\n", fv.head(20).round(4).to_string(index=False))
        if exp != exps[0]:
            continue
        # ---- binned dependence tables for the unconstrained new features + encodings
        y = sub["y"].astype(int).to_numpy()
        bins = []
        for f in FREE_NEW + ENC:
            x = X[f]
            q = pd.qcut(x.rank(method="first"), 10, labels=False)
            zero = (x == 0)
            g = pd.DataFrame({"f": f, "bin": np.where(x.isna(), "NaN", np.where(zero & (f in FREE_NEW), "exact 0",
                                                                                   "D" + (q + 1).astype("Int64").astype(str))),
                              "x": x, "shap": sv[f], "y": y})
            t = g.groupby("bin").agg(n=("y", "size"), x_min=("x", "min"), x_median=("x", "median"), x_max=("x", "max"),
                                     mean_shap=("shap", "mean"), sd_shap=("shap", "std"),
                                     dr_2010=("y", "mean")).reset_index()
            t.insert(0, "feature", f)
            bins.append(t)
        pd.concat(bins).to_csv(os.path.join(OUT, "dependence_bins.csv"), index=False)
        # ---- tfr_ratio: what does it separate? by sector / zero-TFR / personnel intensity
        sec = B._section(sub["ateco_sector"]) if hasattr(B, "_section") else sub["ateco_sector"]
        seg = pd.DataFrame({"section": np.asarray(sec), "tfr": X["tfr_ratio"], "shap_tfr": sv["tfr_ratio"],
                            "personnel_to_sales": X["personnel_to_sales"], "asset_turnover": X["asset_turnover"],
                            "tangible_ratio": X["tangible_ratio"], "y": y})
        seg["tfr_zero"] = seg["tfr"] <= 0
        seg["tfr_band"] = pd.cut(seg["tfr"], [-np.inf, 0, 0.02, 0.05, 0.10, np.inf],
                                 labels=["0 (no staff fund)", "0-2%", "2-5%", "5-10%", ">10%"])
        t1 = seg.groupby("tfr_band", observed=True).agg(n=("y", "size"), dr=("y", "mean"), mean_shap=("shap_tfr", "mean"),
                                                         share_section_L=("section", lambda s: (s == "L").mean()),
                                                         med_personnel_to_sales=("personnel_to_sales", "median"),
                                                         med_asset_turnover=("asset_turnover", "median"),
                                                         med_tangible=("tangible_ratio", "median")).reset_index()
        t1.insert(0, "table", "tfr band (all sections)")
        segL = seg.assign(grp=np.where(seg["section"] == "L", "real estate (L)", "other sections"))
        t2 = segL.groupby(["grp", "tfr_band"], observed=True).agg(n=("y", "size"), dr=("y", "mean"),
                                                                  mean_shap=("shap_tfr", "mean")).reset_index()
        t2.insert(0, "table", "tfr band x real estate")
        # interaction proxy: within-band residual SHAP correlated with other features
        resid = seg["shap_tfr"] - seg.groupby("tfr_band", observed=True)["shap_tfr"].transform("mean")
        inter = pd.Series({f: spearmanr(X[f], resid, nan_policy="omit").correlation for f in feats
                           if f not in B.CAT_FEATURES and f != "tfr_ratio"}).abs().sort_values(ascending=False)
        inter.head(8).rename("abs_spearman_with_tfr_shap_residual").to_csv(os.path.join(OUT, "tfr_interaction_proxy.csv"),
                                                                           index_label="feature")
        pd.concat([t1, t2]).to_csv(os.path.join(OUT, "tfr_by_segment.csv"), index=False)
        # ---- reason codes: raw top-3 vs family-merged top-3 (sector percentile and base ratio are one concept)
        c = art["calibrator"]
        p = G._sigmoid(c["a"] + c["b"] * G._logit(bst.predict(X)))
        order = np.argsort(p)
        concept = {f: B.META[f].get("family", f) if f in B.META else f for f in feats}
        concept.update({f"{b_}_sect_pct": concept[b_] for b_ in B.PCT_BASE})
        fam_sv = sv[feats].T.groupby(pd.Series(concept)).sum().T
        top3 = np.argsort(-sv[feats].to_numpy(), axis=1)[:, :3]
        conc = np.array([concept[f] for f in feats])
        dup = np.array([len(set(conc[t])) < 3 for t in top3])
        new_any = np.array([any(feats[i] in B.NEW_FEATURES_V2 for i in t) for t in top3])
        pd.DataFrame([{"n_firms": len(top3), "share_top3_repeats_a_family": dup.mean(),
                       "share_top3_with_a_new_feature": new_any.mean()}]).to_csv(
            os.path.join(OUT, "reason_code_stats.csv"), index=False)
        rc = []
        for lab, qq in (("low PD (5th pct)", 0.05), ("median PD", 0.5), ("high PD (99th pct)", 0.99)):
            j = order[int(qq * (len(p) - 1))]
            s_ = sv.loc[j, feats].sort_values(ascending=False)
            for rnk, f in enumerate([f for f in s_.index if s_[f] > 0][:3], 1):
                val = X.loc[j, f]
                pct = np.nan if f in B.CAT_FEATURES else float((X[f] <= val).mean() * 100)
                rc.append({"firm": lab, "pd": p[j], "view": "raw feature", "rank": rnk, "driver": f,
                           "plain_words": PLAIN.get(f, B.META[f].get("reason", f) if f in B.META else f),
                           "value": val, "percentile_in_2010_sample": pct, "shap_logodds": s_[f]})
            sf = fam_sv.loc[j].sort_values(ascending=False)
            for rnk, f in enumerate([f for f in sf.index if sf[f] > 0][:3], 1):
                rc.append({"firm": lab, "pd": p[j], "view": "family-merged", "rank": rnk, "driver": f,
                           "plain_words": f"{f} family, sum of its features", "shap_logodds": sf[f]})
        pd.DataFrame(rc).to_csv(os.path.join(OUT, "reason_codes.csv"), index=False)
        tfr_tables(panel, art)


def tfr_tables(panel, art):
    """What does tfr_ratio separate? TRAINING rows of the fold-3 model only (2007-09), so samples are large."""
    tr = panel[(panel["fs_year"] <= 2009) & (panel["label_status"] == "ok")].reset_index(drop=True)
    X = X_of(art, tr)
    y = tr["y"].astype(int).to_numpy()
    bands = pd.cut(X["tfr_ratio"], [-np.inf, 0, 0.02, 0.05, 0.10, np.inf],
                   labels=["0", "0-2%", "2-5%", "5-10%", ">10%"])
    sec = pd.Series(np.asarray(B._section(tr["ateco_sector"])), index=tr.index).fillna("?")
    eq = pd.qcut(X["equity_ratio"].rank(method="first"), 3, labels=["low equity", "mid equity", "high equity"])
    out = []

    def tab(name, grp, mask=None):
        d = pd.DataFrame({"g": grp, "band": bands, "y": y})
        d = d[mask] if mask is not None else d
        t = d.groupby(["g", "band"], observed=True)["y"].agg(n="size", defaults="sum", dr="mean").reset_index()
        t.insert(0, "table", name)
        out.append(t)

    tab("all rows", pd.Series("all", index=tr.index))
    tab("sector", sec.where(sec.isin(["C", "F", "G", "L"]), "other"))
    tab("equity tercile, excluding real estate (L)", eq.astype(str), (sec != "L").to_numpy())
    pd.concat(out).to_csv(os.path.join(OUT, "tfr_train_tables.csv"), index=False)
    # share of zero-TFR rows that are real estate; AUC of tfr with and without section L
    aucs = []
    for lab, m in (("all", np.ones(len(y), bool)), ("excl. L", (sec != "L").to_numpy()),
                   ("excl. L, TFR > 0", ((sec != "L") & (X["tfr_ratio"] > 0)).to_numpy()), ("C only", (sec == "C").to_numpy())):
        ok = m & X["tfr_ratio"].notna().to_numpy()
        aucs.append({"subset": lab, "n": int(ok.sum()), "defaults": int(y[ok].sum()),
                     "auc_tfr_ratio": G.M.auc(y[ok], X["tfr_ratio"].to_numpy()[ok])})
    pd.DataFrame(aucs).to_csv(os.path.join(OUT, "tfr_train_auc.csv"), index=False)
    # tenure proxy: fund stock / annual wage bill (wage bill = goodwill - ebitda: D-010/D-091 hypothesis)
    num = lambda c: pd.to_numeric(tr[c], errors="coerce")
    wage = num("goodwill") - num("ebitda")
    ok = (num("liab_lt_emp") > 0) & (wage > 0) & (sec != "L")
    ten = (num("liab_lt_emp") / wage)[ok]
    q = pd.qcut(ten, 10, labels=False, duplicates="drop") + 1
    t = pd.DataFrame({"decile": q, "fund_over_wagebill": ten, "tfr": X["tfr_ratio"][ok], "y": y[ok.to_numpy()],
                      "log_assets": X["log_assets"][ok], "equity_ratio": X["equity_ratio"][ok]}).groupby("decile").agg(
        n=("y", "size"), median_fund_over_wagebill=("fund_over_wagebill", "median"), median_tfr=("tfr", "median"),
        dr=("y", "mean"), median_log_assets=("log_assets", "median"), median_equity=("equity_ratio", "median")).reset_index()
    t.to_csv(os.path.join(OUT, "tfr_tenure_proxy.csv"), index=False)
    print(pd.concat(out).round(4).to_string(), "\n", pd.DataFrame(aucs).round(4).to_string(), "\n", t.round(3).to_string())


def _dep_plot(X, sv, sub, f, grid, curve, exp):
    m = X[f].notna()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    x = X.loc[m, f]
    ax[0].scatter(x, sv.loc[m, f], s=2, alpha=0.15)
    q = pd.qcut(x.rank(method="first"), 20, labels=False)
    b = pd.DataFrame({"x": x, "s": sv.loc[m, f], "q": q}).groupby("q").mean()
    ax[0].plot(b["x"], b["s"], color="k", lw=1.5, label="20-bin mean")
    ax[0].axhline(0, color="grey", lw=0.5)
    lo, hi = np.nanquantile(x, [0.005, 0.995])
    ax[0].set_xlim(lo, hi)
    ax[0].set_title(f"{exp} fold 3: SHAP dependence, {f}")
    ax[0].set_xlabel(f), ax[0].set_ylabel("SHAP (log-odds)"), ax[0].legend()
    ax[1].plot(grid, curve, marker="o", ms=3)
    ax[1].set_title(f"partial dependence (sign: {B.expected_signs([f]).get(f, 0)})")
    ax[1].set_xlabel(f), ax[1].set_ylabel("mean raw log-odds")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"dep_{f}.png"), dpi=90)
    plt.close(fig)


def tfr_interaction():
    """Does the fitted model use the TFR x equity interaction seen in the training tables?
    Mean tfr_ratio SHAP by TFR band x equity tercile, plus the top-features-excluded check: the raw default-rate
    pattern and the model's mean PD in each cell, 2010 sample (excluding section L)."""
    panel = pd.read_parquet(os.path.join(ROOT, "data/interim/panel.parquet"), filters=[("fs_year", "<", 2011)])
    assert panel["fs_year"].max() <= 2010
    te = panel[(panel["fs_year"] == 2010) & (panel["label_status"] == "ok")].reset_index(drop=True)
    sub = te.iloc[np.random.default_rng(0).choice(len(te), N_SHAP, replace=False)].reset_index(drop=True)
    art = load_art("GB-20")
    bst, feats = art["booster"], art["features"]
    X = X_of(art, sub)
    c = bst.predict(X, pred_contrib=True)
    sv = pd.DataFrame(c, columns=feats + ["_bias"])
    sec = pd.Series(np.asarray(B._section(sub["ateco_sector"])), index=sub.index).fillna("?")
    edges = np.quantile(X["equity_ratio"].dropna(), [1 / 3, 2 / 3])
    d = pd.DataFrame({"equity": pd.cut(X["equity_ratio"], [-np.inf, edges[0], edges[1], np.inf],
                                       labels=["low equity", "mid equity", "high equity"]),
                      "tfr_band": pd.cut(X["tfr_ratio"], [-np.inf, 0, 0.02, 0.05, 0.10, np.inf],
                                         labels=["0", "0-2%", "2-5%", "5-10%", ">10%"]),
                      "shap_tfr": sv["tfr_ratio"], "y": sub["y"].astype(int), "raw_pd": bst.predict(X)})[sec != "L"]
    t = d.groupby(["equity", "tfr_band"], observed=True).agg(n=("y", "size"), defaults=("y", "sum"), dr_2010=("y", "mean"),
                                                            mean_pd_raw=("raw_pd", "mean"),
                                                            mean_shap_tfr=("shap_tfr", "mean")).reset_index()
    t.to_csv(os.path.join(OUT, "tfr_shap_by_equity.csv"), index=False)
    edges = np.quantile(X["asset_turnover"].dropna(), [1 / 3, 2 / 3])
    d2 = d.assign(turnover=pd.cut(X["asset_turnover"], [-np.inf, edges[0], edges[1], np.inf],
                                  labels=["low turnover", "mid turnover", "high turnover"])[sec != "L"])
    d2.groupby(["turnover", "tfr_band"], observed=True).agg(n=("y", "size"), defaults=("y", "sum"), dr_2010=("y", "mean"),
                                                            mean_pd_raw=("raw_pd", "mean"), mean_shap_tfr=("shap_tfr", "mean")
                                                            ).reset_index().to_csv(os.path.join(OUT, "tfr_shap_by_turnover.csv"), index=False)
    print(t.round(4).to_string())
    tfr_suppression(panel, art)


def tfr_suppression(panel, art):
    """Suppression test on TRAINING rows (2007-09, excl. real estate): sign of tfr_ratio in a logit alone and next
    to the labour-intensity / asset-structure ratios it is correlated with; DR by TFR band x turnover tercile."""
    from sklearn.linear_model import LogisticRegression
    tr = panel[(panel["fs_year"] <= 2009) & (panel["label_status"] == "ok")].reset_index(drop=True)
    X = X_of(art, tr)
    y = tr["y"].astype(int).to_numpy()
    sec = pd.Series(np.asarray(B._section(tr["ateco_sector"])), index=tr.index).fillna("?")
    specs = {"tfr alone": ["tfr_ratio"], "+ asset_turnover": ["tfr_ratio", "asset_turnover"],
             "+ personnel_to_sales": ["tfr_ratio", "personnel_to_sales"],
             "+ turnover, personnel": ["tfr_ratio", "asset_turnover", "personnel_to_sales"],
             "+ turnover, personnel, equity, roa, tangible, st debt":
                 ["tfr_ratio", "asset_turnover", "personnel_to_sales", "equity_ratio", "roa_net", "tangible_ratio", "st_debt_ratio"]}
    keep = ["tfr_ratio", "asset_turnover", "personnel_to_sales", "equity_ratio", "roa_net", "tangible_ratio", "st_debt_ratio"]
    ok = (sec != "L").to_numpy() & X[keep].notna().all(axis=1).to_numpy()
    Z = X.loc[ok, keep]
    Z = (Z - Z.mean()) / Z.std()
    rows = []
    for lab, cols in specs.items():
        m = LogisticRegression(C=1e6, max_iter=2000).fit(Z[cols], y[ok])
        rows.append({"spec": lab, "n": int(ok.sum()), "defaults": int(y[ok].sum()),
                     "coef_tfr_per_sd": float(m.coef_[0][0]),
                     **{f"coef_{c}": float(b) for c, b in zip(cols[1:], m.coef_[0][1:])}})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tfr_suppression_logit.csv"), index=False)
    bands = pd.cut(X["tfr_ratio"], [-np.inf, 0, 0.02, 0.05, 0.10, np.inf], labels=["0", "0-2%", "2-5%", "5-10%", ">10%"])
    turn = pd.qcut(X["asset_turnover"].rank(method="first"), 3, labels=["low turnover", "mid turnover", "high turnover"])
    d = pd.DataFrame({"turn": turn, "band": bands, "y": y})[(sec != "L").to_numpy()]
    t = d.groupby(["turn", "band"], observed=True)["y"].agg(n="size", defaults="sum", dr="mean").reset_index()
    t.to_csv(os.path.join(OUT, "tfr_by_turnover.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(), "\n", t.round(4).to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["tfr_interaction"]:
        tfr_interaction()
    else:
        main(sys.argv[1:] or ["GB-20"])
