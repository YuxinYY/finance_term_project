"""Explainability for the GBM challengers (fold-3 models: train 2007-2009, explained on test 2010 rows).

usage: python experiments/gbm/explain.py GB-01 [GB-02 ...]
For every exp: TreeSHAP (LightGBM pred_contrib) global importance, SHAP-dependence trend per feature
vs the expected sign, partial-dependence face-validity check, and (first exp) reason codes for three
example firms + prediction timing. Writes experiments/gbm/<exp>/explain_*.csv|png and explain.md
for the first exp.
"""
from __future__ import annotations

import os
import sys
import time

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
N_SHAP, N_PD, GRID_Q = 40000, 3000, np.linspace(0.02, 0.98, 25)

PLAIN = {  # credit-officer wording: (what it is, wording when it raises PD)
    "log_assets": ("size (total assets)", "small balance sheet"),
    "equity_ratio": ("equity / assets", "thin equity cushion"),
    "st_debt_ratio": ("short-term liabilities / assets", "heavy short-term liabilities"),
    "bank_debt_ratio": ("bank debt / assets", "high bank borrowing"),
    "st_debt_share": ("short-term share of debt", "debt concentrated in the short term"),
    "wc_ratio": ("net working capital / assets", "weak working capital"),
    "current_ratio": ("current ratio", "low current ratio"),
    "cash_ratio": ("cash / assets", "little cash"),
    "roa_net": ("net profit / assets", "losses / low net profitability"),
    "roa_oper": ("operating profit / assets", "weak operating profitability"),
    "ebitda_margin": ("EBITDA margin", "low EBITDA margin"),
    "interest_coverage": ("EBITDA / interest expense", "EBITDA negative or barely covers interest"),
    "cfo_to_debt": ("cash flow / debt", "low cash flow relative to debt"),
    "interest_to_assets": ("interest expense / assets", "high interest burden"),
    "asset_turnover": ("sales / assets", "sales-to-assets in a riskier range"),
    "ar_days": ("days receivables outstanding", "slow collection of receivables"),
    "ap_days": ("days payables outstanding", "stretched supplier payments"),
    "tangible_ratio": ("tangible fixed assets / assets", "little tangible collateral"),
    "intangible_ratio": ("intangible assets / assets", "high share of intangibles"),
    "extraord_to_assets": ("extraordinary items / assets", "unusual extraordinary items"),
    "neg_equity": ("negative equity flag", "negative book equity"),
    "debt_split_missing": ("debt breakdown missing", "debt breakdown not reported"),
    "rev_nonpositive": ("no operating revenue flag", "no operating revenue"),
    "ateco_group": ("industry (ATECO section)", "riskier industry"),
    "legal_group": ("legal form", "riskier legal form"),
    "macro_area": ("region (macro-area)", "riskier region"),
}


def load_art(exp):
    import pickle
    import lightgbm as lgb
    with open(os.path.join(HERE, f".{exp}_fold3_art.pkl"), "rb") as fh:
        d = pickle.load(fh)
    d["booster"] = lgb.Booster(model_str=d.pop("booster_str"))
    return d


def pdp(art, X, feat):
    """Partial dependence on the raw log-odds scale over training-quantile grid of `feat`."""
    v = X[feat].dropna()
    grid = np.unique(np.quantile(v, GRID_Q))
    out = []
    for g in grid:
        Xg = X.copy()
        Xg[feat] = g
        out.append(np.mean(art["booster"].predict(Xg, raw_score=True)))
    return grid, np.array(out)


def main(exps):
    panel = pd.read_parquet(os.path.join(ROOT, "data/interim/panel.parquet"), filters=[("fs_year", "<", 2011)])  # lockbox (round 2)
    te = panel[(panel["fs_year"] == 2010) & (panel["label_status"] == "ok")].reset_index(drop=True)
    rng = np.random.default_rng(0)
    signs = B.expected_signs()
    for i, exp in enumerate(exps):
        art = load_art(exp)
        od = os.path.join(HERE, exp)
        sub = te.iloc[rng.choice(len(te), N_SHAP, replace=False)].reset_index(drop=True)
        X = B.transform(sub, art["feature_params"], impute=art["impute"])[art["features"]]
        sv = G.shap_values(art, sub)
        feats = art["features"]
        imp = sv[feats].abs().mean().sort_values(ascending=False)
        rows = []
        Xpd = X.iloc[:N_PD]
        for f in feats:
            if f in B.CAT_FEATURES:
                lv = sv[f].groupby(X[f].astype(str)).mean().sort_values()
                rows.append(dict(feature=f, mean_abs_shap=imp[f], expected="none (categorical)",
                                 shap_spearman=np.nan, pd_trend="levels: " + ", ".join(f"{k} {v:+.2f}" for k, v in lv.items()),
                                 pd_violation_share=np.nan, verdict="n/a"))
                continue
            m = X[f].notna()
            rho = spearmanr(X.loc[m, f], sv.loc[m, f]).correlation if X.loc[m, f].nunique() > 1 else np.nan
            grid, curve = pdp(art, Xpd, f)
            d = np.diff(curve)
            s = signs.get(f, 0)
            tol = 1e-4
            if s == 0 or len(d) == 0:
                viol, verdict = np.nan, "no expectation"
            else:
                bad = (d * s) < -tol
                viol = float(np.abs(d[bad]).sum() / max(np.abs(d).sum(), 1e-12))
                total = (curve[-1] - curve[0]) * s
                verdict = ("agrees" if viol < 0.05 and total >= 0 else
                           "mostly agrees" if total > 0 else "CONTRADICTS")
            rows.append(dict(feature=f, mean_abs_shap=imp[f], expected={1: "+", -1: "-", 0: "none"}[s],
                             shap_spearman=rho, pd_trend=f"{curve[0]:+.2f} -> {curve[-1]:+.2f} (range {curve.max()-curve.min():.2f})",
                             pd_violation_share=viol, verdict=verdict))
            if i == 0 and f in imp.index[:8]:
                fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
                ax[0].scatter(X.loc[m, f], sv.loc[m, f], s=2, alpha=0.2)
                ax[0].set_title(f"SHAP dependence: {f}")
                ax[0].set_xlabel(f), ax[0].set_ylabel("SHAP (log-odds)")
                ax[1].plot(grid, curve, marker="o")
                ax[1].set_title(f"partial dependence (expected sign {rows[-1]['expected']})")
                ax[1].set_xlabel(f)
                fig.tight_layout()
                fig.savefig(os.path.join(od, f"dep_{f}.png"), dpi=90)
                plt.close(fig)
        fv = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
        fv.to_csv(os.path.join(od, "explain_face_validity.csv"), index=False)
        print(exp, "\n", fv.to_string(index=False))
        if i > 0:
            continue

        # ---- reason codes for 3 example test firms (low / mid / high PD) ----
        p = G.predict_pd(art, sub)
        qs = {"low PD (5th pct)": 0.05, "middle PD (median)": 0.50, "high PD (99th pct)": 0.99}
        order = np.argsort(p)
        reasons = []
        for lab, q in qs.items():
            j = order[int(q * (len(p) - 1))]
            c = sv.loc[j, feats].sort_values(ascending=False)
            up = [f for f in c.index if c[f] > 0][:3]
            dn = [f for f in c.sort_values().index if c[f] < 0][:2]
            pct = {f: (X[f] <= X.loc[j, f]).mean() * 100 for f in up + dn if f not in B.CAT_FEATURES}

            def desc(f, up_=True):
                val = X.loc[j, f]
                if f in B.CAT_FEATURES:
                    return f"{PLAIN[f][0]} = {val} ({'raises' if up_ else 'lowers'} PD, SHAP {c[f]:+.2f})"
                txt = PLAIN[f][1] if up_ else f"favourable {PLAIN[f][0]}"
                if B.META[f].get("slog") and pd.notna(val):   # show on the natural scale
                    val = np.sign(val) * np.expm1(abs(val))
                return (f"{txt}: {PLAIN[f][0]} = {val:.3g} (percentile {pct[f]:.0f} of 2010 applicants; "
                        f"SHAP {c[f]:+.2f} log-odds)")
            reasons.append(dict(case=lab, row_id=int(sub.loc[j, "row_id"]), pd=float(p[j]),
                                ateco=str(X.loc[j, "ateco_group"]), y=int(sub.loc[j, "y"]),
                                up=[desc(f) for f in up], down=[desc(f, False) for f in dn]))

        # ---- timing per 100k rows (transform + booster + calibration) ----
        big = te.sample(100000, random_state=1, replace=False)
        t0 = time.time()
        G.predict_pd(art, big)
        t_full = time.time() - t0
        t0 = time.time()
        B.transform(big, art["feature_params"], impute=art["impute"])
        t_tr = time.time() - t0
        write_md(exp, art, fv, imp, reasons, t_full, t_tr)


def write_md(exp, art, fv, imp, reasons, t_full, t_tr):
    rec = art["record"]
    n_trees = art["booster"].num_trees()
    L = [f"# {exp} explainability (fold-3 model: trained on fs_year 2007-2009, explained on 2010 test rows)\n",
         "SHAP values are TreeSHAP contributions on the raw log-odds scale, computed by LightGBM "
         "(`pred_contrib=True`) on a random 40,000-row sample of 2010 test rows. The own calibration "
         "is a monotone map of the log-odds (Platt), so the ranking of drivers is the same on the PD "
         "scale. Shapley-value attribution properties: R-23 (Lundberg & Lee 2017, Sec. 3). The tree "
         "algorithm used to compute them is not in R-23 [no VERIFIED source for TreeSHAP yet].\n",
         "Correlated features (e.g. roa_net / cfo_to_debt / roa_oper / ebitda_margin; equity_ratio / "
         "neg_equity) split their attribution between them, so read drivers by FAMILY "
         "(FEATURE_SPEC note).\n",
         "## 1. Global importance and shape vs expected sign\n",
         "`shap_spearman`: rank correlation between the feature value and its SHAP value (sign = "
         "direction the model uses). `pd_trend`: partial dependence (average log-odds, 3,000 rows) "
         "from the 2nd to the 98th percentile of the feature. `pd_violation_share`: share of the "
         "partial-dependence movement that goes AGAINST the expected sign (0 = perfectly monotone in "
         "the expected direction). Features with expected sign 'none' are unconstrained by design.\n",
         fv.to_markdown(index=False, floatfmt=".3f"),
         "\nFamily importance (sum of mean |SHAP|):\n"]
    fam = pd.Series({f: imp[f] for f in art["features"]}).groupby(
        lambda f: B.META[f]["family"].replace(" (flag)", "")).sum().sort_values(ascending=False)
    L.append(fam.to_frame("mean_abs_shap").to_markdown(floatfmt=".3f"))
    L.append("\nDependence plots for the top-8 features: `dep_<feature>.png` in this folder "
             "(left: SHAP dependence scatter; right: partial dependence).\n")
    L.append("## 2. Reason codes for three example 2010 applicants\n")
    L.append("Top-3 factors pushing the PD UP (largest positive SHAP) and the 2 strongest factors "
             "pulling it DOWN, in credit-officer wording. Percentiles are within the 40,000 sampled "
             "2010 applicants. Values are on the natural scale (signed-log features back-transformed; "
             "winsorised at the training 1st/99th percentiles, so e.g. days outstanding are capped).\n")
    for r in reasons:
        L.append(f"### {r['case']}: row_id {r['row_id']}, PD {r['pd']*100:.2f}%, industry section "
                 f"{r['ateco']}, observed default in window: {'yes' if r['y'] else 'no'}\n")
        L.append("Raises PD:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(r["up"])))
        L.append("\nLowers PD:\n" + "\n".join(f"- {s}" for s in r["down"]) + "\n")
    L.append("## 3. Harness-relevant cost\n")
    L.append(f"- Fold-3 booster: {n_trees} trees, max {rec['chosen_cfg']['num_leaves']} leaves "
             f"(depth <= {rec['chosen_cfg']['max_depth']}); {len(art['features'])} features.\n"
             f"- Prediction for 100,000 rows (feature transform + booster + calibration), single "
             f"process, 8 threads, this laptop: {t_full:.1f} s total, of which feature transform "
             f"{t_tr:.1f} s.\n"
             f"- Complexity: O(n x (F + T x D)) for n rows, F features, T trees, depth D; linear in n, "
             f"no dependence on other rows (each row is scored from its own record and frozen params).\n")
    with open(os.path.join(HERE, exp, "explain.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
