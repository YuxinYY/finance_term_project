"""Real-firm sign check: exact TreeSHAP (LightGBM pred_contrib) on 20,000 round-3 TRAIN rows (FY2007-09).
For each candidate-sign feature: Spearman(feature value, SHAP value) and, for the top and bottom deciles of
the feature, the mean SHAP (log-odds). Wrong-signed reason code = SHAP moves against the expected sign."""
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr
from common import load, feature_list, OUT_DIR
feats = feature_list()
st = pd.read_csv(OUT_DIR / "sign_table.csv").set_index("feature")
cand = st[st.candidate != 0]
X = load("round 03", "train").sample(20000, random_state=3)[feats]
rows = []
for mode in ("free", "mono"):
    b = lgb.Booster(model_file=str(OUT_DIR / f"model_{mode}_round_03.txt"))
    sh = pd.DataFrame(b.predict(X.to_numpy(), pred_contrib=True)[:, :-1], columns=feats, index=X.index)
    for f, r in cand.iterrows():
        x, s = X[f], sh[f]
        if x.nunique() <= 2:
            rho = s[x == 1].mean() - s[x == 0].mean(); lo = s[x == 0].mean(); hi = s[x == 1].mean()
        else:
            rho = spearmanr(x, s).statistic
            d = pd.qcut(x.rank(method="first"), 10, labels=False)
            lo, hi = s[d == 0].mean(), s[d == 9].mean()
            mid = s[d.between(4, 5)].mean()
        # rows whose SHAP sign contradicts their position: above-median value with SHAP raising PD for a '-' feature
        med = x.median()
        wrong = ((x > med) & (s * r.candidate < 0)).mean() + ((x < med) & (s * r.candidate > 0)).mean() if x.nunique() > 2 else np.nan
        rows.append(dict(mode=mode, feature=f, expected=int(r.candidate), spearman_value_shap=round(rho, 3),
                         shap_bottom_decile=round(lo, 4), shap_top_decile=round(hi, 4),
                         mean_abs_shap=round(s.abs().mean(), 4), share_rows_wrong_side=round(wrong, 3)))
d = pd.DataFrame(rows); d.to_csv(OUT_DIR / "shap_sign_check.csv", index=False)
p = d.pivot(index="feature", columns="mode", values=["spearman_value_shap", "share_rows_wrong_side", "mean_abs_shap"])
p.columns = [f"{a}_{b}" for a, b in p.columns]
p = p.join(cand.candidate).sort_values("mean_abs_shap_free", ascending=False)
print(p.round(3).to_string())

# --- Family check (added 2026-11-21): sum SHAP over near-twin features, correlate with an anchor ratio ---
FAMILIES = {
    "cash_flow": (["operating_cash_flow_ratio", "operating_cash_flow_to_debt", "operating_cash_flow_margin",
                   "cash_interest_coverage"], "operating_cash_flow_ratio"),
    "profitability": (["roa", "op_margin", "net_prof_margin", "ebitda_to_debt", "int_coverage"], "roa"),
    "liquidity": (["quick_ratio", "cur_debt_ratio", "cash_ratio", "wc_ratio"], "cur_debt_ratio"),
}
fam = []
for mode in ("free", "mono"):
    b = lgb.Booster(model_file=str(OUT_DIR / f"model_{mode}_round_03.txt"))
    sh = pd.DataFrame(b.predict(X.to_numpy(), pred_contrib=True)[:, :-1], columns=feats, index=X.index)
    for name, (cols, anchor) in FAMILIES.items():
        tot = sh[cols].sum(axis=1)
        fam.append(dict(mode=mode, family=name, members=" + ".join(cols), anchor=anchor,
                        spearman_anchor_vs_family_shap=round(spearmanr(X[anchor], tot).statistic, 3),
                        mean_abs_family_shap=round(tot.abs().mean(), 3)))
fam = pd.DataFrame(fam); fam.to_csv(OUT_DIR / "shap_family_check.csv", index=False)
print(fam.to_string(index=False))
