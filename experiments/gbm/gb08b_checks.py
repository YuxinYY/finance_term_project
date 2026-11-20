"""GB-08b fold-3 model: ebitda_margin shape on natural scale, by sector group, and wrong-way reason codes, on 40,000
in-window 2009 rows. Writes GB-08b/checks_ebitda_shape.csv, checks_ebitda_reason_codes.csv."""
import os, pickle, sys
import lightgbm as lgb, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_lib as L, gbm_lib as G
pd.set_option("display.width", 200)
d = pickle.load(open(os.path.join(HERE, ".GB-08b_fold3_art.pkl"), "rb")); d["booster"] = lgb.Booster(model_str=d.pop("booster_str"))
p = L.load_panel(); w = p[(p.fs_year == 2009) & (p.label_status == "ok")].sample(40000, random_state=0).reset_index(drop=True)
X = L.B.transform(w, d["feature_params"])[d["features"]]
f = "ebitda_margin"; Xs = X.sample(3000, random_state=0); rows = []
for q in [.02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98]:
    g = X[f].quantile(q); X2 = Xs.copy(); X2[f] = g
    rows.append(dict(pct=q, slog_x=g, margin_natural=np.sign(g) * np.expm1(abs(g)), pd_logodds=float(np.mean(d["booster"].predict(X2, raw_score=True)))))
s = pd.DataFrame(rows); s["rel_to_median_PD_multiple"] = np.exp(s.pd_logodds - s.pd_logodds[s.pct == .5].iloc[0])
s.to_csv(os.path.join(HERE, "GB-08b", "checks_ebitda_shape.csv"), index=False); print(s.round(3).to_string())
sv = G.shap_values(d, w)[d["features"]]; cols = list(sv.columns); j = cols.index(f)
top3 = np.argsort(-sv.to_numpy(), axis=1)[:, :3]
med = X[f].median()
in3 = (top3 == j).any(axis=1) & (sv[f].to_numpy() > 0)
top = X[f] >= X[f].quantile(.8)
print("share of firms: ebitda_margin in top-3 PD-raising:", in3.mean(), "| and margin above median:", (in3 & (X[f] > med).to_numpy()).mean(), "| and margin in top quintile:", (in3 & top.to_numpy()).mean())
r = pd.DataFrame([dict(top3_up=in3.mean(), top3_up_above_median=float((in3 & (X[f] > med).to_numpy()).mean()), top3_up_top_quintile=float((in3 & top.to_numpy()).mean()), mean_abs_shap=float(sv[f].abs().mean()))])
r.to_csv(os.path.join(HERE, "GB-08b", "checks_ebitda_reason_codes.csv"), index=False)
t = pd.DataFrame({"sect": X["ateco_group"].astype(str), "top": top.values, "shap": sv[f].values, "turn": X["asset_turnover"].values})
print(t[t.top].groupby("sect").agg(n=("shap", "size"), shap=("shap", "mean"), turn=("turn", "median")).sort_values("n", ascending=False).head(5).round(3).to_string())
