"""GB-08 follow-up checks, in-window data only (fs_year <= 2009; lockbox respected).
1. Univariate default rate with a 95% Wilson interval in the top tail of each freed ratio vs its lowest-DR ventile
   (2007 and 2009), so the 'uptick' can be judged against noise.
2. Sector mix of the top quintile of each freed ratio (2007): is the 'property / holding vehicle' story supported?
3. Wrong-way reason codes: on 2009 training rows, share of firms for which a freed ratio is in the top-3 PD-raising
   SHAP factors while the ratio is above its PD minimum (i.e. 'strong ratio raises PD').
4. Additivity of the single-drop costs vs the all-dropped cost (from ablation_summary.csv).
Writes GB-08/checks_*.csv."""
import os, pickle, sys
import lightgbm as lgb, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_lib as L, gbm_lib as G
pd.set_option("display.width", 220)
FREED = ["interest_coverage", "cfo_to_debt", "ebitda_margin"]
OUT = os.path.join(HERE, "GB-08")
p = L.load_panel()

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; d = 1 + z*z/n; c = ph + z*z/(2*n); h = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

rows, sect = [], []
for yr in (2007, 2009):
    d = p[(p.fs_year == yr) & (p.label_status == "ok")].reset_index(drop=True)
    X = L.B.transform(d, L.B.fit(d)); y = d["y"].astype(int).to_numpy()
    for f in FREED + ["roa_net", "roa_oper"]:
        v = X[f]; ok = v.notna().to_numpy()
        q = pd.qcut(v[ok].rank(method="first"), 20, labels=False).to_numpy()
        yy = y[ok]
        g = pd.DataFrame({"q": q, "y": yy}).groupby("q").y.agg(["sum", "size"])
        g["dr"] = g["sum"] / g["size"]
        mn = int(g["dr"].idxmin())
        for name, idx in (("lowest_ventile", [mn]), ("top_ventile", [19]), ("top_two_ventiles", [18, 19])):
            k, n = int(g.loc[idx, "sum"].sum()), int(g.loc[idx, "size"].sum())
            lo, hi = wilson(k, n)
            rows.append(dict(year=yr, feature=f, group=name, lowest_ventile_index=mn, n=n, defaults=k, dr=k/n, ci_lo=lo, ci_hi=hi))
    if yr == 2007:
        for f in FREED:
            top = X[f] >= X[f].quantile(0.8)
            a = X.loc[top, "ateco_group"].astype(str).value_counts(normalize=True)
            b = X.loc[~top & X[f].notna(), "ateco_group"].astype(str).value_counts(normalize=True)
            for s in sorted(set(a.index) | set(b.index)):
                sect.append(dict(feature=f, section=s, share_top_quintile=a.get(s, 0.0), share_rest=b.get(s, 0.0)))
t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "checks_tail_dr.csv"), index=False); print(t.round(5).to_string())
s = pd.DataFrame(sect); s.to_csv(os.path.join(OUT, "checks_sector_mix.csv"), index=False)
print(s[s.section.isin(["L", "C", "F", "G"])].round(3).to_string())

d = pickle.load(open(os.path.join(HERE, ".GB-08_fold3_art.pkl"), "rb")); d["booster"] = lgb.Booster(model_str=d.pop("booster_str"))
w = p[(p.fs_year == 2009) & (p.label_status == "ok")].sample(40000, random_state=0).reset_index(drop=True)
X = L.B.transform(w, d["feature_params"])[d["features"]]
sv = G.shap_values(d, w)[d["features"]]
shape = pd.read_csv(os.path.join(OUT, "freed_shapes.csv"))
top3 = np.argsort(-sv.to_numpy(), axis=1)[:, :3]
cols = list(sv.columns); ww = []
for f in FREED:
    g = shape[shape.feature == f]; xmin = g.loc[g.pd_logodds.idxmin(), "x"]
    j = cols.index(f)
    in_top3 = (top3 == j).any(axis=1) & (sv[f].to_numpy() > 0)
    above = X[f].to_numpy() > xmin
    ww.append(dict(feature=f, pd_min_x=xmin, share_above_min=float(np.nanmean(above)), share_in_top3_up=float(in_top3.mean()),
                   share_wrong_way_top3=float((in_top3 & above).mean())))
any_ww = np.zeros(len(X), bool)
for f in FREED:
    g = shape[shape.feature == f]; xmin = g.loc[g.pd_logodds.idxmin(), "x"]; j = cols.index(f)
    any_ww |= (top3 == j).any(axis=1) & (sv[f].to_numpy() > 0) & (X[f].to_numpy() > xmin)
ww.append(dict(feature="ANY", share_wrong_way_top3=float(any_ww.mean())))
r = pd.DataFrame(ww); r.to_csv(os.path.join(OUT, "checks_wrong_way_reason_codes.csv"), index=False); print(r.round(4).to_string())

a = pd.read_csv(os.path.join(OUT, "ablation_summary.csv"), index_col=0)
sing = a.drop(index=["all_dropped", "none_dropped"])
ad = pd.DataFrame([dict(fold=k, sum_single=sing[f"cost_AB_ll_f{k}"].sum(), all_dropped=a.loc["all_dropped", f"cost_AB_ll_f{k}"],
                        largest_single=sing[f"cost_AB_ll_f{k}"].max(), n_negative_or_zero=int((sing[f"cost_AB_ll_f{k}"] <= 0).sum()),
                        cost_ar_days=a.loc["ar_days", f"cost_AB_ll_f{k}"], cost_current=a.loc["current_ratio", f"cost_AB_ll_f{k}"],
                        cost_ap_days=a.loc["ap_days", f"cost_AB_ll_f{k}"]) for k in (1, 2, 3)])
ad.to_csv(os.path.join(OUT, "checks_additivity.csv"), index=False); print(ad.round(6).to_string())
