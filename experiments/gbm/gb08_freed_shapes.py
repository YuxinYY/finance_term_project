"""GB-08 fold-3 final model: partial dependence of the 3 freed ratios on 3,000 in-window 2009 rows, with the
2009 univariate default rate by ventile for reference. Writes GB-08/freed_shapes.png, freed_shapes.csv."""
import os, sys, pickle
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_lib as L
import gbm_lib as G
exp = sys.argv[1] if len(sys.argv) > 1 else "GB-08"
d = pickle.load(open(os.path.join(HERE, f".{exp}_fold3_art.pkl"), "rb")); d["booster"] = lgb.Booster(model_str=d.pop("booster_str"))
p = L.load_panel(); w = p[(p.fs_year == 2009) & (p.label_status == "ok")].reset_index(drop=True)
X = L.B.transform(w, d["feature_params"])[d["features"]]
Xs = X.sample(3000, random_state=0)
rows = []
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, f in zip(ax, ["interest_coverage", "cfo_to_debt", "ebitda_margin"]):
    grid = np.unique(np.nanquantile(X[f].dropna(), np.linspace(0.01, 0.99, 41)))
    pdv = []
    for g in grid:
        X2 = Xs.copy(); X2[f] = g; pdv.append(np.mean(d["booster"].predict(X2, raw_score=True)))
    pdv = np.array(pdv)
    q = pd.qcut(X[f].rank(method="first"), 20, labels=False)
    u = pd.DataFrame({"x": X[f], "y": w["y"].astype(int)}).groupby(q).agg(x=("x", "median"), dr=("y", "mean"))
    imin = int(np.argmin(pdv))
    for g, v in zip(grid, pdv):
        rows.append(dict(feature=f, x=g, pd_logodds=v, pct=float((X[f] <= g).mean())))
    a.plot(grid, pdv, "b-", label="GB-08 partial dependence (log-odds)")
    a2 = a.twinx(); a2.plot(u.x, 100 * u.dr, "o:", color="grey", ms=3, label="2009 default rate by ventile (%)")
    a.axvline(grid[imin], color="r", lw=0.7)
    a.set_title(f"{f}: PD minimum at x={grid[imin]:.2f} (pct {100*(X[f] <= grid[imin]).mean():.0f})", fontsize=9)
    a2.set_ylabel("2009 DR %")
ax[0].legend(fontsize=7, loc="upper left")
fig.suptitle(f"{exp} fold-3 model: freed ratios (blue) vs univariate 2009 default rate (grey); in-window rows")
fig.tight_layout(); fig.savefig(os.path.join(HERE, exp, "freed_shapes.png"), dpi=100)
out = pd.DataFrame(rows); out.to_csv(os.path.join(HERE, exp, "freed_shapes.csv"), index=False)
for f, g in out.groupby("feature"):
    i = g.pd_logodds.idxmin()
    print(f, "min at pct", round(g.loc[i, "pct"], 2), "x", round(g.loc[i, "x"], 3), "| pd at p1 %.3f  min %.3f  p99 %.3f" % (g.pd_logodds.iloc[0], g.pd_logodds.min(), g.pd_logodds.iloc[-1]))
