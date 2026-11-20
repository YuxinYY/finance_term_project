"""GB-08 step 3: univariate default rate by ventile for the costly constrained features, fs_year 2007 only
(training data in every fold, so the economic judgement is leak-free for all three outer folds).
Also: how small is the denominator in the top 5% of each ratio (interest_to_assets, revenue/assets, debt)?
Writes GB-08/univariate_2007.csv."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_lib as L
panel = L.load_panel()
d = panel[(panel.fs_year == 2007) & (panel.label_status == "ok")].reset_index(drop=True)
fp = L.B.fit(d)
X = L.B.transform(d, fp)
X["y"] = d["y"].astype(int).values
rows = []
for f in ["roa_net", "roa_oper", "interest_coverage", "cfo_to_debt", "ebitda_margin", "wc_ratio", "equity_ratio",
          "asset_turnover", "interest_to_assets"]:
    v = X[f]
    q = pd.qcut(v.rank(method="first"), 20, labels=False)
    g = X.groupby(q).agg(lo=(f, "min"), hi=(f, "max"), n=("y", "size"), dr=("y", "mean"),
                         interest_to_assets=("interest_to_assets", "median"), asset_turnover=("asset_turnover", "median"),
                         equity_ratio=("equity_ratio", "median"), bank_debt_ratio=("bank_debt_ratio", "median"),
                         roa_net=("roa_net", "median"))
    g.insert(0, "feature", f); g["ventile"] = g.index
    rows.append(g)
    na = v.isna()
    rows.append(pd.DataFrame([dict(feature=f, ventile="NaN", n=int(na.sum()), dr=X.loc[na, "y"].mean())]))
out = pd.concat(rows, ignore_index=True)
out.to_csv(os.path.join(HERE, "GB-08", "univariate_2007.csv"), index=False)
pd.set_option("display.width", 250)
print(out.round(4).to_string())
