"""GB-20 vs GB-01 (and GB-23 vs GB-01): 1,000-draw firm-clustered bootstrap, year x seen/new cells and the
seen-minus-new gain gap. Scores nothing new: reads saved OOT predictions. Writes comparison_r2b_bootstrap_1000.csv."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_round2b as R  # noqa: E402
from src.common import metrics as M  # noqa: E402

ROOT = R.ROOT
panel = pd.read_parquet(os.path.join(ROOT, "data/interim/panel.parquet"), columns=["row_id", "id", "fs_year", "y"],
                        filters=[("fs_year", "<", 2011)])
base = pd.read_parquet(os.path.join(ROOT, "data/splits/walkforward_test_rows.parquet")).merge(panel, on="row_id", validate="one_to_one")
P = {}
for e in ("GB-01", "GB-20", "GB-23"):
    d = base[["row_id", "fold"]].merge(pd.read_parquet(os.path.join(HERE, e, "oot_predictions.parquet")), on=["row_id", "fold"], how="left")
    assert d["pd"].notna().all()
    P[e] = d["pd"].to_numpy()
y, yr, firm = base["y"].astype(int).to_numpy(), base["test_year"].to_numpy(), base["id"].to_numpy()
seen = base["firm_seen_in_train"].to_numpy().astype(bool)
sub = {"ALL": np.ones(len(y), bool), "seen": seen, "new": ~seen}
for t in (2008, 2009, 2010):
    sub[str(t)] = yr == t
    sub[f"{t} seen"] = (yr == t) & seen
    sub[f"{t} new"] = (yr == t) & ~seen
S = R.referee_funcs()
dr = S["bootstrap_draws"](y, firm, P, sub, n_boot=1000, seed=0)
rows = []
for new in ("GB-20", "GB-23"):
    for s, m in sub.items():
        da, dl = dr[("auc", new, s)] - dr[("auc", "GB-01", s)], dr[("log_loss", new, s)] - dr[("log_loss", "GB-01", s)]
        rows.append({"new": new, "old": "GB-01", "subset": s, "n": int(m.sum()), "defaults": int(y[m].sum()),
                     "d_auc": M.auc(y[m], P[new][m]) - M.auc(y[m], P["GB-01"][m]), "d_auc_lo": S["ci"](da)[0], "d_auc_hi": S["ci"](da)[1],
                     "d_ll": M.log_loss(y[m], P[new][m]) - M.log_loss(y[m], P["GB-01"][m]), "d_ll_lo": S["ci"](dl)[0], "d_ll_hi": S["ci"](dl)[1],
                     "n_draws": len(da)})
    for lab, a, b in (("gain seen minus gain new (AUC), all years", "seen", "new"),
                      *[(f"gain seen minus gain new (AUC), {t}", f"{t} seen", f"{t} new") for t in (2008, 2009, 2010)]):
        g = (dr[("auc", new, a)] - dr[("auc", "GB-01", a)]) - (dr[("auc", new, b)] - dr[("auc", "GB-01", b)])
        pt = (M.auc(y[sub[a]], P[new][sub[a]]) - M.auc(y[sub[a]], P["GB-01"][sub[a]])) - (M.auc(y[sub[b]], P[new][sub[b]]) - M.auc(y[sub[b]], P["GB-01"][sub[b]]))
        rows.append({"new": new, "old": "GB-01", "subset": lab, "d_auc": pt, "d_auc_lo": S["ci"](g)[0], "d_auc_hi": S["ci"](g)[1], "n_draws": len(g)})
r = pd.DataFrame(rows)
r.to_csv(os.path.join(HERE, "comparison_r2b_bootstrap_1000.csv"), index=False)
print(r.round(5).to_string())
