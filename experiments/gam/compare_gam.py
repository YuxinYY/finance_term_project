"""Paired firm-clustered bootstrap (200 draws, seed 0) using the referee's fast verified engine
(AUCEngine, row_loss, bootstrap_draws copied by source extraction from reports/tournament/score_r1.py, D-069)."""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.common import metrics as M
src = open(os.path.join(ROOT, "reports/tournament/score_r1.py")).read().split("\n")
def block(start, end): return "\n".join(src[start - 1:end])
i0 = next(i for i, l in enumerate(src, 1) if l.startswith("class AUCEngine"))
i1 = next(i for i, l in enumerate(src, 1) if l.startswith("# 4a."))
N_BOOT, SEED = 200, 0
exec(block(i0, i1 - 1))
panel = pd.read_parquet(os.path.join(ROOT, "data/interim/panel.parquet"), columns=["row_id", "id", "y"])
def ld(path): return pd.read_parquet(os.path.join(ROOT, path)).set_index("row_id")["pd"]
P = {"GAM-01": ld("experiments/gam/GAM-01/oot_predictions.parquet"), "GAM-02": ld("experiments/gam/GAM-02/oot_predictions.parquet"),
     "GB-01": ld("experiments/gbm/GB-01/oot_predictions.parquet"), "CAL-01": ld("experiments/calibration/CAL-01/oot_predictions.parquet"),
     "GAM-01+CAL": ld("experiments/gam/GAM-01/oot_predictions_calibrated.parquet"), "GAM-02+CAL": ld("experiments/gam/GAM-02/oot_predictions_calibrated.parquet")}
ix = P["GAM-01"].index
d = panel.set_index("row_id").loc[ix]
y = d["y"].astype(int).to_numpy(); firm = d["id"].to_numpy()
Pn = {k: v.loc[ix].to_numpy() for k, v in P.items()}
dr = bootstrap_draws(y, firm, Pn, {"ALL": np.ones(len(y), bool)}, n_boot=N_BOOT, seed=SEED)
pairs = [("GAM-02", "GAM-01"), ("GAM-01", "GB-01"), ("GAM-02", "GB-01"), ("GAM-01+CAL", "CAL-01"), ("GAM-02+CAL", "CAL-01"), ("GAM-02+CAL", "GAM-01+CAL")]
rows = []
for a, b in pairs:
    r = {"pair": f"{a} - {b}"}
    for m_ in ("auc", "log_loss"):
        pt = (M.auc(y, Pn[a]) - M.auc(y, Pn[b])) if m_ == "auc" else (M.log_loss(y, Pn[a]) - M.log_loss(y, Pn[b]))
        dd = dr[(m_, a, "ALL")] - dr[(m_, b, "ALL")]
        r.update({f"d_{m_}": pt, f"d_{m_}_lo": np.percentile(dd, 2.5), f"d_{m_}_hi": np.percentile(dd, 97.5)})
    rows.append(r)
out = pd.DataFrame(rows); out.to_csv(os.path.join(ROOT, "experiments/gam/paired_bootstrap.csv"), index=False)
print(out.to_string(float_format=lambda x: f"{x:.5f}"))
