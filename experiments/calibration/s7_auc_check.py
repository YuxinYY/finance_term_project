"""Stage 7: verify that monotone maps leave the per-year AUC unchanged (R-07 pp. 89-91: monotone
recalibration cannot add power). Pooled AUC can move because the shift differs by fold."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/calibration"
r = pd.read_csv(OUT / "methods_by_year.csv")
base = r[r.method == "M0"][["model", "group", "auc"]].rename(columns={"auc": "auc_M0"})
m = r.merge(base, on=["model", "group"])
m["d_auc"] = m.auc - m.auc_M0
m["scope"] = np.where(m.group == "ALL", "pooled 2008-2010", "within test year")
out = m.groupby(["model", "method", "scope"]).d_auc.agg(lambda s: s.abs().max()).rename("max_abs_d_auc").reset_index()
out.to_csv(OUT / "auc_invariance_check.csv", index=False)
print(out.to_string())
