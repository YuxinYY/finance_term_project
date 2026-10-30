"""Stage 8: frozen level parameters for harness-engineer (no holdout data used; nothing re-estimated
at prediction time). PD_final = clip(expit(logit(PD_model) + delta), floor, cap), with
delta = logit(tau) - logit(r_w), r_w = default rate of the rows the final model is fitted on, and the
model's own calibrator anchored to r_w (GB-01: Platt slope + anchor; RC-02: final transform F).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/calibration"


def logit(p):
    return float(np.log(p / (1 - p)))


lv = pd.read_csv(OUT / "q5_label_window_levels.csv").set_index("label")
P = pd.read_parquet(ROOT / "data/interim/panel.parquet", columns=["fs_year", "y", "label_status", "win_start", "def_date"])
ok = P[P.label_status == "ok"]
c11 = P[(P.fs_year == 2011) & (P.label_status == "censored")]
d11 = int(((c11.def_date >= c11.win_start) & (c11.def_date <= pd.Timestamp("2013-05-31"))).sum())
r_w = {"fit_2007_2010": float(ok.y.mean()),
       "fit_2007_2011_with_11m_label": float((ok.y.sum() + d11) / (len(ok) + len(c11)))}
out = {"created_by": "calibration-specialist, experiments/calibration/s8_harness_level_params.py",
       "method": "prior shift (Saerens et al. 2002, eq. 4) = logit intercept shift by the log prior-odds ratio",
       "floor": 0.0003, "cap": 0.99, "primary_label": "s6", "tau": {}, "r_w": r_w, "delta": {}}
for lab in ("s6", "s3", "s0"):
    tau = float(lv.loc[lab, "recommended_tau_same_rule"])
    out["tau"][lab] = tau
    out["delta"][lab] = {k: logit(tau) - logit(v) for k, v in r_w.items()}
out["note"] = ("Use tau['s6'] unless Roger/TA confirm a different label window (D-002a). r_w must be the "
               "default rate of the exact rows passed to the final fit; recompute delta if those rows change.")
json.dump(out, open(ROOT / "experiments/calibration/harness_level_params.json", "w"), indent=1)
print(json.dumps(out, indent=1))
