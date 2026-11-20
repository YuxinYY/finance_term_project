"""Round 1 (Chloe LightGBM), part 2: firm-clustered paired bootstrap of every variant vs her method (b_platt),
using src/common/metrics.paired_bootstrap (1000 draws, seed 0), on her label and on our start-6 label.
FY2009 has one row per firm (checked in 2026-11-08_file_checks.csv), so clustering by firm equals row resampling here.

Part 3: label-only preview of the level rules on her window for test years 2009-2011 (no model): what each rule
would have targeted and its level-only extra log loss (Bernoulli KL, as s2_rate_series.py). Rounds 2-3 test 2010/2011.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.common import metrics as M  # noqa: E402

HERE = Path(__file__).resolve().parent
P_ = "2026-11-08_"
N_BOOT = 1000
te = pd.read_parquet(HERE / f"{P_}fy2009_predictions.parquet")
VARS = ["raw", "c_ours", "d_ours_e", "d_ours_b", "d_ours_rw", "d_s6level", "iso_val", "iso_val_e", "d_ours_e_cleanB"]
LABS = {"her_label": (np.ones(len(te), bool), te.y_chloe.to_numpy()),
        "our_label_s6": ((te.status_s6 == "ok").fillna(False).to_numpy(), te.y_s6.to_numpy())}


def one(lab, v):
    mk, yy = LABS[lab]
    out = M.paired_bootstrap(yy[mk].astype(int), te[f"pd_{v}"].to_numpy()[mk], te["pd_b_platt"].to_numpy()[mk],
                             te.id.to_numpy()[mk], n_boot=N_BOOT, seed=0)
    out.insert(0, "variant_vs_b_platt", v)
    out.insert(0, "label", lab)
    return out


res = pd.concat(Parallel(n_jobs=6)(delayed(one)(lab, v) for lab in LABS for v in VARS))
res.to_csv(HERE / f"{P_}bootstrap_vs_b_platt.csv", index=False)
print(res.round(6).to_string())

# ---------------------------------------------------------------- part 3: rule preview (labels only)
R = pd.read_csv(HERE / f"{P_}rates_by_cohort_her_window.csv").set_index("fs_year")


def kl(pi, q):
    return pi * np.log(pi / q) + (1 - pi) * np.log((1 - pi) / (1 - q))


rows = []
for t in (2009, 2010, 2011):
    prev = R.loc[: t - 1]
    rules = {"a_pooled": prev.defaults.sum() / prev.n.sum(), "b_last_year": prev.rate.iloc[-1],
             "e_last_6m_annualised": prev.half_year_e_months7_12.iloc[-1],
             "train_rate_of_her_round_model (FY2007..t-2)": R.loc[: t - 2].defaults.sum() / R.loc[: t - 2].n.sum()}
    for k, v in rules.items():
        pi = R.rate[t]
        rows.append({"test_year": t, "rule": k, "target": v, "observed_her_label": pi, "obs_over_target": pi / v,
                     "level_kl_1e4": kl(pi, v) * 1e4})
pv = pd.DataFrame(rows)
pv.to_csv(HERE / f"{P_}rule_preview_her_window.csv", index=False)
print(pv.round(5).to_string())
