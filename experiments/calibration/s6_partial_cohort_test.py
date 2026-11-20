"""Stage 6 (calibration-specialist), Q6: is it safe and useful to train on a cohort whose label
window is only 11 of 12 months observed (fs_year 2011 in the final fit)?

Walk-forward analogue on fold 3 (test fs_year 2010, all labels complete):
  F3-full  : the standard fold-3 fit (train 2007-2009, full 12-month labels)        [= submitted]
  F3-trunc : same rows, but the LAST training cohort (2009) has an 11-month label: a default in the
             12th month of its window is set to 0 (exactly what fs_year 2011 would look like)
  F3-drop  : last training cohort dropped (train 2007-2008): what we lose by excluding the newest
             cohort from the fit
Every variant gets the same level treatment for scoring (prior shift to the same target, rule e),
so the comparison isolates ranking/shape. Labels are built from the panel's own win_start/def_date
(not src/common/labels.make_labels, which has the D-011/D-068 censoring bug).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/gbm"))
sys.path.insert(0, str(ROOT / "experiments/riskcalc"))
import gbm_lib as G  # noqa: E402
import riskcalc as RC  # noqa: E402
from src.common import metrics as M  # noqa: E402

OUT = ROOT / "reports/calibration"
panel = pd.read_parquet(ROOT / "data/interim/panel.parquet")
testrows = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet")
ok = panel[panel.label_status == "ok"].copy()
te = panel.set_index("row_id").loc[testrows.loc[testrows.fold == 3, "row_id"]].reset_index()
y_te = te.y.astype(int).to_numpy()
tau = float(pd.read_csv(OUT / "ct_targets.csv").query("label=='s6' and test_year==2010 and rule=='e_last_6m_annualised'").target.iloc[0])

tr_full = ok[ok.fs_year.isin([2007, 2008, 2009])].reset_index(drop=True)
tr_trunc = tr_full.copy()
cut = tr_trunc.win_start + pd.DateOffset(months=11)
m12 = (tr_trunc.fs_year == 2009) & (tr_trunc.y == 1) & (tr_trunc.def_date >= cut)
tr_trunc.loc[m12, "y"] = 0.0
tr_drop = ok[ok.fs_year.isin([2007, 2008])].reset_index(drop=True)
cfg = {**RC.DEFAULT_CONFIG, **{"ratios": list(RC.DEFAULT_CONFIG["ratios"]) + ["bank_debt_ratio", "ap_days"]}}


def shift_to(p, r_w, tau):
    z = np.log(p / (1 - p)) + np.log(tau / (1 - tau)) - np.log(r_w / (1 - r_w))
    return 1 / (1 + np.exp(-z))


rows = []
for name, tr in (("F3-full", tr_full), ("F3-trunc", tr_trunc), ("F3-drop", tr_drop)):
    r_w = tr.y.mean()
    art = G.fit_pd_model(tr, exp="GB-01", log=lambda *a: None)
    prm = RC.fit_riskcalc(tr, cfg)
    for model, p in (("GB-01", G.predict_pd(art, te)), ("RC-02", RC.predict_riskcalc(te, prm))):
        q = shift_to(p, r_w, tau)
        rows.append({"variant": name, "model": model, "n_train": len(tr), "train_defaults": int(tr.y.sum()),
                     "labels_set_to_0": int(m12.sum()) if name == "F3-trunc" else 0, "r_w": r_w,
                     "auc": M.auc(y_te, p), "log_loss_raw_level": M.log_loss(y_te, p),
                     "log_loss_level_e": M.log_loss(y_te, q), "mean_pd_level_e": q.mean(),
                     **M.calibration_slope_intercept(y_te, q)})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "q6_partial_cohort_test.csv", index=False)
