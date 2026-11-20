"""Referee spot-check (round 1): refit fold 3 of SC-03, RC-03 and GB-01 with the challengers' CURRENT
code on the fold-3 training rows only (fs_year 2007-2009, label_status ok) and compare with the
submitted fold-3 OOT predictions. Confirms (a) the submitted file was produced by the code we audited
and (b) the fit consumes nothing but the training frame passed in.
    python reports/tournament/repro_check_r1.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments/riskcalc")); sys.path.insert(0, str(ROOT / "experiments/gbm"))
import experiments.scorecard.scorecard_lib as L
import riskcalc as RC
import gbm_lib as G
from run import EXPERIMENTS as RCEXP

panel = pd.read_parquet(ROOT / "data/interim/panel.parquet")
sp = json.load(open(ROOT / "data/splits/walkforward.json"))
f = [x for x in sp["folds"] if x["fold"] == 3][0]
trn = panel[panel.fs_year.isin(f["train_years"]) & (panel.label_status == "ok")].copy()
assert trn.fs_year.max() < f["test_year"] and len(trn) == f["n_train"]
tr_rows = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet").query("fold == 3")
te = panel.set_index("row_id").loc[tr_rows.row_id].reset_index()
# poison test labels: if any code path read them, refit would not matter, but we make sure the
# fitting frame contains no test-year rows at all (asserted above) and give predict a label-free frame
te_nolab = te.drop(columns=[c for c in te.columns if c.startswith(("y", "status", "def_date", "label_status", "win_"))])
out = []
def cmp(name, p_new, path):
    o = pd.read_parquet(ROOT / path).query("fold == 3").set_index("row_id")["pd"]
    d = np.abs(o.loc[te.row_id].to_numpy() - p_new)
    out.append({"exp": name, "n": len(d), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())})
    print(out[-1], flush=True)

prm = L.fit_woe_scorecard(trn); cmp("SC-03", L.predict_woe(te_nolab, prm), "experiments/scorecard/SC-03/oot_predictions.parquet")
cfg = {**RC.DEFAULT_CONFIG, **RCEXP["RC-03"]["config"]}
prc = RC.fit_riskcalc(trn, cfg); cmp("RC-03", RC.predict_riskcalc(te_nolab, prc), "experiments/riskcalc/RC-03/oot_predictions.parquet")
G.BASE["num_threads"] = 8
art = G.fit_pd_model(trn, "GB-01", log=lambda s: None); cmp("GB-01", G.predict_pd(art, te_nolab), "experiments/gbm/GB-01/oot_predictions.parquet")
pd.DataFrame(out).to_csv(ROOT / "reports/tournament/r1_repro_check.csv", index=False)
