"""Run SC-01..SC-04 over the 3 frozen walk-forward folds; write predictions, metrics, config."""
import json, sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/kietnguyen/Downloads/pd-agent-kit")
from src.common import metrics as M
import experiments.scorecard.scorecard_lib as L

ROOT = pathlib.Path("/Users/kietnguyen/Downloads/pd-agent-kit")
OUT = ROOT / "experiments/scorecard"
sp = json.load(open(ROOT / "data/splits/walkforward.json"))
tr = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet")
panel = pd.read_parquet(ROOT / "data/interim/panel.parquet")
P = panel.set_index("row_id")
ids = sys.argv[1:] or list(L.FIT)
for eid in ids:
    d = OUT / eid; d.mkdir(exist_ok=True)
    preds, fitlog = [], []
    for f in sp["folds"]:
        trn = panel[panel.fs_year.isin(f["train_years"]) & (panel.label_status == "ok")]
        te = tr[tr.fold == f["fold"]]
        prm = L.FIT[eid](trn)
        test = P.loc[te.row_id]
        pd_ = L.PRED[eid](test.reset_index(), prm)
        preds.append(pd.DataFrame({"row_id": te.row_id.values, "fold": f["fold"], "pd": pd_}))
        fitlog.append({"fold": f["fold"], "n_train": len(trn), "train_years": f["train_years"],
                       "chosen": prm.get("chosen"), "coef": dict(zip(prm["glm"]["names"], prm["glm"]["coef"]))})
        if f["fold"] == 3:
            L.save_params(prm, d / "params_fold3.json")
            if eid == "SC-03": L.scorecard_points(prm).to_csv(d / "scorecard_points_fold3.csv", index=False)
    pr = pd.concat(preds, ignore_index=True)
    assert pr.pd.between(0, 1, inclusive="neither").all() and pr.pd.notna().all()
    pr.to_parquet(d / "oot_predictions.parquet", index=False)
    m = pr.merge(P[["y", "fs_year"]], left_on="row_id", right_index=True)
    sc = M.score(m.y, m.pd, group=m.fold)
    lines = [f"# {eid} metrics (frozen walk-forward, test 2008/2009/2010)\n",
             "AR = 2*AUC-1. Brier = overall probability accuracy (not calibration). Group = fold (1=2008, 2=2009, 3=2010).\n",
             sc.to_markdown(index=False, floatfmt=".4f"), "\n\n## Decile table (pooled)\n",
             M.decile_table(m.y, m.pd).to_markdown(index=False, floatfmt=".4f")]
    (d / "metrics.md").write_text("\n".join(lines))
    sc.to_csv(d / "metrics.csv", index=False)
    json.dump({"exp_id": eid, "folds": fitlog, "split": "data/splits/walkforward.json",
               "tuning": "none: no tunable hyper-parameter (unpenalised GLM); SC-03 uses pre-declared 20 initial quantile bins, min bin 2%, PAV monotone; all folds use the same pre-declared settings",
               "features_version": L.B.VERSION}, open(d / "config.json", "w"), indent=1, default=str)
    print(eid); print(sc[["group", "n", "defaults", "auc", "log_loss", "brier", "mean_pd", "observed_dr", "cal_intercept", "cal_slope"]].round(4).to_string())
