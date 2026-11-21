"""Robustness of the free-vs-mono gap for the shared winner setting: full-data binning
(bin_construct_sample_cnt = all rows) and 3 seeds. Validation only (test reported nowhere here)."""
import importlib, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from common import load, feature_list, OUT_DIR, ROUNDS
grid = importlib.import_module("2026-11-20_02_grid")
feats = feature_list()
signs = pd.read_csv(OUT_DIR / "sign_table.csv").set_index("feature").loc[feats, "constraint"].astype(int).tolist()
S = dict(num_leaves=31, min_child_samples=3000, feature_fraction=0.5, bagging_fraction=1.0, lambda_l2=10.0)
rows = []
for rnd in ROUNDS:
    tr, va = load(rnd, "train"), load(rnd, "validation")
    dsp = {"max_bin": 255, "feature_pre_filter": False, "bin_construct_sample_cnt": 10**7}
    ds_tr = lgb.Dataset(tr[feats], tr.label, params=dsp, free_raw_data=False).construct()
    ds_va = lgb.Dataset(va[feats], va.label, reference=ds_tr, free_raw_data=False).construct()
    for seed in (1, 2, 3):
        for mode in ("mono", "free"):
            p = dict(grid.FIXED, **S, seed=seed)
            if mode == "mono":
                p.update(monotone_constraints=signs, monotone_constraints_method="intermediate")
            m = lgb.train(p, ds_tr, num_boost_round=grid.MAX_ROUNDS, valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(grid.PATIENCE, first_metric_only=True, verbose=False)])
            pv = m.predict(va[feats], num_iteration=m.best_iteration)
            rows.append(dict(round=rnd, seed=seed, mode=mode, auc=roc_auc_score(va.label, pv),
                             logloss=log_loss(va.label, pv), best_iter=m.best_iteration))
            print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.to_csv(OUT_DIR / "seed_check.csv", index=False)
w = d.pivot_table(index=["round", "seed"], columns="mode", values=["auc", "logloss"])
gap = pd.DataFrame({"dAUC": w["auc"]["free"] - w["auc"]["mono"], "dLL": w["logloss"]["free"] - w["logloss"]["mono"]})
print(gap.round(5)); print("mean per round:\n", gap.groupby(level=0).mean().round(5)); print("overall", gap.mean().round(5).to_dict())
print("best_iter by mode:\n", d.groupby(["mode", "round"]).best_iter.agg(["min", "max"]))
