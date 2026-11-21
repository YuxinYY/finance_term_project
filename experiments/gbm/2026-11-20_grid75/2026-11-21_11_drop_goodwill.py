"""What does dropping goodwill_ratio cost? (KN 2026-11-21: the `goodwill` field behaves like value added, not
accounting goodwill, D-010 / D-091, so the feature cannot be explained under its name.)

Same protocol as the constraint ablation (2026-11-21_06): D-158 winner setting, full-data binning, seeds 1-2,
rounds 1-3, early stopping on validation AUC, 20 monotone constraints (D-157). Configurations:
  mono        the 75-feature mono20 model
  mono_nogw   the same without goodwill_ratio (74 features)
Validation = selection years FY2008-10; test years (FY2009-11) reported only.
Paired bootstrap (500 draws, rows = firms, one row per firm per year) on seed-1 validation predictions.
Outputs: drop_goodwill/results.csv, summary.csv, bootstrap.csv
"""
import importlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

from common import load, feature_list, OUT_DIR, ROUNDS

grid = importlib.import_module("2026-11-20_02_grid")
OUT = OUT_DIR / "drop_goodwill"
OUT.mkdir(exist_ok=True)
feats = feature_list()
cons = pd.read_csv(OUT_DIR / "sign_table.csv").set_index("feature").loc[feats, "constraint"].astype(int).to_dict()
S = dict(num_leaves=31, min_child_samples=3000, feature_fraction=0.5, bagging_fraction=1.0, lambda_l2=10.0)
CONFIGS = {"mono": feats, "mono_nogw": [f for f in feats if f != "goodwill_ratio"]}
DSP = {"max_bin": 255, "feature_pre_filter": False, "bin_construct_sample_cnt": 10**7}

rows, preds = [], {}
t0 = time.time()
for rnd in ROUNDS:
    tr, va, te = load(rnd, "train"), load(rnd, "validation"), load(rnd, "test")
    for name, fs in CONFIGS.items():
        ds_tr = lgb.Dataset(tr[fs], tr.label, params=DSP, free_raw_data=False).construct()
        ds_va = lgb.Dataset(va[fs], va.label, reference=ds_tr, free_raw_data=False).construct()
        for seed in (1, 2):
            p = dict(grid.FIXED, **S, seed=seed, monotone_constraints=[cons[f] for f in fs],
                     monotone_constraints_method="intermediate")
            m = lgb.train(p, ds_tr, num_boost_round=grid.MAX_ROUNDS, valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(grid.PATIENCE, first_metric_only=True, verbose=False)])
            pv = m.predict(va[fs], num_iteration=m.best_iteration)
            pt = m.predict(te[fs], num_iteration=m.best_iteration)
            rows.append(dict(config=name, round=rnd, seed=seed, n_features=len(fs),
                             val_auc=roc_auc_score(va.label, pv), val_logloss=log_loss(va.label, pv),
                             test_auc=roc_auc_score(te.label, pt), test_logloss=log_loss(te.label, pt),
                             best_iter=m.best_iteration))
            if seed == 1:
                preds[(name, rnd)] = (va.label.to_numpy(), pv)
            print(rows[-1], f"{(time.time() - t0) / 60:.1f} min", flush=True)

d = pd.DataFrame(rows)
d.to_csv(OUT / "results.csv", index=False)
g = d.groupby("config")[["val_auc", "val_logloss", "test_auc", "test_logloss", "best_iter"]].mean()
g.loc["difference (nogw - mono)"] = g.loc["mono_nogw"] - g.loc["mono"]
g.to_csv(OUT / "summary.csv")
print(g.round(5).to_string())

rng = np.random.default_rng(0)
b = []
for rnd in ROUNDS:
    y, p0 = preds[("mono", rnd)]
    _, p1 = preds[("mono_nogw", rnd)]
    da, dl = [], []
    for _ in range(500):
        i = rng.integers(0, len(y), len(y))
        da.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i]))
        dl.append(log_loss(y[i], p1[i]) - log_loss(y[i], p0[i]))
    b.append(dict(round=rnd, dAUC=roc_auc_score(y, p1) - roc_auc_score(y, p0), dAUC_lo=np.quantile(da, .025),
                  dAUC_hi=np.quantile(da, .975), dLL=log_loss(y, p1) - log_loss(y, p0),
                  dLL_lo=np.quantile(dl, .025), dLL_hi=np.quantile(dl, .975)))
pd.DataFrame(b).to_csv(OUT / "bootstrap.csv", index=False)
print(pd.DataFrame(b).round(5).to_string(index=False))
