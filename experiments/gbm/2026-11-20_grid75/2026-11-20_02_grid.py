"""Phase-1 grid search: LightGBM on the 75 features, monotone vs unconstrained, rounds 1-3.

Per (setting, mode, round): fit on the round's train file, early-stop on its validation file using AUC
(first_metric_only), record validation and test metrics. Test metrics are REPORTED ONLY, never used to
select (round-3 test = FY2011 lockbox). Results are appended row by row to grid_results.csv, so the run
can be stopped and resumed (finished keys are skipped).

Usage: python 2026-11-20_02_grid.py [--time-budget-hours H] [--limit N]
"""
import argparse, itertools, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from common import load, feature_list, OUT_DIR, ROUNDS

GRID = {
    "num_leaves": [7, 15, 31, 63],
    "min_child_samples": [100, 300, 1000, 3000],
    "feature_fraction": [0.5, 0.8, 1.0],
    "bagging_fraction": [0.7, 1.0],
    "lambda_l2": [0.0, 10.0, 50.0],
}
FIXED = dict(objective="binary", metric=["auc", "binary_logloss"], first_metric_only=True,
             learning_rate=0.05, max_bin=255, seed=42, deterministic=True, force_row_wise=True,
             num_threads=8, verbose=-1)
MAX_ROUNDS, PATIENCE = 5000, 200
RESULTS = OUT_DIR / "grid_results.csv"


def settings():
    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    # priority order: the setting closest to the old bundle first, then a seeded shuffle,
    # so a time-limited run is still a uniform random sample of the grid
    rng = np.random.default_rng(20261120)
    order = rng.permutation(len(combos))
    combos = [combos[i] for i in order]
    old = dict(num_leaves=15, min_child_samples=300, feature_fraction=1.0, bagging_fraction=1.0, lambda_l2=0.0)
    combos.sort(key=lambda c: c != old)
    return combos


def key(s):
    return "nl{num_leaves}_mcs{min_child_samples}_ff{feature_fraction}_bf{bagging_fraction}_l2{lambda_l2}".format(**s)


def metrics(y, p, prefix):
    return {f"{prefix}_auc": roc_auc_score(y, p), f"{prefix}_logloss": log_loss(y, p),
            f"{prefix}_brier": brier_score_loss(y, p), f"{prefix}_obs_pred": y.mean() / p.mean(),
            f"{prefix}_mean_pd": p.mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-budget-hours", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N settings (timing test)")
    a = ap.parse_args()
    feats = feature_list()
    signs = pd.read_csv(OUT_DIR / "sign_table.csv").set_index("feature").loc[feats, "constraint"].astype(int).tolist()
    modes = {"mono": signs, "free": None}
    combos = settings()[: a.limit or None]
    done = set()
    if RESULTS.exists():
        r = pd.read_csv(RESULTS)
        done = set(zip(r.setting, r["mode"], r["round"]))
    t_start, deadline = time.time(), time.time() + a.time_budget_hours * 3600
    data = {}
    for rnd in ROUNDS:
        tr, va, te = (load(rnd, s) for s in ["train", "validation", "test"])
        ds_tr = lgb.Dataset(tr[feats], tr.label, params={"max_bin": 255, "feature_pre_filter": False},
                            free_raw_data=False).construct()
        ds_va = lgb.Dataset(va[feats], va.label, reference=ds_tr, free_raw_data=False).construct()
        data[rnd] = (ds_tr, ds_va, va, te)
    print(f"loaded data in {time.time()-t_start:.0f}s; {len(combos)} settings x 2 modes x 3 rounds", flush=True)

    n = 0
    for s in combos:
        for mode, mono in modes.items():
            for rnd in ROUNDS:
                if (key(s), mode, rnd) in done:
                    continue
                if time.time() > deadline:
                    print("time budget reached; stopping cleanly", flush=True)
                    return
                ds_tr, ds_va, va, te = data[rnd]
                p = dict(FIXED, **s, bagging_freq=1 if s["bagging_fraction"] < 1 else 0)
                if mono is not None:
                    p.update(monotone_constraints=mono, monotone_constraints_method="intermediate")
                t0 = time.time()
                m = lgb.train(p, ds_tr, num_boost_round=MAX_ROUNDS, valid_sets=[ds_va],
                              callbacks=[lgb.early_stopping(PATIENCE, first_metric_only=True, verbose=False)])
                bi = m.best_iteration
                row = dict(setting=key(s), mode=mode, round=rnd, **s, best_iter=bi, fit_sec=round(time.time() - t0, 1))
                row.update(metrics(va.label.to_numpy(), m.predict(va[feats], num_iteration=bi), "val"))
                row.update(metrics(te.label.to_numpy(), m.predict(te[feats], num_iteration=bi), "test"))
                pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
                n += 1
                print(f"[{n}] {key(s)} {mode} {rnd}: val AUC {row['val_auc']:.4f} LL {row['val_logloss']:.5f} "
                      f"iter {bi} {row['fit_sec']}s (elapsed {(time.time()-t_start)/60:.0f} min)", flush=True)
    print("grid complete", flush=True)


if __name__ == "__main__":
    main()
