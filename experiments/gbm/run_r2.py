"""Round-2 outer runs (GB-08m, GB-09, GB-10, GB-11, GB-12). Every choice is made by r2_lib.fit_general on
the inner slices of each fold's training window; the outer test year is scored once, here.

usage: python experiments/gbm/run_r2.py GB-09 [--features full|core|FULL2_FEATURES] [--tag suffix]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.model_selection import ParameterSampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gbm_lib as G  # noqa: E402
import r2_lib as L  # noqa: E402
import run_experiment as R  # noqa: E402  (metrics_md, fmt only)
from src.common import metrics as M  # noqa: E402

# ---------------------------------------------------------------------------------------------- search space
SEARCH_SPACE = {
    "learning_rate": loguniform(0.015, 0.10),
    "max_depth": [2, 3, 4],
    "num_leaves": [4, 6, 8, 12, 16],
    "min_data_in_leaf": [200, 500, 1000, 2000, 4000],
    "feature_fraction": uniform(0.5, 0.5),
    "bagging_fraction": uniform(0.5, 0.5),
    "lambda_l1": [0.0, 0.1, 1.0, 10.0],
    "lambda_l2": loguniform(0.1, 100.0),
    "min_gain_to_split": [0.0, 0.01, 0.1, 0.5],
}
N_RANDOM = 32


def search_cfgs():
    out = [dict(G.DEFAULT)] + [dict(c) for c in G.GRID]          # incumbent's own candidates first
    for c in ParameterSampler(SEARCH_SPACE, n_iter=N_RANDOM, random_state=0):
        c = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in c.items()}
        c["num_leaves"] = int(min(c["num_leaves"], 2 ** c["max_depth"]))
        out.append(c)
    return out


def gb01_cfgs(k):
    return [dict(G.DEFAULT)] if k == 1 else [dict(c) for c in G.GRID]


HALF_LIVES = [None, 4.0, 2.0, 1.0, 0.5]       # None first: ties (1 inner-train year) -> no weighting


def recency_cfgs(k):
    return [{**c, "_hl": h} for h in HALF_LIVES for c in gb01_cfgs(k)]


RF_CFGS = [dict(num_leaves=nl, max_depth=-1, min_data_in_leaf=md, feature_fraction=0.5, bagging_fraction=0.632,
                bagging_freq=1, lambda_l2=0.0, min_sum_hessian_in_leaf=1e-3)
           for nl in (63, 255) for md in (200, 1000)]

SKRF_CFGS = [dict(n_estimators=300, min_samples_leaf=msl, max_features="sqrt", max_samples=0.5)
             for msl in (50, 200, 1000)]

# minimal defensible constraint set (GB-08): filled from gb08_ablation.py judgement, see decisions.md D-102
MINIMAL_RELAX: list[str] = []
_mr = os.path.join(HERE, "GB-08", "minimal_relax.json")
if os.path.exists(_mr):
    MINIMAL_RELAX = json.load(open(_mr))["relax"]

def gb12_cfgs(k):
    """GB-09's 5 best cfgs on this fold's inner A (inner-only information) + GB-01's own candidates."""
    c = json.load(open(os.path.join(HERE, "GB-09", "config.json")))
    t = sorted([f for f in c["folds"] if f["fold"] == k][0]["trials"], key=lambda t: t["A_log_loss"])[:5]
    out = gb01_cfgs(k) + [x["cfg"] for x in t]
    return [dict(x) for i, x in enumerate(out) if x not in out[:i]]


SPECS = {
    "GB-01r": dict(relax=[], cfgs=gb01_cfgs, desc="GB-01 rebuilt with r2_lib (reproduction check)"),
    "GB-08": dict(relax="MINIMAL", cfgs=gb01_cfgs,
                  desc="minimal defensible monotone constraints (economically ambiguous constraints dropped)"),
    "GB-08b": dict(relax=["ebitda_margin"], cfgs=gb01_cfgs,
                   desc="GB-01 with only the ebitda_margin constraint dropped (the one drop with a supported economic "
                        "reason; conceived AFTER GB-08's outer score was seen, see D-103)"),
    "GB-09": dict(relax=[], cfgs=lambda k: search_cfgs(),
                  desc="GB-01 constraints + random search (37 cfgs incl. GB-01's own) on inner slice A"),
    "GB-10": dict(relax=[], cfgs=recency_cfgs,
                  desc="GB-01 + recency weights, half-life chosen on inner slice A from {none,4,2,1,0.5} yrs"),
    "GB-12": dict(relax="MINIMAL", cfgs=gb12_cfgs,
                  desc="diagnostic combination: GB-08 minimal constraints + GB-09 top-5 inner cfgs per fold (+GB-01 cfgs)"),
    "GB-11": dict(relax="ALL", cfgs=lambda k: SKRF_CFGS, rf="sklearn",
                  desc="random-forest benchmark (sklearn RandomForest, 300 trees, sqrt features, 50% rows/tree, "
                       "min_samples_leaf chosen on inner A; one-hot categoricals; D-053 calibration)"),
    "GB-11lgbrf": dict(relax="ALL", cfgs=lambda k: RF_CFGS, rf=True,
                       desc="REJECTED implementation: LightGBM boosting='rf' (raw probabilities near 0/1, see D-106)"),
}


def resolve_relax(relax, feats):
    if relax == "ALL":
        return list(feats)
    if relax == "MINIMAL":
        return list(MINIMAL_RELAX)
    return list(relax)


def run(exp, spec, features="full", tag=""):
    name = exp + tag
    out_dir = os.path.join(HERE, name)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train_log.md"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    feats = L.feature_list(features)
    relax = resolve_relax(spec["relax"], feats)
    mono = L.monotone_vector(feats, True, relax)
    panel = L.load_panel()
    wf, test_rows = L.load_splits()
    log(f"# {name} training log\n\n{spec['desc']}\nfeature set: {features} ({len(feats)} features)\n"
        f"relaxed (unconstrained) features: {relax}\nmonotone vector: {dict(zip(feats, mono))}\n"
        f"BASE params: {json.dumps(L.BASE)}\nLockbox: panel read with fs_year < {L.LOCKBOX_YEAR}; "
        f"max fs_year read = {int(panel['fs_year'].max())}\n")
    preds, raws, inner, recs = [], [], [], []
    for f in wf["folds"]:
        k = f["fold"]
        t0 = time.time()
        f, tr, te = L.fold_frames(panel, wf, test_rows, k)
        cfgs = spec["cfgs"](k)
        log(f"\n## Fold {k}: train {f['train_years']} ({len(tr):,} rows) -> test {f['test_year']}; {len(cfgs)} candidate cfgs")
        art = L.fit_general(tr, feats, mono, cfgs, rf=spec.get("rf", False), log=log, tag=name)
        rec = art["record"]
        p = G.predict_pd(art, te)
        pr = G.predict_pd(art, te, raw=True)
        assert np.all((p > 0) & (p < 1)) and len(p) == len(te)
        yte = te["y"].astype(int).to_numpy()
        preds.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(), "fold": k, "pd": p, "y": yte,
                                   "test_year": f["test_year"], "seen": te["firm_seen_in_train"].to_numpy()}))
        raws.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(), "fold": k, "pd": pr}))
        ib = rec.pop("inner_B_predictions")
        ib["fold"] = k
        inner.append(ib)
        ch = [t for t in rec["trials"] if t["i"] == rec["chosen_index"]][0]
        rec.update(fold=k, test_year=f["test_year"], wall_secs=round(time.time() - t0, 1),
                   oot_auc=M.auc(yte, p), oot_log_loss=M.log_loss(yte, p))
        recs.append(rec)
        log(f"  CHOSEN #{rec['chosen_index']} {rec['chosen_cfg']} best_iter={rec['best_iter']} final_trees={rec['n_final_trees']}\n"
            f"  inner: A_ll {ch['A_log_loss']:.5f} A_auc {ch['A_auc']:.4f} | B_ll {ch['B_log_loss']:.5f} B_auc {ch['B_auc']:.4f}\n"
            f"  Platt on B b={rec['platt_on_B']['b']:.3f}; in-sample AUC {rec['in_sample']['auc']:.4f}\n"
            f"  OUTER test {f['test_year']} (scored once): AUC {rec['oot_auc']:.4f} log loss {rec['oot_log_loss']:.5f} "
            f"mean pd {p.mean():.4f} vs DR {yte.mean():.4f} ({rec['wall_secs']}s)")
        if k == 3 and spec.get("rf") != "sklearn":
            G.save_model(art, os.path.join(out_dir, "model_fold3"))
            with open(os.path.join(HERE, f".{name}_fold3_art.pkl"), "wb") as fh:
                pickle.dump({kk: vv for kk, vv in art.items() if kk != "booster"}
                            | {"booster_str": art["booster"].model_to_string()}, fh)
    oot = pd.concat(preds, ignore_index=True)
    raw = pd.concat(raws, ignore_index=True)
    oot[["row_id", "fold", "pd"]].to_parquet(os.path.join(out_dir, "oot_predictions.parquet"), index=False)
    raw.to_parquet(os.path.join(out_dir, "oot_predictions_raw.parquet"), index=False)
    pd.concat(inner, ignore_index=True).to_parquet(os.path.join(out_dir, "inner_B_predictions.parquet"), index=False)
    md, s = R.metrics_md(name, oot, raw, spec["desc"])
    open(os.path.join(out_dir, "metrics.md"), "w").write(md)
    s.to_csv(os.path.join(out_dir, "metrics_by_year.csv"), index=False)
    cfg = {"exp": name, "desc": spec["desc"], "features": features, "relax": relax, "rf": spec.get("rf", False),
           "base_params": L.BASE, "max_rounds": G.MAX_ROUNDS, "early_stopping_rounds": G.EARLY_STOP,
           "n_final_factor": G.N_FINAL_FACTOR, "selection": "min log loss on inner slice A (early stopping also on A); "
           "B = Platt slope + untouched check", "search_space": {k: str(v) for k, v in SEARCH_SPACE.items()} if exp == "GB-09" else None,
           "half_lives": HALF_LIVES if exp == "GB-10" else None,
           "class_weighting": "none" if exp != "GB-10" else "recency weights only (no class weights)",
           "lockbox": f"fs_year >= {L.LOCKBOX_YEAR} never read", "folds": recs}
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1, default=str)
    log("\n## Pooled / per-year (own calibration)\n" + R.fmt(s))
    logf.close()
    return oot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--features", default="full")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(a.exp, SPECS[a.exp], a.features, a.tag)


if __name__ == "__main__":
    main()
