"""challenger-gam: Explainable Boosting Machine (EBM) PD model, reusable estimation API.

    art = fit_pd_model(train_df, exp="GAM-01")     # train_df: panel rows with y, fs_year, id (+ raw fields)
    save_model(art, dir); art = load_model(dir)
    p = predict_pd(art, any_df, delta=0.0)         # frozen: no re-estimation, row-wise, PDs in (0,1)

Inner scheme (mirrors GB-01, D-052/D-053; everything inside the training window passed in):
  * >= 2 training years: inner-train = years before the last; last year split by FIRM (crc32 of id)
    into A (early stopping + grid choice by A log loss) and B (Platt slope). Time-ordered.
  * 1 training year (fold 1): random firm split 70/15/15 and the pre-declared DEFAULT config.
  * EBM validation/bagging: passed explicitly through `bags` so that no row outside the training window
    can enter. Inner model: validation = A for every bag (time-ordered), bootstrap counts over inner-train.
    Final model: refit on ALL training rows; per bag a different firm-level 15% validation slice
    (rotating offset, so every row is used by most bags), bootstrap over the rest.
  * Calibration (D-053): Platt slope b from B (inner model), intercept anchored so that the mean PD on
    the training window equals the training default rate. Level shifts (prior shift, rule e) are applied
    separately via `delta` (D-072), computed from training labels only (reports/calibration/ct_targets.csv).
Monotone constraints = B.expected_signs() (+1/-1; 0 = unconstrained), on the shape function (log-odds).
No class re-weighting or resampling.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import zlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.features import build as B  # noqa: E402
from src.common import metrics as M  # noqa: E402
import interpret  # noqa: E402
from interpret.glassbox import ExplainableBoostingClassifier as EBC  # noqa: E402

SEED = 0
EPS = 1e-6
BASE = dict(max_bins=128, max_interaction_bins=32, learning_rate=0.03, smoothing_rounds=75,
            interaction_smoothing_rounds=75, early_stopping_rounds=100, max_rounds=30000,
            inner_bags=0, validation_size=0.15, n_jobs=8, random_state=SEED)
INNER_BAGS, FINAL_BAGS, VAL_FRAC = 4, 8, 0.15
GRID = [dict(max_leaves=ml, min_samples_leaf=msl) for ml in (2, 3) for msl in (200, 1000)]
DEFAULT = dict(max_leaves=2, min_samples_leaf=500)     # fold 1 (one training year), pre-declared
EXPERIMENTS = {
    "GAM-01": dict(interactions=0, monotone=True,
                   desc="EBM main effects only, monotone constraints from the expected-sign table"),
    "GAM-02": dict(interactions=10, monotone=True,
                   desc="GAM-01 + 10 pairwise interaction terms (FAST-selected inside the training window)"),
}


def _firm_u(ids):
    return np.array([zlib.crc32(str(int(i)).encode()) / 2**32 for i in ids])


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def mono_vector(feats, on=True):
    s = B.expected_signs()
    return [int(s.get(f, 0)) if on else 0 for f in feats]


def _boot_counts(rng, idx_pool, n):
    c = np.bincount(rng.integers(0, len(idx_pool), len(idx_pool)), minlength=len(idx_pool))
    out = np.zeros(n, np.int8)
    out[idx_pool] = np.minimum(c, 100)
    return out


def _bags_inner(n, m_tr, m_val, n_bags, seed):
    rng = np.random.default_rng(seed)
    pool = np.where(m_tr)[0]
    bags = np.stack([_boot_counts(rng, pool, n) for _ in range(n_bags)])
    bags[:, m_val] = -1
    return bags


def _bags_final(n, u, n_bags, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bags):
        val = ((u + b / n_bags) % 1.0) < VAL_FRAC
        r = _boot_counts(rng, np.where(~val)[0], n)
        r[val] = -1
        rows.append(r)
    return np.stack(rows)


def _fit_ebm(X, y, bags, cfg, mono, interactions):
    p = {**BASE, **cfg, "outer_bags": bags.shape[0], "interactions": interactions}
    if mono is not None:
        p["monotone_constraints"] = mono
    m = EBC(**p)
    m.fit(X, y, bags=np.ascontiguousarray(bags.T))
    return m


def _p(m, X):
    return np.clip(m.predict_proba(X)[:, 1], EPS, 1 - EPS)


def _platt(z, y):
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def _anchor(z, b, rate):
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _sig(mid + b * z).mean() > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def fit_pd_model(train_df, exp="GAM-01", fixed_cfg=None, log=print):
    spec = EXPERIMENTS[exp]
    feats = list(B.ALL_FEATURES)
    mono = mono_vector(feats, spec["monotone"])
    inter = spec["interactions"]
    df = train_df.reset_index(drop=True)
    y = df["y"].astype(int).to_numpy()
    n = len(df)
    years = sorted(df["fs_year"].unique())
    u = _firm_u(df["id"].to_numpy())
    rec = {"exp": exp, "train_years": [int(t) for t in years], "n_train": n, "train_defaults": int(y.sum()),
           "features": feats, "monotone": mono, "interactions": inter}
    if len(years) >= 2:
        last = years[-1]
        m_tr = (df["fs_year"] < last).to_numpy()
        m_a = (~m_tr) & (u < 0.5)
        m_b = (~m_tr) & (u >= 0.5)
        rec["inner_scheme"] = f"time: inner-train {years[:-1]}, last year {last} split by firm into A (early stop/grid) and B (Platt slope)"
        grid = [fixed_cfg] if fixed_cfg else GRID
    else:
        m_tr, m_a, m_b = u < 0.70, (u >= 0.70) & (u < 0.85), u >= 0.85
        rec["inner_scheme"] = "single training year: random firm split 70/15/15, pre-declared DEFAULT config"
        grid = [fixed_cfg] if fixed_cfg else [DEFAULT]
    for k, m in (("inner_train", m_tr), ("A", m_a), ("B", m_b)):
        rec[f"n_{k}"], rec[f"defaults_{k}"] = int(m.sum()), int(y[m].sum())

    fp_in = B.fit(df[m_tr])
    X_in = B.transform(df, fp_in)[feats]            # transform of all rows with inner-train params
    bags_in = _bags_inner(n, m_tr, m_a, INNER_BAGS, SEED)
    trials = []
    for cfg in grid:
        t0 = time.time()
        m = _fit_ebm(X_in, y, bags_in, cfg, mono, inter)
        pa = _p(m, X_in[m_a])
        trials.append(dict(cfg=cfg, A_log_loss=M.log_loss(y[m_a], pa), A_auc=M.auc(y[m_a], pa),
                           secs=round(time.time() - t0, 1), _m=m))
        log(f"  [{exp}] cfg={cfg} A_logloss={trials[-1]['A_log_loss']:.5f} A_auc={trials[-1]['A_auc']:.4f} "
            f"({trials[-1]['secs']}s)")
    best = min(trials, key=lambda t: t["A_log_loss"])
    rec["grid"] = [{k: v for k, v in t.items() if k != "_m"} for t in trials]
    rec["chosen_cfg"] = best["cfg"]
    pb = _p(best["_m"], X_in[m_b])
    a_b, b_b = _platt(_logit(pb), y[m_b])
    rec["platt_on_B"] = {"a": a_b, "b": b_b}
    rec["B_log_loss_raw"] = M.log_loss(y[m_b], pb)
    rec["B_log_loss_platt"] = M.log_loss(y[m_b], _sig(a_b + b_b * _logit(pb)))

    fp = B.fit(df)
    Xall = B.transform(df, fp)[feats]
    t0 = time.time()
    ebm = _fit_ebm(Xall, y, _bags_final(n, u, FINAL_BAGS, SEED + 1), best["cfg"], mono, inter)
    rec["final_fit_secs"] = round(time.time() - t0, 1)
    z_all = _logit(_p(ebm, Xall))
    a_anchor = _anchor(z_all, b_b, y.mean())
    cal = {"method": "platt_slope_from_B_intercept_anchored_to_train_rate", "a": a_anchor, "b": b_b,
           "target_rate": float(y.mean())}
    rec["calibrator"] = cal
    rec["inner_B_predictions"] = pd.DataFrame({"row_id": df.loc[m_b, "row_id"].to_numpy(), "y": y[m_b],
                                               "pd_raw_inner": pb})
    return {"ebm": ebm, "feature_params": fp, "calibrator": cal, "features": feats, "record": rec}


def predict_pd(art, df, raw=False, delta=0.0):
    """Frozen prediction. delta = logit-intercept shift of the level (prior shift, D-072); 0 = none."""
    X = B.transform(df, art["feature_params"])[art["features"]]
    p = _p(art["ebm"], X)
    if raw:
        return p
    c = art["calibrator"]
    return np.clip(_sig(c["a"] + c["b"] * _logit(p) + delta), 0.0003, 0.99)


def save_model(art, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ebm.pkl"), "wb") as fh:
        pickle.dump(art["ebm"], fh)
    B.save_params(art["feature_params"], os.path.join(out_dir, "feature_params.json"))
    with open(os.path.join(out_dir, "calibrator.json"), "w") as fh:
        json.dump(art["calibrator"], fh, indent=1)
    meta = {"features": art["features"], "record": {k: v for k, v in art["record"].items() if k != "inner_B_predictions"},
            "interpret_version": interpret.__version__, "feature_version": B.VERSION}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)


def load_model(out_dir):
    with open(os.path.join(out_dir, "meta.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(out_dir, "calibrator.json")) as fh:
        cal = json.load(fh)
    with open(os.path.join(out_dir, "ebm.pkl"), "rb") as fh:
        ebm = pickle.load(fh)
    return {"ebm": ebm, "feature_params": B.load_params(os.path.join(out_dir, "feature_params.json")),
            "calibrator": cal, "features": meta["features"], "record": meta["record"]}
