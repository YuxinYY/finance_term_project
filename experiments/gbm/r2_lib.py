"""challenger-gbm, round 2: generalised fitter for GB-07..GB-12 (gbm_lib.py is left untouched so GB-01
stays reproducible; this module reuses its helpers).

Same inner scheme as GB-01 (D-052): inside the training window only.
  * >= 2 training years: inner-train = all training years but the last; last training year split by
    firm hash into A (early stopping + every model choice) and B (Platt slope; also an untouched check
    of whether the choice made on A holds).
  * 1 training year (fold 1): firm split 70/15/15 (train / A / B) inside 2007.
Outer test years are scored once per final configuration (run_r2.py).

LOCKBOX: load_panel() reads only fs_year < 2011 (pyarrow filter at read time) and asserts it.
Feature set by NAME: "full" (B.ALL_FEATURES), "core" (B.CORE_FEATURES) or any list attribute of
src/features/build.py, e.g. "FULL2_FEATURES" (added by feature-architect).
"""
from __future__ import annotations

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gbm_lib as G  # noqa: E402
from src.features import build as B  # noqa: E402
from src.common import metrics as M  # noqa: E402

ROOT = G.ROOT
LOCKBOX_YEAR = 2011
THREADS = int(os.environ.get("GBM_THREADS", "4"))
BASE = {**G.BASE, "num_threads": THREADS}


# ---------------------------------------------------------------------------------------------- data
def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(os.path.join(ROOT, "data/interim/panel.parquet"),
                        filters=[("fs_year", "<", LOCKBOX_YEAR)])
    assert p["fs_year"].max() < LOCKBOX_YEAR, "lockbox violated"
    return p


def load_splits():
    with open(os.path.join(ROOT, "data/splits/walkforward.json")) as fh:
        wf = json.load(fh)
    tr = pd.read_parquet(os.path.join(ROOT, "data/splits/walkforward_test_rows.parquet"))
    return wf, tr


def fold_frames(panel, wf, test_rows, k):
    f = [f for f in wf["folds"] if f["fold"] == k][0]
    tr = panel[panel["fs_year"].isin(f["train_years"]) & (panel["label_status"] == "ok")].reset_index(drop=True)
    te_ids = test_rows[test_rows["fold"] == k][["row_id", "firm_seen_in_train"]]
    te = panel.merge(te_ids, on="row_id", how="inner")
    assert len(tr) == f["n_train"] and len(te) == f["n_test"], (len(tr), len(te))
    return f, tr, te


def feature_list(name: str) -> list[str]:
    if name == "full":
        return list(B.ALL_FEATURES)
    if name == "core":
        return list(B.CORE_FEATURES)
    v = getattr(B, name, None)
    if not isinstance(v, (list, tuple)):
        raise ValueError(f"unknown feature set {name!r} in src/features/build.py")
    return list(v)


def monotone_vector(feats, on=True, relax=()):
    s = B.expected_signs([f for f in feats if f in B.META])
    return [int(s.get(f, 0)) if on and f not in relax else 0 for f in feats]


def inner_masks(df):
    years = sorted(df["fs_year"].unique())
    u = G._firm_u(df["id"].to_numpy())
    if len(years) >= 2:
        last = years[-1]
        m_tr = (df["fs_year"] < last).to_numpy()
        m_a, m_b = (~m_tr) & (u < 0.5), (~m_tr) & (u >= 0.5)
        scheme = f"time: inner-train {years[:-1]}, validate {last} (firm halves A / B)"
    else:
        m_tr, m_a, m_b = u < 0.70, (u >= 0.70) & (u < 0.85), u >= 0.85
        scheme = "single training year: firm split 70/15/15 (train / A / B)"
    return m_tr, m_a, m_b, u, scheme


def recency_weights(fs_year, half_life):
    """w = 0.5 ** ((newest_year - fs_year) / half_life); half_life None/inf -> all ones."""
    fs_year = np.asarray(fs_year)
    if half_life is None or not np.isfinite(half_life):
        return np.ones(len(fs_year))
    return 0.5 ** ((fs_year.max() - fs_year) / float(half_life))


# ---------------------------------------------------------------------------------------------- sklearn RF
class SkRF:
    """sklearn RandomForest on B.design_matrix (one-hot categoricals with the training reference level, NaN
    passed through: sklearn >= 1.4 trees route missing values). predict() mimics the Booster call signature
    and returns the vote fraction (mean leaf default frequency), clipped to (EPS, 1-EPS)."""

    def __init__(self, cfg, fp, feats):
        from sklearn.ensemble import RandomForestClassifier
        self.fp, self.feats = fp, feats
        self.m = RandomForestClassifier(n_estimators=cfg.get("n_estimators", 300),
                                        min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
                                        max_samples=cfg.get("max_samples"), n_jobs=THREADS, random_state=G.SEED)

    def fit(self, X, y, w=None):
        self.cols = list(B.design_matrix(X.iloc[:5], self.fp, self.feats).columns)
        self.m.fit(B.design_matrix(X, self.fp, self.feats)[self.cols].to_numpy(np.float32), y, sample_weight=w)
        return self

    def predict(self, X, num_iteration=None, raw_score=False):
        p = self.m.predict_proba(B.design_matrix(X, self.fp, self.feats)[self.cols].to_numpy(np.float32))[:, 1]
        return np.clip(p, G.EPS, 1 - G.EPS)


# ---------------------------------------------------------------------------------------------- fit
def _lgb_cfg(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def train_booster(X, y, cfg, mono, Xv=None, yv=None, n_rounds=None, w=None, rf=False, record=None,
                  max_rounds=G.MAX_ROUNDS, early_stop=G.EARLY_STOP):
    p = {**BASE, **_lgb_cfg(cfg), "monotone_constraints": mono}
    if rf:
        p.update(boosting="rf")
        p.pop("learning_rate", None)
    dtr = lgb.Dataset(X, y, weight=w, categorical_feature="auto", free_raw_data=False)
    cbs = []
    if record is not None:
        cbs.append(lgb.record_evaluation(record))
    if n_rounds is None:
        dva = lgb.Dataset(Xv, yv, reference=dtr)
        if early_stop:
            cbs.append(lgb.early_stopping(early_stop, verbose=False))
        return lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dtr, dva] if record is not None else [dva],
                         valid_names=["train", "valid"] if record is not None else ["valid"], callbacks=cbs)
    return lgb.train(p, dtr, num_boost_round=int(n_rounds))


def fit_general(train_df, feats, mono, cfgs, rf=False, rf_trees=300, log=print, tag=""):
    """cfg keys starting with '_' are not LightGBM params: '_hl' = recency half-life in years (None = no
    weighting). Ties on A (e.g. weights irrelevant with one inner-train year) go to the earliest cfg.
    Select cfg on A log loss (early stopping on A), Platt slope from B, refit on the whole window.
    Returns art (same keys as gbm_lib, usable with G.predict_pd / G.shap_values) with a record."""
    df = train_df.reset_index(drop=True)
    y = df["y"].astype(int).to_numpy()
    m_tr, m_a, m_b, u, scheme = inner_masks(df)
    rec = {"train_years": sorted(int(t) for t in df["fs_year"].unique()), "n_train": int(len(df)),
           "train_defaults": int(y.sum()), "features": feats, "monotone": mono, "inner_scheme": scheme,
           "rf": rf}
    for k, m in (("inner_train", m_tr), ("A", m_a), ("B", m_b)):
        rec[f"n_{k}"], rec[f"defaults_{k}"] = int(m.sum()), int(y[m].sum())
    fp_in = B.fit(df[m_tr])
    Xtr, Xa, Xb = (G._X(df[m], fp_in, feats, False) for m in (m_tr, m_a, m_b))
    trials = []
    for i, cfg in enumerate(cfgs):
        t0 = time.time()
        w_in = recency_weights(df.loc[m_tr, "fs_year"], cfg.get("_hl"))
        if rf == "sklearn":
            bst = SkRF(cfg, fp_in, feats).fit(Xtr, y[m_tr], w_in)
            bi = None
        elif rf:
            bst = train_booster(Xtr, y[m_tr], cfg, mono, n_rounds=rf_trees, w=w_in, rf=True)
            bi = rf_trees
        else:
            bst = train_booster(Xtr, y[m_tr], cfg, mono, Xa, y[m_a], w=w_in)
            bi = int(bst.best_iteration)
        pa = np.clip(bst.predict(Xa, num_iteration=bi), G.EPS, 1 - G.EPS)
        pb = np.clip(bst.predict(Xb, num_iteration=bi), G.EPS, 1 - G.EPS)
        t = dict(i=i, cfg=cfg, best_iter=bi, A_log_loss=M.log_loss(y[m_a], pa), A_auc=M.auc(y[m_a], pa),
                 B_log_loss=M.log_loss(y[m_b], pb), B_auc=M.auc(y[m_b], pb), secs=round(time.time() - t0, 1))
        trials.append({**t, "_bst": bst, "_pb": pb})
        log(f"  [{tag}] #{i} best_iter={bi} A_ll={t['A_log_loss']:.5f} A_auc={t['A_auc']:.4f} "
            f"B_ll={t['B_log_loss']:.5f} B_auc={t['B_auc']:.4f} ({t['secs']}s) cfg={cfg}")
    best = min(trials, key=lambda t: t["A_log_loss"])
    rec["trials"] = [{k: v for k, v in t.items() if not k.startswith("_")} for t in trials]
    rec["chosen_cfg"], rec["best_iter"], rec["chosen_index"] = best["cfg"], best["best_iter"], best["i"]
    pb = best["_pb"]
    rec["calibration_compare_B"] = G._compare_calibrators(pb, y[m_b], u[m_b])
    a_b, b_b = G._platt_fit(G._logit(pb), y[m_b])
    rec["platt_on_B"] = {"a": a_b, "b": b_b}

    fp = B.fit(df)
    Xall = G._X(df, fp, feats, False)
    w_all = recency_weights(df["fs_year"], best["cfg"].get("_hl"))
    n_final = rf_trees if rf else int(round(G.N_FINAL_FACTOR * best["best_iter"]))
    t0 = time.time()
    if rf == "sklearn":
        bst = SkRF(best["cfg"], fp, feats).fit(Xall, y, w_all)
    else:
        bst = train_booster(Xall, y, best["cfg"], mono, n_rounds=n_final, w=w_all, rf=rf)
    rec["n_final_trees"], rec["final_fit_secs"] = n_final, round(time.time() - t0, 1)
    p_in = np.clip(bst.predict(Xall), G.EPS, 1 - G.EPS)
    z_all = G._logit(p_in)
    a_anchor = G._anchor_intercept(z_all, b_b, y.mean())
    cal = {"method": "platt_slope_from_B_intercept_anchored_to_train_rate", "a": a_anchor, "b": b_b,
           "target_rate": float(y.mean())}
    rec["calibrator"] = cal
    p_cal_in = G._sigmoid(a_anchor + b_b * z_all)
    rec["in_sample"] = {"auc": M.auc(y, p_cal_in), "log_loss": M.log_loss(y, p_cal_in)}
    rec["inner_B_predictions"] = pd.DataFrame({"row_id": df.loc[m_b, "row_id"].to_numpy(), "y": y[m_b],
                                               "pd_raw_inner": pb})
    return {"booster": bst, "feature_params": fp, "calibrator": cal, "features": feats, "impute": False,
            "record": rec}


# ---------------------------------------------------------------------------------------------- bootstrap
# Copied from reports/tournament/score_r1.py (referee's fast firm-clustered paired bootstrap; verified there
# against src/common/metrics.paired_bootstrap, r1_bootstrap_equivalence.csv). score_r1.py runs on import,
# so the functions are copied verbatim rather than imported.
class AUCEngine:
    def __init__(self, yv, pv):
        o = np.argsort(pv, kind="mergesort")
        self.o, self.y = o, yv[o]
        ps = pv[o]
        self.starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])

    def __call__(self, w):
        w = w[self.o]
        gp = np.add.reduceat(w * self.y, self.starts)
        gn = np.add.reduceat(w * (1 - self.y), self.starts)
        below = np.cumsum(gn) - gn
        return float((gp * (below + 0.5 * gn)).sum() / (gp.sum() * gn.sum()))


def row_loss(yv, pv):
    pv = np.clip(pv, M.EPS, 1 - M.EPS)
    return -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))


def bootstrap_draws(yv, firmv, preds: dict, subsets: dict, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    firms, inv = np.unique(firmv, return_inverse=True)
    eng = {(e, s): AUCEngine(yv[m], p[m]) for e, p in preds.items() for s, m in subsets.items()}
    L = {e: row_loss(yv, p) for e, p in preds.items()}
    out = {}
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(firms), len(firms)), minlength=len(firms))
        w = cnt[inv].astype(float)
        for s, m in subsets.items():
            ws = w[m]
            if (ws * yv[m]).sum() == 0 or (ws * (1 - yv[m])).sum() == 0:
                continue
            for e in preds:
                out.setdefault(("auc", e, s), []).append(eng[(e, s)](ws))
            for k, l in L.items():
                out.setdefault(("log_loss", k, s), []).append(float((ws * l[m]).sum() / ws.sum()))
    return {k: np.asarray(v) for k, v in out.items()}


def ci(d):
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
