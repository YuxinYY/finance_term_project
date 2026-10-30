"""challenger-gbm: monotone-constrained LightGBM PD model with post-hoc calibration.

Reusable estimation API (for the tournament AND for harness-engineer's final refit):

    art = fit_pd_model(train_df, exp="GB-01")      # train_df: panel rows with label 'y', 'fs_year', 'id'
    save_model(art, "path/to/dir")                 # booster.txt + feature_params.json + calibrator.json + meta.json
    art = load_model("path/to/dir")
    pd_ = predict_pd(art, any_df)                  # (n,) PDs in (0,1), same row order, never re-estimates

Inner scheme (all inside the training window passed in; nothing else is read):
  * >= 2 training years: inner-train = all years before the last one; the last training year is split
    by FIRM (hash of id) into half A (early stopping + grid selection) and half B (calibration
    comparison and Platt slope). Time-ordered: inner validation is later than inner training.
  * 1 training year (fold 1 only): random firm split 70/15/15 (train / A / B) and the pre-declared
    DEFAULT config (no grid), because there is no later year inside the window.
  * Final booster: refit on ALL training rows with n_trees = round(1.2 x best_iter) (pre-declared;
    more data supports slightly more trees, D-052).
  * Own calibration (D-053): Platt on the logit, y ~ a + b*logit(p). Slope b is learned on half B from
    the inner model (held out from both fitting and early stopping). The intercept is then re-anchored
    so that the mean calibrated PD on the full training window equals the training default rate
    (level = training-window average; final level-setting belongs to calibration-specialist).
    Raw / Platt / isotonic are compared on B with 2-fold firm cross-fitting and reported.
No class re-weighting or resampling (re-weighting distorts probability levels).
"""
from __future__ import annotations

import json
import os
import sys
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.features import build as B  # noqa: E402
from src.common import metrics as M  # noqa: E402

SEED = 0
EPS = 1e-6
N_FINAL_FACTOR = 1.2
ISOTONIC_MIN_DEFAULTS = 500

# ---- pre-declared hyper-parameters --------------------------------------------------------------
BASE = dict(objective="binary", learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=10.0, min_sum_hessian_in_leaf=1.0, max_bin=255,
            cat_smooth=50.0, cat_l2=10.0, min_data_per_group=500, max_cat_threshold=16,
            monotone_constraints_method="intermediate", monotone_penalty=0.0,
            seed=SEED, deterministic=True, force_col_wise=True, num_threads=8, verbose=-1)
GRID = [dict(num_leaves=nl, max_depth=md, min_data_in_leaf=mdl)
        for (nl, md) in ((4, 2), (8, 3)) for mdl in (500, 2000)]
DEFAULT = dict(num_leaves=8, max_depth=3, min_data_in_leaf=1000)   # fold 1 (one training year)
MAX_ROUNDS, EARLY_STOP = 4000, 150

EXPERIMENTS = {
    "GB-01": dict(features="full", monotone=True, impute=False,
                  desc="monotone LightGBM, full feature set, native NaN, Platt-slope calibration"),
    "GB-02": dict(features="full", monotone=False, impute=False,
                  desc="GB-01 without monotone constraints (face-validity cost diagnostic)"),
    "GB-03": dict(features="core", monotone=True, impute=False,
                  desc="GB-01 on the 13 core features only"),
    "GB-04": dict(features="full", monotone=True, impute=True,
                  desc="GB-01 with the shared build.py imputation instead of native NaN"),
    "GB-05": dict(features="full", monotone=True, impute=False, extra={"monotone_constraints_method": "advanced"},
                  desc="GB-01 with LightGBM's 'advanced' monotone method (still fully monotone, less restrictive splits)"),
    "GB-06": dict(features="full", monotone=True, impute=False,
                  relax=["current_ratio", "tangible_ratio", "st_debt_share"],
                  desc="diagnostic: GB-01 with constraints dropped on the 3 decile-break features (feature-architect warning)"),
}


def feature_list(kind: str) -> list[str]:
    return list(B.CORE_FEATURES) if kind == "core" else list(B.ALL_FEATURES)


def monotone_vector(feats: list[str], on: bool, relax=()) -> list[int]:
    s = B.expected_signs()
    return [int(s.get(f, 0)) if on and f not in relax else 0 for f in feats]


def _firm_u(ids) -> np.ndarray:
    """Deterministic uniform(0,1) per firm id (crc32 hash) -> firm-level splits."""
    return np.array([zlib.crc32(str(int(i)).encode()) / 2**32 for i in ids])


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def _X(df, fparams, feats, impute):
    return B.transform(df, fparams, impute=impute)[feats]


def _train(X, y, Xv, yv, cfg, mono, n_rounds=None, extra=None):
    p = {**BASE, **(extra or {}), **cfg, "monotone_constraints": mono}
    dtr = lgb.Dataset(X, y, categorical_feature="auto", free_raw_data=False)
    if n_rounds is None:
        dva = lgb.Dataset(Xv, yv, reference=dtr)
        bst = lgb.train(p, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    else:
        bst = lgb.train(p, dtr, num_boost_round=int(n_rounds))
    return bst


def _platt_fit(z, y, slope_only_from=None):
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def _anchor_intercept(z, b, target_rate):
    """Find a so that mean(sigmoid(a + b z)) == target_rate (bisection)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _sigmoid(mid + b * z).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _compare_calibrators(p_raw, y, firm_u):
    """2-fold firm cross-fitted log loss on slice B for raw / Platt / isotonic."""
    z = _logit(p_raw)
    half = firm_u < np.median(firm_u)
    out = {"raw": M.log_loss(y, p_raw)}
    pp, pi = np.empty_like(p_raw), np.empty_like(p_raw)
    for fit_m in (half, ~half):
        app = ~fit_m
        a, b = _platt_fit(z[fit_m], y[fit_m])
        pp[app] = _sigmoid(a + b * z[app])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(z[fit_m], y[fit_m])
        pi[app] = np.clip(iso.predict(z[app]), EPS, 1 - EPS)
    out["platt"] = M.log_loss(y, pp)
    out["isotonic"] = M.log_loss(y, pi)
    out["auc_raw"], out["auc_platt"], out["auc_isotonic"] = M.auc(y, p_raw), M.auc(y, pp), M.auc(y, pi)
    return out


def fit_pd_model(train_df: pd.DataFrame, exp: str = "GB-01", fixed_cfg: dict | None = None,
                 log=print) -> dict:
    """Fit features + booster + calibrator on the training frame passed in (and nothing else)."""
    spec = EXPERIMENTS[exp]
    feats = feature_list(spec["features"])
    mono = monotone_vector(feats, spec["monotone"], spec.get("relax", ()))
    df = train_df.reset_index(drop=True)
    y = df["y"].astype(int).to_numpy()
    years = sorted(df["fs_year"].unique())
    u = _firm_u(df["id"].to_numpy())
    rec: dict = {"exp": exp, "train_years": [int(t) for t in years], "n_train": int(len(df)),
                 "train_defaults": int(y.sum()), "features": feats, "monotone": mono}

    if len(years) >= 2:
        last = years[-1]
        m_tr = (df["fs_year"] < last).to_numpy()
        m_a = (~m_tr) & (u < 0.5)
        m_b = (~m_tr) & (u >= 0.5)
        rec["inner_scheme"] = f"time: inner-train {years[:-1]}, last year {last} split by firm into A (early stop/grid) and B (calibration)"
        grid = [fixed_cfg] if fixed_cfg else GRID
    else:
        m_tr, m_a, m_b = u < 0.70, (u >= 0.70) & (u < 0.85), u >= 0.85
        rec["inner_scheme"] = "single training year: random firm split 70/15/15 (train/A/B), pre-declared DEFAULT config"
        grid = [fixed_cfg] if fixed_cfg else [DEFAULT]
    for k, m in (("inner_train", m_tr), ("A_early_stop", m_a), ("B_calibration", m_b)):
        rec[f"n_{k}"], rec[f"defaults_{k}"] = int(m.sum()), int(y[m].sum())

    # inner model: feature params fitted on the inner-train rows only
    fp_in = B.fit(df[m_tr])
    Xtr, Xa, Xb = (_X(df[m], fp_in, feats, spec["impute"]) for m in (m_tr, m_a, m_b))
    trials = []
    for cfg in grid:
        t0 = time.time()
        bst = _train(Xtr, y[m_tr], Xa, y[m_a], cfg, mono, extra=spec.get("extra"))
        pa = bst.predict(Xa, num_iteration=bst.best_iteration)
        trials.append(dict(cfg=cfg, best_iter=int(bst.best_iteration), A_log_loss=M.log_loss(y[m_a], pa),
                           A_auc=M.auc(y[m_a], pa), secs=round(time.time() - t0, 1), _bst=bst))
        log(f"  [{exp}] cfg={cfg} best_iter={bst.best_iteration} A_logloss={trials[-1]['A_log_loss']:.5f} "
            f"A_auc={trials[-1]['A_auc']:.4f} ({trials[-1]['secs']}s)")
    best = min(trials, key=lambda t: t["A_log_loss"])
    rec["grid"] = [{k: v for k, v in t.items() if k != "_bst"} for t in trials]
    rec["chosen_cfg"], rec["best_iter"] = best["cfg"], best["best_iter"]

    # calibration comparison on B (inner model, held out from fit and early stopping)
    pb = best["_bst"].predict(Xb, num_iteration=best["best_iter"])
    cmp_ = _compare_calibrators(pb, y[m_b], u[m_b])
    rec["calibration_compare_B"] = cmp_
    rec["isotonic_eligible"] = bool(y[m_b].sum() >= ISOTONIC_MIN_DEFAULTS)
    a_b, b_b = _platt_fit(_logit(pb), y[m_b])
    rec["platt_on_B"] = {"a": a_b, "b": b_b}

    # final booster on all training rows
    fp = B.fit(df)
    n_final = int(round(N_FINAL_FACTOR * best["best_iter"]))
    Xall = _X(df, fp, feats, spec["impute"])
    t0 = time.time()
    bst = _train(Xall, y, None, None, best["cfg"], mono, n_rounds=n_final, extra=spec.get("extra"))
    rec["n_final_trees"], rec["final_fit_secs"] = n_final, round(time.time() - t0, 1)
    z_all = _logit(bst.predict(Xall))
    a_anchor = _anchor_intercept(z_all, b_b, y.mean())
    cal = {"method": "platt_slope_from_B_intercept_anchored_to_train_rate", "a": a_anchor, "b": b_b,
           "target_rate": float(y.mean())}
    rec["calibrator"] = cal
    rec["inner_B_predictions"] = pd.DataFrame({"row_id": df.loc[m_b, "row_id"].to_numpy(),
                                               "y": y[m_b], "pd_raw_inner": pb})
    return {"booster": bst, "feature_params": fp, "calibrator": cal, "features": feats,
            "impute": spec["impute"], "record": rec}


def predict_pd(art: dict, df: pd.DataFrame, raw: bool = False) -> np.ndarray:
    """Frozen prediction: transform with stored params, booster score, stored calibration map."""
    X = B.transform(df, art["feature_params"], impute=art["impute"])[art["features"]]
    p = art["booster"].predict(X)
    if raw:
        return np.clip(p, EPS, 1 - EPS)
    c = art["calibrator"]
    return np.clip(_sigmoid(c["a"] + c["b"] * _logit(p)), EPS, 1 - EPS)


def shap_values(art: dict, df: pd.DataFrame) -> pd.DataFrame:
    """TreeSHAP contributions on the log-odds scale (LightGBM pred_contrib); last col = bias."""
    X = B.transform(df, art["feature_params"], impute=art["impute"])[art["features"]]
    c = art["booster"].predict(X, pred_contrib=True)
    return pd.DataFrame(c, columns=art["features"] + ["_bias"], index=df.index)


def save_model(art: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    art["booster"].save_model(os.path.join(out_dir, "booster.txt"))
    B.save_params(art["feature_params"], os.path.join(out_dir, "feature_params.json"))
    with open(os.path.join(out_dir, "calibrator.json"), "w") as fh:
        json.dump(art["calibrator"], fh, indent=1)
    meta = {"features": art["features"], "impute": art["impute"],
            "record": {k: v for k, v in art["record"].items() if k != "inner_B_predictions"},
            "lightgbm_version": lgb.__version__, "feature_version": B.VERSION}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)


def load_model(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "meta.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(out_dir, "calibrator.json")) as fh:
        cal = json.load(fh)
    return {"booster": lgb.Booster(model_file=os.path.join(out_dir, "booster.txt")),
            "feature_params": B.load_params(os.path.join(out_dir, "feature_params.json")),
            "calibrator": cal, "features": meta["features"], "impute": meta["impute"], "record": meta["record"]}
