"""Stage 3 (calibration-specialist): central-tendency rule x calibration method, scored on the
three walk-forward test years for GB-01 (incumbent) and RC-02 (transparent second model).

Leakage rule: every number used to build a calibrated PD for test year t comes from
(i) training-window labels (fs_year <= t-1), (ii) predictions of the fold's own models on training
rows or on the in-window held-out slice B, or (iii) the target level from s2 (training labels only).
Test labels are used only to SCORE.

Methods (all monotone, so AUC must not change except for isotonic ties):
  M0  none          : the submitted PDs (level = training-window average, by construction)
  M1  prior shift   : Saerens et al. (2002) eq. (4): odds x [tau/(1-tau)] / [r_w/(1-r_w)], i.e. a logit
                      shift delta = logit(tau) - logit(r_w), r_w = training-window default rate
  M2  mean match    : logit shift delta such that the mean PD of the final model on the LAST training
                      year's rows equals tau (uses the model's own view of the latest population)
  M3  Platt on B    : y_B ~ a + b*logit(p_inner_B) on slice B (in-window, out-of-time for folds 2-3),
                      applied to the final model's PD; then level moved to tau as in M1
                      (M3raw: Platt as fitted, its level = the B year's rate; no tau)
  M4  isotonic      : PAV of y on the final model's in-sample training PD (whole window), then level
                      moved to tau as in M1 (shape + level)
Rules (targets tau, s2): a_pooled, b_last_year, c_trend_logit, c2_half_trend, d_ttc_equal_weight,
e_last_6m_annualised, f_mean_last2.
Diagnostic only (NOT allowed in the harness, it uses the batch of test-year features): Saerens EM
estimate of the test-year prior from unlabeled test rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import metrics as M  # noqa: E402

CACHE = ROOT / "experiments/calibration/cache"
OUT = ROOT / "reports/calibration"
EPS = M.EPS
N_BOOT, SEED = 200, 0


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def solve_shift(z, target):
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if expit(z + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


splits = json.load(open(ROOT / "data/splits/walkforward.json"))
testrows = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet")
panel = pd.read_parquet(ROOT / "data/interim/panel.parquet", columns=["row_id", "id", "y", "fs_year"])
base = testrows.merge(panel, on="row_id", how="left", validate="one_to_one")
ct = pd.read_csv(OUT / "ct_targets.csv").query("label == 's6'")
RULES = sorted(ct.rule.unique())

oot = {"GB-01": pd.read_parquet(ROOT / "experiments/gbm/GB-01/oot_predictions.parquet"),
       "RC-02": pd.read_parquet(ROOT / "experiments/riskcalc/RC-02/oot_predictions.parquet")}
gb_raw = pd.read_parquet(ROOT / "experiments/gbm/GB-01/oot_predictions_raw.parquet")
gb_B = pd.read_parquet(ROOT / "experiments/gbm/GB-01/inner_B_predictions.parquet")

preds = {}      # (model, rule, method) -> array aligned with base
params = []     # what each calibrated variant used (for the decision log / harness)
for model in ("GB-01", "RC-02"):
    out = {}
    for f in splits["folds"]:
        k, t = f["fold"], f["test_year"]
        m = (base.fold == k).to_numpy()
        rid = base.loc[m, "row_id"]
        p0 = oot[model].query("fold == @k").set_index("row_id").loc[rid, "pd"].to_numpy()
        if model == "GB-01":
            tr = pd.read_parquet(CACHE / f"gb_train_{k}.parquet")
            ptr = tr["pd"].to_numpy()
            Bk = gb_B.query("fold == @k")
            zB, yB = logit(Bk.pd_raw_inner), Bk.y.to_numpy()
            p_final_raw_te = gb_raw.query("fold == @k").set_index("row_id").loc[rid, "pd"].to_numpy()
            p_final_raw_tr = tr["pd_raw"].to_numpy()
        else:
            tr = pd.read_parquet(CACHE / f"rc_train_{k}.parquet")
            ptr = tr["pd"].to_numpy()
            Bk = pd.read_parquet(CACHE / f"rc_innerB_{k}.parquet")
            zB, yB = logit(Bk.pd_inner), Bk.y.to_numpy()
            p_final_raw_te, p_final_raw_tr = p0, ptr
        ytr = tr["y"].to_numpy()
        r_w = ytr.mean()
        last = (tr.fs_year == tr.fs_year.max()).to_numpy()
        # M3: Platt on B (inner model, held out); slope b and intercept a
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(zB.reshape(-1, 1), yB)
        a_B, b_B = float(lr.intercept_[0]), float(lr.coef_[0, 0])
        z3_te = a_B + b_B * logit(p_final_raw_te)
        z3_tr = a_B + b_B * logit(p_final_raw_tr)
        # re-anchor M3 to r_w on training rows, then prior-shift (so the level is set by tau only)
        a3 = solve_shift(z3_tr, r_w)
        # M4: isotonic on in-sample training predictions (whole window)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5).fit(logit(ptr), ytr)
        p4_te = iso.predict(logit(p0))
        p4_tr = iso.predict(logit(ptr))
        a4 = solve_shift(logit(p4_tr), r_w)
        # baseline and raw Platt
        out.setdefault(("none", "M0"), np.empty(len(base)))[m] = p0
        out.setdefault(("B-year", "M3raw"), np.empty(len(base)))[m] = expit(z3_te)
        out.setdefault(("pooled", "M4raw"), np.empty(len(base)))[m] = p4_te
        cts = ct.query("test_year == @t").set_index("rule")["target"]
        for rule in RULES:
            tau = float(cts[rule])
            d1 = logit(tau) - logit(r_w)
            d2 = solve_shift(logit(ptr[last]), tau)
            out.setdefault((rule, "M1"), np.empty(len(base)))[m] = expit(logit(p0) + d1)
            out.setdefault((rule, "M2"), np.empty(len(base)))[m] = expit(logit(p0) + d2)
            out.setdefault((rule, "M3"), np.empty(len(base)))[m] = expit(z3_te + a3 + d1)
            out.setdefault((rule, "M4"), np.empty(len(base)))[m] = expit(logit(p4_te) + a4 + d1)
            params.append({"model": model, "fold": k, "test_year": t, "rule": rule, "tau": tau, "r_w": r_w,
                           "delta_M1": d1, "delta_M2": d2, "platt_B_a": a_B, "platt_B_b": b_B,
                           "M3_anchor": a3, "M4_anchor": a4,
                           "mean_pd_train_last_year": ptr[last].mean(), "rate_last_year": ytr[last].mean()})
    for (rule, meth), v in out.items():
        preds[(model, rule, meth)] = np.clip(v, EPS, 1 - EPS)
pd.DataFrame(params).to_csv(OUT / "method_params.csv", index=False)

# ---------------------------------------------------------------- scoring
y = base["y"].astype(int).to_numpy()
yr = base["test_year"].to_numpy()
firm = base["id"].to_numpy()
rows = []
for (model, rule, meth), p in preds.items():
    for g in ("ALL", 2008, 2009, 2010):
        mm = np.ones(len(y), bool) if g == "ALL" else (yr == g)
        r = {"model": model, "rule": rule, "method": meth, "group": str(g),
             "auc": M.auc(y[mm], p[mm]), "log_loss": M.log_loss(y[mm], p[mm]), "brier": M.brier(y[mm], p[mm])}
        r.update(M.calibration_in_the_large(y[mm], p[mm]))
        r.update(M.calibration_slope_intercept(y[mm], p[mm]))
        rows.append(r)
res = pd.DataFrame(rows)

# ---------------------------------------------------------------- firm-clustered paired bootstrap
# (same estimator/RNG sequence as reports/tournament/score_r1.py: np.unique firms, rng.integers per
#  draw, firm-multiplicity weights; log loss only, the monotone maps leave AUC unchanged)
firms, inv = np.unique(firm, return_inverse=True)
rng = np.random.default_rng(SEED)
W = [np.bincount(rng.integers(0, len(firms), len(firms)), minlength=len(firms)) for _ in range(N_BOOT)]
groups = {"ALL": np.ones(len(y), bool), "2008": yr == 2008, "2009": yr == 2009, "2010": yr == 2010}


def rowloss(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


boot = []
for model in ("GB-01", "RC-02"):
    L0 = rowloss(preds[(model, "none", "M0")])
    for (mdl, rule, meth), p in preds.items():
        if mdl != model or meth == "M0":
            continue
        dl = rowloss(p) - L0
        for g, gm in groups.items():
            sd = np.bincount(inv[gm], weights=dl[gm], minlength=len(firms))
            sn = np.bincount(inv[gm], minlength=len(firms)).astype(float)
            draws = np.array([(w * sd).sum() / (w * sn).sum() for w in W])
            boot.append({"model": model, "rule": rule, "method": meth, "group": g,
                         "d_log_loss": dl[gm].mean(), "ci_low": np.percentile(draws, 2.5),
                         "ci_high": np.percentile(draws, 97.5), "share_draws_better": float((draws < 0).mean())})
boot = pd.DataFrame(boot)
res = res.merge(boot, on=["model", "rule", "method", "group"], how="left")
res.to_csv(OUT / "methods_by_year.csv", index=False)

# ---------------------------------------------------------------- Saerens EM diagnostic (not harness-legal)
em = []
for model in ("GB-01", "RC-02"):
    p0 = preds[(model, "none", "M0")]
    for t in (2008, 2009, 2010):
        mm = yr == t
        prm = pd.read_csv(OUT / "method_params.csv").query("model == @model and test_year == @t").iloc[0]
        r_old = prm.r_w
        q, pt = r_old, p0[mm]
        for _ in range(200):
            w1, w0 = q / r_old, (1 - q) / (1 - r_old)
            post = w1 * pt / (w1 * pt + w0 * (1 - pt))
            q_new = post.mean()
            if abs(q_new - q) < 1e-9:
                break
            q = q_new
        em.append({"model": model, "test_year": t, "train_rate": r_old, "model_mean_pd_on_test": pt.mean(),
                   "em_prior_estimate": q, "observed_rate": y[mm].mean()})
pd.DataFrame(em).to_csv(OUT / "saerens_em_diagnostic.csv", index=False)

# ---------------------------------------------------------------- write a few prediction files (contract)
store = {}
for (model, rule, meth), p in preds.items():
    store[f"{model}|{rule}|{meth}"] = p
np.savez_compressed(CACHE / "s3_preds.npz", row_id=base.row_id.to_numpy(), fold=base.fold.to_numpy(), **store)
print(pd.DataFrame(em).round(5))
show = res[res.group == "ALL"].sort_values(["model", "log_loss"])
print(show[["model", "rule", "method", "auc", "log_loss", "mean_pd", "ratio_obs_to_pred", "cal_intercept",
            "cal_slope", "d_log_loss", "ci_low", "ci_high"]].round(5).to_string())
