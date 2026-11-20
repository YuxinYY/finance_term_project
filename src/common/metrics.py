"""Shared evaluation metrics for the tournament. Frozen: only the referee may propose changes.

Power:       auc, accuracy_ratio (AR = 2*AUC - 1; same information as AUC)
Probability: log_loss, brier, calibration_in_the_large, calibration_slope_intercept,
             decile_table (diagnostic, with binomial bands)
Comparison:  paired_bootstrap  (resamples FIRMS, not rows, because firms repeat across years)
Summary:     score  (all of the above for one prediction vector, optionally per group)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def accuracy_ratio(y, p) -> float:
    a = auc(y, p)
    return 2 * a - 1 if not np.isnan(a) else np.nan


def log_loss(y, p) -> float:
    y, p = np.asarray(y, float), _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def calibration_in_the_large(y, p) -> dict:
    """Mean predicted PD vs observed default rate, with an exact binomial test p-value."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    n, k = len(y), int(y.sum())
    exp_rate = float(p.mean())
    pval = float(stats.binomtest(k, n, min(max(exp_rate, EPS), 1 - EPS)).pvalue) if n else np.nan
    return {"mean_pd": exp_rate, "observed_dr": k / n if n else np.nan,
            "ratio_obs_to_pred": (k / n) / exp_rate if n and exp_rate > 0 else np.nan,
            "binom_pvalue": pval}


def calibration_slope_intercept(y, p) -> dict:
    """Logistic recalibration y ~ a + b*logit(p). Perfect calibration: a=0, b=1.
    b < 1 -> predictions too extreme (overfit); b > 1 -> too timid."""
    y = np.asarray(y, int)
    if y.sum() == 0 or y.sum() == len(y):
        return {"cal_intercept": np.nan, "cal_slope": np.nan}
    lp = np.log(_clip(p) / (1 - _clip(p))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(lp, y)
    slope = float(m.coef_[0, 0])
    # intercept-only recalibration with slope fixed at 1 (calibration-in-the-large on logit scale)
    off = lp.ravel()
    a = 0.0
    for _ in range(50):  # Newton steps on intercept with offset
        mu = 1 / (1 + np.exp(-(a + off)))
        g, h = np.sum(y - mu), np.sum(mu * (1 - mu))
        if h <= 0:
            break
        step = g / h
        a += step
        if abs(step) < 1e-10:
            break
    return {"cal_intercept": float(a), "cal_slope": slope}


def decile_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Rows sorted into PD quantile bins; observed vs expected defaults with 95% binomial
    (Clopper-Pearson) interval for the observed rate. Diagnostic only."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(n=("y", "size"), defaults=("y", "sum"),
                              mean_pd=("p", "mean"), min_pd=("p", "min"), max_pd=("p", "max"))
    g["observed_dr"] = g["defaults"] / g["n"]
    lo, hi = [], []
    for k, n in zip(g["defaults"].astype(int), g["n"]):
        ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        lo.append(ci.low)
        hi.append(ci.high)
    g["dr_ci_low"], g["dr_ci_high"] = lo, hi
    g["pd_inside_ci"] = (g["mean_pd"] >= g["dr_ci_low"]) & (g["mean_pd"] <= g["dr_ci_high"])
    return g.reset_index()


def score(y, p, group=None) -> pd.DataFrame:
    """All headline metrics, pooled and (optionally) per group (e.g. test year, seen/new firm)."""
    def one(yy, pp):
        r = {"n": len(yy), "defaults": int(np.sum(yy)), "auc": auc(yy, pp),
             "ar": accuracy_ratio(yy, pp), "log_loss": log_loss(yy, pp), "brier": brier(yy, pp)}
        r.update(calibration_in_the_large(yy, pp))
        r.update(calibration_slope_intercept(yy, pp))
        return r

    y, p = np.asarray(y), np.asarray(p)
    rows = [{"group": "ALL", **one(y, p)}]
    if group is not None:
        group = np.asarray(group)
        for g in pd.unique(group):
            m = group == g
            rows.append({"group": g, **one(y[m], p[m])})
    return pd.DataFrame(rows)


def paired_bootstrap(y, p_new, p_old, cluster, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Firm-clustered paired bootstrap of (new - old) for AUC and log loss.

    Resamples whole firms with replacement so repeated firm-years stay together.
    Positive dAUC favours new; NEGATIVE dlog_loss favours new.
    Returns point estimate, 2.5% and 97.5% percentiles, and share of draws favouring new.
    """
    rng = np.random.default_rng(seed)
    y, p_new, p_old, cluster = map(np.asarray, (y, p_new, p_old, cluster))
    firms, inv = np.unique(cluster, return_inverse=True)
    idx_by_firm = [np.flatnonzero(inv == i) for i in range(len(firms))]
    point = {"d_auc": auc(y, p_new) - auc(y, p_old),
             "d_log_loss": log_loss(y, p_new) - log_loss(y, p_old)}
    draws = {"d_auc": [], "d_log_loss": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(firms), len(firms))
        ix = np.concatenate([idx_by_firm[i] for i in pick])
        yy = y[ix]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        draws["d_auc"].append(auc(yy, p_new[ix]) - auc(yy, p_old[ix]))
        draws["d_log_loss"].append(log_loss(yy, p_new[ix]) - log_loss(yy, p_old[ix]))
    rows = []
    for k, better in (("d_auc", np.greater), ("d_log_loss", np.less)):
        d = np.asarray(draws[k])
        rows.append({"metric": k, "point": point[k], "ci_low": np.percentile(d, 2.5),
                     "ci_high": np.percentile(d, 97.5), "share_favouring_new": float(np.mean(better(d, 0))),
                     "n_valid_draws": len(d)})
    return pd.DataFrame(rows)
