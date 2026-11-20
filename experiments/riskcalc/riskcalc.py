"""RiskCalc-style PD model (challenger-riskcalc, 2026-10-31).

Architecture (R-01 Dwyer, Kocagil & Stein 2004, Sec. 3.1 p. 14, VERIFIED in docs/REFERENCES.md N-01):

    PD = F( Phi( sum_i beta_i * T_i(x_i) + sum_j gamma_j * I_j ) )

* T_i  : univariate non-parametric transform of ratio i. Here: the ratio's training-window
         default rate by quantile bin, shrunk towards the base rate (Beta prior with
         `prior_defaults` pseudo-defaults), made monotone in the expected direction
         (weighted pool-adjacent-violators) where FEATURE_SPEC gives a sign, linearly
         interpolated between bin centres and flat beyond the outer bin centres (this is what
         neutralises outliers). Missing values get their own (shrunk) empirical default rate.
         The transform is stored and fed to the probit on the probit scale, Phi^-1(T_i(x)), so
         every T_i is a PD-scale measure and every beta_i should be positive.
* probit: statsmodels Probit on the transformed ratios + one-hot industry (and optionally
         legal-form / macro-area) indicators.
* F    : final monotone map from probit index to PD, fitted on the training window: index
         quantile bins -> shrunk default rates -> PAV (increasing) -> one knot per PAV block
         (at its weighted mean index) -> linear interpolation on the probit scale, with slope-1
         extrapolation beyond the outer knots (so ranks are never tied off in the tails).

Everything is fitted from the rows passed to `fit_riskcalc` only. `predict_riskcalc` is a pure
row-wise function of (row, params): no statistic of the scoring frame is used.

Usage
-----
    params = fit_riskcalc(train_rows, config)          # train rows: panel rows with label col 'y'
    save_params(params, "riskcalc_params.json")
    pd_hat = predict_riskcalc(new_rows, load_params("riskcalc_params.json"))

Complexity of prediction: O(n * (p * log K + m)) for n rows, p transformed ratios with K knots
each (binary search in np.interp), m indicator columns, plus O(log B) for the F lookup.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.features import build as B  # noqa: E402

VERSION = "riskcalc-v1 (2026-10-31)"
PD_FLOOR, PD_CAP = 1e-5, 0.99

DEFAULT_CONFIG = {
    "ratios": ["log_assets", "equity_ratio", "st_debt_ratio", "cash_ratio", "roa_net",
               "interest_coverage", "interest_to_assets", "ar_days", "tangible_ratio",
               "neg_equity"],
    "categoricals": ["ateco_group"],
    "link": "probit",               # 'probit' or 'logit' (variant check)
    "min_defaults_per_bin": 30,     # K = clip(train_defaults // 30, 10, max_bins)
    "max_bins": 50,
    "prior_defaults": 2.0,          # Beta prior strength (pseudo-defaults) toward base rate
    "monotone": True,               # PAV in the FEATURE_SPEC direction where sign != 0
    "final_bins": 100,              # quantile bins of the probit index for F
    "drop_nonpositive": False,      # in-window selection: drop T_i with coef <= 0, refit (repeat)
}


# ------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------
def _pav(v: np.ndarray, w: np.ndarray, increasing: bool = True):
    """Weighted pool-adjacent-violators. Returns (fitted values per input, block list)."""
    v = np.asarray(v, float) if increasing else -np.asarray(v, float)
    blocks = []  # [sum_wv, sum_w, start, end]
    for i, (vi, wi) in enumerate(zip(v, w)):
        blocks.append([vi * wi, wi, i, i])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][3] = b[3]
    out = np.empty(len(v))
    for b in blocks:
        out[b[2]:b[3] + 1] = b[0] / b[1]
    return (out if increasing else -out), [(b[2], b[3]) for b in blocks]


def _shrunk_rate(d, n, p0, prior_defaults):
    m = prior_defaults / p0  # pseudo-rows
    return (np.asarray(d, float) + prior_defaults) / (np.asarray(n, float) + m)


def _quantile_bins(x: np.ndarray, k: int):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, k + 1)))
    inner = edges[1:-1]
    return inner, np.searchsorted(inner, x, side="right")


def _link_inv(z, link):
    return norm.cdf(z) if link == "probit" else 1 / (1 + np.exp(-z))


def _link(p, link):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return norm.ppf(p) if link == "probit" else np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------
# univariate transforms T_i
# ------------------------------------------------------------------------------------------
def fit_transform_table(x: pd.Series, y: np.ndarray, sign: int, p0: float, k: int,
                        prior_defaults: float, monotone: bool, is_flag: bool) -> dict:
    x = x.to_numpy(dtype=float)
    miss = np.isnan(x)
    tab = {"sign": int(sign), "n_missing": int(miss.sum()), "d_missing": int(y[miss].sum())}
    tab["missing_rate"] = float(_shrunk_rate(y[miss].sum(), miss.sum(), p0, prior_defaults))
    xp, yp = x[~miss], y[~miss]
    if is_flag:
        lv = np.unique(xp)
        n = np.array([(xp == v).sum() for v in lv])
        d = np.array([yp[xp == v].sum() for v in lv])
        knots_x = lv.astype(float)
        bin_n, bin_d = n, d
    else:
        _, b = _quantile_bins(xp, k)
        nb = b.max() + 1
        bin_n = np.bincount(b, minlength=nb)
        bin_d = np.bincount(b, weights=yp, minlength=nb)
        sx = np.bincount(b, weights=xp, minlength=nb)
        keep = bin_n > 0
        bin_n, bin_d, knots_x = bin_n[keep], bin_d[keep], sx[keep] / bin_n[keep]
    raw = _shrunk_rate(bin_d, bin_n, p0, prior_defaults)
    vals = raw
    if monotone and sign != 0 and len(raw) > 1:
        vals, _ = _pav(raw, bin_n, increasing=sign > 0)
    tab.update({"knots_x": [float(v) for v in knots_x], "rate_raw": [float(v) for v in raw],
                "rate": [float(v) for v in vals], "bin_n": [int(v) for v in bin_n],
                "bin_d": [int(v) for v in bin_d]})
    return tab


def apply_transform(x: np.ndarray, tab: dict) -> np.ndarray:
    """PD-scale transform T(x) (a default rate). Interpolates on the probit scale."""
    x = np.asarray(x, float)
    kx = np.asarray(tab["knots_x"])
    kz = norm.ppf(np.asarray(tab["rate"]))
    out = np.full(len(x), tab["missing_rate"])
    ok = ~np.isnan(x)
    if len(kx) == 1:
        out[ok] = tab["rate"][0]
    else:
        out[ok] = norm.cdf(np.interp(x[ok], kx, kz))  # flat beyond the outer knots
    return out


# ------------------------------------------------------------------------------------------
# design, fit, predict
# ------------------------------------------------------------------------------------------
def _features(df: pd.DataFrame, feat_params: dict) -> pd.DataFrame:
    return B.transform(df, feat_params, impute=False)


def design(X: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Probit design: Phi^-1(T_i(x_i)) for each ratio + indicator columns (fixed by params)."""
    cols = {}
    keep = params.get("design_columns")
    for c in params["config"]["ratios"]:
        if keep is not None and f"T_{c}" not in keep:
            continue
        cols[f"T_{c}"] = norm.ppf(apply_transform(X[c].to_numpy(float), params["transforms"][c]))
    Z = pd.DataFrame(cols, index=X.index)
    cats = params["config"]["categoricals"]
    if cats:
        D = B.design_matrix(X, params["features"], cats)
        Z = pd.concat([Z, D], axis=1)
    Z.insert(0, "const", 1.0)
    return Z[params["design_columns"]] if "design_columns" in params else Z


def _fit_glm(y, Z, link):
    import statsmodels.api as sm
    fam = sm.Probit if link == "probit" else sm.Logit
    res = fam(y, Z).fit(method="newton", maxiter=100, disp=False)
    return res


def fit_final_transform(score: np.ndarray, y: np.ndarray, link: str, nbins: int,
                        prior_defaults: float) -> dict:
    p0 = y.mean()
    _, b = _quantile_bins(score, nbins)
    nb = b.max() + 1
    n = np.bincount(b, minlength=nb)
    d = np.bincount(b, weights=y, minlength=nb)
    s = np.bincount(b, weights=score, minlength=nb)
    keep = n > 0
    n, d, s = n[keep], d[keep], s[keep] / n[keep]   # s = mean index per bin
    r = _shrunk_rate(d, n, p0, prior_defaults)
    _, blocks = _pav(r, n, increasing=True)
    kx, kv = [], []
    for a, e in blocks:
        w = n[a:e + 1]
        kx.append(float((s[a:e + 1] * w).sum() / w.sum()))
        kv.append(float((r[a:e + 1] * w).sum() / w.sum()))
    return {"knots_index": kx, "knots_pd": kv, "link": link, "n_bins": int(len(n)),
            "n_blocks": len(kx)}


def apply_final_transform(score: np.ndarray, F: dict) -> np.ndarray:
    kx = np.asarray(F["knots_index"])
    kz = norm.ppf(np.asarray(F["knots_pd"]))
    z = np.interp(score, kx, kz)
    lo, hi = score < kx[0], score > kx[-1]
    # slope-1 extrapolation on the probit scale keeps ranks strict in the tails
    z[lo] = kz[0] + (score[lo] - kx[0])
    z[hi] = kz[-1] + (score[hi] - kx[-1])
    return np.clip(norm.cdf(z), PD_FLOOR, PD_CAP)


def fit_riskcalc(train_df: pd.DataFrame, config: dict | None = None, label_col: str = "y") -> dict:
    """Fit feature params, transforms T_i, probit and final transform F on `train_df` only."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    y = pd.to_numeric(train_df[label_col]).to_numpy(float)
    feat = B.fit(train_df, label_col=label_col)
    X = _features(train_df, feat)
    p0 = float(y.mean())
    k = int(np.clip(int(y.sum()) // cfg["min_defaults_per_bin"], 10, cfg["max_bins"]))
    signs = B.expected_signs()
    transforms = {}
    for c in cfg["ratios"]:
        transforms[c] = fit_transform_table(X[c], y, signs.get(c, 0), p0, k, cfg["prior_defaults"],
                                            cfg["monotone"], c in B.FLAG_FEATURES)
    params = {"version": VERSION, "config": cfg, "features": feat, "transforms": transforms,
              "base_rate": p0, "n_train": int(len(y)), "train_defaults": int(y.sum()),
              "n_bins": k, "dropped": []}
    Z = design(X, params)
    while True:
        res = _fit_glm(y, Z, cfg["link"])
        bad = [c for c in Z.columns if c.startswith("T_") and res.params[c] <= 0]
        if not (cfg["drop_nonpositive"] and bad):
            break
        worst = min(bad, key=lambda c: res.params[c] / res.bse[c])   # most negative z first
        params["dropped"].append({"term": worst, "coef": float(res.params[worst]),
                                  "z": float(res.params[worst] / res.bse[worst])})
        Z = Z.drop(columns=worst)
    params["design_columns"] = list(Z.columns)
    params["coef"] = {c: float(v) for c, v in res.params.items()}
    params["se"] = {c: float(v) for c, v in res.bse.items()}
    params["llf"] = float(res.llf)
    params["converged"] = bool(res.mle_retvals.get("converged", True))
    score = Z.to_numpy() @ res.params.to_numpy()
    params["final"] = fit_final_transform(score, y, cfg["link"], cfg["final_bins"],
                                          cfg["prior_defaults"])
    return params


def index_riskcalc(df: pd.DataFrame, params: dict, X: pd.DataFrame | None = None):
    X = _features(df, params["features"]) if X is None else X
    Z = design(X, params)
    beta = np.array([params["coef"][c] for c in params["design_columns"]])
    return Z, Z.to_numpy() @ beta


def predict_riskcalc(df: pd.DataFrame, params: dict) -> np.ndarray:
    """PD in (0,1) for every row of df, same order. Row-wise; no refitting."""
    _, s = index_riskcalc(df, params)
    s = np.where(np.isfinite(s), s, params["final"]["knots_index"][len(params["final"]["knots_index"]) // 2])
    return apply_final_transform(s, params["final"])


def save_params(params: dict, path) -> None:
    with open(path, "w") as fh:
        json.dump(params, fh, indent=1)


def load_params(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------
# best configuration (RC-03) and a one-call estimator for harness-engineer
# ------------------------------------------------------------------------------------------
BEST_CONFIG = {**DEFAULT_CONFIG,
               "ratios": DEFAULT_CONFIG["ratios"] + ["bank_debt_ratio", "ap_days"],
               "drop_nonpositive": True}


def fit_and_save(train_df: pd.DataFrame, path, config: dict | None = None) -> dict:
    """Fit RC-03 (or `config`) on any training frame (label column 'y', label_status=='ok' rows
    only) and write the frozen params JSON consumed by `predict_riskcalc`."""
    params = fit_riskcalc(train_df, config or BEST_CONFIG)
    save_params(params, path)
    return params
