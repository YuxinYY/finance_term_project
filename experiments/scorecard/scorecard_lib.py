"""Benchmark models SC-01..SC-04 (challenger-scorecard). All fitting happens on the training frame
passed in. Each fit_* returns a JSON-serialisable params dict; predict_* is a pure function of
(raw frame, params), row by row. Reusable: fit_woe_scorecard(train_df) / save_params for harness."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import src.features.build as B

EPS = 1e-6
NUM_CORE = [f for f in B.CORE_FEATURES if f in B.NUM_FEATURES]
FLAG_CORE = [f for f in B.CORE_FEATURES if f in B.FLAG_FEATURES]
CAT_CORE = [f for f in B.CORE_FEATURES if f in B.CAT_FEATURES]
Z_FEATURES = ["wc_ratio", "equity_ratio", "roa_oper", "asset_turnover"]  # SC-04 Z'-style proxies


def _sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _clip(p):
    return np.clip(p, EPS, 1 - EPS)


def _glm(Z, y):
    Zc = sm.add_constant(Z, has_constant="add")
    res = sm.GLM(np.asarray(y, float), Zc, family=sm.families.Binomial()).fit(maxiter=100)
    return res


def _coef_dict(res, cols):
    names = ["const"] + list(cols)
    return {"names": names, "coef": [float(x) for x in res.params], "se": [float(x) for x in res.bse]}


def _lin_predict(Z, cd):
    b = np.asarray(cd["coef"])
    return _clip(_sig(b[0] + Z[cd["names"][1:]].to_numpy(float) @ b[1:]))


# ---------------- SC-01 / SC-02 / SC-04: logits on the feature layer ----------------
def fit_logit(train_df, kind: str):
    fp = B.fit(train_df)
    X = B.transform(train_df, fp, impute=True)
    y = train_df["y"].to_numpy()
    if kind == "single":
        aucs = {c: roc_auc_score(y, X[c]) for c in NUM_CORE}
        best = max(aucs, key=lambda c: abs(aucs[c] - 0.5))
        feats = [best]
        extra = {"candidate_auc": aucs, "chosen": best}
    elif kind == "core":
        feats, extra = B.CORE_FEATURES, {}
    elif kind == "altman":
        feats, extra = Z_FEATURES, {}
    Z = B.design_matrix(X, fp, feats)
    res = _glm(Z, y)
    return {"model": kind, "feature_params": fp, "features": feats, "glm": _coef_dict(res, Z.columns), **extra}


def predict_logit(df, params):
    fp = params["feature_params"]
    Z = B.design_matrix(B.transform(df, fp, impute=True), fp, params["features"])
    return _lin_predict(Z, params["glm"])


# ---------------- SC-03: WoE scorecard ----------------
N_INIT_BINS = 20      # pre-declared: 20 quantile bins, pooled to monotone by isotonic regression
MIN_BIN_SHARE = 0.02  # a final bin must hold >= 2% of training rows (else merged with neighbour)


def _woe_table(bad, tot, B_tot, G_tot):
    """WoE = ln( (bad_i/B) / (good_i/G) ): positive = riskier than average (so beta > 0)."""
    good = tot - bad
    return np.log(((bad + 0.5) / (B_tot + 0.5)) / ((good + 0.5) / (G_tot + 0.5)))


def _numeric_bins(x, y, sign):
    x, y = np.asarray(x, float), np.asarray(y, float)
    qs = np.unique(np.quantile(x, np.linspace(0, 1, N_INIT_BINS + 1)[1:-1]))
    idx = np.searchsorted(qs, x, side="right")
    k = len(qs) + 1
    n = np.bincount(idx, minlength=k).astype(float)
    d = np.bincount(idx, weights=y, minlength=k)
    keep = n > 0
    rate = np.where(n > 0, d / np.maximum(n, 1), 0)
    if sign != 0:   # monotone WoE in the expected direction: pool-adjacent-violators
        iso = IsotonicRegression(increasing=(sign > 0))
        fit = iso.fit_transform(np.arange(k)[keep], rate[keep], sample_weight=n[keep])
        grp = np.zeros(k); grp[keep] = fit
    else:
        grp = rate
    # merge adjacent initial bins with identical pooled rate -> final bins
    labels = np.zeros(k, int); cur = 0
    for i in range(1, k):
        if not np.isclose(grp[i], grp[i - 1]):
            cur += 1
        labels[i] = cur
    # enforce min share by merging a small final bin into its neighbour with the closest rate
    while True:
        nf = np.bincount(labels, weights=n); df_ = np.bincount(labels, weights=d)
        if len(nf) == 1 or nf.min() / n.sum() >= MIN_BIN_SHARE:
            break
        j = int(nf.argmin()); r = df_ / np.maximum(nf, 1)
        cand = [t for t in (j - 1, j + 1) if 0 <= t < len(nf)]
        t = min(cand, key=lambda t: abs(r[t] - r[j]))
        labels[labels == j] = t
        _, labels = np.unique(labels, return_inverse=True)
    nf = np.bincount(labels, weights=n); df_ = np.bincount(labels, weights=d)
    edges = [float(qs[i - 1]) for i in range(1, k) if labels[i] != labels[i - 1]]
    return edges, nf, df_


def _bin_index(x, edges):
    return np.searchsorted(np.asarray(edges), np.asarray(x, float), side="right")


def fit_woe_scorecard(train_df, pdo=20.0, base_score=600.0, base_odds=50.0):
    """Fit feature params, monotone-WoE bins, logistic regression on WoE; return params dict."""
    fp = B.fit(train_df)
    X = B.transform(train_df, fp, impute=True)
    y = train_df["y"].to_numpy(); Bt = y.sum(); Gt = len(y) - Bt
    bins = {}
    for c in NUM_CORE + FLAG_CORE + CAT_CORE:
        if c in CAT_CORE:
            lv = fp["categories"][c]
            idx = pd.Categorical(X[c].astype(str), categories=lv).codes
            n = np.bincount(idx, minlength=len(lv)).astype(float); d = np.bincount(idx, weights=y, minlength=len(lv))
            bins[c] = {"type": "cat", "levels": lv, "n": n.tolist(), "bad": d.tolist(), "woe": _woe_table(d, n, Bt, Gt).tolist()}
        elif c in FLAG_CORE:
            v = X[c].to_numpy(float).astype(int)
            n = np.bincount(v, minlength=2).astype(float); d = np.bincount(v, weights=y, minlength=2)
            bins[c] = {"type": "flag", "n": n.tolist(), "bad": d.tolist(), "woe": _woe_table(d, n, Bt, Gt).tolist()}
        else:
            edges, n, d = _numeric_bins(X[c], y, B.META[c]["sign"])
            bins[c] = {"type": "num", "edges": edges, "n": n.tolist(), "bad": d.tolist(), "woe": _woe_table(d, n, Bt, Gt).tolist(), "sign": B.META[c]["sign"]}
    W = _woe_matrix(X, bins)
    res = _glm(W, y)
    cd = _coef_dict(res, W.columns)
    k = len(W.columns); factor = pdo / np.log(2); offset = base_score - factor * np.log(base_odds)
    return {"model": "woe", "feature_params": fp, "bins": bins, "glm": cd,
            "scaling": {"pdo": pdo, "base_score": base_score, "base_odds_good_to_bad": base_odds,
                        "factor": factor, "offset": offset, "k": k}}


def _woe_matrix(X, bins):
    cols = {}
    for c, b in bins.items():
        w = np.asarray(b["woe"])
        if b["type"] == "cat":
            idx = pd.Categorical(X[c].astype(str), categories=b["levels"]).codes
            idx = np.where(idx < 0, len(b["levels"]) - 1, idx)  # unseen -> OTHER (last)
        elif b["type"] == "flag":
            idx = np.nan_to_num(X[c].to_numpy(float)).astype(int)
        else:
            idx = _bin_index(np.nan_to_num(X[c].to_numpy(float)), b["edges"])
        cols[c] = w[idx]
    return pd.DataFrame(cols, index=X.index)


def predict_woe(df, params):
    X = B.transform(df, params["feature_params"], impute=True)
    return _lin_predict(_woe_matrix(X, params["bins"]), params["glm"])


def scorecard_points(params):
    """Points table: score = offset - factor*ln(PD/(1-PD)); split intercept equally over features."""
    s = params["scaling"]; cd = params["glm"]; beta = dict(zip(cd["names"][1:], cd["coef"][1:])); b0 = cd["coef"][0]
    rows = []
    for c, b in params["bins"].items():
        for i, w in enumerate(b["woe"]):
            if b["type"] == "num":
                e = b["edges"]
                lab = (f"<= {e[0]:.4g}" if i == 0 else f"> {e[-1]:.4g}" if i == len(e) else f"({e[i-1]:.4g}, {e[i]:.4g}]") if e else "all"
            elif b["type"] == "cat":
                lab = b["levels"][i]
            else:
                lab = f"{c}={i}"
            logit_c = beta[c] * w + b0 / s["k"]
            pts = s["offset"] / s["k"] - s["factor"] * logit_c
            rows.append({"feature": c, "bin": lab, "n": b["n"][i], "defaults": b["bad"][i],
                         "bad_rate": b["bad"][i] / max(b["n"][i], 1), "woe": w, "points": pts})
    return pd.DataFrame(rows)


def save_params(params, path):
    with open(path, "w") as fh:
        json.dump(params, fh, indent=1)


def load_params(path):
    with open(path) as fh:
        return json.load(fh)


FIT = {"SC-01": lambda d: fit_logit(d, "single"), "SC-02": lambda d: fit_logit(d, "core"),
       "SC-03": fit_woe_scorecard, "SC-04": lambda d: fit_logit(d, "altman")}
PRED = {"SC-01": predict_logit, "SC-02": predict_logit, "SC-03": predict_woe, "SC-04": predict_logit}
