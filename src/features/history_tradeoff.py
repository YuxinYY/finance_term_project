"""Firm-history (growth / lag) trade-off, evidence for D-019 (feature-architect, 2026-10-30).

Question: how much would lagged features from a FROZEN TRAINING-DATA LOOKUP add, and for whom?
Design (inside the fold-3 training window only; fs_year 2010 never read):
  - fit rows   = fs_year 2008 (label ok); lag source = the firm's fs_year 2007 statement
  - eval rows  = fs_year 2009 (label ok); lag source = the firm's fs_year 2008 statement
  - feature params fitted on fit rows only; logit on core set vs core + lag features
  - rows without a prior-year statement (new firms) get lag features = 0 + has_history = 0,
    which is exactly what the harness would do for a firm not in the lookup.
Run: python -m src.features.history_tradeoff  ->  reports/features/history_tradeoff.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.common.metrics import auc, log_loss
from src.features import build as B

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAGS = ["equity_ratio", "roa_net", "st_debt_ratio", "log_assets"]


def lag_feats(rows, prev, params):
    cur = B.transform(rows, params, impute=True)
    pv = B.transform(prev, params, impute=True)
    pv.index = prev["id"].values
    ids = rows["id"].values
    has = pd.Series(np.isin(ids, pv.index), index=rows.index)
    out = pd.DataFrame(index=rows.index)
    for c in LAGS:
        prevv = pd.Series(pv[c].reindex(ids).values, index=rows.index)
        out[f"d_{c}"] = (cur[c] - prevv).where(has, 0.0).fillna(0.0)
    out["has_history"] = has.astype(float)
    return cur, out


def main():
    p = pd.read_parquet(os.path.join(ROOT, "data", "interim", "panel.parquet"))
    p = p[p["fs_year"].isin([2007, 2008, 2009])]
    fit_rows = p[(p.fs_year == 2008) & (p.label_status == "ok")].reset_index(drop=True)
    ev_rows = p[(p.fs_year == 2009) & (p.label_status == "ok")].reset_index(drop=True)
    prev07 = p[p.fs_year == 2007].drop_duplicates("id")
    prev08 = p[p.fs_year == 2008].drop_duplicates("id")
    params = B.fit(fit_rows)
    Xa, La = lag_feats(fit_rows, prev07, params)
    Xb, Lb = lag_feats(ev_rows, prev08, params)
    res = []
    for name, use_lag in [("core", False), ("core + lags", True)]:
        Za = B.design_matrix(Xa, params, B.CORE_FEATURES)
        Zb = B.design_matrix(Xb, params, B.CORE_FEATURES)
        if use_lag:
            Za, Zb = pd.concat([Za, La], axis=1), pd.concat([Zb, Lb], axis=1)
        mu, sd = Za.mean(), Za.std().replace(0, 1)
        lr = LogisticRegression(max_iter=3000).fit((Za - mu) / sd, fit_rows.y)
        pb = lr.predict_proba((Zb - mu) / sd)[:, 1]
        for grp, m in [("all", np.ones(len(ev_rows), bool)), ("seen firm (has 2008 stmt)", Lb.has_history.values == 1),
                       ("new firm (no 2008 stmt)", Lb.has_history.values == 0)]:
            res.append(dict(model=name, group=grp, n=int(m.sum()), defaults=int(ev_rows.y[m].sum()),
                            auc=auc(ev_rows.y[m], pb[m]), logloss=log_loss(ev_rows.y[m], pb[m])))
    out = pd.DataFrame(res)
    out.to_csv(os.path.join(ROOT, "reports", "features", "history_tradeoff.csv"), index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
