"""GB-25 checks (pre-specified question: what does restoring interest_to_sales' + sign cost?).
Reads saved predictions and fold-3 models; fits nothing. Writes experiments/gbm/GB-25/gb25_*.csv.
1 metrics by year; 2 paired firm-clustered bootstrap vs GB-20 and GB-01, 1,000 draws; 3 monotone check;
4 same-year gap; 5 fold-3 SHAP of interest_to_sales and interest_to_assets, GB-20 vs GB-25, on the same 20,000
2010 rows and the same bins; 6 whether the top-decile reversal moved to other features."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_round2b as R  # noqa: E402
import explain_r2b as E  # noqa: E402
import gbm_lib as G  # noqa: E402
from src.common import metrics as M  # noqa: E402
from src.features import build as B  # noqa: E402

ROOT, OUT = R.ROOT, os.path.join(HERE, "GB-25")
panel = pd.read_parquet(os.path.join(ROOT, "data/interim/panel.parquet"), filters=[("fs_year", "<", 2011)])
assert panel["fs_year"].max() <= 2010
base = pd.read_parquet(os.path.join(ROOT, "data/splits/walkforward_test_rows.parquet")).merge(
    panel[["row_id", "id", "y"]], on="row_id", validate="one_to_one")
FILES = {"GB-01": "GB-01/oot_predictions.parquet", "GB-20": "GB-20/oot_predictions.parquet",
         "GB-25": "GB-25/oot_predictions.parquet", "CAL-01": "../calibration/CAL-01/oot_predictions.parquet",
         "GB-20+CAL": "GB-20/oot_predictions_common_cal.parquet", "GB-25+CAL": "GB-25/oot_predictions_common_cal.parquet"}
P = {}
for e, f in FILES.items():
    d = base[["row_id", "fold"]].merge(pd.read_parquet(os.path.join(HERE, f)), on=["row_id", "fold"], how="left")
    assert d["pd"].notna().all()
    P[e] = d["pd"].to_numpy()
assert ((P["GB-25"] > 0) & (P["GB-25"] < 1)).all()
y, yr, firm = base["y"].astype(int).to_numpy(), base["test_year"].to_numpy(), base["id"].to_numpy()
seen = base["firm_seen_in_train"].to_numpy().astype(bool)
sub = {"ALL": np.ones(len(y), bool), "2008": yr == 2008, "2009": yr == 2009, "2010": yr == 2010, "seen": seen, "new": ~seen}
for t in (2008, 2009, 2010):
    sub[f"{t} new"] = (yr == t) & ~seen

# 1 metrics
rows = []
for e in P:
    for s, m in sub.items():
        r = M.score(y[m], P[e][m]).iloc[0].to_dict()
        r.update(model=e, subset=s)
        rows.append(r)
pd.DataFrame(rows).drop(columns="group").to_csv(os.path.join(OUT, "gb25_metrics.csv"), index=False)

# 2 bootstrap
S = R.referee_funcs()
dr = S["bootstrap_draws"](y, firm, P, sub, n_boot=1000, seed=0)
rows = []
for a, b in (("GB-25", "GB-20"), ("GB-25", "GB-01"), ("GB-25+CAL", "GB-20+CAL"), ("GB-25+CAL", "CAL-01")):
    for s, m in sub.items():
        da, dl = dr[("auc", a, s)] - dr[("auc", b, s)], dr[("log_loss", a, s)] - dr[("log_loss", b, s)]
        rows.append({"new": a, "old": b, "subset": s, "d_auc": M.auc(y[m], P[a][m]) - M.auc(y[m], P[b][m]),
                     "d_auc_lo": S["ci"](da)[0], "d_auc_hi": S["ci"](da)[1],
                     "d_ll": M.log_loss(y[m], P[a][m]) - M.log_loss(y[m], P[b][m]),
                     "d_ll_lo": S["ci"](dl)[0], "d_ll_hi": S["ci"](dl)[1], "n_draws": len(da)})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "gb25_bootstrap_1000.csv"), index=False)
print(pd.DataFrame(rows).round(5).to_string())

# 4 same-year gap (referee definition; out-of-fold encodings as the primary view)
ins = pd.read_parquet(os.path.join(OUT, "in_sample_predictions.parquet"))
ov = pd.read_csv(os.path.join(HERE, "comparison_r2b_overfit.csv"))
ov = ov[ov.model.isin(["GB-01", "GB-20"]) & ov.view.isin(["referee r1b (calibration cache)", "pd_oof_enc", "pd_fulltable_enc"])]
new = []
for t, k in ((2008, 2), (2008, 3), (2009, 3)):
    c = ins[(ins.fold == k) & (ins.fs_year == t)]
    for v in ("pd_oof_enc", "pd_fulltable_enc"):
        new.append({"model": "GB-25", "year": t, "in_sample_fold": k, "view": v, "n_in": len(c),
                    "auc_in_sample": M.auc(c.y, c[v]), "auc_oot": M.auc(y[yr == t], P["GB-25"][yr == t])})
gap = pd.concat([ov, pd.DataFrame(new)], ignore_index=True)
gap["gap_auc"] = gap.auc_in_sample - gap.auc_oot
gap.to_csv(os.path.join(OUT, "gb25_same_year_gap.csv"), index=False)
print(gap.round(4).to_string())

# fold-3 models on the same rows as GB-22
te = panel[(panel["fs_year"] == 2010) & (panel["label_status"] == "ok")].reset_index(drop=True)
smp = te.iloc[np.random.default_rng(0).choice(len(te), E.N_SHAP, replace=False)].reset_index(drop=True)
A = {e: E.load_art(e) for e in ("GB-20", "GB-25")}
X = {e: E.X_of(A[e], smp) for e in A}
pd.testing.assert_frame_equal(X["GB-20"], X["GB-25"])     # identical features: only the sign differs
X0 = X["GB-20"]
SV = {e: pd.DataFrame(A[e]["booster"].predict(X0, pred_contrib=True), columns=A[e]["features"] + ["_bias"]) for e in A}
RAW = {e: A[e]["booster"].predict(X0) for e in A}
feats = A["GB-25"]["features"]

# 3 monotone check: every constrained feature, sign taken from the model's own vector
signs = dict(zip(feats, A["GB-25"]["record"]["monotone"]))
assert signs["interest_to_sales"] == 1
assert all(signs[f] == A["GB-20"]["record"]["monotone"][i] for i, f in enumerate(feats) if f != "interest_to_sales")
bst = A["GB-25"]["booster"]
spl = dict(zip(bst.feature_name(), bst.feature_importance("split")))
mono = []
Xp, Xi = X0.iloc[:E.N_PD], X0.iloc[:E.N_ICE]
for f, s in signs.items():
    if s == 0:
        continue
    v = X0[f].dropna()
    grid = np.unique(np.r_[v.min(), np.quantile(v, E.GRID_Q), v.max()])
    curve = E.pdp(bst, Xp, f, grid)
    d = np.diff(curve)
    viol = float(np.abs(d[(d * s) < -1e-9]).sum() / max(np.abs(d).sum(), 1e-12)) if len(d) else 0.0
    ice = np.stack([bst.predict(Xi.assign(**{f: g}), raw_score=True) for g in grid], axis=1)
    bad = (np.diff(ice, axis=1) * s) < -1e-9
    mono.append({"feature": f, "sign": s, "n_splits": int(spl[f]), "grid_points": len(grid), "pd_start": curve[0],
                 "pd_end": curve[-1], "pd_violation_share": viol, "ice_rows": E.N_ICE,
                 "ice_rows_violating": int(bad.any(axis=1).sum())})
mono = pd.DataFrame(mono)
mono.to_csv(os.path.join(OUT, "gb25_monotone.csv"), index=False)
print("constrained", len(mono), "violating", int(((mono.pd_violation_share > 1e-9) | (mono.ice_rows_violating > 0)).sum()),
      "unused", list(mono[mono.n_splits == 0].feature))

# 5 SHAP by decile, same bins as GB-22/dependence_bins.csv
yy = smp["y"].astype(int).to_numpy()


def bins(f):
    x = X0[f]
    q = pd.qcut(x.rank(method="first"), 10, labels=False)
    return pd.Series(np.where(x.isna(), "NaN", np.where(x == 0, "exact 0", "D" + (q + 1).astype("Int64").astype(str))), index=x.index)


order = ["exact 0"] + [f"D{i}" for i in range(1, 11)] + ["NaN"]
tabs = []
for f in ("interest_to_sales", "interest_to_assets"):
    b = bins(f)
    t = pd.DataFrame({"feature": f, "bin": b, "x": X0[f], "y": yy, "shap_GB20": SV["GB-20"][f], "shap_GB25": SV["GB-25"][f],
                      "pd_raw_GB20": RAW["GB-20"], "pd_raw_GB25": RAW["GB-25"]}).groupby(["feature", "bin"]).agg(
        n=("y", "size"), x_median=("x", "median"), dr_2010=("y", "mean"), shap_GB20=("shap_GB20", "mean"), shap_GB25=("shap_GB25", "mean"),
        pd_raw_GB20=("pd_raw_GB20", "mean"), pd_raw_GB25=("pd_raw_GB25", "mean")).reset_index()
    t["o"] = t["bin"].map({k: i for i, k in enumerate(order)})
    tabs.append(t.sort_values("o").drop(columns="o"))
pd.concat(tabs).to_csv(os.path.join(OUT, "gb25_shap_deciles.csv"), index=False)
print(pd.concat(tabs).round(4).to_string())
gb01 = E.load_art("GB-01")
sv01 = pd.DataFrame(gb01["booster"].predict(E.X_of(gb01, smp), pred_contrib=True), columns=gb01["features"] + ["_bias"])
imp = []
for e, sv in (("GB-01", sv01), ("GB-20", SV["GB-20"]), ("GB-25", SV["GB-25"])):
    ab = sv.drop(columns="_bias").abs().mean().sort_values(ascending=False)
    a = {"GB-01": gb01, "GB-20": A["GB-20"], "GB-25": A["GB-25"]}[e]
    Xe = E.X_of(a, smp)
    for f in ("interest_to_sales", "interest_to_assets", "interest_coverage", "interest_coverage_sect_pct", "asset_turnover"):
        if f not in ab.index:
            continue
        g = np.unique(np.quantile(Xe[f].dropna(), E.GRID_Q))
        c = E.pdp(a["booster"], Xe.iloc[:E.N_PD], f, g)
        imp.append({"model": e, "feature": f, "mean_abs_shap": ab[f], "rank": list(ab.index).index(f) + 1,
                    "pd_start": c[0], "pd_end": c[-1], "pd_range": np.ptp(c), "n_splits": int(dict(zip(a["booster"].feature_name(), a["booster"].feature_importance("split")))[f])})
pd.DataFrame(imp).to_csv(os.path.join(OUT, "gb25_interest_family.csv"), index=False)
print(pd.DataFrame(imp).round(4).to_string())

# 6 did the top-decile effect just move? Change in mean SHAP in interest_to_sales D10 and D9
b = bins("interest_to_sales")
rows = []
for lab in ("D9", "D10"):
    m = (b == lab).to_numpy()
    d = (SV["GB-25"][feats].loc[m].mean() - SV["GB-20"][feats].loc[m].mean())
    top = d.reindex(d.abs().sort_values(ascending=False).index).head(8)
    for f, v in top.items():
        rows.append({"bin": lab, "n": int(m.sum()), "feature": f, "shap_GB20": SV["GB-20"][f][m].mean(), "shap_GB25": SV["GB-25"][f][m].mean(), "change": v})
    rows.append({"bin": lab, "n": int(m.sum()), "feature": "TOTAL raw log-odds", "shap_GB20": SV["GB-20"].loc[m].sum(axis=1).mean(),
                 "shap_GB25": SV["GB-25"].loc[m].sum(axis=1).mean(), "change": SV["GB-25"].loc[m].sum(axis=1).mean() - SV["GB-20"].loc[m].sum(axis=1).mean()})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "gb25_top_decile_moves.csv"), index=False)
print(pd.DataFrame(rows).round(4).to_string())
