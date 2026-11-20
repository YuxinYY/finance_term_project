"""Referee scoring, tournament round 1 (referee, 2026-10-31).

    /Users/kietnguyen/miniconda3/envs/datascience/bin/python reports/tournament/score_r1.py

Reads ONLY: experiments/*/<exp>/oot_predictions.parquet, data/interim/panel.parquet (row_id, id,
fs_year, y, label_status, stmt_date, def_date), data/splits/walkforward*.{json,parquet}, the
challengers' saved parameter files (face validity) and src/common/{metrics,labels}.py.
All metrics come from src/common/metrics.py. The paired bootstrap uses a fast, exactly equivalent
firm-multiplicity-weight implementation (verified against metrics.paired_bootstrap below).

Writes reports/tournament/r1_*.csv and reports/tournament/fig/r1_*.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import metrics as M  # noqa: E402
from src.common.labels import make_labels  # noqa: E402

OUT = ROOT / "reports/tournament"
FIG = OUT / "fig"
FIG.mkdir(parents=True, exist_ok=True)
N_BOOT, SEED = 200, 0           # D-007: 200 firm-clustered draws
INCUMBENT = "SC-03"
EXPS = {
    "SC-01": "scorecard", "SC-02": "scorecard", "SC-03": "scorecard", "SC-04": "scorecard",
    "RC-01": "riskcalc", "RC-02": "riskcalc", "RC-03": "riskcalc",
    "GB-01": "gbm", "GB-02": "gbm", "GB-03": "gbm", "GB-04": "gbm", "GB-05": "gbm", "GB-06": "gbm",
}
KEY_PAIRS = [("RC-03", "RC-01"), ("RC-02", "RC-01"), ("RC-03", "RC-02"), ("GB-01", "RC-01"),
             ("GB-01", "RC-03"), ("GB-02", "GB-01"), ("GB-03", "RC-01"), ("SC-03", "SC-02"),
             ("GB-03", "GB-01"), ("GB-04", "GB-01"), ("GB-05", "GB-01"), ("GB-06", "GB-01")]


def logit(p):
    p = np.clip(np.asarray(p, float), M.EPS, 1 - M.EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------------------------------
# 0. Data and contract checks
# --------------------------------------------------------------------------------------------
splits = json.load(open(ROOT / "data/splits/walkforward.json"))
testrows = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet")
panel = pd.read_parquet(ROOT / "data/interim/panel.parquet",
                        columns=["row_id", "id", "fs_year", "y", "label_status", "stmt_date", "def_date"])
base = testrows.merge(panel[["row_id", "id", "y", "fs_year"]], on="row_id", how="left", validate="one_to_one")
assert base["y"].notna().all() and (base["fs_year"] == base["test_year"]).all()
y = base["y"].astype(int).to_numpy()
firm = base["id"].to_numpy()
year = base["test_year"].to_numpy()
seen = base["firm_seen_in_train"].to_numpy(bool)
fold = base["fold"].to_numpy()

contract, P = [], {}
for e, agent in EXPS.items():
    o = pd.read_parquet(ROOT / f"experiments/{agent}/{e}/oot_predictions.parquet")
    r = {"exp": e, "n_rows": len(o), "expected_rows": len(testrows),
         "dup_row_fold": int(o.duplicated(["row_id", "fold"]).sum()),
         "nan_pd": int(o["pd"].isna().sum()),
         "pd_outside_open_01": int((~o["pd"].between(0, 1, inclusive="neither")).sum()),
         "min_pd": float(o["pd"].min()), "max_pd": float(o["pd"].max())}
    m = testrows[["row_id", "fold"]].merge(o, on=["row_id", "fold"], how="left", indicator=True)
    r["missing_test_rows"] = int((m["_merge"] == "left_only").sum())
    r["extra_rows"] = len(o) - int((m["_merge"] == "both").sum())
    for k in (1, 2, 3):
        r[f"rows_fold{k}"] = int((o["fold"] == k).sum())
    r["PASS"] = (r["n_rows"] == r["expected_rows"] and r["dup_row_fold"] == 0 and r["nan_pd"] == 0
                 and r["pd_outside_open_01"] == 0 and r["missing_test_rows"] == 0 and r["extra_rows"] == 0)
    contract.append(r)
    P[e] = base[["row_id", "fold"]].merge(o, on=["row_id", "fold"], how="left")["pd"].to_numpy(float)
pd.DataFrame(contract).to_csv(OUT / "r1_contract.csv", index=False)
print(pd.DataFrame(contract)[["exp", "n_rows", "missing_test_rows", "nan_pd", "pd_outside_open_01", "PASS"]])

# --------------------------------------------------------------------------------------------
# 1. D-011 check: is the panel's y for fs_year 2007-2010 identical to make_labels' y?
# --------------------------------------------------------------------------------------------
lab = make_labels(panel[["row_id", "fs_year", "stmt_date", "def_date"]], start_months=6,
                  horizon_months=12, data_end="2013-05-31")
d11 = []
for t in range(2007, 2013):
    s = (lab["fs_year"] == t).to_numpy()
    a, b = panel.loc[s], lab.loc[s]
    same_status = (a["label_status"].to_numpy() == b["label_status"].to_numpy())
    both_ok = (a["label_status"] == "ok").to_numpy() & (b["label_status"] == "ok").to_numpy()
    d11.append({"fs_year": t, "rows": int(s.sum()),
                "panel_ok": int((a["label_status"] == "ok").sum()),
                "make_labels_ok": int((b["label_status"] == "ok").sum()),
                "status_mismatch_rows": int((~same_status & ~((a["label_status"] == "after_default").to_numpy())).sum()),
                "y_mismatch_on_common_ok": int((a["y"].to_numpy()[both_ok] != b["y"].to_numpy()[both_ok]).sum()),
                "make_labels_ok_with_incomplete_window": int(((b["label_status"] == "ok") & (b["win_end"] > pd.Timestamp("2013-06-01"))).sum()),
                "panel_defaults_ok": int(a.loc[a["label_status"] == "ok", "y"].sum()),
                "make_labels_defaults_ok": int(b.loc[b["label_status"] == "ok", "y"].sum())})
pd.DataFrame(d11).to_csv(OUT / "r1_d011_label_check.csv", index=False)
print(pd.DataFrame(d11))

# --------------------------------------------------------------------------------------------
# 2. Level adjustments (diagnostic only; the level decision belongs to calibration-specialist)
#    L1 (leakage-free): common logit shift per fold, delta_k = logit(DR of the LAST training year)
#        - logit(DR of the whole training window); training labels only; same delta for every model.
#        Fold 1 has one training year -> delta = 0.
#    L2 (oracle, uses test labels, NOT a model result): each model's own calibration-in-the-large
#        intercept per test year (metrics.calibration_slope_intercept) -> "level-free" log loss.
# --------------------------------------------------------------------------------------------
pok = panel[panel["label_status"] == "ok"]
delta = {}
for f in splits["folds"]:
    tr = pok[pok["fs_year"].isin(f["train_years"])]
    last = tr[tr["fs_year"] == max(f["train_years"])]
    delta[f["fold"]] = float(logit(last["y"].mean()) - logit(tr["y"].mean()))
pd.DataFrame([{"fold": k, "delta_logit": v} for k, v in delta.items()]).to_csv(OUT / "r1_level_shift_L1.csv", index=False)
print("L1 deltas", delta)
P_L1 = {e: sigmoid(logit(p) + np.vectorize(delta.get)(fold)) for e, p in P.items()}
P_L2, oracle_icpt = {}, []
for e, p in P.items():
    q = np.empty_like(p)
    for t in (2008, 2009, 2010):
        m = year == t
        a = M.calibration_slope_intercept(y[m], p[m])["cal_intercept"]
        q[m] = sigmoid(logit(p[m]) + a)
        oracle_icpt.append({"exp": e, "test_year": t, "oracle_intercept": a})
    P_L2[e] = q
pd.DataFrame(oracle_icpt).to_csv(OUT / "r1_level_shift_L2_oracle.csv", index=False)

# --------------------------------------------------------------------------------------------
# 3. Point metrics (src/common/metrics.py): pooled, per test year, seen/new
# --------------------------------------------------------------------------------------------
grp = np.where(seen, "seen", "new")
rows = []
for e, p in P.items():
    s1 = M.score(y, p, year)
    s2 = M.score(y, p, grp).iloc[1:]
    s = pd.concat([s1, s2])
    s["group"] = s["group"].astype(str)
    s.insert(0, "exp", e)
    ll1 = {"ALL": M.log_loss(y, P_L1[e])}
    ll2 = {"ALL": M.log_loss(y, P_L2[e])}
    for g in ("2008", "2009", "2010", "seen", "new"):
        m = (year.astype(str) == g) | (grp == g)
        ll1[g], ll2[g] = M.log_loss(y[m], P_L1[e][m]), M.log_loss(y[m], P_L2[e][m])
    s["log_loss_L1"] = s["group"].map(ll1)
    s["log_loss_L2_oracle"] = s["group"].map(ll2)
    rows.append(s)
met = pd.concat(rows, ignore_index=True)
met.to_csv(OUT / "r1_metrics_long.csv", index=False)

dec = []
for e, p in P.items():
    d = M.decile_table(y, p)
    d.insert(0, "exp", e)
    dec.append(d)
pd.concat(dec).to_csv(OUT / "r1_deciles_pooled.csv", index=False)

# --------------------------------------------------------------------------------------------
# 4. Fast firm-clustered paired bootstrap (equivalent to metrics.paired_bootstrap)
# --------------------------------------------------------------------------------------------
class AUCEngine:
    """Weighted Mann-Whitney AUC (ties = 1/2) with the sort done once. With integer firm
    multiplicity weights this equals roc_auc_score on the concatenated resample."""

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


def bootstrap_draws(yv, firmv, preds: dict, subsets: dict, n_boot=N_BOOT, seed=SEED, extra_ll=None):
    """Return dict[(metric, exp, subset)] -> array of n_boot draws. Uses the same RNG call
    sequence as metrics.paired_bootstrap (np.unique firms; rng.integers(0, F, F) per draw)."""
    rng = np.random.default_rng(seed)
    firms, inv = np.unique(firmv, return_inverse=True)
    eng = {(e, s): AUCEngine(yv[m], p[m]) for e, p in preds.items() for s, m in subsets.items()}
    L = {e: row_loss(yv, p) for e, p in preds.items()}
    if extra_ll:
        for tag, dd in extra_ll.items():
            for e, p in dd.items():
                L[(tag, e)] = row_loss(yv, p)
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
                name = ("log_loss", k, s) if not isinstance(k, tuple) else (f"log_loss_{k[0]}", k[1], s)
                out.setdefault(name, []).append(float((ws * l[m]).sum() / ws.sum()))
    return {k: np.asarray(v) for k, v in out.items()}


# 4a. Equivalence check against the shared function on a firm subsample
rng0 = np.random.default_rng(123)
uf = np.unique(firm)
sub_f = rng0.choice(uf, 15000, replace=False)
ms = np.isin(firm, sub_f)
ref = M.paired_bootstrap(y[ms], P["GB-01"][ms], P["SC-03"][ms], firm[ms], n_boot=40, seed=0)
dr = bootstrap_draws(y[ms], firm[ms], {"GB-01": P["GB-01"][ms], "SC-03": P["SC-03"][ms]},
                     {"ALL": np.ones(ms.sum(), bool)}, n_boot=40, seed=0)
fast = {"d_auc": dr[("auc", "GB-01", "ALL")] - dr[("auc", "SC-03", "ALL")],
        "d_log_loss": dr[("log_loss", "GB-01", "ALL")] - dr[("log_loss", "SC-03", "ALL")]}
eq = []
for _, r in ref.iterrows():
    d = fast[r["metric"]]
    eq.append({"metric": r["metric"], "shared_ci_low": r["ci_low"], "fast_ci_low": np.percentile(d, 2.5),
               "shared_ci_high": r["ci_high"], "fast_ci_high": np.percentile(d, 97.5),
               "shared_share": r["share_favouring_new"],
               "fast_share": float(np.mean(d > 0) if r["metric"] == "d_auc" else np.mean(d < 0)),
               "n_rows": int(ms.sum()), "n_firms": len(sub_f), "n_boot": 40})
eq = pd.DataFrame(eq)
eq["max_abs_diff"] = np.maximum(abs(eq.shared_ci_low - eq.fast_ci_low), abs(eq.shared_ci_high - eq.fast_ci_high))
eq.to_csv(OUT / "r1_bootstrap_equivalence.csv", index=False)
print(eq)
assert (eq["max_abs_diff"] < 1e-9).all(), "fast bootstrap does not match metrics.paired_bootstrap"

# 4b. Full bootstrap: all models, pooled + per test year + seen/new, 200 draws
subsets = {"ALL": np.ones(len(y), bool), "2008": year == 2008, "2009": year == 2009, "2010": year == 2010,
           "seen": seen, "new": ~seen}
draws = bootstrap_draws(y, firm, P, subsets, extra_ll={"L1": P_L1})
np.savez_compressed(OUT / "r1_bootstrap_draws.npz", **{"|".join(k): v for k, v in draws.items()})


def ci(d):
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def pair_row(new, old, s):
    r = {"new": new, "old": old, "subset": s}
    for met_, better in (("auc", 1), ("log_loss", -1), ("log_loss_L1", -1)):
        if met_ == "auc":
            pt = M.auc(y[subsets[s]], P[new][subsets[s]]) - M.auc(y[subsets[s]], P[old][subsets[s]])
        else:
            src = P if met_ == "log_loss" else P_L1
            pt = M.log_loss(y[subsets[s]], src[new][subsets[s]]) - M.log_loss(y[subsets[s]], src[old][subsets[s]])
        d = draws[(met_, new, s)] - draws[(met_, old, s)]
        lo, hi = ci(d)
        r[f"d_{met_}"], r[f"d_{met_}_lo"], r[f"d_{met_}_hi"] = pt, lo, hi
        r[f"d_{met_}_share_fav_new"] = float(np.mean(d * better > 0))
    r["n_draws"] = len(d)
    return r


pairs = [(e, INCUMBENT) for e in EXPS if e != INCUMBENT] + KEY_PAIRS
prow = [pair_row(a, b, s) for a, b in pairs for s in subsets]
pairs_df = pd.DataFrame(prow)
pairs_df.to_csv(OUT / "r1_paired_bootstrap.csv", index=False)

# per-model CIs (marginal) for the leaderboard + seen-new gap CI
lb = []
for e in EXPS:
    r = {"exp": e}
    for s in subsets:
        m = subsets[s]
        r[f"auc_{s}"] = M.auc(y[m], P[e][m])
        r[f"auc_{s}_lo"], r[f"auc_{s}_hi"] = ci(draws[("auc", e, s)])
        r[f"ll_{s}"] = M.log_loss(y[m], P[e][m])
        r[f"ll_{s}_lo"], r[f"ll_{s}_hi"] = ci(draws[("log_loss", e, s)])
    g = draws[("auc", e, "seen")] - draws[("auc", e, "new")]
    r["seen_minus_new_auc"] = r["auc_seen"] - r["auc_new"]
    r["seen_minus_new_lo"], r["seen_minus_new_hi"] = ci(g)
    lb.append(r)
lb = pd.DataFrame(lb)
lb.to_csv(OUT / "r1_leaderboard_ci.csv", index=False)

# --------------------------------------------------------------------------------------------
# 5. Face validity
# --------------------------------------------------------------------------------------------
sys.path.insert(0, str(ROOT / "experiments/gbm"))
from src.features import build as B  # noqa: E402
SIGNS = B.expected_signs()
fv = []
# 5a. scorecard GLMs: coefficient signs per fold (config.json fit log)
for e in ("SC-01", "SC-02", "SC-04"):
    cfg = json.load(open(ROOT / f"experiments/scorecard/{e}/config.json"))
    for fl in cfg["folds"]:
        for name, c in fl["coef"].items():
            if name in SIGNS and SIGNS[name] != 0:
                fv.append({"exp": e, "fold": fl["fold"], "feature": name, "expected": SIGNS[name],
                           "evidence": "logit coef", "value": c, "violates": bool(np.sign(c) != SIGNS[name])})
# SC-03: beta on risk-oriented WoE must be > 0 and WoE monotone in the expected direction
cfg = json.load(open(ROOT / "experiments/scorecard/SC-03/config.json"))
for fl in cfg["folds"]:
    for name, c in fl["coef"].items():
        if name in SIGNS and SIGNS[name] != 0:
            fv.append({"exp": "SC-03", "fold": fl["fold"], "feature": name, "expected": SIGNS[name],
                       "evidence": "beta on WoE (must be > 0)", "value": c, "violates": bool(c <= 0)})
p3 = json.load(open(ROOT / "experiments/scorecard/SC-03/params_fold3.json"))
woe_mono = {}
for name, b in p3["bins"].items():
    if b["type"] == "num" and SIGNS.get(name, 0) != 0:
        w = np.diff(b["woe"])
        woe_mono[name] = bool(np.all(w * SIGNS[name] >= -1e-12))
# 5b. RiskCalc: T_i coefficient > 0 (every fold) and T_i rates monotone in the expected direction
for e in ("RC-01", "RC-02", "RC-03"):
    co = pd.read_csv(ROOT / f"experiments/riskcalc/{e}/coefficients.csv")
    for _, r in co[co.term.str.startswith("T_")].iterrows():
        f = r["term"][2:]
        fv.append({"exp": e, "fold": int(r["fold"]), "feature": f, "expected": SIGNS.get(f, 0),
                   "evidence": "probit coef on Phi^-1(T_i) (must be > 0)", "value": r["coef"],
                   "z": r["z"], "violates": bool(r["coef"] <= 0)})
    for k in (1, 2, 3):
        prm = json.load(open(ROOT / f"experiments/riskcalc/{e}/params_fold{k}.json"))
        for f, tab in prm["transforms"].items():
            s = SIGNS.get(f, 0)
            if s != 0 and len(tab["rate"]) > 1:
                ok = bool(np.all(np.diff(tab["rate"]) * s >= -1e-12))
                woe_mono[f"{e}|fold{k}|{f}"] = ok
# 5c. GBM: partial dependence of the saved fold-3 models (only fold 3 is saved) on a 20k sample of
#     fold-3 TRAINING rows; violation share = movement against the expected sign / total movement.
import gbm_lib as G  # noqa: E402
f3 = [f for f in splits["folds"] if f["fold"] == 3][0]
trs = pd.read_parquet(ROOT / "data/interim/panel.parquet")
trs = trs[trs["fs_year"].isin(f3["train_years"]) & (trs["label_status"] == "ok")].sample(20000, random_state=0)
gb_pd = []
for e in [x for x in EXPS if x.startswith("GB")]:
    art = G.load_model(str(ROOT / f"experiments/gbm/{e}/model_fold3"))
    X = B.transform(trs, art["feature_params"], impute=art["impute"])[art["features"]]
    mono = dict(zip(art["features"], art["record"]["monotone"]))
    for f in art["features"]:
        s = SIGNS.get(f, 0)
        if s == 0:
            continue
        v = X[f].dropna().to_numpy(float)
        grid = np.unique(np.quantile(v, np.linspace(0.01, 0.99, 25))) if f not in B.FLAG_FEATURES else np.array([0.0, 1.0])
        curve = []
        for g in grid:
            Xg = X.copy()
            Xg[f] = g
            curve.append(np.mean(np.log(np.clip(art["booster"].predict(Xg), 1e-9, 1)) - np.log(np.clip(1 - art["booster"].predict(Xg), 1e-9, 1))))
        dv = np.diff(np.asarray(curve)) * s       # positive = moves in the expected direction
        tot = np.abs(dv).sum()
        share = float(np.clip(-dv, 0, None).sum() / tot) if tot > 1e-12 else 0.0
        gb_pd.append({"exp": e, "feature": f, "expected": s, "constrained": int(mono.get(f, 0)) != 0,
                      "pd_logodds_lo": curve[0], "pd_logodds_hi": curve[-1], "range": float(np.ptp(curve)),
                      "violation_share": share})
        fv.append({"exp": e, "fold": 3, "feature": f, "expected": s, "evidence": "PD log-odds, violation share",
                   "value": share, "violates": bool(share > 0.5), "partial": bool(0.1 < share <= 0.5)})
gb_pd = pd.DataFrame(gb_pd)
gb_pd.to_csv(OUT / "r1_face_validity_gbm_pd.csv", index=False)
fv = pd.DataFrame(fv)
fv.to_csv(OUT / "r1_face_validity_long.csv", index=False)
json.dump(woe_mono, open(OUT / "r1_monotone_shape_checks.json", "w"), indent=1)

# tangible_ratio sanity check (training rows of fold 3 only): univariate AUC overall and within ATECO section
tr3 = pd.read_parquet(ROOT / "data/interim/panel.parquet")
tr3 = tr3[tr3["fs_year"].isin(f3["train_years"]) & (tr3["label_status"] == "ok")]
fp3 = B.fit(tr3)
X3 = B.transform(tr3, fp3, impute=False)
tg = []
for sec, g in X3.assign(y=tr3["y"].to_numpy()).groupby("ateco_group", observed=True):
    gg = g.dropna(subset=["tangible_ratio"])
    if gg["y"].sum() >= 30:
        tg.append({"ateco_group": sec, "n": len(gg), "defaults": int(gg["y"].sum()),
                   "median_tangible": float(gg["tangible_ratio"].median()),
                   "auc_tangible_within": M.auc(gg["y"], -gg["tangible_ratio"])})
gg = X3.assign(y=tr3["y"].to_numpy()).dropna(subset=["tangible_ratio"])
tg.append({"ateco_group": "ALL", "n": len(gg), "defaults": int(gg["y"].sum()),
           "median_tangible": float(gg["tangible_ratio"].median()),
           "auc_tangible_within": M.auc(gg["y"], -gg["tangible_ratio"])})
pd.DataFrame(tg).to_csv(OUT / "r1_tangible_ratio_check.csv", index=False)

# --------------------------------------------------------------------------------------------
# 6. Figures
# --------------------------------------------------------------------------------------------
SHOW = ["SC-02", "SC-03", "RC-01", "RC-03", "GB-01", "GB-02"]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
for e in SHOW:
    p = P[e]
    o = np.argsort(-p, kind="mergesort")
    yy = y[o]
    tpr = np.r_[0, np.cumsum(yy) / yy.sum()]
    fpr = np.r_[0, np.cumsum(1 - yy) / (1 - yy).sum()]
    pop = np.r_[0, np.arange(1, len(yy) + 1) / len(yy)]
    a = M.auc(y, p)
    ax[0].plot(fpr, tpr, lw=1.2, label=f"{e} AUC {a:.4f}")
    ax[1].plot(pop, tpr, lw=1.2, label=f"{e} AR {2 * a - 1:.3f}")
ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
br = y.mean()
ax[1].plot([0, br, 1], [0, 1, 1], "k--", lw=0.8, label="perfect")
ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
ax[0].set(title="ROC, pooled 2008-2010", xlabel="false positive rate", ylabel="true positive rate")
ax[1].set(title="CAP, pooled 2008-2010 (AR = 2*AUC - 1)", xlabel="share of firms, riskiest first",
          ylabel="share of defaults captured")
for a_ in ax:
    a_.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(FIG / "r1_roc_cap.png", dpi=130)
plt.close()

fig, axs = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
for a_, e in zip(axs.ravel(), SHOW):
    for t, c in ((2008, "C0"), (2009, "C1"), (2010, "C2")):
        m = year == t
        d = M.decile_table(y[m], P[e][m])
        a_.errorbar(d["mean_pd"], d["observed_dr"],
                    yerr=[d["observed_dr"] - d["dr_ci_low"], d["dr_ci_high"] - d["observed_dr"]],
                    fmt="o-", ms=3, lw=0.8, color=c, label=str(t))
    a_.plot([1e-4, 0.3], [1e-4, 0.3], "k:", lw=0.8)
    a_.set(xscale="log", yscale="log", title=e)
    a_.legend(fontsize=7)
for a_ in axs[1]:
    a_.set_xlabel("mean predicted PD (decile)")
for a_ in axs[:, 0]:
    a_.set_ylabel("observed default rate")
fig.suptitle("Calibration by PD decile and test year (95% Clopper-Pearson bands); points above the diagonal = under-prediction")
plt.tight_layout()
plt.savefig(FIG / "r1_calibration_deciles.png", dpi=130)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for e in EXPS:
    ls = "-" if e in SHOW else ":"
    au = [M.auc(y[year == t], P[e][year == t]) for t in (2008, 2009, 2010)]
    ll = [M.log_loss(y[year == t], P[e][year == t]) for t in (2008, 2009, 2010)]
    ax[0].plot([2008, 2009, 2010], au, ls, marker="o", ms=3, label=e)
    ax[1].plot([2008, 2009, 2010], ll, ls, marker="o", ms=3, label=e)
ax[0].set(title="AUC by test year", xticks=[2008, 2009, 2010])
ax[1].set(title="Log loss by test year (own level)", xticks=[2008, 2009, 2010])
ax[0].legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(FIG / "r1_by_year.png", dpi=130)
plt.close()
print("done")
