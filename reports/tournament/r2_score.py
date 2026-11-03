"""Referee round 2: contract, metrics, 1,000-draw paired firm bootstrap, segments, level, grades, ledger, figures.

    OMP_NUM_THREADS=1 python reports/tournament/r2_score.py

LOCKBOX: the panel is read with a parquet filter fs_year < 2011 and asserted; prediction rows are asserted to be
fs_year 2008-2010; the only rule-e targets used are those in the challengers' config.json for folds 1-3 (cross-checked
against ct_targets.csv reading only lines with test_year 2008-2010). Metrics: src/common/metrics.py only. Fast bootstrap
functions are exec'd from score_r1.py (without its body) and re-verified against metrics.paired_bootstrap.
Writes reports/tournament/r2_*.csv|json and fig/r2_*.png.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import metrics as M  # noqa: E402

OUT = ROOT / "reports/tournament"
FIG = OUT / "fig"
N_BOOT, SEED = 1000, 0
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def load_defs(path, names, ns):
    tree = ast.parse(open(path).read())
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    assert set(names) <= set(ns), set(names) - set(ns)
    return ns


S = load_defs(OUT / "score_r1.py", {"AUCEngine", "row_loss", "bootstrap_draws", "ci", "logit", "sigmoid"},
              {"np": np, "M": M, "N_BOOT": N_BOOT, "SEED": SEED})
S = load_defs(OUT / "r1b_diagnostics.py", {"assign", "cp_ci", "grade_table"}, {**S, "stats": stats, "pd": pd})
logit, sigmoid, ci, bootstrap_draws, grade_table, assign, cp_ci = (S[k] for k in
    ("logit", "sigmoid", "ci", "bootstrap_draws", "grade_table", "assign", "cp_ci"))

# ---------------------------------------------------------------------------------- data
splits = json.load(open(ROOT / "data/splits/walkforward.json"))
testrows = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet")
panel = pd.read_parquet(ROOT / "data/interim/panel.parquet",
                        columns=["row_id", "id", "fs_year", "y", "label_status", "asst_tot", "ateco_section"],
                        filters=[("fs_year", "<", 2011)])
assert panel["fs_year"].max() <= 2010, "LOCKBOX"
assert set(testrows["test_year"].unique()) == {2008, 2009, 2010}
base = testrows.merge(panel, on="row_id", how="left", validate="one_to_one")
assert base["y"].notna().all() and (base["fs_year"] == base["test_year"]).all()
y = base["y"].astype(int).to_numpy()
firm, year, fold = base["id"].to_numpy(), base["test_year"].to_numpy(), base["fold"].to_numpy()
seen = base["firm_seen_in_train"].to_numpy(bool)
G = "experiments/gbm"
FILES = {"GB-20": f"{G}/GB-20/oot_predictions.parquet", "GB-25": f"{G}/GB-25/oot_predictions.parquet",
         "GB-23": f"{G}/GB-23/oot_predictions.parquet", "GB-08": f"{G}/GB-08/oot_predictions.parquet",
         "GB-08b": f"{G}/GB-08b/oot_predictions.parquet", "GB-09": f"{G}/GB-09/oot_predictions.parquet",
         "GB-10": f"{G}/GB-10/oot_predictions.parquet", "GB-11": f"{G}/GB-11/oot_predictions.parquet",
         "GB-01": f"{G}/GB-01/oot_predictions.parquet", "GB-02": f"{G}/GB-02/oot_predictions.parquet",
         "GAM-01": "experiments/gam/GAM-01/oot_predictions.parquet",
         "CAL-01": "experiments/calibration/CAL-01/oot_predictions.parquet",
         "GB-20+CAL": f"{G}/GB-20/oot_predictions_common_cal.parquet",
         "GB-25+CAL": f"{G}/GB-25/oot_predictions_common_cal.parquet",
         "GAM-01+CAL": "experiments/gam/GAM-01/oot_predictions_calibrated.parquet"}
OWN = ["GB-25", "GB-20", "GB-23", "GB-08", "GB-08b", "GB-09", "GB-10", "GB-11", "GB-01", "GB-02", "GAM-01"]
SHIFT = ["GB-25+CAL", "GB-20+CAL", "CAL-01", "GAM-01+CAL"]

# ---------------------------------------------------------------------------------- 1. contract
contract, P = [], {}
for e, f in FILES.items():
    o = pd.read_parquet(ROOT / f, columns=["row_id", "fold", "pd"])
    r = {"exp": e, "file": f, "sha256_12": hashlib.sha256(open(ROOT / f, "rb").read()).hexdigest()[:12],
         "n_rows": len(o), "expected_rows": len(testrows), "dup_row_fold": int(o.duplicated(["row_id", "fold"]).sum()),
         "nan_pd": int(o["pd"].isna().sum()), "pd_outside_open_01": int((~o["pd"].between(0, 1, inclusive="neither")).sum()),
         "min_pd": float(o["pd"].min()), "max_pd": float(o["pd"].max()),
         "rows_fs_year_ge_2011": int(((o["row_id"] % 10000) >= 2011).sum())}
    m = testrows[["row_id", "fold"]].merge(o, on=["row_id", "fold"], how="left", indicator=True)
    r["missing_test_rows"] = int((m["_merge"] == "left_only").sum())
    r["extra_rows"] = len(o) - int((m["_merge"] == "both").sum())
    r["PASS"] = (r["n_rows"] == r["expected_rows"] and not r["dup_row_fold"] and not r["nan_pd"] and not r["pd_outside_open_01"]
                 and not r["missing_test_rows"] and not r["extra_rows"] and not r["rows_fs_year_ge_2011"])
    contract.append(r)
    P[e] = base[["row_id", "fold"]].merge(o, on=["row_id", "fold"], how="left")["pd"].to_numpy(float)
contract = pd.DataFrame(contract)
contract.to_csv(OUT / "r2_contract.csv", index=False)
log(contract[["exp", "n_rows", "missing_test_rows", "nan_pd", "pd_outside_open_01", "PASS"]].to_string())
assert contract["PASS"].all()

# ---------------------------------------------------------------------------------- shipped level check
cfg = {e: json.load(open(ROOT / f"{G}/{e}/config.json")) for e in ("GB-20", "GB-25")}
delta_rows = []
ct_lines = [ln.strip().split(",") for ln in open(ROOT / "reports/calibration/ct_targets.csv")]
ct_ok = {int(l[1]): float(l[3]) for l in ct_lines[1:] if l[0] == "s6" and l[1] in ("2008", "2009", "2010") and l[2] == "e_last_6m_annualised"}
pok = panel[panel["label_status"] == "ok"]
for f in splits["folds"]:
    k, t = f["fold"], f["test_year"]
    tr = pok[pok["fs_year"].isin(f["train_years"])]
    assert len(tr) == f["n_train"]
    rw = float(tr["y"].mean())
    tau = ct_ok[t]
    dlt = float(logit(tau) - logit(rw))
    for e in ("GB-20", "GB-25"):
        c = [x for x in cfg[e]["folds"] if x["fold"] == k][0]
        m = fold == k
        pre = np.clip(sigmoid(logit(P[e][m]) + dlt), 0.0003, 0.99)
        delta_rows.append({"exp": e, "fold": k, "test_year": t, "r_w_recomputed": rw, "tau_rule_e_ct_targets": tau, "delta_recomputed": dlt,
                           "delta_config": c["common_cal"]["delta"], "max_abs_diff_vs_common_cal_file": float(np.abs(pre - P[e + "+CAL"][m]).max())})
pd.DataFrame(delta_rows).to_csv(OUT / "r2_level_shift_check.csv", index=False)
log(pd.DataFrame(delta_rows).to_string())

# ---------------------------------------------------------------------------------- 2. segments
edges = pd.read_csv(OUT / "r1b_size_quintile_edges.csv").set_index("fold")
ta = pd.to_numeric(base["asst_tot"], errors="coerce").to_numpy(float)
size = np.full(len(base), "missing", dtype=object)
for k in (1, 2, 3):
    m = fold == k
    q = np.searchsorted(edges.loc[k].to_numpy(float), ta[m], side="right") + 1
    lab = np.array([f"Q{i}" for i in q], dtype=object)
    lab[~np.isfinite(ta[m]) | (ta[m] <= 0)] = "missing"
    size[m] = lab
sec = base["ateco_section"].astype("string").fillna("missing").to_numpy(object)
dc = pd.Series(y).groupby(sec).sum()
small = [s for s, v in dc.items() if v < 30 and s != "L"]
sec_lab = np.array([f"pooled({','.join(sorted(small))})" if s in small else s for s in sec], dtype=object)
segs = {}
for t in (2008, 2009, 2010):
    segs[f"year|{t}"] = year == t
for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    segs[f"size|{q}"] = size == q
for s in sorted(set(sec_lab)):
    segs[f"ateco|{s}"] = sec_lab == s
segs["firm|seen"], segs["firm|new"] = seen, ~seen
for t in (2008, 2009, 2010):
    segs[f"firmyear|new|{t}"] = (~seen) & (year == t)
    segs[f"firmyear|seen|{t}"] = seen & (year == t)
    segs[f"sizeyear|Q1|{t}"] = (size == "Q1") & (year == t)
    segs[f"ateco_year|F|{t}"] = (sec_lab == "F") & (year == t)
assert all((y[m].sum() > 0) for m in segs.values())

import os
if not os.environ.get("R2_TAIL"):
    # ---------------------------------------------------------------------------------- 3. metrics
    met = []
    for e in FILES:
        p = P[e]
        for g, m in {"pooled": np.ones(len(y), bool), **segs}.items():
            s = M.score(y[m], p[m]).iloc[0].to_dict()
            s.pop("group")
            met.append({"exp": e, "cell": g, **s})
    met = pd.DataFrame(met)
    met.to_csv(OUT / "r2_metrics_long.csv", index=False)
    log("metrics done")
    dec = []
    for e in FILES:
        for g, m in {"pooled": np.ones(len(y), bool), "2010": year == 2010}.items():
            d = M.decile_table(y[m], P[e][m]).assign(exp=e, cell=g)
            dec.append(d)
    pd.concat(dec).to_csv(OUT / "r2_deciles.csv", index=False)

    # ---------------------------------------------------------------------------------- 4. bootstrap
    rng0 = np.random.default_rng(123)
    sub_f = rng0.choice(np.unique(firm), 15000, replace=False)
    ms = np.isin(firm, sub_f)
    ref = M.paired_bootstrap(y[ms], P["GB-25"][ms], P["GB-01"][ms], firm[ms], n_boot=40, seed=0)
    dr0 = bootstrap_draws(y[ms], firm[ms], {"a": P["GB-25"][ms], "b": P["GB-01"][ms]}, {"ALL": np.ones(ms.sum(), bool)}, n_boot=40, seed=0)
    fast = {"d_auc": dr0[("auc", "a", "ALL")] - dr0[("auc", "b", "ALL")], "d_log_loss": dr0[("log_loss", "a", "ALL")] - dr0[("log_loss", "b", "ALL")]}
    eq = [{"metric": r["metric"], "shared_ci_low": r["ci_low"], "fast_ci_low": np.percentile(fast[r["metric"]], 2.5), "shared_ci_high": r["ci_high"],
           "fast_ci_high": np.percentile(fast[r["metric"]], 97.5)} for _, r in ref.iterrows()]
    eq = pd.DataFrame(eq)
    eq["max_abs_diff"] = np.maximum(abs(eq.shared_ci_low - eq.fast_ci_low), abs(eq.shared_ci_high - eq.fast_ci_high))
    eq.to_csv(OUT / "r2_bootstrap_equivalence.csv", index=False)
    assert (eq.max_abs_diff < 1e-9).all()
    log("equivalence OK", eq.max_abs_diff.max())

    main_subs = {"ALL": np.ones(len(y), bool), "2008": year == 2008, "2009": year == 2009, "2010": year == 2010, "seen": seen, "new": ~seen}


    def load_npz(fs):
        zs = [np.load(OUT / f) for f in fs]
        return {tuple(k.split("|")): np.concatenate([z[k] for z in zs]) for k in zs[0].files}


    draws_prim = load_npz([f"r2_boot_prim{i}.npz" for i in range(3)])      # 3 x 334 = 1,002 draws, seeds 0,1,2
    draws_rest = load_npz(["r2_boot_rest.npz"])                            # 200 draws, seed 3
    PRIM = {k[1] for k in draws_prim}


    def src(a, b):
        return draws_prim if a in PRIM and b in PRIM else draws_rest


    draws = {**draws_rest, **draws_prim}
    log("bootstrap loaded")


    def pair(new, old, s, m, dd):
        da, dl = dd[("auc", new, s)] - dd[("auc", old, s)], dd[("log_loss", new, s)] - dd[("log_loss", old, s)]
        r = {"new": new, "old": old, "subset": s, "n": int(m.sum()), "defaults": int(y[m].sum()),
             "auc_new": M.auc(y[m], P[new][m]), "auc_old": M.auc(y[m], P[old][m]),
             "d_auc": M.auc(y[m], P[new][m]) - M.auc(y[m], P[old][m]), "d_auc_lo": ci(da)[0], "d_auc_hi": ci(da)[1],
             "d_ar": 2 * (M.auc(y[m], P[new][m]) - M.auc(y[m], P[old][m])),
             "d_auc_lo_995": float(np.percentile(da, 0.25)), "d_auc_hi_995": float(np.percentile(da, 99.75)),
             "share_auc_fav_new": float(np.mean(da > 0)),
             "d_ll": M.log_loss(y[m], P[new][m]) - M.log_loss(y[m], P[old][m]), "d_ll_lo": ci(dl)[0], "d_ll_hi": ci(dl)[1],
             "d_ll_lo_995": float(np.percentile(dl, 0.25)), "d_ll_hi_995": float(np.percentile(dl, 99.75)),
             "share_ll_fav_new": float(np.mean(dl < 0)), "n_draws": len(da)}
        return r


    PAIRS = [(a, b) for a in ("GB-25", "GB-20") for b in ("GB-01", "GB-02", "GB-11", "GAM-01")] + [("GB-25", "GB-20")]
    PAIRS += [(a, "GB-01") for a in ("GB-23", "GB-08", "GB-08b", "GB-09", "GB-10", "GB-11", "GB-02")] + [("GB-23", "GB-20"), ("GB-23", "GB-25"), ("GB-08b", "GB-25")]
    PAIRS += [("GB-25+CAL", "CAL-01"), ("GB-20+CAL", "CAL-01"), ("GB-25+CAL", "GB-20+CAL"), ("GB-25+CAL", "GAM-01+CAL")]
    pt = pd.DataFrame([pair(a, b, s, m, src(a, b)) for a, b in PAIRS for s, m in main_subs.items()])
    pt.to_csv(OUT / "r2_paired_bootstrap.csv", index=False)
    lb = []
    for e in FILES:
        for s, m in main_subs.items():
            a, l = draws[("auc", e, s)], draws[("log_loss", e, s)]
            lb.append({"exp": e, "subset": s, "auc": M.auc(y[m], P[e][m]), "auc_lo": ci(a)[0], "auc_hi": ci(a)[1],
                       "ll": M.log_loss(y[m], P[e][m]), "ll_lo": ci(l)[0], "ll_hi": ci(l)[1]})
    pd.DataFrame(lb).to_csv(OUT / "r2_marginal_ci.csv", index=False)
    log(pt[pt.subset.isin(["ALL", "2010"])][["new", "old", "subset", "d_auc", "d_auc_lo", "d_auc_hi", "d_ll", "d_ll_lo", "d_ll_hi"]].round(5).to_string())

    # segment bootstrap: GB-25, GB-20, GB-01 own level, same firm draws (same seed)
    trio = {e: P[e] for e in ("GB-25", "GB-20", "GB-01")}
    ds = bootstrap_draws(y, firm, trio, segs, n_boot=200, seed=SEED)
    log("segment bootstrap done")
    sg = []
    for s, m in segs.items():
        r = {"cell": s, "n": int(m.sum()), "defaults": int(y[m].sum())}
        for e in trio:
            a = ds[("auc", e, s)]
            r[f"ar_{e}"], r[f"ar_{e}_lo"], r[f"ar_{e}_hi"] = M.accuracy_ratio(y[m], P[e][m]), 2 * ci(a)[0] - 1, 2 * ci(a)[1] - 1
        for new, old in (("GB-25", "GB-01"), ("GB-25", "GB-20")):
            da = ds[("auc", new, s)] - ds[("auc", old, s)]
            dl = ds[("log_loss", new, s)] - ds[("log_loss", old, s)]
            tag = f"{new}_vs_{old}"
            r[f"d_ar_{tag}"] = M.accuracy_ratio(y[m], P[new][m]) - M.accuracy_ratio(y[m], P[old][m])
            r[f"d_ar_{tag}_lo"], r[f"d_ar_{tag}_hi"] = 2 * ci(da)[0], 2 * ci(da)[1]
            r[f"d_ll_{tag}"] = M.log_loss(y[m], P[new][m]) - M.log_loss(y[m], P[old][m])
            r[f"d_ll_{tag}_lo"], r[f"d_ll_{tag}_hi"] = ci(dl)
        r["n_draws"] = len(ds[("auc", "GB-25", s)])
        sg.append(r)
    sg = pd.DataFrame(sg)
    sg.to_csv(OUT / "r2_segments.csv", index=False)

else:
    sg = pd.read_csv(OUT / "r2_segments.csv")

# ---------------------------------------------------------------------------------- 5. level and grades
ge = json.load(open(OUT / "r1b_grade_edges.json"))["edges_new"]
lv, gt, mid = [], [], []
for e in SHIFT + ["GB-25", "GB-01"]:
    for t in (2008, 2009, 2010):
        m = year == t
        c = M.calibration_in_the_large(y[m], P[e][m])
        lo, hi = cp_ci(int(y[m].sum()), int(m.sum()))
        lv.append({"exp": e, "year": t, "n": int(m.sum()), "defaults": int(y[m].sum()), "expected_defaults": float(P[e][m].sum()),
                   "excess_defaults": float(y[m].sum() - P[e][m].sum()), "mean_pd": c["mean_pd"], "observed_dr": c["observed_dr"], "dr_lo": lo, "dr_hi": hi,
                   "obs_over_pred": c["ratio_obs_to_pred"], **M.calibration_slope_intercept(y[m], P[e][m])})
    if e in SHIFT:
        for t in (2008, 2009, 2010):
            m = year == t
            g = grade_table(P[e][m], y[m], ge, f"{e}|{t}")
            g["exp"], g["year"] = e, t
            gt.append(g)
            ga = assign(P[e][m], ge)
            for name, sel in (("grades 6-14 (PD 0.5%-10%)", (ga >= 6) & (ga <= 14)), ("grades 1-5 (PD<0.5%)", ga <= 5), ("grades 15-16 (PD>=10%)", ga >= 15)):
                yy, pp = y[m][sel], P[e][m][sel]
                mid.append({"exp": e, "year": t, "band": name, "n": len(yy), "share_firms": len(yy) / m.sum(), "defaults": int(yy.sum()),
                            "expected": float(pp.sum()), "excess": float(yy.sum() - pp.sum()), "obs_over_pred": float(yy.sum() / pp.sum())})
pd.DataFrame(lv).to_csv(OUT / "r2_level_by_year.csv", index=False)
gt = pd.concat(gt, ignore_index=True)
gt.to_csv(OUT / "r2_grade_table.csv", index=False)
pd.DataFrame(mid).to_csv(OUT / "r2_grade_bands.csv", index=False)
# same firms: the firms CAL-01 puts in grades 6-14 in 2010
m10 = year == 2010
sel = (assign(P["CAL-01"][m10], ge) >= 6) & (assign(P["CAL-01"][m10], ge) <= 14)
same = [{"exp": e, "firms": "CAL-01 grades 6-14, 2010", "n": int(sel.sum()), "defaults": int(y[m10][sel].sum()), "expected": float(P[e][m10][sel].sum()),
         "obs_over_pred": float(y[m10][sel].sum() / P[e][m10][sel].sum())} for e in SHIFT]
pd.DataFrame(same).to_csv(OUT / "r2_grade_same_firms.csv", index=False)
# CAP points (diagnostic only, used in no decision)
cap = []
for e in FILES:
    for t in ("pooled", 2008, 2009, 2010):
        m = np.ones(len(y), bool) if t == "pooled" else year == t
        o = np.argsort(-P[e][m], kind="mergesort")
        yy = y[m][o]
        for sh in (0.05, 0.1, 0.2):
            k = int(round(sh * len(yy)))
            cap.append({"exp": e, "cell": t, "riskiest_share": sh, "defaults_captured_share": float(yy[:k].sum() / yy.sum())})
pd.DataFrame(cap).to_csv(OUT / "r2_cap_points.csv", index=False)
log("level done")

# ---------------------------------------------------------------------------------- 6. selection ledger
files_now = len(list((ROOT / "experiments").glob("*/*/oot_predictions*.parquet")))
prev = json.load(open(OUT / "r1b_selection_ledger.json"))
inner = {}
for name, f, cols in (("GB-21 ablation variants (2 inner slices, 2008 and 2009 B)", f"{G}/GB-21/ablation_runs.csv", ["fold_inner", "variant"]),
                      ("GB-24 extras variants (2 inner slices)", f"{G}/GB-24/extras_runs.csv", ["fold_inner", "variant"]),
                      ("GB-24 extras2 variants (2 inner slices)", f"{G}/GB-24/extras2_runs.csv", ["fold_inner", "variant"])):
    inner[name] = int(pd.read_csv(ROOT / f)[cols].drop_duplicates().shape[0])
try:
    inner["feature-architect LightGBM sets scored on 2009 (round2_lgb_ablation.csv rows)"] = int(len(pd.read_csv(ROOT / "reports/features/round2_lgb_ablation.csv")))
except Exception as ex:
    inner["feature-architect ablation"] = str(ex)
new_cfg = ["GB-08", "GB-08b", "GB-09", "GB-10", "GB-11", "GB-11lgbrf", "GB-20", "GB-23", "GB-25"]
json.dump({"r1b_oot_files": prev["oot_prediction_files_covering_2008_2010"], "r1b_cal_configs": prev["calibration_configs_scored_on_2008_2010"],
           "oot_files_now": files_now, "added_oot_files": files_now - prev["oot_prediction_files_covering_2008_2010"],
           "new_configurations_scored_on_2008_2010_in_round_2": new_cfg, "n_new_configurations": len(new_cfg),
           "n_new_common_level_vectors": 3, "looks_total_files_plus_cal": files_now + prev["calibration_configs_scored_on_2008_2010"],
           "full2_configurations_scored_on_2008_2010": ["GB-20", "GB-23", "GB-25"], "inner_looks": inner, "referee_new_configurations_this_round": 0},
          open(OUT / "r2_selection_ledger.json", "w"), indent=1)

# ---------------------------------------------------------------------------------- 7. figures
SHOW = ["GB-25", "GB-20", "GB-01", "GB-02", "GB-11", "GAM-01"]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
for e in SHOW:
    o = np.argsort(-P[e], kind="mergesort")
    yy = y[o]
    tpr = np.r_[0, np.cumsum(yy) / yy.sum()]
    fpr = np.r_[0, np.cumsum(1 - yy) / (1 - yy).sum()]
    pop = np.r_[0, np.arange(1, len(yy) + 1) / len(yy)]
    a = M.auc(y, P[e])
    ax[0].plot(fpr, tpr, lw=1.1, label=f"{e} AUC {a:.4f}")
    ax[1].plot(pop, tpr, lw=1.1, label=f"{e} AR {2 * a - 1:.3f}")
ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
ax[1].plot([0, y.mean(), 1], [0, 1, 1], "k--", lw=0.8, label="perfect")
ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
ax[0].set(title="ROC, pooled 2008-2010", xlabel="false positive rate", ylabel="true positive rate")
ax[1].set(title="CAP, pooled 2008-2010 (AR = 2*AUC - 1)", xlabel="share of firms, riskiest first", ylabel="share of defaults captured")
for a_ in ax:
    a_.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "r2_roc_cap.png", dpi=130); plt.close()

fig, axs = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
for a_, e in zip(axs.ravel(), SHOW):
    for t, c in ((2008, "C0"), (2009, "C1"), (2010, "C2")):
        m = year == t
        d = M.decile_table(y[m], P[e][m])
        a_.errorbar(d["mean_pd"], d["observed_dr"], yerr=[d["observed_dr"] - d["dr_ci_low"], d["dr_ci_high"] - d["observed_dr"]], fmt="o-", ms=3, lw=0.8, color=c, label=str(t))
    a_.plot([1e-4, 0.3], [1e-4, 0.3], "k:", lw=0.8)
    a_.set(xscale="log", yscale="log", title=e)
    a_.legend(fontsize=7)
fig.suptitle("Calibration by PD decile and test year, own level (95% Clopper-Pearson); above the diagonal = under-prediction")
plt.tight_layout(); plt.savefig(FIG / "r2_calibration_deciles.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for e in OWN:
    ls = "-" if e in SHOW else ":"
    ax[0].plot([2008, 2009, 2010], [M.auc(y[year == t], P[e][year == t]) for t in (2008, 2009, 2010)], ls, marker="o", ms=3, label=e)
    ax[1].plot([2008, 2009, 2010], [M.log_loss(y[year == t], P[e][year == t]) for t in (2008, 2009, 2010)], ls, marker="o", ms=3, label=e)
ax[0].set(title="AUC by test year", xticks=[2008, 2009, 2010])
ax[1].set(title="Log loss by test year (own level)", xticks=[2008, 2009, 2010])
ax[0].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(FIG / "r2_by_year.png", dpi=130); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, sharex=True)
for a_, t in zip(axs, (2008, 2009, 2010)):
    for e, c in (("GB-25+CAL", "C0"), ("CAL-01", "C1")):
        g = gt[(gt.exp == e) & (gt.year == t) & (gt.n > 0)]
        a_.errorbar(g["mean_pd"], g["observed_dr"], yerr=[g["observed_dr"] - g["dr_ci_low"], g["dr_ci_high"] - g["observed_dr"]], fmt="o-", ms=3, lw=0.8, color=c, label=e)
    a_.plot([3e-4, 0.5], [3e-4, 0.5], "k:", lw=0.8)
    a_.set(xscale="log", yscale="log", title=f"{t}: 16-grade grid, rule-e level", xlabel="mean PD in grade")
    a_.legend(fontsize=8)
axs[0].set_ylabel("observed default rate")
plt.tight_layout(); plt.savefig(FIG / "r2_grade_level.png", dpi=130); plt.close()

seg_cells = [c for c in segs if c.split("|")[0] in ("size",) or c in ("firm|seen", "firm|new", "ateco|C", "ateco|F", "ateco|G", "ateco|L", "ateco|M")]
fig, ax = plt.subplots(figsize=(11, 4.6))
s2 = sg.set_index("cell").loc[seg_cells]
for i, e in enumerate(("GB-01", "GB-25", "GB-20")):
    xx = np.arange(len(seg_cells)) + (i - 1) * 0.22
    ax.errorbar(xx, s2[f"ar_{e}"], yerr=[s2[f"ar_{e}"] - s2[f"ar_{e}_lo"], s2[f"ar_{e}_hi"] - s2[f"ar_{e}"]], fmt="o", ms=3, lw=0.8, label=e)
ax.set_xticks(np.arange(len(seg_cells))); ax.set_xticklabels([c.split("|")[1] for c in seg_cells]); ax.set_ylabel("AR (95% firm-clustered CI)"); ax.legend(); ax.set_title("AR by segment, pooled 2008-2010")
plt.tight_layout(); plt.savefig(FIG / "r2_segments.png", dpi=130); plt.close()
log("ALL DONE")
