"""challenger-gbm, round 2 phase B: the round-2 feature set `full2` (45 features, D-090..D-097).

Own driver (phase A owns run_experiment.py / r2_lib.py / compare*.py); gbm_lib.py is imported, never edited.
GB-01's estimation recipe (gbm_lib.fit_pd_model, D-052/D-053) is reproduced step by step here; the only
differences are the feature set and the out-of-fold target encodings for TRAINING rows:

    params = B.fit(train, feature_set="full2")                       # inside every fit window
    X_train = B.transform(train, params, feature_set="full2", oof=True)
    X_score = B.transform(rows,  params, feature_set="full2")        # A/B slices, test rows: oof=False

usage (python = /Users/kietnguyen/miniconda3/envs/datascience/bin/python):
    run_round2b.py outer GB-20                     # GB-01 grid/scheme on full2, outer folds scored once
    run_round2b.py outer GB-23 --fixed-from GB-09  # full2 + GB-09's tuned cfg per fold
    run_round2b.py ablation                        # GB-21: drop one group F1..F5, inner splits only
    run_round2b.py tuneinner                       # GB-23 choice, inner only, run BEFORE its outer folds
    run_round2b.py extras                          # GB-24: single drops, unconstrained twin, GB-08 relax; inner only
    run_round2b.py encpersist                      # diagnostic: do encodings predict next year's rates?
    run_round2b.py compare                         # bootstrap vs GB-01, tables, overfit gap
    explain_r2b.py GB-20 GB-01                     # GB-22 face validity + SHAP (separate script)

LOCKBOX: the panel is read with a parquet filter fs_year < 2011 and asserted.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gbm_lib as G  # noqa: E402  (helpers only: _train, _platt_fit, _anchor_intercept, ...)
from src.features import build as B  # noqa: E402
from src.common import metrics as M  # noqa: E402

ROOT = G.ROOT
THREADS = 4
G.BASE["num_threads"] = THREADS          # in-process only; gbm_lib.py on disk is unchanged
FS = "full2"
FEATS = list(B.FULL2_FEATURES)
SIGNS = B.expected_signs(FEATS)
GROUPS = {
    "F1_riskcalc_france": ["operating_margin", "interest_to_sales", "payables_to_sales", "personnel_to_sales"],
    "F2_sector_percentiles": [f"{b}_sect_pct" for b in B.PCT_BASE],
    "F3_target_encodings": ["ateco_div_te", "province_te", "region_te"],
    "F4_data_quality": ["n_core_imputed", "bs_gap"],
    "F5_unused_fields": ["nonbank_fin_debt", "ap_lt_ratio", "fin_fixed_ratio", "tfr_ratio"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(B.NEW_FEATURES_V2)
EXPS = {
    "GB-20": "GB-01 settings (grid, inner scheme, Platt-slope calibration, constraints from expected_signs; "
             "sign 0 unconstrained) on feature set full2 (45 features)",
    "GB-23": "GB-20 with GB-09's tuned cfg per fold (chosen by phase A on v1 features, inner slice A)",
    "GB-25": "GB-20 with interest_to_sales constrained to +1 (R-33 sign; pre-specified fix of D-125); all else identical",
}


def mono_vec(feats):
    return [int(SIGNS.get(f, 0)) for f in feats]


def load():
    panel = pd.read_parquet(os.path.join(ROOT, "data/interim/panel.parquet"), filters=[("fs_year", "<", 2011)])
    assert panel["fs_year"].max() <= 2010, "LOCKBOX violated"
    with open(os.path.join(ROOT, "data/splits/walkforward.json")) as fh:
        wf = json.load(fh)
    tr = pd.read_parquet(os.path.join(ROOT, "data/splits/walkforward_test_rows.parquet"))
    return panel, wf, tr


def fold_frames(panel, wf, test_rows, f):
    tr = panel[panel["fs_year"].isin(f["train_years"]) & (panel["label_status"] == "ok")].reset_index(drop=True)
    te_ids = test_rows[test_rows["fold"] == f["fold"]][["row_id", "firm_seen_in_train"]]
    te = panel.merge(te_ids, on="row_id", how="inner")
    assert len(tr) == f["n_train"] and len(te) == f["n_test"]
    return tr, te


def inner_masks(df):
    """Exactly gbm_lib.fit_pd_model's inner scheme (D-052)."""
    years = sorted(df["fs_year"].unique())
    u = G._firm_u(df["id"].to_numpy())
    if len(years) >= 2:
        last = years[-1]
        m_tr = (df["fs_year"] < last).to_numpy()
        return m_tr, (~m_tr) & (u < 0.5), (~m_tr) & (u >= 0.5), u, True
    return u < 0.70, (u >= 0.70) & (u < 0.85), u >= 0.85, u, False


def fit_full2(df, grid_or_fixed=None, feats=FEATS, log=print, tag="GB-20"):
    """GB-01 recipe on full2. grid_or_fixed: None -> GB-01 GRID (>=2 yrs) / DEFAULT (1 yr); dict -> fixed cfg."""
    mono = mono_vec(feats)
    y = df["y"].astype(int).to_numpy()
    m_tr, m_a, m_b, u, multi = inner_masks(df)
    if grid_or_fixed is not None:
        grid = [grid_or_fixed]
    else:
        grid = G.GRID if multi else [G.DEFAULT]
    rec = {"exp": tag, "train_years": sorted(int(t) for t in df["fs_year"].unique()), "n_train": int(len(df)),
           "train_defaults": int(y.sum()), "features": feats, "monotone": mono,
           "inner_scheme": "as gbm_lib (D-052)" + ("" if multi else ": single year, firm split 70/15/15")}
    for k, m in (("inner_train", m_tr), ("A_early_stop", m_a), ("B_calibration", m_b)):
        rec[f"n_{k}"], rec[f"defaults_{k}"] = int(m.sum()), int(y[m].sum())
    t0 = time.time()
    fp_in = B.fit(df[m_tr], feature_set=FS)
    Xtr = B.transform(df[m_tr], fp_in, feature_set=FS, oof=True)[feats]
    Xa = B.transform(df[m_a], fp_in, feature_set=FS)[feats]
    Xb = B.transform(df[m_b], fp_in, feature_set=FS)[feats]
    log(f"  [{tag}] inner features fitted ({time.time() - t0:.0f}s)")
    trials = []
    for cfg in grid:
        t0 = time.time()
        bst = G._train(Xtr, y[m_tr], Xa, y[m_a], cfg, mono)
        pa = bst.predict(Xa, num_iteration=bst.best_iteration)
        pb = bst.predict(Xb, num_iteration=bst.best_iteration)
        trials.append(dict(cfg=cfg, best_iter=int(bst.best_iteration), A_log_loss=M.log_loss(y[m_a], pa),
                           A_auc=M.auc(y[m_a], pa), B_log_loss=M.log_loss(y[m_b], pb), B_auc=M.auc(y[m_b], pb),
                           secs=round(time.time() - t0, 1), _bst=bst, _pb=pb))
        log(f"  [{tag}] cfg={cfg} best_iter={bst.best_iteration} A_ll={trials[-1]['A_log_loss']:.5f} "
            f"A_auc={trials[-1]['A_auc']:.4f} B_ll={trials[-1]['B_log_loss']:.5f} ({trials[-1]['secs']}s)")
    best = min(trials, key=lambda t: t["A_log_loss"])
    rec["grid"] = [{k: v for k, v in t.items() if not k.startswith("_")} for t in trials]
    rec["chosen_cfg"], rec["best_iter"] = best["cfg"], best["best_iter"]
    pb = best["_pb"]
    rec["calibration_compare_B"] = G._compare_calibrators(pb, y[m_b], u[m_b])
    rec["isotonic_eligible"] = bool(y[m_b].sum() >= G.ISOTONIC_MIN_DEFAULTS)
    a_b, b_b = G._platt_fit(G._logit(pb), y[m_b])
    rec["platt_on_B"] = {"a": a_b, "b": b_b}
    # final model on the whole window
    fp = B.fit(df, feature_set=FS)
    Xall = B.transform(df, fp, feature_set=FS, oof=True)[feats]
    n_final = int(round(G.N_FINAL_FACTOR * best["best_iter"]))
    t0 = time.time()
    bst = G._train(Xall, y, None, None, best["cfg"], mono, n_rounds=n_final)
    rec["n_final_trees"], rec["final_fit_secs"] = n_final, round(time.time() - t0, 1)
    # level anchor on the matrix the booster was trained on (OOF encodings): no own-label leakage
    p_in = bst.predict(Xall)
    z_all = G._logit(p_in)
    a_anchor = G._anchor_intercept(z_all, b_b, y.mean())
    cal = {"method": "platt_slope_from_B_intercept_anchored_to_train_rate (anchor on OOF-encoded train matrix)",
           "a": a_anchor, "b": b_b, "target_rate": float(y.mean())}
    rec["calibrator"] = cal
    # in-sample PDs (for the same-year overfit check): OOF view (what the booster saw) and full-table view
    p_in_cal = np.clip(G._sigmoid(a_anchor + b_b * z_all), G.EPS, 1 - G.EPS)
    Xall_full = B.transform(df, fp, feature_set=FS)[feats]
    p_in_full = np.clip(G._sigmoid(a_anchor + b_b * G._logit(bst.predict(Xall_full))), G.EPS, 1 - G.EPS)
    in_sample = pd.DataFrame({"row_id": df["row_id"].to_numpy(), "fs_year": df["fs_year"].to_numpy(), "y": y,
                              "pd_oof_enc": p_in_cal, "pd_fulltable_enc": p_in_full})
    inner_b = pd.DataFrame({"row_id": df.loc[m_b, "row_id"].to_numpy(), "y": y[m_b], "pd_raw_inner": pb})
    return {"booster": bst, "feature_params": fp, "calibrator": cal, "features": feats, "impute": False,
            "record": rec, "in_sample": in_sample, "inner_B": inner_b}


def predict_full2(art, df, raw=False):
    X = B.transform(df, art["feature_params"], feature_set=FS)[art["features"]]
    p = art["booster"].predict(X)
    if raw:
        return np.clip(p, G.EPS, 1 - G.EPS)
    c = art["calibrator"]
    return np.clip(G._sigmoid(c["a"] + c["b"] * G._logit(p)), G.EPS, 1 - G.EPS)


def common_calibration(pd_own, fold_test_year, train_rate):
    """CAL-01's common level method (D-071/D-072/D-079): prior shift to rule e, floor 0.03%, cap 99%."""
    ct = pd.read_csv(os.path.join(ROOT, "reports/calibration/ct_targets.csv"))
    tau = float(ct[(ct.label == "s6") & (ct.test_year == fold_test_year) & (ct.rule == "e_last_6m_annualised")]
                ["target"].iloc[0])
    delta = G._logit(tau) - G._logit(train_rate)
    return np.clip(G._sigmoid(G._logit(pd_own) + delta), 0.0003, 0.99), tau, float(delta)


# ================================================================================================ outer
def run_outer(exp, fixed_from=None):
    if exp == "GB-25":                       # the only difference from GB-20; in-process, build.py is unchanged
        SIGNS["interest_to_sales"] = 1
    out = os.path.join(HERE, exp)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "train_log.md"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    fixed = {}
    if fixed_from:
        with open(os.path.join(HERE, fixed_from, "config.json")) as fh:
            fixed = {f["fold"]: f["chosen_cfg"] for f in json.load(fh)["folds"]}
    panel, wf, test_rows = load()
    log(f"# {exp} training log\n\n{EXPS[exp]}\n\nfeature_set={FS} ({len(FEATS)} features), threads={THREADS}\n"
        f"BASE: {json.dumps(G.BASE)}\nGRID (folds 2-3): {G.GRID}\nDEFAULT (fold 1): {G.DEFAULT}\n"
        f"monotone: {dict(zip(FEATS, mono_vec(FEATS)))}\n"
        + (f"fixed cfg per fold from {fixed_from}: {fixed}\n" if fixed else ""))
    preds, raws, commons, inner, ins, recs = [], [], [], [], [], []
    for f in wf["folds"]:
        k = f["fold"]
        t0 = time.time()
        tr, te = fold_frames(panel, wf, test_rows, f)
        log(f"\n## Fold {k}: train {f['train_years']} ({len(tr):,} rows, {int(tr['y'].sum()):,} def) -> test {f['test_year']}")
        art = fit_full2(tr, fixed.get(k), log=log, tag=exp)
        rec = art["record"]
        tp = time.time()
        p = predict_full2(art, te)
        t_pred = time.time() - tp
        pr = predict_full2(art, te, raw=True)
        pc, tau, delta = common_calibration(p, f["test_year"], rec["calibrator"]["target_rate"])
        assert np.all((p > 0) & (p < 1)) and len(p) == len(te) and np.isfinite(p).all()
        yv = te["y"].astype(int).to_numpy()
        base = {"row_id": te["row_id"].to_numpy(), "fold": k}
        preds.append(pd.DataFrame({**base, "pd": p, "y": yv, "test_year": f["test_year"],
                                   "seen": te["firm_seen_in_train"].to_numpy()}))
        raws.append(pd.DataFrame({**base, "pd": pr}))
        commons.append(pd.DataFrame({**base, "pd": pc}))
        ib = art["inner_B"]
        ib["fold"] = k
        inner.append(ib)
        si = art["in_sample"]
        si["fold"] = k
        ins.append(si)
        rec.update(fold=k, test_year=f["test_year"], common_cal={"tau_rule_e": tau, "delta": delta},
                   predict_secs_per_100k=round(t_pred / len(te) * 1e5, 2), wall_secs=round(time.time() - t0, 1),
                   oot_auc=M.auc(yv, p), oot_log_loss=M.log_loss(yv, p))
        recs.append(rec)
        c = rec["calibration_compare_B"]
        log(f"  chosen {rec['chosen_cfg']} best_iter={rec['best_iter']} final_trees={rec['n_final_trees']}\n"
            f"  B ({rec['defaults_B_calibration']} def; isotonic eligible={rec['isotonic_eligible']}): cross-fit "
            f"log loss raw {c['raw']:.5f} platt {c['platt']:.5f} isotonic {c['isotonic']:.5f}; "
            f"Platt b={rec['platt_on_B']['b']:.3f}; anchored a={rec['calibrator']['a']:.3f}\n"
            f"  test {f['test_year']}: AUC {rec['oot_auc']:.4f} log loss {rec['oot_log_loss']:.5f} "
            f"(raw AUC {M.auc(yv, pr):.4f}; common-cal log loss {M.log_loss(yv, pc):.5f}); "
            f"{rec['predict_secs_per_100k']}s/100k; wall {rec['wall_secs']}s")
        if k == 3:
            G.save_model(art, os.path.join(out, "model_fold3"))
            with open(os.path.join(HERE, f".{exp}_fold3_art.pkl"), "wb") as fh:
                pickle.dump({kk: vv for kk, vv in art.items() if kk not in ("booster",)}
                            | {"booster_str": art["booster"].model_to_string()}, fh)
    oot = pd.concat(preds, ignore_index=True)
    oot[["row_id", "fold", "pd"]].to_parquet(os.path.join(out, "oot_predictions.parquet"), index=False)
    pd.concat(raws).to_parquet(os.path.join(out, "oot_predictions_raw.parquet"), index=False)
    pd.concat(commons).to_parquet(os.path.join(out, "oot_predictions_common_cal.parquet"), index=False)
    pd.concat(inner).to_parquet(os.path.join(out, "inner_B_predictions.parquet"), index=False)
    pd.concat(ins).to_parquet(os.path.join(out, "in_sample_predictions.parquet"), index=False)
    cfg = {"exp": exp, "desc": EXPS[exp], "feature_set": FS, "features": FEATS,
           "monotone": dict(zip(FEATS, mono_vec(FEATS))), "base_params": G.BASE, "grid": G.GRID,
           "default_cfg_fold1": G.DEFAULT, "max_rounds": G.MAX_ROUNDS, "early_stopping_rounds": G.EARLY_STOP,
           "n_final_factor": G.N_FINAL_FACTOR, "fixed_cfg_from": fixed_from,
           "class_weighting": "none", "target_encoding": "oof=True on training rows, oof=False on A/B/test",
           "lockbox": "fs_year >= 2011 never read", "folds": recs}
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1, default=str)
    y, p, yr = oot["y"].to_numpy(), oot["pd"].to_numpy(), oot["test_year"].to_numpy()
    s = M.score(y, p, yr)
    s.to_csv(os.path.join(out, "metrics_by_year.csv"), index=False)
    log("\n## Pooled / per year (own calibration)\n" + s.to_markdown(index=False, floatfmt=".5g"))
    logf.close()


# ================================================================================================ GB-21
def run_ablation():
    """Drop one group at a time from full2; inner splits of folds 2 and 3 only (no outer test year)."""
    panel, wf, test_rows = load()
    out = os.path.join(HERE, "GB-21")
    os.makedirs(out, exist_ok=True)
    gb01 = json.load(open(os.path.join(HERE, "GB-01", "config.json")))
    cfg01 = {f["fold"]: f["chosen_cfg"] for f in gb01["folds"]}
    variants = {"full2": FEATS, "v1 (GB-01 set)": list(B.ALL_FEATURES)}
    for g, cols in GROUPS.items():
        variants[f"drop {g}"] = [c for c in FEATS if c not in cols]
    rows, preds = [], {}
    for f in wf["folds"][1:]:
        k = f["fold"]
        tr, _ = fold_frames(panel, wf, test_rows, f)
        m_tr, m_a, m_b, u, _ = inner_masks(tr)
        y = tr["y"].astype(int).to_numpy()
        fp = B.fit(tr[m_tr], feature_set=FS)
        Xtr = B.transform(tr[m_tr], fp, feature_set=FS, oof=True)
        Xa = B.transform(tr[m_a], fp, feature_set=FS)
        Xb = B.transform(tr[m_b], fp, feature_set=FS)
        for vname, feats in variants.items():
            pbs = []
            for seed in (0, 1, 2):
                t0 = time.time()
                bst = G._train(Xtr[feats], y[m_tr], Xa[feats], y[m_a], cfg01[k], mono_vec(feats),
                               extra={"seed": seed})
                pb = bst.predict(Xb[feats], num_iteration=bst.best_iteration)
                pa = bst.predict(Xa[feats], num_iteration=bst.best_iteration)
                pbs.append(pb)
                rows.append({"fold_inner": k, "variant": vname, "seed": seed, "n_features": len(feats),
                             "best_iter": bst.best_iteration, "A_auc": M.auc(y[m_a], pa),
                             "A_log_loss": M.log_loss(y[m_a], pa), "B_auc": M.auc(y[m_b], pb),
                             "B_log_loss": M.log_loss(y[m_b], pb), "secs": round(time.time() - t0, 1)})
                print(rows[-1], flush=True)
            preds[(k, vname)] = np.mean(pbs, axis=0)
        preds[(k, "_y")] = y[m_b]
        preds[(k, "_firm")] = tr.loc[m_b, "id"].to_numpy()
        pd.DataFrame(rows).to_csv(os.path.join(out, "ablation_runs.csv"), index=False)
    with open(os.path.join(out, "ablation_B_preds.pkl"), "wb") as fh:
        pickle.dump(preds, fh)
    # paired firm-clustered bootstrap on B (seed-averaged predictions), d = variant - full2
    S = referee_funcs()
    summ = []
    for k in (2, 3):
        yb, fb = preds[(k, "_y")], preds[(k, "_firm")]
        P = {v: preds[(k, v)] for v in variants}
        dr = S["bootstrap_draws"](yb, fb, P, {"B": np.ones(len(yb), bool)}, n_boot=200, seed=0)
        for v in variants:
            r = {"fold_inner": k, "variant": v, "n_features": len(variants[v]), "B_auc_seedavg": M.auc(yb, P[v]),
                 "B_log_loss_seedavg": M.log_loss(yb, P[v])}
            if v != "full2":
                da = dr[("auc", v, "B")] - dr[("auc", "full2", "B")]
                dl = dr[("log_loss", v, "B")] - dr[("log_loss", "full2", "B")]
                r.update(d_auc_vs_full2=r["B_auc_seedavg"] - M.auc(yb, P["full2"]),
                         d_auc_lo=S["ci"](da)[0], d_auc_hi=S["ci"](da)[1],
                         d_ll_vs_full2=r["B_log_loss_seedavg"] - M.log_loss(yb, P["full2"]),
                         d_ll_lo=S["ci"](dl)[0], d_ll_hi=S["ci"](dl)[1])
            summ.append(r)
    runs = pd.DataFrame(rows)
    sd = runs.groupby(["fold_inner", "variant"]).agg(B_auc_mean=("B_auc", "mean"), B_auc_sd=("B_auc", "std"),
                                                     B_ll_mean=("B_log_loss", "mean"),
                                                     best_iter_mean=("best_iter", "mean")).reset_index()
    summ = pd.DataFrame(summ).merge(sd, on=["fold_inner", "variant"])
    summ.to_csv(os.path.join(out, "ablation_summary.csv"), index=False)
    print(summ.to_string())


# ================================================================================================ GB-23 choice
def run_tuneinner():
    """Inner-only, run BEFORE GB-23's outer folds: GB-09's tuned cfg vs GB-20's grid choice on full2.
    Same slices as GB-20. Selection = log loss on A (D-052); B = untouched check. Writes GB-23/inner_choice.csv."""
    panel, wf, test_rows = load()
    out = os.path.join(HERE, "GB-23")
    os.makedirs(out, exist_ok=True)
    c20 = {f["fold"]: f for f in json.load(open(os.path.join(HERE, "GB-20", "config.json")))["folds"]}
    c09 = {f["fold"]: f for f in json.load(open(os.path.join(HERE, "GB-09", "config.json")))["folds"]}
    v1 = pd.read_csv(os.path.join(HERE, "GB-09", "selection_check.csv")).set_index("fold")
    mono = mono_vec(FEATS)
    rows = []
    for f in wf["folds"]:
        k = f["fold"]
        tr, _ = fold_frames(panel, wf, test_rows, f)
        m_tr, m_a, m_b, _, _ = inner_masks(tr)
        y = tr["y"].astype(int).to_numpy()
        fp = B.fit(tr[m_tr], feature_set=FS)
        Xtr = B.transform(tr[m_tr], fp, feature_set=FS, oof=True)[FEATS]
        Xa = B.transform(tr[m_a], fp, feature_set=FS)[FEATS]
        Xb = B.transform(tr[m_b], fp, feature_set=FS)[FEATS]
        cfg = c09[k]["chosen_cfg"]
        bst = G._train(Xtr, y[m_tr], Xa, y[m_a], cfg, mono)
        pa = bst.predict(Xa, num_iteration=bst.best_iteration)
        pb = bst.predict(Xb, num_iteration=bst.best_iteration)
        g = min(c20[k]["grid"], key=lambda t: t["A_log_loss"])
        ra, rb = M.log_loss(y[m_a], pa), M.log_loss(y[m_b], pb)
        rows.append({"fold": k, "GB09_cfg": json.dumps(cfg), "GB20_cfg": json.dumps(g["cfg"]),
                     "A_ll_GB20cfg": g["A_log_loss"], "A_ll_GB09cfg": ra, "A_gain_full2": g["A_log_loss"] - ra,
                     "A_gain_v1(GB-09 selection_check)": v1.loc[k, "A_gain_vs_GB01"],
                     "B_ll_GB20cfg": g["B_log_loss"], "B_ll_GB09cfg": rb, "B_gain_full2": g["B_log_loss"] - rb,
                     "B_gain_v1(GB-09 selection_check)": v1.loc[k, "B_gain_vs_GB01"],
                     "A_auc_GB20cfg": g["A_auc"], "A_auc_GB09cfg": M.auc(y[m_a], pa),
                     "B_auc_GB20cfg": g["B_auc"], "B_auc_GB09cfg": M.auc(y[m_b], pb),
                     "best_iter_GB20cfg": g["best_iter"], "best_iter_GB09cfg": int(bst.best_iteration),
                     "inner_choice_A": "GB-09 cfg" if ra < g["A_log_loss"] else "GB-20 cfg"})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "inner_choice.csv"), index=False)


# ================================================================================================ extras (inner only)
def run_extras():
    """Inner splits of folds 2-3 only; no outer year. Fixed GB-01 cfg per fold, 3 seeds, metrics on B.
    (a) single-feature drops: tfr_ratio and the encodings. (b) unconstrained twins of v1 and full2: is the
    full2 gain new information, or does it just buy back the cost of the constraints?
    (c) full2 and v1 with GB-08's three relaxed constraints: do the two gains add up?"""
    panel, wf, test_rows = load()
    out = os.path.join(HERE, "GB-24")
    os.makedirs(out, exist_ok=True)
    cfg01 = {f["fold"]: f["chosen_cfg"] for f in json.load(open(os.path.join(HERE, "GB-01", "config.json")))["folds"]}
    relax = json.load(open(os.path.join(HERE, "GB-08", "config.json")))["relax"]
    V1 = list(B.ALL_FEATURES)
    zero = lambda fs: [0] * len(fs)
    rel = lambda fs: [0 if f in relax else int(SIGNS.get(f, 0)) for f in fs]
    variants = {"full2": (FEATS, mono_vec(FEATS)), "v1": (V1, mono_vec(V1)),
                "full2 unconstrained": (FEATS, zero(FEATS)), "v1 unconstrained": (V1, zero(V1)),
                "full2 + GB-08 relax": (FEATS, rel(FEATS)), "v1 + GB-08 relax": (V1, rel(V1))}
    for d in ("tfr_ratio", "ateco_div_te", "province_te", "region_te"):
        fs = [c for c in FEATS if c != d]
        variants[f"drop {d}"] = (fs, mono_vec(fs))
    fs = [c for c in FEATS if c not in ("province_te", "region_te")]
    variants["drop province_te+region_te"] = (fs, mono_vec(fs))
    fs = [c for c in FEATS if c not in ("tfr_ratio", "personnel_to_sales")]
    variants["drop tfr_ratio+personnel_to_sales"] = (fs, mono_vec(fs))
    prev = {}
    pf = os.path.join(HERE, "GB-21", "ablation_B_preds.pkl")
    with open(pf, "rb") as fh:
        prev = pickle.load(fh)
    rows, preds = [], {}
    for f in wf["folds"][1:]:
        k = f["fold"]
        tr, _ = fold_frames(panel, wf, test_rows, f)
        m_tr, m_a, m_b, _, _ = inner_masks(tr)
        y = tr["y"].astype(int).to_numpy()
        fp = B.fit(tr[m_tr], feature_set=FS)
        Xtr = B.transform(tr[m_tr], fp, feature_set=FS, oof=True)
        Xa = B.transform(tr[m_a], fp, feature_set=FS)
        Xb = B.transform(tr[m_b], fp, feature_set=FS)
        for v, (fs, mono) in variants.items():
            if v in ("full2", "v1"):                  # identical fits already in GB-21: reuse, do not refit
                preds[(k, v)] = prev[(k, {"full2": "full2", "v1": "v1 (GB-01 set)"}[v])]
                continue
            pbs = []
            for seed in (0, 1, 2):
                t0 = time.time()
                bst = G._train(Xtr[fs], y[m_tr], Xa[fs], y[m_a], cfg01[k], mono, extra={"seed": seed})
                pb = bst.predict(Xb[fs], num_iteration=bst.best_iteration)
                pbs.append(pb)
                rows.append({"fold_inner": k, "variant": v, "seed": seed, "n_features": len(fs),
                             "best_iter": bst.best_iteration, "B_auc": M.auc(y[m_b], pb),
                             "B_log_loss": M.log_loss(y[m_b], pb), "secs": round(time.time() - t0, 1)})
                print(rows[-1], flush=True)
            preds[(k, v)] = np.mean(pbs, axis=0)
        pd.DataFrame(rows).to_csv(os.path.join(out, "extras_runs.csv"), index=False)
        assert (prev[(k, "_y")] == y[m_b]).all()
    S = referee_funcs()
    pairs = [(v, "full2") for v in variants if v != "full2"] + [
        ("full2 unconstrained", "v1 unconstrained"), ("full2", "v1"), ("v1 unconstrained", "v1"),
        ("full2 unconstrained", "full2"), ("full2 + GB-08 relax", "v1 + GB-08 relax"), ("v1 + GB-08 relax", "v1")]
    summ = []
    for k in (2, 3):
        yb, fb = prev[(k, "_y")], prev[(k, "_firm")]
        P = {v: preds[(k, v)] for v in variants}
        dr = S["bootstrap_draws"](yb, fb, P, {"B": np.ones(len(yb), bool)}, n_boot=200, seed=0)
        for a, b_ in dict.fromkeys(pairs):
            da = dr[("auc", a, "B")] - dr[("auc", b_, "B")]
            dl = dr[("log_loss", a, "B")] - dr[("log_loss", b_, "B")]
            summ.append({"fold_inner": k, "a": a, "b": b_, "auc_a": M.auc(yb, P[a]), "auc_b": M.auc(yb, P[b_]),
                         "d_auc": M.auc(yb, P[a]) - M.auc(yb, P[b_]), "d_auc_lo": S["ci"](da)[0], "d_auc_hi": S["ci"](da)[1],
                         "d_ll": M.log_loss(yb, P[a]) - M.log_loss(yb, P[b_]), "d_ll_lo": S["ci"](dl)[0], "d_ll_hi": S["ci"](dl)[1]})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out, "extras_summary.csv"), index=False)
    print(summ.round(5).to_string())


def run_extras2():
    """Inner only. The three unconstrained new features whose shapes are hardest to defend: what do they add?
    Reuses full2 predictions from GB-21. Also interest_to_sales constrained to R-33's upward direction."""
    panel, wf, test_rows = load()
    out = os.path.join(HERE, "GB-24")
    cfg01 = {f["fold"]: f["chosen_cfg"] for f in json.load(open(os.path.join(HERE, "GB-01", "config.json")))["folds"]}
    with open(os.path.join(HERE, "GB-21", "ablation_B_preds.pkl"), "rb") as fh:
        prev = pickle.load(fh)

    def drop(*names):
        fs = [c for c in FEATS if c not in names]
        return fs, mono_vec(fs)

    variants = {"drop interest_to_sales": drop("interest_to_sales"), "drop bs_gap": drop("bs_gap"),
                "drop fin_fixed_ratio": drop("fin_fixed_ratio"),
                "drop interest_to_sales+bs_gap+fin_fixed_ratio": drop("interest_to_sales", "bs_gap", "fin_fixed_ratio"),
                "interest_to_sales constrained +": (FEATS, [1 if f == "interest_to_sales" else int(SIGNS.get(f, 0)) for f in FEATS])}
    rows, summ = [], []
    S = referee_funcs()
    for f in wf["folds"][1:]:
        k = f["fold"]
        tr, _ = fold_frames(panel, wf, test_rows, f)
        m_tr, m_a, m_b, _, _ = inner_masks(tr)
        y = tr["y"].astype(int).to_numpy()
        fp = B.fit(tr[m_tr], feature_set=FS)
        Xtr = B.transform(tr[m_tr], fp, feature_set=FS, oof=True)
        Xa = B.transform(tr[m_a], fp, feature_set=FS)
        Xb = B.transform(tr[m_b], fp, feature_set=FS)
        P = {"full2": prev[(k, "full2")]}
        for v, (fs, mono) in variants.items():
            pbs = []
            for seed in (0, 1, 2):
                bst = G._train(Xtr[fs], y[m_tr], Xa[fs], y[m_a], cfg01[k], mono, extra={"seed": seed})
                pbs.append(bst.predict(Xb[fs], num_iteration=bst.best_iteration))
                rows.append({"fold_inner": k, "variant": v, "seed": seed, "best_iter": bst.best_iteration,
                             "B_auc": M.auc(y[m_b], pbs[-1]), "B_log_loss": M.log_loss(y[m_b], pbs[-1])})
                print(rows[-1], flush=True)
            P[v] = np.mean(pbs, axis=0)
        yb, fb = prev[(k, "_y")], prev[(k, "_firm")]
        assert (yb == y[m_b]).all()
        dr = S["bootstrap_draws"](yb, fb, P, {"B": np.ones(len(yb), bool)}, n_boot=200, seed=0)
        for v in variants:
            da = dr[("auc", v, "B")] - dr[("auc", "full2", "B")]
            dl = dr[("log_loss", v, "B")] - dr[("log_loss", "full2", "B")]
            summ.append({"fold_inner": k, "a": v, "b": "full2", "auc_a": M.auc(yb, P[v]), "auc_b": M.auc(yb, P["full2"]),
                         "d_auc": M.auc(yb, P[v]) - M.auc(yb, P["full2"]), "d_auc_lo": S["ci"](da)[0], "d_auc_hi": S["ci"](da)[1],
                         "d_ll": M.log_loss(yb, P[v]) - M.log_loss(yb, P["full2"]), "d_ll_lo": S["ci"](dl)[0], "d_ll_hi": S["ci"](dl)[1]})
    pd.DataFrame(rows).to_csv(os.path.join(out, "extras2_runs.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(out, "extras2_summary.csv"), index=False)
    print(pd.DataFrame(summ).round(5).to_string())


def run_enc_persist():
    """Do default-rate encodings fitted on the past predict the NEXT year's default rates? Diagnostic only:
    no choice uses it (the next-year labels of the outer years are read here, after GB-20 was already scored)."""
    panel, wf, test_rows = load()
    out = os.path.join(HERE, "GB-22")
    os.makedirs(out, exist_ok=True)
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    rows, tabs = [], []
    for f in wf["folds"]:
        tr, te = fold_frames(panel, wf, test_rows, f)
        fp = B.fit(tr, feature_set=FS)
        X = B.transform(te, fp, feature_set=FS)
        y = te["y"].astype(int).to_numpy()
        lev = {"ateco_div_te": te["ateco_sector"].astype(str).str[:2], "province_te": te["HQ_city"].astype(str)}
        for enc in ("ateco_div_te", "province_te", "region_te"):
            z = X[enc].to_numpy()
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
            r = {"fit_years": str(f["train_years"]), "next_year": f["test_year"], "encoding": enc,
                 "row_slope_y_on_te": float(lr.coef_[0, 0]), "row_auc": M.auc(y, z), "te_sd_rows": float(z.std())}
            if enc in lev:
                g = pd.DataFrame({"l": lev[enc].to_numpy(), "te": z, "y": y}).groupby("l").agg(
                    n=("y", "size"), d=("y", "sum"), te=("te", "mean")).reset_index()
                g = g[g["n"] >= 200]
                g["dr"] = g["d"] / g["n"]
                r.update(n_levels=len(g), level_spearman_te_vs_next_dr=float(spearmanr(g["te"], g["dr"]).correlation))
                if f["fold"] == 3:
                    tabs.append(g.sort_values("te").assign(encoding=enc))
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out, "encoding_persistence.csv"), index=False)
    t = pd.concat(tabs)
    pd.concat([t.groupby("encoding").head(6), t.groupby("encoding").tail(6)]).sort_values(["encoding", "te"]).to_csv(
        os.path.join(out, "encoding_extremes_fold3.csv"), index=False)
    print(pd.DataFrame(rows).round(4).to_string())


# ================================================================================================ compare
def referee_funcs():
    """Load AUCEngine / row_loss / bootstrap_draws / ci / pair_row from the referee's score_r1.py WITHOUT
    executing the script body (it scores and writes files at import). Only those defs are exec'd."""
    path = os.path.join(ROOT, "reports/tournament/score_r1.py")
    tree = ast.parse(open(path).read())
    keep = {"AUCEngine", "row_loss", "bootstrap_draws", "ci", "pair_row", "logit", "sigmoid"}
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in keep]
    ns = {"np": np, "M": M, "N_BOOT": 200, "SEED": 0}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    assert keep <= set(ns), keep - set(ns)
    return ns


def run_compare(n_boot=200):
    panel = pd.read_parquet(os.path.join(ROOT, "data/interim/panel.parquet"),
                            columns=["row_id", "id", "fs_year", "y"], filters=[("fs_year", "<", 2011)])
    test_rows = pd.read_parquet(os.path.join(ROOT, "data/splits/walkforward_test_rows.parquet"))
    base = test_rows.merge(panel, on="row_id", how="left", validate="one_to_one")
    models = {"GB-01": "GB-01/oot_predictions.parquet", "GB-20": "GB-20/oot_predictions.parquet",
              "GB-09": "GB-09/oot_predictions.parquet", "GB-23": "GB-23/oot_predictions.parquet",
              "CAL-01": "../calibration/CAL-01/oot_predictions.parquet",
              "GB-20+CAL": "GB-20/oot_predictions_common_cal.parquet",
              "GB-23+CAL": "GB-23/oot_predictions_common_cal.parquet"}
    P = {}
    for e, f in models.items():
        fp = os.path.join(HERE, f)
        if not os.path.exists(fp):
            continue
        d = base[["row_id", "fold"]].merge(pd.read_parquet(fp), on=["row_id", "fold"], how="left")
        assert d["pd"].notna().all() and len(d) == len(base), e
        P[e] = d["pd"].to_numpy()
    y = base["y"].astype(int).to_numpy()
    year = base["test_year"].to_numpy()
    seen = base["firm_seen_in_train"].to_numpy().astype(bool)
    firm = base["id"].to_numpy()
    subsets = {"ALL": np.ones(len(y), bool), "2008": year == 2008, "2009": year == 2009, "2010": year == 2010,
               "seen": seen, "new": ~seen}
    S = referee_funcs()
    t0 = time.time()
    draws = S["bootstrap_draws"](y, firm, P, subsets, n_boot=n_boot, seed=0)
    print(f"bootstrap {n_boot} draws in {time.time() - t0:.0f}s", flush=True)
    # pair_row reads globals y, subsets, P, P_L1, draws: give it the referee's namespace with ours
    S.update(y=y, subsets=subsets, P=P, P_L1=P, draws=draws)
    for kk in [k for k in draws if k[0] == "log_loss"]:
        draws[("log_loss_L1",) + kk[1:]] = draws[kk]      # L1 column unused here: mirror log loss
    pairs = [(a, "GB-01") for a in ("GB-20", "GB-23", "GB-09") if a in P] + \
            [(a, b) for a, b in (("GB-23", "GB-20"), ("GB-20", "GB-09"), ("GB-20+CAL", "CAL-01"),
                                 ("GB-23+CAL", "CAL-01")) if a in P and b in P]
    prow = pd.DataFrame([S["pair_row"](a, b, s) for a, b in pairs for s in subsets])
    prow = prow.drop(columns=[c for c in prow.columns if "L1" in c])
    prow.to_csv(os.path.join(HERE, "comparison_r2b_bootstrap.csv"), index=False)
    met = []
    for e in P:
        for s, m in subsets.items():
            sc = M.score(y[m], P[e][m])
            r = sc[sc["group"] == "ALL"].iloc[0].to_dict() if "group" in sc else sc.iloc[0].to_dict()
            r.update(model=e, subset=s)
            met.append(r)
    met = pd.DataFrame(met)
    met.to_csv(os.path.join(HERE, "comparison_r2b_metrics.csv"), index=False)
    # overfit: same-year in-sample vs OOT (referee definition, r1b_addons.py)
    ov = []
    for e in ("GB-20", "GB-23"):
        fp = os.path.join(HERE, e, "in_sample_predictions.parquet")
        if not os.path.exists(fp) or e not in P:
            continue
        ins = pd.read_parquet(fp)
        for t, k_in in ((2008, 2), (2008, 3), (2009, 3)):
            c = ins[(ins["fold"] == k_in) & (ins["fs_year"] == t)]
            m = year == t
            for view in ("pd_oof_enc", "pd_fulltable_enc"):
                ov.append({"model": e, "year": t, "in_sample_fold": k_in, "view": view, "n_in": len(c),
                           "auc_in_sample": M.auc(c["y"], c[view]), "auc_oot": M.auc(y[m], P[e][m])})
    ref = pd.read_csv(os.path.join(ROOT, "reports/tournament/r1b_same_year_overfit.csv"))
    ref = ref[ref["model"] == "GB-01"].assign(view="referee r1b (calibration cache)")
    ov = pd.concat([ref[["model", "year", "in_sample_fold", "view", "n_in", "auc_in_sample", "auc_oot"]],
                    pd.DataFrame(ov)], ignore_index=True)
    ov["gap_auc"] = ov["auc_in_sample"] - ov["auc_oot"]
    ov.to_csv(os.path.join(HERE, "comparison_r2b_overfit.csv"), index=False)
    print(prow.round(5).to_string())
    print(ov.round(4).to_string())


# ================================================================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("exp", nargs="?")
    ap.add_argument("--fixed-from", default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "outer":
        run_outer(a.exp, a.fixed_from)
    elif a.cmd == "ablation":
        run_ablation()
    elif a.cmd == "tuneinner":
        run_tuneinner()
    elif a.cmd == "extras":
        run_extras()
    elif a.cmd == "extras2":
        run_extras2()
    elif a.cmd == "encpersist":
        run_enc_persist()
    elif a.cmd == "compare":
        run_compare(a.n_boot)
    else:
        raise SystemExit(f"unknown command {a.cmd}")
