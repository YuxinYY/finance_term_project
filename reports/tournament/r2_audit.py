"""Referee round 2 audit of the full2 pipeline (no experiment or src file is edited; nothing outside reports/tournament is written).

    OMP_NUM_THREADS=3 python reports/tournament/r2_audit.py light     # saved-model checks, single-row vs batch, face validity
    python reports/tournament/r2_audit.py refit                       # refit folds with train rows only, instrumented (4 threads)

LOCKBOX: panel read with fs_year < 2011. 2010 rows are used here as FEATURE rows only, labels removed; default rates in
the shape tables come from TRAINING rows 2007-2009.
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/gbm"))
import run_round2b as R  # noqa: E402  (imports only; sets G.BASE threads in-process)
import gbm_lib as G  # noqa: E402
from src.features import build as B  # noqa: E402
from src.common import metrics as M  # noqa: E402

OUT = ROOT / "reports/tournament"
LABELS = ["y", "def_date", "label_status", "win_start", "win_end", "src_index"] + [c for c in () ]
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def strip(df):
    cols = [c for c in df.columns if c in LABELS or c.startswith("y_") or c.startswith("status_")]
    return df.drop(columns=cols)


def lo(art, X):
    return G._logit(art["booster"].predict(X))


def load():
    panel, wf, test_rows = R.load()
    assert panel["fs_year"].max() <= 2010
    return panel, wf, test_rows


# ================================================================================================ light
def light():
    panel, wf, test_rows = load()
    f3 = wf["folds"][2]
    tr3, te3 = R.fold_frames(panel, wf, test_rows, f3)
    FEATS = list(B.FULL2_FEATURES)
    arts = {e: G.load_model(str(ROOT / f"experiments/gbm/{e}/model_fold3")) for e in ("GB-25", "GB-20")}
    art01 = G.load_model(str(ROOT / "experiments/gbm/GB-01/model_fold3"))

    # A. saved models reproduce fold-3 OOT with labels removed and rows shuffled
    rep = []
    te_s = strip(te3).sample(frac=1.0, random_state=7)
    for e, a in arts.items():
        p = R.predict_full2(a, te_s)
        o = pd.read_parquet(ROOT / f"experiments/gbm/{e}/oot_predictions.parquet")
        o = o[o.fold == 3].set_index("row_id")["pd"].reindex(te_s["row_id"]).to_numpy()
        rep.append({"exp": e, "fold": 3, "source": "saved model_fold3", "n_test": len(p), "max_abs_diff": float(np.abs(p - o).max())})
        # no id column needed either
        p2 = R.predict_full2(a, te_s.drop(columns=["id"]))
        rep[-1]["max_abs_diff_no_id_column"] = float(np.abs(p2 - p).max())
    pd.DataFrame(rep).to_csv(OUT / "r2_repro_saved.csv", index=False)
    log(rep)

    # B. saved feature params = refit on training rows only
    prov = []
    fp = B.fit(tr3, feature_set="full2")
    for e, a in arts.items():
        s1 = json.dumps(json.loads(json.dumps(fp)), sort_keys=True)
        s2 = json.dumps(a["feature_params"], sort_keys=True)
        rec = a["record"]
        prov.append({"exp": e, "params_identical_to_refit_on_train_rows": s1 == s2, "train_years_in_params": a["feature_params"].get("train_years"),
                     "n_train_rows_params": a["feature_params"]["n_train_rows"], "n_train_expected": f3["n_train"], "record_train_years": rec["train_years"],
                     "te_p0_division": a["feature_params"]["v2"]["te"]["ateco_div_te"]["full"]["p0"], "train_rate": float(tr3["y"].mean()),
                     "n_levels_division": a["feature_params"]["v2"]["te"]["ateco_div_te"]["full"]["n_levels"],
                     "m_division": a["feature_params"]["v2"]["te"]["ateco_div_te"]["full"]["m"],
                     "m_province": a["feature_params"]["v2"]["te"]["province_te"]["full"]["m"], "m_region": a["feature_params"]["v2"]["te"]["region_te"]["full"]["m"],
                     "calibrator_target_equals_train_rate": bool(abs(a["calibrator"]["target_rate"] - tr3["y"].mean()) < 1e-12),
                     "features_equal_FULL2": a["features"] == FEATS, "banned_in_features": [c for c in a["features"] if c in ("def_date", "stmt_date", "fs_year", "id", "y", "src_index")]})
    # inner and in-sample files contain training rows only, in every fold
    train_ids = {f["fold"]: set(panel[panel["fs_year"].isin(f["train_years"]) & (panel["label_status"] == "ok")]["row_id"]) for f in wf["folds"]}
    test_ids = {f["fold"]: set(test_rows[test_rows.fold == f["fold"]]["row_id"]) for f in wf["folds"]}
    cross = []
    for e in ("GB-20", "GB-25"):
        ib = pd.read_parquet(ROOT / f"experiments/gbm/{e}/inner_B_predictions.parquet", columns=["row_id", "fold"])
        ins = pd.read_parquet(ROOT / f"experiments/gbm/{e}/in_sample_predictions.parquet", columns=["row_id", "fold"])
        for k in (1, 2, 3):
            a = set(ib[ib.fold == k]["row_id"]); b = set(ins[ins.fold == k]["row_id"])
            cross.append({"exp": e, "fold": k, "inner_B_rows": len(a), "inner_B_outside_train": len(a - train_ids[k]), "inner_B_in_test": len(a & test_ids[k]),
                          "in_sample_rows": len(b), "in_sample_equals_train": b == train_ids[k], "in_sample_in_test": len(b & test_ids[k])})
    pd.DataFrame(prov).to_csv(OUT / "r2_param_provenance.csv", index=False)
    pd.DataFrame(cross).to_csv(OUT / "r2_inner_rows_check.csv", index=False)
    log(prov); log(pd.DataFrame(cross).to_string())

    # C. OOF encodings: flip the labels of one firm-hash fold; that fold's own encodings must not move; firms are not split
    f1 = wf["folds"][0]
    tr1, _ = R.fold_frames(panel, wf, test_rows, f1)
    fold_ = B._firm_fold(tr1["id"], B.TE_FOLDS)
    split = pd.DataFrame({"id": tr1["id"].to_numpy(), "f": fold_}).groupby("id")["f"].nunique().max()
    tr3f = B._firm_fold(tr3["id"], B.TE_FOLDS)
    split3 = pd.DataFrame({"id": tr3["id"].to_numpy(), "f": tr3f}).groupby("id")["f"].nunique().max()
    te_cols = B.V2_TE
    fp0 = B.fit(tr1, feature_set="full2")
    X0 = B.transform(tr1, fp0, feature_set="full2", oof=True)[te_cols]
    Xf0 = B.transform(tr1, fp0, feature_set="full2", oof=False)[te_cols]
    res = []
    for k in (0, 3):
        t2 = tr1.copy()
        t2.loc[fold_ == k, "y"] = 1 - t2.loc[fold_ == k, "y"].astype(int)
        fp1 = B.fit(t2, feature_set="full2")
        X1 = B.transform(t2, fp1, feature_set="full2", oof=True)[te_cols]
        Xf1 = B.transform(t2, fp1, feature_set="full2", oof=False)[te_cols]
        res.append({"flipped_fold": k, "rows_flipped": int((fold_ == k).sum()), "max_firms_folds_per_firm_fold1": int(split), "max_firms_folds_per_firm_fold3": int(split3),
                    "own_fold_max_change_oof": float(np.abs(X1[fold_ == k].to_numpy() - X0[fold_ == k].to_numpy()).max()),
                    "other_folds_max_change_oof": float(np.abs(X1[fold_ != k].to_numpy() - X0[fold_ != k].to_numpy()).max()),
                    "full_table_max_change": float(np.abs(Xf1.to_numpy() - Xf0.to_numpy()).max())})
    oof_auc = {c: {"oof": M.auc(tr1["y"], X0[c]), "full_table": M.auc(tr1["y"], Xf0[c])} for c in te_cols}
    pd.DataFrame(res).to_csv(OUT / "r2_oof_label_flip.csv", index=False)
    json.dump(oof_auc, open(OUT / "r2_oof_auc_fold1.json", "w"), indent=1)
    log(res, oof_auc)

    # D. single-row vs batch, label columns removed, with deliberately awkward rows included
    a = arts["GB-25"]
    teX = strip(te3).reset_index(drop=True)
    Xb = B.transform(teX, a["feature_params"], feature_set="full2")[FEATS]
    pb = R.predict_full2(a, teX)
    rs = np.random.RandomState(11)
    idx = list(rs.choice(len(teX), 800, replace=False))
    rev = pd.to_numeric(teX["rev_operating"], errors="coerce")
    awk = np.flatnonzero((rev <= 0).to_numpy() | teX["AP_st"].isna().to_numpy() | teX["ateco_sector"].isna().to_numpy() | teX["HQ_city"].isna().to_numpy())
    idx += list(rs.choice(awk, min(200, len(awk)), replace=False))
    idx = sorted(set(int(i) for i in idx))
    dX, dP, dXs, dPs, nan_mis = 0.0, 0.0, 0.0, 0.0, 0
    num_c = [c for c in FEATS if c not in B.CAT_FEATURES]
    cat_mis = 0
    for i in idx:
        row = teX.iloc[[i]]
        x1 = B.transform(row, a["feature_params"], feature_set="full2")[FEATS]
        p1 = R.predict_full2(a, row)[0]
        v1, v0 = x1[num_c].to_numpy(float)[0], Xb.iloc[[i]][num_c].to_numpy(float)[0]
        nan_mis += int((np.isnan(v1) != np.isnan(v0)).sum())
        dX = max(dX, float(np.nanmax(np.abs(v1 - v0))))
        cat_mis += int((x1[B.CAT_FEATURES].astype(str).to_numpy() != Xb.iloc[[i]][B.CAT_FEATURES].astype(str).to_numpy()).sum())
        dP = max(dP, abs(float(p1) - float(pb[i])))
        rs_ = row.astype(object).where(row.notna(), "NA").astype(str)      # every field as a string; blanks as "NA"
        x2 = B.transform(rs_, a["feature_params"], feature_set="full2")[FEATS]
        v2 = x2[num_c].to_numpy(float)[0]
        nan_mis += int((np.isnan(v2) != np.isnan(v0)).sum())
        dXs = max(dXs, float(np.nanmax(np.abs(v2 - v0))))
        dPs = max(dPs, abs(float(R.predict_full2(a, rs_)[0]) - float(pb[i])))
    messy = pd.DataFrame([{c: "NA" for c in teX.columns}, {c: "" for c in teX.columns}, {"asst_tot": "#DIV/0!", "HQ_city": "ZZZ", "ateco_sector": "99.9", "legal_struct": "unseen"}, {"foo": 1}])
    pm = R.predict_full2(a, messy)
    sb = pd.DataFrame([{"rows_tested": len(idx), "awkward_rows_in_pool": len(awk), "max_abs_feature_diff_single_vs_batch": dX, "max_abs_pd_diff_single_vs_batch": dP,
                        "max_abs_feature_diff_string_row": dXs, "max_abs_pd_diff_string_row": dPs, "nan_pattern_mismatches": nan_mis, "cat_mismatches": cat_mis,
                        "messy_rows_pd": [round(float(x), 5) for x in pm], "messy_ok": bool(((pm > 0) & (pm < 1)).all())}])
    sb.to_csv(OUT / "r2_single_vs_batch.csv", index=False)
    log(sb.T)
    # timing
    ts = time.time(); R.predict_full2(a, teX.iloc[:100000]); t_pred = time.time() - ts
    log("predict 100k rows secs", t_pred)

    # E. face validity
    te_ = teX.sample(20000, random_state=0).reset_index(drop=True)
    fv, nanr, ice = [], [], []
    shapes = []
    tr_raw = tr3.reset_index(drop=True)
    ytr = tr_raw["y"].astype(int).to_numpy()
    for e, art in {**arts, "GB-01": art01}.items():
        feats = art["features"]
        fs = "full2" if e != "GB-01" else "full"
        X = B.transform(te_, art["feature_params"], feature_set=fs)[feats]
        signs = dict(zip(feats, art["record"]["monotone"]))
        txt = open(ROOT / f"experiments/gbm/{e}/model_fold3/booster.txt").read()
        line = [l for l in txt.splitlines() if l.startswith("[monotone_constraints:")]
        mth = [l for l in txt.splitlines() if l.startswith("[monotone_constraints_method:")]
        file_mono = [int(v) for v in line[0].split(":")[1].strip(" ]").split(",")]
        assert file_mono == [int(signs[f]) for f in feats], (e, "booster file monotone != config")
        spl = dict(zip(feats, art["booster"].feature_importance("split")))
        Xs = X.sample(1000, random_state=1).reset_index(drop=True)
        Xpd = X.sample(3000, random_state=2).reset_index(drop=True)
        full_tr = B.transform(tr_raw, art["feature_params"], feature_set=fs)[feats]   # full-table view, for decile default rates
        for f in feats:
            s = int(signs[f])
            spec = int(B.META[f]["sign"]) if f in B.META else 0
            if f in B.CAT_FEATURES:
                lv = list(X[f].cat.categories)
                c, drt = [], []
                for l_ in lv:
                    Xg = Xpd.copy(); Xg[f] = pd.Categorical([l_] * len(Xg), categories=lv)
                    c.append(float(lo(art, Xg).mean()))
                    mm = (full_tr[f].astype(str) == l_).to_numpy()
                    drt.append(float(ytr[mm].mean()) if mm.sum() else np.nan)
                for l_, v_, d_ in zip(lv, c, drt):
                    shapes.append({"exp": e, "feature": f, "kind": "level", "point": l_, "x": np.nan, "pd_logodds": v_, "train_dr_2007_09": d_})
                fv.append({"exp": e, "feature": f, "kind": "cat", "spec_sign": 0, "constraint": 0, "splits": int(spl[f]), "pd_range": float(np.ptp(c))})
                continue
            v = X[f].dropna().to_numpy(float)
            grid = np.unique(np.r_[np.quantile(v, np.linspace(0, 1, 41)), v.min(), v.max()])
            if len(grid) < 2:
                grid = np.array([0.0, 1.0])
            curve = []
            for g in np.unique(np.quantile(v, np.linspace(0.01, 0.99, 25))) if len(np.unique(v)) > 3 else grid:
                Xg = Xpd.copy(); Xg[f] = g
                curve.append(float(lo(art, Xg).mean()))
            curve = np.asarray(curve)
            sign_for_share = s if s != 0 else spec
            dv = np.diff(curve)
            ap = int(+1) if (f == "interest_to_sales") else sign_for_share
            against = float(np.clip(-(dv * ap), 0, None).sum() / np.abs(dv).sum()) if (ap != 0 and np.abs(dv).sum() > 1e-12) else np.nan
            Xn = Xpd.copy(); Xn[f] = np.nan
            pn = float(lo(art, Xn).mean())
            viol_rows = 0
            if s != 0:
                M_ = np.vstack([lo(art, Xs.assign(**{f: g})) for g in grid])
                st = np.diff(M_, axis=0) * s
                viol_rows = int((st < -1e-9).any(axis=0).sum())
            fv.append({"exp": e, "feature": f, "kind": "num", "spec_sign": spec, "constraint": s, "splits": int(spl[f]), "pd_lo": float(curve[0]), "pd_hi": float(curve[-1]),
                       "pd_range": float(np.ptp(curve)), "against_share_vs_apriori": against, "ice_rows": len(Xs), "ice_grid": len(grid), "ice_violating_rows": viol_rows,
                       "pd_at_nan": pn, "nan_outside_curve": bool(pn < curve.min() - 1e-9 or pn > curve.max() + 1e-9), "pct_nan_2010": float(X[f].isna().mean())})
            if s == 0 or f == "interest_to_sales":
                qs = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
                if len(qs) >= 3:
                    mids = (qs[:-1] + qs[1:]) / 2
                    trv = full_tr[f].to_numpy(float)
                    for j, m_ in enumerate(mids):
                        Xg = Xpd.copy(); Xg[f] = m_
                        sel = (trv >= qs[j]) & (trv <= qs[j + 1])
                        shapes.append({"exp": e, "feature": f, "kind": "decile", "point": j + 1, "x": float(m_), "pd_logodds": float(lo(art, Xg).mean()),
                                       "train_dr_2007_09": float(ytr[sel].mean()) if sel.sum() else np.nan, "train_n": int(sel.sum())})
                else:
                    for g in np.unique(v):
                        Xg = Xpd.copy(); Xg[f] = g
                        sel = full_tr[f].to_numpy(float) == g
                        shapes.append({"exp": e, "feature": f, "kind": "value", "point": g, "x": float(g), "pd_logodds": float(lo(art, Xg).mean()), "train_dr_2007_09": float(ytr[sel].mean()) if sel.sum() else np.nan, "train_n": int(sel.sum())})
        log("face validity", e, "file monotone method", mth)
    fv = pd.DataFrame(fv); fv.to_csv(OUT / "r2_face_validity.csv", index=False)
    pd.DataFrame(shapes).to_csv(OUT / "r2_shapes_unconstrained.csv", index=False)
    log(fv.groupby("exp").agg(constrained=("constraint", lambda s: int((s != 0).sum())), violating_rows=("ice_violating_rows", "sum")))

    # F. raw-input test: raise ONLY the firm's interest expense; every derived ratio moves together
    raw = teX[pd.to_numeric(teX["exp_financing"], errors="coerce") > 0].sample(3000, random_state=3).reset_index(drop=True)
    factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    out = []
    for e, art in {**arts, "GB-01": art01}.items():
        fs = "full2" if e != "GB-01" else "full"
        curves = []
        for fc in factors:
            r2 = raw.copy()
            r2["exp_financing"] = pd.to_numeric(r2["exp_financing"]) * fc
            curves.append(lo(art, B.transform(r2, art["feature_params"], feature_set=fs)[art["features"]]))
        C = np.vstack(curves)
        d = np.diff(C, axis=0)
        net = C[-1] - C[0]
        out.append({"exp": e, "firms": len(raw), "any_step_down_share": float((d < -1e-9).any(axis=0).mean()), "worst_step_logodds": float(d.min()),
                    "net_fall_share_0.25x_to_4x": float((net < -1e-9).mean()), "mean_net_logodds_0.25x_to_4x": float(net.mean()),
                    "mean_step_1x_to_2x": float((C[6] - C[3]).mean()), "share_pd_falls_1x_to_2x": float((C[6] < C[3] - 1e-9).mean())})
    pd.DataFrame(out).to_csv(OUT / "r2_raw_interest_test.csv", index=False)
    log(pd.DataFrame(out).T)

    # G. reason codes: how often an unconstrained feature is a top-3 PD raiser (fold 3, 20,000 2010 rows, no labels)
    rc = []
    for e, art in arts.items():
        X = B.transform(te_, art["feature_params"], feature_set="full2")[art["features"]]
        sh = art["booster"].predict(X, pred_contrib=True)[:, :-1]
        top = np.argsort(-sh, axis=1)[:, :3]
        mono = np.asarray(art["record"]["monotone"])
        for j, f in enumerate(art["features"]):
            if mono[j] == 0 and f not in B.CAT_FEATURES:
                pos = (top == j).any(axis=1) & (sh[:, j] > 0)
                rc.append({"exp": e, "feature": f, "share_top3_raiser": float(pos.mean()), "mean_abs_shap": float(np.abs(sh[:, j]).mean())})
    pd.DataFrame(rc).to_csv(OUT / "r2_unconstrained_reason_codes.csv", index=False)
    log(pd.DataFrame(rc).round(4).to_string())
    log("LIGHT DONE")


# ================================================================================================ refit
def refit():
    panel, wf, test_rows = load()
    calls = []
    fit0, tr0 = B.fit, B.transform

    def wrap_fit(df, *a, **k):
        calls.append({"call": "fit", "n": len(df), "years": sorted(pd.unique(df["fs_year"]).tolist()), "oof": None, "row_ids": set(df["row_id"].tolist()), "feature_set": k.get("feature_set")})
        return fit0(df, *a, **k)

    def wrap_tr(df, *a, **k):
        if k.get("_clean") is None and k.get("feature_set") == "full2":
            calls.append({"call": "transform", "n": len(df), "years": sorted(pd.unique(df["fs_year"]).tolist()) if "fs_year" in df else None, "oof": bool(k.get("oof", False)),
                          "row_ids": set(df["row_id"].tolist()) if "row_id" in df else None})
        return tr0(df, *a, **k)

    B.fit, B.transform = wrap_fit, wrap_tr
    rows, callog, ice = [], [], []
    for exp in ("GB-25", "GB-20"):
        R.SIGNS["interest_to_sales"] = 1 if exp == "GB-25" else 0
        cfg = {f["fold"]: f for f in json.load(open(ROOT / f"experiments/gbm/{exp}/config.json"))["folds"]}
        oot = pd.read_parquet(ROOT / f"experiments/gbm/{exp}/oot_predictions.parquet")
        for f in wf["folds"]:
            k = f["fold"]
            if exp == "GB-20" and k != 1:
                continue
            ts = time.time()
            calls.clear()
            # the fit sees ONLY training rows: everything else is removed from memory for this call
            tr = panel[panel["fs_year"].isin(f["train_years"]) & (panel["label_status"] == "ok")].reset_index(drop=True)
            art = R.fit_full2(tr, None, log=lambda s: None, tag=exp)
            fit_calls = list(calls)
            ids = test_rows[test_rows.fold == k]["row_id"]
            te = strip(panel[panel["row_id"].isin(ids)]).sample(frac=1.0, random_state=k)
            calls.clear()
            p = R.predict_full2(art, te)
            pc = list(calls)
            o = oot[oot.fold == k].set_index("row_id")["pd"].reindex(te["row_id"]).to_numpy()
            test_set = set(ids.tolist())
            trset = set(tr["row_id"].tolist())
            fit_on_test = sum(len(c["row_ids"] & test_set) for c in fit_calls if c["row_ids"] is not None)
            max_year = max(max(c["years"]) for c in fit_calls if c["years"])
            oof_true_outside = sum(len(c["row_ids"] - trset) for c in fit_calls if c["oof"] is True)
            fits = [c for c in fit_calls if c["call"] == "fit"]
            # each oof=True transform must be on exactly the rows of the matching fit; each oof=False transform in fit must be rows NOT in the inner fit (A/B) or the full-table view
            detail = [{"exp": exp, "fold": k, "phase": "fit", "call": c["call"], "n": c["n"], "years": c["years"], "oof": c["oof"]} for c in fit_calls] + \
                     [{"exp": exp, "fold": k, "phase": "score_test", "call": c["call"], "n": c["n"], "years": c["years"], "oof": c["oof"]} for c in pc]
            callog += detail
            rows.append({"exp": exp, "fold": k, "n_test": len(p), "max_abs_diff_vs_submitted": float(np.abs(p - o).max()),
                         "chosen_cfg_same": art["record"]["chosen_cfg"] == cfg[k]["chosen_cfg"], "n_final_trees": art["record"]["n_final_trees"],
                         "n_final_trees_config": cfg[k]["n_final_trees"], "rows_of_test_fold_in_fit_calls": fit_on_test, "max_year_in_fit_calls": int(max_year), "train_max_year": max(f["train_years"]),
                         "oof_true_rows_outside_train": oof_true_outside, "test_scoring_oof_flags": sorted(set(str(c["oof"]) for c in pc)), "wall_secs": round(time.time() - ts, 1)})
            log(rows[-1])
            if exp == "GB-25":          # ICE on folds 1-3 of the refit models (features only, test rows)
                feats = art["features"]
                X = tr0(te.sample(1000, random_state=5), art["feature_params"], feature_set="full2")[feats].reset_index(drop=True)
                Xfull = tr0(te.sample(20000, random_state=6), art["feature_params"], feature_set="full2")[feats]
                bad = 0
                spl = dict(zip(feats, art["booster"].feature_importance("split")))
                for j, fn in enumerate(feats):
                    s = int(R.SIGNS.get(fn, 0))
                    if s == 0:
                        continue
                    v = Xfull[fn].dropna().to_numpy(float)
                    grid = np.unique(np.r_[np.quantile(v, np.linspace(0, 1, 41)), v.min(), v.max()]) if len(v) else np.array([0.0, 1.0])
                    if len(grid) < 2:
                        grid = np.array([0.0, 1.0])
                    C = np.vstack([G._logit(art["booster"].predict(X.assign(**{fn: g}))) for g in grid])
                    vr = int(((np.diff(C, axis=0) * s) < -1e-9).any(axis=0).sum())
                    bad += vr
                    ice.append({"fold": k, "feature": fn, "sign": s, "splits": int(spl[fn]), "ice_rows": len(X), "grid": len(grid), "violating_rows": vr})
                log("ICE fold", k, "violations", bad)
            pd.DataFrame(rows).to_csv(OUT / "r2_repro_refit.csv", index=False)
            pd.DataFrame(callog).to_csv(OUT / "r2_refit_call_log.csv", index=False)
            pd.DataFrame(ice).to_csv(OUT / "r2_ice_refit_gb25.csv", index=False)
    log("REFIT DONE")


if __name__ == "__main__":
    {"light": light, "refit": refit}[sys.argv[1]]()
