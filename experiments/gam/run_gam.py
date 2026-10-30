"""usage: python experiments/gam/run_gam.py GAM-01 [--fixed-from GAM-01] [--folds 1,2,3]
Writes experiments/gam/<exp>/{oot_predictions.parquet (own calibration), oot_predictions_raw.parquet,
oot_predictions_calibrated.parquet (own calibration + rule-e prior shift, comparable to CAL-01),
config.json, metrics.md, metrics_calibrated.md, train_log.md, model_fold{k}/}."""
import argparse, importlib.util, json, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gam_lib as G
from src.common import metrics as M
ROOT = G.ROOT
spec_ = importlib.util.spec_from_file_location("gbm_run", os.path.join(ROOT, "experiments/gbm/run_experiment.py"))
GR = importlib.util.module_from_spec(spec_); sys.path.insert(0, os.path.join(ROOT, "experiments/gbm")); spec_.loader.exec_module(GR)


def tau_rule_e(test_year):
    ct = pd.read_csv(os.path.join(ROOT, "reports/calibration/ct_targets.csv"))
    r = ct[(ct.label == "s6") & (ct.rule == "e_last_6m_annualised") & (ct.test_year == test_year)]
    return float(r.target.iloc[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--fixed-from", default=None)
    ap.add_argument("--folds", default="1,2,3"); a = ap.parse_args()
    exp, spec = a.exp, G.EXPERIMENTS[a.exp]
    out_dir = os.path.join(HERE, exp); os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train_log.md"), "w")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    fixed = {}
    if a.fixed_from:
        for f in json.load(open(os.path.join(HERE, a.fixed_from, "config.json")))["folds"]:
            fixed[f["fold"]] = f["chosen_cfg"]
    panel, wf, test_rows = GR.load()
    log(f"# {exp} training log\n\n{spec['desc']}\n\nBASE: {json.dumps(G.BASE)}\ninner bags {G.INNER_BAGS}, final bags {G.FINAL_BAGS}\n"
        f"GRID (folds 2-3): {G.GRID}\nDEFAULT (fold 1): {G.DEFAULT}\n" + (f"Fixed cfg from {a.fixed_from}: {fixed}\n" if fixed else ""))
    P, R, C, I, CF = [], [], [], [], []
    for f in wf["folds"]:
        k = f["fold"]
        if str(k) not in a.folds.split(","): continue
        t0 = time.time()
        tr = panel[panel["fs_year"].isin(f["train_years"]) & (panel["label_status"] == "ok")]
        te = panel.merge(test_rows[test_rows["fold"] == k][["row_id", "firm_seen_in_train"]], on="row_id", how="inner")
        assert len(tr) == f["n_train"] and len(te) == f["n_test"]
        log(f"\n## Fold {k}: train {f['train_years']} ({len(tr):,} rows, {int(tr.y.sum()):,} def) -> test {f['test_year']} ({len(te):,})")
        art = G.fit_pd_model(tr, exp, fixed_cfg=fixed.get(k), log=log)
        rec = art["record"]
        tp = time.time(); p = G.predict_pd(art, te); t_pred = time.time() - tp
        pr = G.predict_pd(art, te, raw=True)
        r_w = float(tr.y.mean()); tau = tau_rule_e(f["test_year"])
        delta = float(np.log(tau/(1-tau)) - np.log(r_w/(1-r_w)))
        pc = G.predict_pd(art, te, delta=delta)
        assert np.all((p > 0) & (p < 1)) and np.all((pc > 0) & (pc < 1))
        base = {"row_id": te.row_id.to_numpy(), "fold": k, "y": te.y.astype(int).to_numpy(), "test_year": f["test_year"], "seen": te.firm_seen_in_train.to_numpy()}
        P.append(pd.DataFrame({**base, "pd": p})); R.append(pd.DataFrame({**base, "pd": pr})); C.append(pd.DataFrame({**base, "pd": pc}))
        ib = rec.pop("inner_B_predictions"); ib["fold"] = k; I.append(ib)
        rec.update(fold=k, test_year=f["test_year"], tau_rule_e=tau, r_w=r_w, delta_prior_shift=delta,
                   predict_secs_test=round(t_pred, 2), predict_secs_per_100k=round(t_pred/len(te)*1e5, 2),
                   wall_secs=round(time.time()-t0, 1), fit_secs_total_incl_features_and_grid=round(time.time()-t0 - t_pred, 1))
        CF.append(rec)
        log(f"  chosen {rec['chosen_cfg']}; Platt on B a={rec['platt_on_B']['a']:.3f} b={rec['platt_on_B']['b']:.3f}; B LL raw {rec['B_log_loss_raw']:.5f} platt {rec['B_log_loss_platt']:.5f}\n"
            f"  test {f['test_year']}: AUC {M.auc(te.y, p):.4f} LL {M.log_loss(te.y, p):.5f} mean pd {p.mean():.4f} vs DR {te.y.mean():.4f}; "
            f"after rule-e shift (tau {tau:.4f}, delta {delta:.3f}) LL {M.log_loss(te.y, pc):.5f}; raw AUC {M.auc(te.y, pr):.4f} LL {M.log_loss(te.y, pr):.5f}; "
            f"predict {rec['predict_secs_per_100k']}s/100k; final fit {rec['final_fit_secs']}s; wall {rec['wall_secs']}s")
        G.save_model(art, os.path.join(out_dir, f"model_fold{k}"))
    oot, raw, cal = (pd.concat(x, ignore_index=True) for x in (P, R, C))
    oot[["row_id", "fold", "pd"]].to_parquet(os.path.join(out_dir, "oot_predictions.parquet"), index=False)
    raw[["row_id", "fold", "pd"]].to_parquet(os.path.join(out_dir, "oot_predictions_raw.parquet"), index=False)
    cal[["row_id", "fold", "pd"]].to_parquet(os.path.join(out_dir, "oot_predictions_calibrated.parquet"), index=False)
    pd.concat(I, ignore_index=True).to_parquet(os.path.join(out_dir, "inner_B_predictions.parquet"), index=False)
    md, s = GR.metrics_md(exp, oot, raw, spec["desc"] + " (own calibration: Platt slope from inner B, level = training rate; 'raw' table = EBM probabilities without Platt)")
    open(os.path.join(out_dir, "metrics.md"), "w").write(md)
    mdc, sc = GR.metrics_md(exp + "+CAL", cal, oot, spec["desc"] + " + prior shift to rule e (D-071/D-072, same taus as CAL-01); the 'raw' table here = the own-calibrated PDs before the shift")
    open(os.path.join(out_dir, "metrics_calibrated.md"), "w").write(mdc)
    s.to_csv(os.path.join(out_dir, "metrics_by_year.csv"), index=False); sc.to_csv(os.path.join(out_dir, "metrics_calibrated_by_year.csv"), index=False)
    json.dump({"exp": exp, "spec": spec, "base_params": G.BASE, "inner_bags": G.INNER_BAGS, "final_bags": G.FINAL_BAGS,
               "grid": G.GRID, "default_cfg_fold1": G.DEFAULT, "fixed_cfg_from": a.fixed_from,
               "class_weighting": "none", "interpret_version": G.interpret.__version__, "folds": CF},
              open(os.path.join(out_dir, "config.json"), "w"), indent=1, default=str)
    log("\n## Own calibration\n" + GR.fmt(s) + "\n\n## With rule-e prior shift\n" + GR.fmt(sc))
if __name__ == "__main__":
    main()
