"""Run one GBM experiment over the frozen walk-forward folds.

usage: python experiments/gbm/run_experiment.py GB-01 [--threads 8]
Writes experiments/gbm/<exp>/{oot_predictions.parquet, oot_predictions_raw.parquet,
inner_B_predictions.parquet, config.json, metrics.md, train_log.md, model_fold3/}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gbm_lib as G  # noqa: E402
from src.common import metrics as M  # noqa: E402

ROOT = G.ROOT


def load():
    panel = pd.read_parquet(os.path.join(ROOT, "data/interim/panel.parquet"), filters=[("fs_year", "<", 2011)])  # lockbox (round 2)
    with open(os.path.join(ROOT, "data/splits/walkforward.json")) as fh:
        wf = json.load(fh)
    tr = pd.read_parquet(os.path.join(ROOT, "data/splits/walkforward_test_rows.parquet"))
    return panel, wf, tr


def fmt(df):
    return df.to_markdown(index=False, floatfmt=".5g")


def metrics_md(exp, oot, raw, desc):
    y, p, yr, seen = oot["y"].to_numpy(), oot["pd"].to_numpy(), oot["test_year"].to_numpy(), oot["seen"].to_numpy()
    cols = ["group", "n", "defaults", "auc", "ar", "log_loss", "brier", "mean_pd", "observed_dr",
            "ratio_obs_to_pred", "binom_pvalue", "cal_intercept", "cal_slope"]
    s_year = M.score(y, p, yr)[cols]
    s_seen = M.score(y, p, np.where(seen, "seen_firm", "new_firm"))[cols].iloc[1:]
    s_raw = M.score(y, raw["pd"].to_numpy(), yr)[cols]
    dec = M.decile_table(y, p)
    out = [f"# {exp} metrics (src/common/metrics.py)\n", f"{desc}\n",
           "AR = 2*AUC - 1. Brier = overall probability accuracy (not a calibration measure). "
           "Calibration: mean_pd vs observed_dr (calibration-in-the-large, exact binomial p-value) and "
           "logistic recalibration intercept/slope (perfect: 0 / 1).\n",
           "## Own calibration (submitted `pd`): pooled (ALL) and per test year\n", fmt(s_year),
           "\n## Own calibration: seen vs new firms (pooled over folds)\n", fmt(s_seen),
           "\n## Raw booster probabilities (no post-hoc map), for comparison\n", fmt(s_raw),
           "\n## Decile table, pooled over the three test years (own calibration)\n", fmt(dec)]
    for t in sorted(np.unique(yr)):
        m = yr == t
        out += [f"\n## Decile table, test year {t}\n", fmt(M.decile_table(y[m], p[m]))]
    return "\n".join(out) + "\n", s_year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fixed-from", default=None, help="reuse chosen cfg per fold from another exp's config.json")
    a = ap.parse_args()
    G.BASE["num_threads"] = a.threads
    exp, spec = a.exp, G.EXPERIMENTS[a.exp]
    out_dir = os.path.join(HERE, exp)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train_log.md"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    fixed = {}
    if a.fixed_from:
        with open(os.path.join(HERE, a.fixed_from, "config.json")) as fh:
            for f in json.load(fh)["folds"]:
                fixed[f["fold"]] = f["chosen_cfg"]

    panel, wf, test_rows = load()
    log(f"# {exp} training log\n\n{spec['desc']}\n\nBASE params: {json.dumps(G.BASE)}\n"
        f"GRID (folds 2-3): {G.GRID}\nDEFAULT (fold 1): {G.DEFAULT}\n"
        f"max rounds {G.MAX_ROUNDS}, early stopping {G.EARLY_STOP} on slice A log loss; "
        f"final trees = round({G.N_FINAL_FACTOR} x best_iter)\n"
        + (f"Fixed cfg per fold reused from {a.fixed_from}: {fixed}\n" if fixed else ""))
    preds, raws, inner, cfg_out = [], [], [], []
    for f in wf["folds"]:
        k = f["fold"]
        t0 = time.time()
        tr = panel[panel["fs_year"].isin(f["train_years"]) & (panel["label_status"] == "ok")]
        te_ids = test_rows[test_rows["fold"] == k][["row_id", "firm_seen_in_train"]]
        te = panel.merge(te_ids, on="row_id", how="inner")
        assert len(tr) == f["n_train"] and len(te) == f["n_test"], (len(tr), len(te))
        log(f"\n## Fold {k}: train {f['train_years']} ({len(tr):,} rows, {int(tr['y'].sum()):,} defaults) "
            f"-> test {f['test_year']} ({len(te):,} rows)")
        art = G.fit_pd_model(tr, exp, fixed_cfg=fixed.get(k), log=log)
        rec = art["record"]
        tp = time.time()
        p = G.predict_pd(art, te)
        t_pred = time.time() - tp
        pr = G.predict_pd(art, te, raw=True)
        assert np.all((p > 0) & (p < 1)) and len(p) == len(te)
        preds.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(), "fold": k, "pd": p,
                                   "y": te["y"].astype(int).to_numpy(), "test_year": f["test_year"],
                                   "seen": te["firm_seen_in_train"].to_numpy()}))
        raws.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(), "fold": k, "pd": pr}))
        ib = rec.pop("inner_B_predictions")
        ib["fold"] = k
        inner.append(ib)
        rec.update(fold=k, test_year=f["test_year"], predict_secs_test=round(t_pred, 2),
                   predict_secs_per_100k=round(t_pred / len(te) * 1e5, 2),
                   wall_secs=round(time.time() - t0, 1))
        cfg_out.append(rec)
        c = rec["calibration_compare_B"]
        log(f"  chosen {rec['chosen_cfg']} best_iter={rec['best_iter']} final_trees={rec['n_final_trees']}\n"
            f"  slices: inner-train {rec['n_inner_train']:,} ({rec['defaults_inner_train']} def), "
            f"A {rec['n_A_early_stop']:,} ({rec['defaults_A_early_stop']} def), "
            f"B {rec['n_B_calibration']:,} ({rec['defaults_B_calibration']} def; isotonic eligible={rec['isotonic_eligible']})\n"
            f"  calibration on B (2-fold firm cross-fit log loss): raw {c['raw']:.5f}, platt {c['platt']:.5f}, "
            f"isotonic {c['isotonic']:.5f}; Platt on B a={rec['platt_on_B']['a']:.3f} b={rec['platt_on_B']['b']:.3f}\n"
            f"  own calibrator: b={rec['calibrator']['b']:.3f}, a anchored={rec['calibrator']['a']:.3f} "
            f"(train rate {rec['calibrator']['target_rate']:.4f})\n"
            f"  test {f['test_year']}: AUC {M.auc(te['y'], p):.4f}, log loss {M.log_loss(te['y'], p):.5f}, "
            f"mean pd {p.mean():.4f} vs DR {te['y'].mean():.4f}; raw AUC {M.auc(te['y'], pr):.4f} "
            f"log loss {M.log_loss(te['y'], pr):.5f}; predict {rec['predict_secs_per_100k']}s/100k rows")
        if k == 3:
            G.save_model(art, os.path.join(out_dir, "model_fold3"))
            import pickle
            with open(os.path.join(HERE, f".{exp}_fold3_art.pkl"), "wb") as fh:
                pickle.dump({kk: vv for kk, vv in art.items() if kk != "booster"} |
                            {"booster_str": art["booster"].model_to_string()}, fh)

    oot = pd.concat(preds, ignore_index=True)
    raw = pd.concat(raws, ignore_index=True)
    oot[["row_id", "fold", "pd"]].to_parquet(os.path.join(out_dir, "oot_predictions.parquet"), index=False)
    raw.to_parquet(os.path.join(out_dir, "oot_predictions_raw.parquet"), index=False)
    pd.concat(inner, ignore_index=True).to_parquet(os.path.join(out_dir, "inner_B_predictions.parquet"), index=False)
    md, s = metrics_md(exp, oot, raw, spec["desc"])
    with open(os.path.join(out_dir, "metrics.md"), "w") as fh:
        fh.write(md)
    s.to_csv(os.path.join(out_dir, "metrics_by_year.csv"), index=False)
    cfg = {"exp": exp, "spec": spec, "base_params": G.BASE, "grid": G.GRID, "default_cfg_fold1": G.DEFAULT,
           "max_rounds": G.MAX_ROUNDS, "early_stopping_rounds": G.EARLY_STOP,
           "n_final_factor": G.N_FINAL_FACTOR, "isotonic_min_defaults": G.ISOTONIC_MIN_DEFAULTS,
           "class_weighting": "none (no resampling, no scale_pos_weight)",
           "fixed_cfg_from": a.fixed_from, "folds": cfg_out}
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1, default=str)
    log("\n## Pooled / per-year (own calibration)\n" + fmt(s))
    logf.close()


if __name__ == "__main__":
    main()
