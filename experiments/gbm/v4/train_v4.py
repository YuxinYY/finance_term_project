"""LGB-v4-mono / LGB-v4-free: freshly tuned LightGBM on the v4 temporal_year split (D-156).

Usage: python train_v4.py {mono|free} [n_jobs]

Selection uses ONLY the validation years (FY2008 in round 1, FY2009 in round 2):
early stopping on the validation year (AUC, level-invariant), config chosen by mean validation AUC
over rounds 1-2 (tie-break: mean validation log-loss). Test years (FY2009, FY2010) are scored only
after the choice is written to config.json. Round 3 (train FY07-09, valid FY10): model + best
iteration only; the FY2011 lockbox is never read.

Level (PROVISIONAL, needs team sign-off): pd_shift = expit(logit(pd_raw) + delta),
delta = logit(tau) - logit(train default rate), tau = the round's validation-year default rate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.common import metrics as M  # noqa: E402

DATA = ROOT / "data/processed/2026-11-20_v4_temporal_year"
OUT = ROOT / "experiments/gbm/v4"
SEED = 20261120
MAX_ROUNDS = 5000
ES_PATIENCE = 200

# ---- expected signs -------------------------------------------------------------------------
# Team table: MONOTONE_DIRECTIONS in src/finance_project/model_development/lightgbm_model.py (19 features)
TEAM_SIGNS = {
    "equity_ratio": -1, "st_debt_share": 1, "lt_debt_to_capital": 1, "cash_ratio": -1,
    "cash_to_financing_exp_ratio": -1, "AR_ratio": 1, "payables_ratio": 1, "wc_ratio": -1,
    "cur_debt_ratio": -1, "quick_ratio": -1, "roa": -1, "op_margin": -1, "net_prof_margin": -1,
    "int_coverage": -1, "financing_expense_ratio": 1, "operating_cash_flow_ratio": -1,
    "ebitda_to_debt": -1, "loss_flag": 1, "ateco_sector_te": 1,
}
# Decided here (GBv4-02/03), see decisions.md
ADDED = {
    "size": -1,                        # experiments/gbm constraint table (log_assets -1); consistency report "-"
    "tangible_asset_ratio": -1,        # experiments/gbm table (tangible_ratio -1); collateral
    "operating_cash_flow_margin": -1,  # consistency report "-"; same family as op_margin
    "cash_interest_coverage": -1,      # consistency report "-"; same family as int_coverage
    "operating_cash_flow_to_debt": -1, # experiments/gbm table (cfo_to_debt -1)
    "roe_negative": 1,                 # = loss AND negative equity (misnamed, M3); orchestrator correction
    "debt_detail_missing": 1,          # experiments/gbm table (debt_split_missing +1)
}
REMOVED = {"lt_debt_to_capital"}       # sign flips when debt_lt + eqty_tot < 0 (D-154 (1) open, U-shaped)
MONO_SIGNS = {**{k: v for k, v in TEAM_SIGNS.items() if k not in REMOVED}, **ADDED}

# Expected sign for face validity (consistency report §3 column "Expected sign"), incl. features
# deliberately left unconstrained. 0 = ambiguous / no economic direction.
EXPECTED = {**TEAM_SIGNS, **ADDED, "profit": -1, "ebitda": -1, "prof_operations": -1,
            "exp_financing": 1, "neg_bs_flag": 1, "neg_pl_flag": 1}


def load(rnd: int, part: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"round_{rnd:02d}_{part}.parquet")


def feats(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("id", "fs_year", "label")]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def search_space(n: int = 13) -> list[dict]:
    rng = np.random.default_rng(SEED)
    grid = dict(num_leaves=[7, 15, 31], max_depth=[3, 4, 5], learning_rate=[0.02, 0.03, 0.05],
                min_child_samples=[200, 500, 1000, 2000], min_sum_hessian_in_leaf=[0.5, 2.0, 5.0],
                feature_fraction=[0.5, 0.7, 0.9], bagging_fraction=[0.7, 0.8, 1.0],
                lambda_l1=[0.0, 1.0, 5.0], lambda_l2=[0.0, 5.0, 20.0])
    cfgs = [dict(num_leaves=15, max_depth=4, learning_rate=0.03, min_child_samples=500,
                 min_sum_hessian_in_leaf=2.0, feature_fraction=0.7, bagging_fraction=0.8,
                 lambda_l1=1.0, lambda_l2=5.0)]
    seen = {json.dumps(cfgs[0], sort_keys=True)}
    while len(cfgs) < n:
        c = {k: v[rng.integers(len(v))] for k, v in grid.items()}
        c = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in c.items()}
        c["num_leaves"] = min(c["num_leaves"], 2 ** c["max_depth"])  # depth caps leaves
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            cfgs.append(c)
    return cfgs, grid


def params(cfg: dict, cols: list[str], variant: str, n_jobs: int) -> dict:
    p = dict(objective="binary", metric=["auc", "binary_logloss"], first_metric_only=True,
             boosting_type="gbdt", bagging_freq=1 if cfg["bagging_fraction"] < 1 else 0,
             seed=SEED, deterministic=True, force_row_wise=True, num_threads=n_jobs, verbosity=-1,
             **cfg)
    if variant == "mono":
        p["monotone_constraints"] = [MONO_SIGNS.get(c, 0) for c in cols]
        p["monotone_constraints_method"] = "intermediate"
    return p


def fit(cfg, tr, va, cols, variant, n_jobs):
    dtr = lgb.Dataset(tr[cols], tr["label"], free_raw_data=False)
    dva = lgb.Dataset(va[cols], va["label"], reference=dtr)
    booster = lgb.train(params(cfg, cols, variant, n_jobs), dtr, num_boost_round=MAX_ROUNDS,
                        valid_sets=[dva], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(ES_PATIENCE, first_metric_only=True, verbose=False)])
    return booster


def main(variant: str, n_jobs: int):
    model_id = f"LGB-v4-{variant}"
    out = OUT / model_id
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.txt", "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    data = {r: {p: load(r, p) for p in ("train", "valid")} for r in (1, 2, 3)}
    cols = feats(data[1]["train"])
    assert len(cols) == 81 and all(feats(data[r][p]) == cols for r in (1, 2, 3) for p in ("train", "valid"))
    cfgs, grid = search_space()

    # ---------------- search: validation years only ----------------
    rows = []
    for i, cfg in enumerate(cfgs):
        rec = {"config_id": i, **cfg}
        for r in (1, 2):
            t0 = time.time()
            b = fit(cfg, data[r]["train"], data[r]["valid"], cols, variant, n_jobs)
            pv = b.predict(data[r]["valid"][cols], num_iteration=b.best_iteration)
            yv = data[r]["valid"]["label"].values
            rec[f"r{r}_best_iter"] = b.best_iteration
            rec[f"r{r}_valid_auc"] = M.auc(yv, pv)
            rec[f"r{r}_valid_logloss"] = M.log_loss(yv, pv)
            rec[f"r{r}_secs"] = round(time.time() - t0, 1)
        rec["mean_valid_auc"] = (rec["r1_valid_auc"] + rec["r2_valid_auc"]) / 2
        rec["mean_valid_logloss"] = (rec["r1_valid_logloss"] + rec["r2_valid_logloss"]) / 2
        rows.append(rec)
        say(f"cfg {i}: {cfg} -> AUC {rec['r1_valid_auc']:.4f}/{rec['r2_valid_auc']:.4f} "
            f"iters {rec['r1_best_iter']}/{rec['r2_best_iter']}")
    led = pd.DataFrame(rows)
    led = led.sort_values(["mean_valid_auc", "mean_valid_logloss"], ascending=[False, True])
    led["rank"] = np.arange(1, len(led) + 1)
    led = led.sort_values("config_id").reset_index(drop=True)
    led["chosen"] = led["rank"] == 1
    led.insert(0, "variant", model_id)
    led.to_csv(out / "selection_ledger.csv", index=False)
    best = cfgs[int(led.loc[led["chosen"], "config_id"].iloc[0])]
    say("CHOSEN", best)

    config = {
        "model_id": model_id, "split": "v4 temporal_year (D-156)", "data_dir": str(DATA.relative_to(ROOT)),
        "feature_schema": "1.3.0", "n_features": len(cols), "features": cols, "lightgbm": lgb.__version__,
        "objective": "binary, no class reweighting, no resampling",
        "early_stopping": f"validation year, metric AUC, patience {ES_PATIENCE}, max {MAX_ROUNDS} rounds",
        "selection_rule": "max mean validation AUC over rounds 1-2; tie-break mean validation log-loss",
        "search_space": grid, "n_configs": len(cfgs), "search_seed": SEED,
        "chosen_config_id": int(led.loc[led["chosen"], "config_id"].iloc[0]), "chosen_params": best,
        "monotone": variant == "mono",
        "monotone_constraints": {c: MONO_SIGNS.get(c, 0) for c in cols} if variant == "mono" else None,
        "monotone_constraints_method": "intermediate" if variant == "mono" else None,
        "level_rule": {"status": "PROVISIONAL (needs team sign-off)",
                       "formula": "pd_shift = expit(logit(pd_raw) + delta); delta = logit(tau) - logit(train DR); "
                                  "tau = round's validation-year DR"},
        "lockbox": "FY2011 not read",
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))

    # ---------------- refit chosen config, score test years ----------------
    best_iter, oot, val, levels = {}, [], [], {}
    for r in (1, 2, 3):
        tr, va = data[r]["train"], data[r]["valid"]
        b = fit(best, tr, va, cols, variant, n_jobs)
        b.save_model(str(out / f"model_round_{r:02d}.txt"), num_iteration=b.best_iteration)
        best_iter[f"round_{r:02d}"] = int(b.best_iteration)
        r_w, tau = float(tr["label"].mean()), float(va["label"].mean())
        delta = float(logit(tau) - logit(r_w))
        levels[f"round_{r:02d}"] = {"train_dr": r_w, "tau_valid_dr": tau, "delta": delta}
        parts = [("valid", va)] + ([("test", load(r, "test"))] if r in (1, 2) else [])
        for name, df in parts:
            p = b.predict(df[cols], num_iteration=b.best_iteration)
            o = pd.DataFrame({"row_id": df["id"].astype("int64") * 10000 + df["fs_year"].astype("int64"),
                              "round": r, "fs_year": df["fs_year"].astype(int), "label": df["label"].astype(int),
                              "pd_raw": p, "pd_shift": expit(logit(p) + delta)})
            (oot if name == "test" else val).append(o)
        say(f"round {r}: best_iter {b.best_iteration}, delta {delta:+.4f}")
    (out / "best_iteration.json").write_text(json.dumps({"best_iteration": best_iter, "level": levels}, indent=1))
    pd.concat(oot).reset_index(drop=True).to_parquet(out / "oot_predictions.parquet", index=False)
    pd.concat(val).reset_index(drop=True).to_parquet(out / "valid_predictions.parquet", index=False)
    say("done")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
