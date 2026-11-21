"""Phase-1 selection on the grid results (rule fixed by KN 2026-11-20 BEFORE seeing results).

Step 2  Pick by ranking: score = mean validation AUC over rounds 1-3 (only settings complete in all 3 rounds).
        Tie band = within 0.001 AUC of the best. Inside the band: lowest mean validation log loss, then
        fewer leaves, then larger min_child_samples (simpler wins).
        Guardrail (reported, not used to swap): winner's 2009 (round-02 validation) AUC rank.
Step 3  Level check: anchor the winner's PDs so the mean PD on its own training rows equals the training
        default rate (log-odds intercept shift; training data only), then on validation: obs/pred should be
        within 0.8-1.25 every round, and anchored log loss is reported next to raw.
Step 4  Monotone vs unconstrained: paired bootstrap (500 draws, rows = firms, validation files have one row
        per firm) of free - mono per round; partial-dependence sign check for the 27 candidate-sign features.
        Default = monotone; switch only if dAUC interval excludes 0 AND no top-10 driver moves >50% against
        its expected sign.
Test-year metrics are REPORTED ONLY (round-3 test = FY2011 lockbox).
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, log_loss
from common import load, feature_list, OUT_DIR, ROUNDS

IMPORT = __import__("importlib").import_module
grid = IMPORT("2026-11-20_02_grid")
TIE, B = 0.001, 500
YEARS = {"round 01": (2008, 2009), "round 02": (2009, 2010), "round 03": (2010, 2011)}


def summarise(r):
    full = r.groupby(["setting", "mode"]).filter(lambda g: g["round"].nunique() == 3)
    g = full.groupby(["setting", "mode"]).agg(
        val_auc=("val_auc", "mean"), val_logloss=("val_logloss", "mean"), val_auc_min=("val_auc", "min"),
        test_auc=("test_auc", "mean"), test_logloss=("test_logloss", "mean"),
        iter_max=("best_iter", "max"), iter_mean=("best_iter", "mean"),
        num_leaves=("num_leaves", "first"), min_child_samples=("min_child_samples", "first"),
        feature_fraction=("feature_fraction", "first"), bagging_fraction=("bagging_fraction", "first"),
        lambda_l2=("lambda_l2", "first")).reset_index()
    return full, g


def pick(g):
    out = {}
    for mode, gm in g.groupby("mode"):
        band = gm[gm.val_auc >= gm.val_auc.max() - TIE]
        w = band.sort_values(["val_logloss", "num_leaves", "min_child_samples"], ascending=[True, True, False]).iloc[0]
        out[mode] = (w, band)
    return out


def anchor(p_train, rate):
    lo = np.log(p_train / (1 - p_train))
    d = brentq(lambda d: (1 / (1 + np.exp(-(lo + d)))).mean() - rate, -5, 5)
    return d


def shift(p, d):
    lo = np.log(p / (1 - p)) + d
    return 1 / (1 + np.exp(-lo))


def refit(w, mode, feats, signs):
    s = {k: w[k] for k in grid.GRID}
    s["num_leaves"], s["min_child_samples"] = int(s["num_leaves"]), int(s["min_child_samples"])
    p = dict(grid.FIXED, **s, bagging_freq=1 if s["bagging_fraction"] < 1 else 0)
    if mode == "mono":
        p.update(monotone_constraints=signs, monotone_constraints_method="intermediate")
    res = {}
    for rnd in ROUNDS:
        tr, va, te = (load(rnd, x) for x in ["train", "validation", "test"])
        ds_tr = lgb.Dataset(tr[feats], tr.label, params={"max_bin": 255, "feature_pre_filter": False})
        ds_va = lgb.Dataset(va[feats], va.label, reference=ds_tr)
        m = lgb.train(p, ds_tr, num_boost_round=grid.MAX_ROUNDS, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(grid.PATIENCE, first_metric_only=True, verbose=False)])
        bi = m.best_iteration
        d = anchor(m.predict(tr[feats], num_iteration=bi), tr.label.mean())
        res[rnd] = dict(model=m, bi=bi, delta=d, train_rate=tr.label.mean(),
                        va=va[["id", "fs_year", "label"]].assign(pd_raw=m.predict(va[feats], num_iteration=bi)),
                        te=te[["id", "fs_year", "label"]].assign(pd_raw=m.predict(te[feats], num_iteration=bi)))
        for k in ("va", "te"):
            res[rnd][k]["pd_anchored"] = shift(res[rnd][k].pd_raw.to_numpy(), d)
        m.save_model(str(OUT_DIR / f"model_{mode}_{rnd.replace(' ', '_')}.txt"), num_iteration=bi)
    return res


def level_table(res, mode):
    rows = []
    for rnd, r in res.items():
        for k, yr in zip(("va", "te"), YEARS[rnd]):
            df = r[k]
            rows.append(dict(mode=mode, round=rnd, split="validation" if k == "va" else "test (report only)", year=yr,
                             dr=df.label.mean(), mean_pd_raw=df.pd_raw.mean(), mean_pd_anch=df.pd_anchored.mean(),
                             obs_pred_raw=df.label.mean() / df.pd_raw.mean(),
                             obs_pred_anch=df.label.mean() / df.pd_anchored.mean(),
                             ll_raw=log_loss(df.label, df.pd_raw), ll_anch=log_loss(df.label, df.pd_anchored),
                             auc=roc_auc_score(df.label, df.pd_raw), best_iter=r["bi"], delta=r["delta"]))
    return pd.DataFrame(rows)


def bootstrap(res_m, res_f):
    rng = np.random.default_rng(7)
    rows, pooled_auc, pooled_ll = [], np.zeros(B), np.zeros(B)
    for rnd in ROUNDS:
        y = res_m[rnd]["va"].label.to_numpy()
        pm, pf = res_m[rnd]["va"].pd_raw.to_numpy(), res_f[rnd]["va"].pd_raw.to_numpy()
        da, dl = np.empty(B), np.empty(B)
        for b in range(B):
            i = rng.integers(0, len(y), len(y))
            da[b] = roc_auc_score(y[i], pf[i]) - roc_auc_score(y[i], pm[i])
            dl[b] = log_loss(y[i], pf[i]) - log_loss(y[i], pm[i])
        pooled_auc += da / 3; pooled_ll += dl / 3
        rows.append(dict(round=rnd, dAUC=roc_auc_score(y, pf) - roc_auc_score(y, pm),
                         dAUC_lo=np.percentile(da, 2.5), dAUC_hi=np.percentile(da, 97.5),
                         dLL=log_loss(y, pf) - log_loss(y, pm), dLL_lo=np.percentile(dl, 2.5), dLL_hi=np.percentile(dl, 97.5)))
    rows.append(dict(round="mean of 3", dAUC=np.mean([r["dAUC"] for r in rows]),
                     dAUC_lo=np.percentile(pooled_auc, 2.5), dAUC_hi=np.percentile(pooled_auc, 97.5),
                     dLL=np.mean([r["dLL"] for r in rows]),
                     dLL_lo=np.percentile(pooled_ll, 2.5), dLL_hi=np.percentile(pooled_ll, 97.5)))
    return pd.DataFrame(rows)


def sign_check(model, feats, cand, mode):
    """Partial dependence on 5,000 round-3 validation rows over the 5th-95th percentile grid of round-3 TRAIN."""
    tr, va = load("round 03", "train"), load("round 03", "validation")
    X = va[feats].sample(5000, random_state=1).to_numpy(copy=True)
    gain = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    top10 = list(gain.index[:10])
    rows = []
    for f, s in cand.items():
        j = feats.index(f)
        q = np.unique(np.quantile(tr[f], np.linspace(0.05, 0.95, 19)))
        if len(q) < 2:
            continue
        pdp = []
        for v in q:
            Xv = X.copy(); Xv[:, j] = v
            pdp.append(model.predict(Xv).mean())
        steps = np.diff(pdp)
        against = np.abs(steps[np.sign(steps) == -s]).sum() / max(np.abs(steps).sum(), 1e-12)
        rows.append(dict(mode=mode, feature=f, expected=s, share_movement_against=round(against, 3),
                         top10_gain=f in top10, gain_rank=int(gain.index.get_loc(f)) + 1))
    return pd.DataFrame(rows), gain


def main():
    feats = feature_list()
    st = pd.read_csv(OUT_DIR / "sign_table.csv").set_index("feature").loc[feats]
    signs = st.constraint.astype(int).tolist()
    cand = {f: int(s) for f, s in st.candidate.items() if s != 0}
    r = pd.read_csv(grid.RESULTS)
    full, g = summarise(r)
    g.sort_values(["mode", "val_auc"], ascending=[True, False]).to_csv(OUT_DIR / "grid_summary.csv", index=False)
    winners = pick(g)
    rep = ["# 75-feature LightGBM grid, Phase 1 selection (2026-11-20)\n",
           f"Settings complete in all 3 rounds: mono {int((g['mode']=='mono').sum())}, free {int((g['mode']=='free').sum())} "
           f"(of {len(grid.settings())} in the full grid). learning_rate {grid.FIXED['learning_rate']}, early stopping on "
           f"validation AUC (patience {grid.PATIENCE}). Sign table: {sum(1 for s in signs if s)} of 75 constrained "
           "(`sign_table.md`).\n",
           "Selection rule (fixed before results): mean validation AUC over rounds 1-3; tie band 0.001; "
           "in band lowest mean validation log loss, then simpler. Test years reported only (FY2011 = lockbox).\n"]
    res, levels, signs_tab = {}, [], []
    r2 = full[full["round"] == "round 02"]
    for mode, (w, band) in winners.items():
        rank09 = int((r2[r2["mode"] == mode].val_auc > r2[(r2["mode"] == mode) & (r2.setting == w.setting)].val_auc.iloc[0]).sum()) + 1
        n09 = int((r2["mode"] == mode).sum())
        top = g[g["mode"] == mode].sort_values("val_auc", ascending=False).head(10)
        rep += [f"\n## Winner, {mode}: `{w.setting}`\n",
                f"mean val AUC {w.val_auc:.4f} (min round {w.val_auc_min:.4f}), mean val log loss {w.val_logloss:.5f}, "
                f"best_iter max {int(w.iter_max)} / mean {w.iter_mean:.0f}. Tie band size {len(band)}. "
                f"2009 crisis guardrail: rank {rank09} of {n09} on round-02 validation AUC.\n",
                f"\nTop 10 by mean validation AUC ({mode}):\n\n",
                top[["setting", "val_auc", "val_auc_min", "val_logloss", "test_auc", "test_logloss", "iter_max"]]
                .to_markdown(index=False, floatfmt=".5f") + "\n"]
        res[mode] = refit(w, mode, feats, signs)
        levels.append(level_table(res[mode], mode))
        sc, gain = sign_check(res[mode]["round 03"]["model"], feats, cand, mode)
        signs_tab.append(sc)
        gain.rename("gain").to_csv(OUT_DIR / f"gain_{mode}_round03.csv")
        for rnd in ROUNDS:
            for k in ("va", "te"):
                res[mode][rnd][k].to_parquet(OUT_DIR / f"pred_{mode}_{rnd.replace(' ', '_')}_{k}.parquet")
    lv = pd.concat(levels); lv.to_csv(OUT_DIR / "level_check.csv", index=False)
    bs = bootstrap(res["mono"], res["free"]); bs.to_csv(OUT_DIR / "bootstrap_free_minus_mono.csv", index=False)
    sc = pd.concat(signs_tab); sc.to_csv(OUT_DIR / "sign_violation_check.csv", index=False)

    va_lv = lv[lv.split == "validation"]
    level_ok = {m: bool(va_lv[va_lv["mode"] == m].obs_pred_anch.between(0.8, 1.25).all()) for m in ("mono", "free")}
    pooled = bs[bs["round"] == "mean of 3"].iloc[0]
    free_bad = sc[(sc["mode"] == "free") & sc.top10_gain & (sc.share_movement_against > 0.5)]
    switch = pooled.dAUC_lo > 0 and free_bad.empty
    rep += ["\n## Level check (anchored to the training default rate; training data only)\n\n",
            lv.to_markdown(index=False, floatfmt=".5f") + "\n",
            f"\nValidation obs/pred (anchored) within 0.8-1.25 in all rounds: mono {level_ok['mono']}, free {level_ok['free']}. "
            "Anchoring to the training rate under-predicts when the next year is worse (cycle); the final level rule "
            "(prior shift to tau vs old bundle method) is a separate open decision.\n",
            "\n## Monotone vs unconstrained (paired bootstrap, free - mono, validation, 500 draws)\n\n",
            bs.to_markdown(index=False, floatfmt=".5f") + "\n",
            "\n## Sign check: partial dependence, share of movement against the expected sign (round-3 model)\n\n",
            sc.sort_values(["mode", "gain_rank"]).to_markdown(index=False) + "\n",
            f"\n## Decision under the pre-set rule\n\nfree dAUC interval excludes 0: {pooled.dAUC_lo > 0}. "
            f"free top-10 drivers moving >50% against sign: {list(free_bad.feature)}. "
            f"=> **{'UNCONSTRAINED' if switch else 'MONOTONE'}**.\n",
            f"\nFinal-fit tree count candidates (learning_rate {grid.FIXED['learning_rate']}): "
            + "; ".join(f"{m}: per-round best_iter {[int(res[m][x]['bi']) for x in ROUNDS]}" for m in res) + ".\n"]
    (OUT_DIR / "2026-11-20_grid75_report.md").write_text("".join(rep))
    json.dump({"choice": "free" if switch else "mono",
               "settings": {m: winners[m][0].setting for m in winners},
               "best_iter": {m: [int(res[m][x]["bi"]) for x in ROUNDS] for m in res}},
              open(OUT_DIR / "selection.json", "w"), indent=2)
    print("".join(rep[-2:]))


if __name__ == "__main__":
    main()
