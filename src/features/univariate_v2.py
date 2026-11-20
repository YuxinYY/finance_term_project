"""Round-2 evidence for the features-v2 set (feature-architect, 2026-11-02).

Run:  OMP_NUM_THREADS=3 python -m src.features.univariate_v2
LOCKBOX: the panel is read with a parquet filter fs_year <= 2009, so no row with fs_year >= 2010 is
ever loaded. Statistics use training rows fs_year 2007-2009, label_status == 'ok' (as round 1,
D-062); the inner check fits on 2007-2008 and scores 2009.
Writes reports/features/round2_*.csv, round2_deciles.png, round2_univariate.md and the
"Round 2" section of docs/FEATURE_SPEC.md (between the ROUND2 markers).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.metrics import auc, log_loss  # noqa: E402
from src.features import build as B  # noqa: E402
from src.features.univariate import decile_rates, md, vif_table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "reports", "features")
SPEC = os.path.join(ROOT, "docs", "FEATURE_SPEC.md")
SIGN = {1: "+", -1: "−", 0: "none"}
MARK0, MARK1 = "<!-- ROUND2-START -->", "<!-- ROUND2-END -->"
NEW = B.V2_CANDIDATES          # incl. rejected candidates (marked in the tables)
OLD = B.NUM_FEATURES + B.FLAG_FEATURES


def load_train() -> pd.DataFrame:
    p = pd.read_parquet(os.path.join(ROOT, "data", "interim", "panel.parquet"),
                        filters=[("fs_year", "<=", 2009)])
    tr = p[(p["label_status"] == "ok") & p["fs_year"].isin([2007, 2008, 2009])].reset_index(drop=True)
    assert tr["fs_year"].max() <= 2009
    return tr


def lgb_auc(Xtr, ytr, Xte, yte, cols, signs, seed=0):
    import lightgbm as lgb
    mono = [int(signs.get(c, 0)) if c not in B.CAT_FEATURES else 0 for c in cols]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           monotone_constraints=mono, n_jobs=3, verbose=-1, random_state=seed)
    m.fit(Xtr[cols], ytr)
    p = m.predict_proba(Xte[cols])[:, 1]
    return auc(yte, p), log_loss(yte, p)


def main() -> None:
    tr = load_train()
    y = tr["y"].astype(float)
    base = float(y.mean())
    out_md = []

    # ---------------- full-window fit (2007-09), OOF encodings for the training rows ----------
    params = B.fit(tr, feature_set="full2")
    X = B.transform(tr, params, feature_set="full2_candidates", oof=True)
    Xi = B.transform(tr, params, feature_set="full2_candidates", oof=True, impute=True)
    X_ins = B.transform(tr[["ateco_sector", "HQ_city"]], params, feature_set="full2")  # in-sample TE

    # ---------------- inner split: fit 2007-08, score 2009 -----------------------------------
    a, b = tr["fs_year"] <= 2008, tr["fs_year"] == 2009
    tra, trb = tr[a].reset_index(drop=True), tr[b].reset_index(drop=True)
    pa = B.fit(tra, feature_set="full2")
    Xa = B.transform(tra, pa, feature_set="full2_candidates", oof=True)
    Xb = B.transform(trb, pa, feature_set="full2_candidates")
    ya, yb = tra["y"].astype(float), trb["y"].astype(float)

    rows, dec = [], {}
    for c in NEW:
        mm = X[c].notna()
        au = auc(y[mm], X[c][mm])
        mb = Xb[c].notna()
        au9 = auc(yb[mb], Xb[c][mb])
        emp = 0 if abs(au - 0.5) < 0.01 else (1 if au > 0.5 else -1)
        exp = B.META[c]["sign"]
        dec[c] = decile_rates(X[c], y)
        rows.append(dict(feature=c, in_full2=c in B.NEW_FEATURES_V2, family=B.META[c]["family"], expected=SIGN[exp],
                         empirical=SIGN[emp], auc_0709=au, auc_imputed=auc(y, Xi[c]),
                         auc_inner_2009=au9, missing_rate=float(1 - mm.mean()),
                         sign_ok=(exp == 0 or emp in (0, exp))))
    st = pd.DataFrame(rows)
    dd = pd.DataFrame(dec, index=[f"D{i+1}" for i in range(10)]).T.reset_index().rename(columns={"index": "feature"})
    st = st.merge(dd, on="feature")

    # ---------------- Spearman vs existing + among new ----------------------------------------
    allc = OLD + NEW
    R = X[allc].replace([np.inf, -np.inf], np.nan).rank().corr()
    R.to_csv(os.path.join(OUT, "round2_spearman.csv"))
    mx = []
    for c in NEW:
        r_old = R.loc[c, OLD].abs().drop(c, errors="ignore")
        r_new = R.loc[c, NEW].abs().drop(c)
        mx.append(dict(feature=c, max_rho_existing=float(r_old.max()), partner_existing=r_old.idxmax(),
                       max_rho_new=float(r_new.max()), partner_new=r_new.idxmax()))
    mx = pd.DataFrame(mx)
    mx["flag_ge_0.7"] = (mx[["max_rho_existing", "max_rho_new"]].max(axis=1) >= 0.7)
    st = st.merge(mx, on="feature")
    st.to_csv(os.path.join(OUT, "round2_univariate_stats.csv"), index=False)
    pairs = []
    for i, c1 in enumerate(allc):
        for c2 in allc[i + 1:]:
            if (c1 in NEW or c2 in NEW) and abs(R.loc[c1, c2]) >= 0.7:
                pairs.append(dict(a=c1, b=c2, spearman=float(R.loc[c1, c2])))
    pairs = pd.DataFrame(pairs, columns=["a", "b", "spearman"])
    pairs.to_csv(os.path.join(OUT, "round2_rho_pairs.csv"), index=False)

    # VIF on imputed numeric full2 (excluding flags with zero variance)
    Z = Xi[[c for c in allc if Xi[c].std() > 0]].astype(float)
    vif = vif_table((Z - Z.mean()) / Z.std()).sort_values(ascending=False)
    vif.rename("vif").to_csv(os.path.join(OUT, "round2_vif.csv"))

    # ---------------- target-encoding evidence ------------------------------------------------
    te_rows = []
    for c in B.V2_TE:
        full = params["v2"]["te"][c]["full"]
        te_rows.append(dict(feature=c, levels=full["n_levels"], m=full["m"], tau2=full["tau2"],
                            p0=full["p0"], enc_min=min(full["enc"].values()),
                            enc_max=max(full["enc"].values()),
                            m_oof=";".join(f"{t['m']:.0f}" for t in params["v2"]["te"][c]["oof"]),
                            auc_train_insample=auc(y, X_ins[c]), auc_train_oof=auc(y, X[c]),
                            auc_inner_2009=auc(yb, Xb[c])))
    te = pd.DataFrame(te_rows)
    te.to_csv(os.path.join(OUT, "round2_target_encoding.csv"), index=False)

    # ---------------- data checks: goodwill hypothesis, gross margin, rounding, counts -------
    r = B._reconstruct(B.coerce_input(tr))
    pc = r["goodwill"] - r["ebitda"]
    z = pc == 0
    sect = B._section(r["ateco_sector"])
    gm = (r["rev_operating"] - r["COGS"]) / r["rev_operating"].where(r["rev_operating"] > 0)
    om = r["prof_operations"] / r["rev_operating"].where(r["rev_operating"] > 0)
    keyamt = ["asst_tot", "eqty_tot", "rev_operating", "debt_st", "asst_current"]
    k1000 = pd.concat([(r[c] % 1000 == 0) & (r[c] != 0) for c in keyamt], axis=1)
    all_round = k1000.all(axis=1)
    some_round = k1000.sum(axis=1) >= 3
    checks = pd.DataFrame([
        ("goodwill - ebitda < 0 (share of rows)", float((pc < 0).mean())),
        ("goodwill - ebitda == 0 (share)", float(z.mean())),
        ("  of which liab_lt_emp (TFR) > 0", float((r["liab_lt_emp"][z] > 0).mean())),
        ("goodwill - ebitda > 0: liab_lt_emp > 0", float((r["liab_lt_emp"][pc > 0] > 0).mean())),
        ("  ==0 rows in section L (real estate)", float((sect[z] == "L").mean())),
        ("  all rows in section L", float((sect == "L").mean())),
        ("Spearman(goodwill - ebitda, liab_lt_emp)", float(pd.concat([pc, r["liab_lt_emp"]], axis=1).corr("spearman").iloc[0, 1])),
        ("median personnel / sales (rev > 0, pc >= 0)", float(((pc.where(pc >= 0)) / r["rev_operating"].where(r["rev_operating"] > 0)).median())),
        ("gross margin == operating margin (|diff| < 1e-6, share)", float(((gm - om).abs() < 1e-6)[gm.notna()].mean())),
        ("all 5 key amounts multiples of 1000 (share)", float(all_round.mean())),
        ("  default rate if all rounded", float(y[all_round].mean())),
        ("  n rows / n defaults if all rounded", f"{int(all_round.sum())} / {int(y[all_round].sum())}"),
        (">= 3 of 5 key amounts multiples of 1000 (share)", float(some_round.mean())),
        ("  default rate if >= 3 rounded", float(y[some_round].mean())),
        ("  n rows / n defaults if >= 3 rounded", f"{int(some_round.sum())} / {int(y[some_round].sum())}"),
        ("base default rate", base),
        ("net liabilities <= 0 (net-cash firms, share)", float((((r["asst_tot"] - r["eqty_tot"]) - r["cash_and_equiv"]) <= 0).mean())),
        ("  default rate of net-cash firms", float(y[((r["asst_tot"] - r["eqty_tot"]) - r["cash_and_equiv"]) <= 0].mean())),
    ], columns=["check", "value"])
    checks.to_csv(os.path.join(OUT, "round2_data_checks.csv"), index=False)

    cnt = []
    for c in B.V2_COUNT:
        v = X[c].clip(upper=8)
        g = y.groupby(v).agg(["count", "mean"]).reset_index()
        g.columns = ["value (8 = 8+)", "rows", "default_rate"]
        g.insert(0, "feature", c)
        cnt.append(g)
    cnt = pd.concat(cnt)
    cnt.to_csv(os.path.join(OUT, "round2_count_rates.csv"), index=False)

    # ---------------- inner LightGBM ablation: v1 full vs + each family ------------------------
    fam = {"F1 RiskCalc France": B.V2_CANDIDATES[0:5], "F2 sector pct": B.V2_PCT,
           "F3 default-rate enc": B.V2_TE, "F4 data quality": ["n_missing_raw", "n_core_imputed", "bs_gap"],
           "F5 unused fields": ["nonbank_fin_debt", "ap_lt_ratio", "fin_fixed_ratio", "tfr_ratio"]}
    signs = B.expected_signs(B.ALL_FEATURES + B.V2_CANDIDATES)
    base_cols = B.ALL_FEATURES
    abl = []
    for name, cols in [("v1 full (26)", base_cols)] + [(f"v1 + {k}", base_cols + v) for k, v in fam.items()] + \
                      [("all 21 candidates", B.ALL_FEATURES + B.V2_CANDIDATES), ("full2 (final, 45)", B.FULL2_FEATURES)]:
        au_, ll_ = lgb_auc(Xa, ya, Xb, yb, cols, signs)
        abl.append(dict(set=name, n_features=len(cols), auc_2009=au_, logloss_2009=ll_))
    # leakage demo: in-sample (non-OOF) encodings on the training rows
    Xa_ins = Xa.copy()
    Xa_ins[B.V2_TE] = B.transform(tra[["ateco_sector", "HQ_city"]], pa, feature_set="full2")[B.V2_TE]
    au_, ll_ = lgb_auc(Xa_ins, ya, Xb, yb, base_cols + B.V2_TE, signs)
    abl.append(dict(set="v1 + F3 with IN-SAMPLE encodings (leaky)", n_features=len(base_cols) + 3,
                    auc_2009=au_, logloss_2009=ll_))
    for name, cols in [("v1 full (26)", base_cols), ("full2 (final, 45)", B.FULL2_FEATURES)]:
        for sd in (1, 2):
            au_, ll_ = lgb_auc(Xa, ya, Xb, yb, cols, signs, seed=sd)
            abl.append(dict(set=f"{name} seed {sd}", n_features=len(cols), auc_2009=au_, logloss_2009=ll_))
    abl = pd.DataFrame(abl)
    abl["d_auc_vs_v1"] = abl["auc_2009"] - abl["auc_2009"].iloc[0]
    abl["d_ll_vs_v1"] = abl["logloss_2009"] - abl["logloss_2009"].iloc[0]
    abl.to_csv(os.path.join(OUT, "round2_lgb_ablation.csv"), index=False)

    # ---------------- plot ---------------------------------------------------------------------
    nf, ncol = len(NEW), 4
    fig, axes = plt.subplots(int(np.ceil(nf / ncol)), ncol, figsize=(16, 3 * np.ceil(nf / ncol)))
    for ax, c in zip(axes.ravel(), NEW):
        ax.bar(range(1, 11), [100 * (v if pd.notna(v) else 0) for v in dec[c]], color="teal")
        ax.axhline(100 * base, color="k", lw=0.8, ls="--")
        ax.set_title(f"{c} (exp {SIGN[B.META[c]['sign']]})", fontsize=9)
        ax.set_ylabel("default %", fontsize=8)
    for ax in axes.ravel()[nf:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "round2_deciles.png"), dpi=110)

    # ---------------- markdown -----------------------------------------------------------------
    show = st[["feature", "in_full2", "family", "expected", "empirical", "auc_0709", "auc_imputed", "auc_inner_2009",
               "missing_rate", "max_rho_existing", "partner_existing", "max_rho_new", "partner_new", "sign_ok"]]
    out_md += [
        "# Round-2 univariate evidence (features-v2)",
        "",
        f"Generated by `python -m src.features.univariate_v2`. Training rows fs_year 2007-2009, label_status ok: "
        f"{len(tr):,} rows, {int(y.sum()):,} defaults, base rate {base:.4f}. The panel is read with a "
        "parquet filter fs_year <= 2009 (LOCKBOX: nothing from 2010+ is loaded). Inner check = fit on "
        f"2007-08 ({len(tra):,} rows), score 2009 ({len(trb):,} rows).",
        "",
        "AUC < 0.5 means PD falls as the feature rises. Default-rate encodings (`*_te`) on training rows are "
        "out-of-fold (5 folds grouped by firm id); the 2009 column uses the 2007-08 full-window table.",
        "",
        "## 1. Univariate table", "", md(show), "",
        "## 2. Default rate by decile (training 2007-09; ties stay together, so mass points leave empty deciles)",
        "", "![deciles](round2_deciles.png)", "",
        md(st[["feature"] + [f"D{i+1}" for i in range(10)]]), "",
        "## 3. Pairs with |Spearman| >= 0.70 involving a new feature", "", md(pairs), "",
        "## 4. VIF (imputed GLM view, all v1 + v2 numeric features; top 20)", "",
        md(vif.head(20).rename("vif").reset_index().rename(columns={"index": "feature"}), "{:.2f}"), "",
        "What we do about it: the full2 set is a **GBM set**; trees do not invert the design, so high VIF "
        "is tolerated there (D-025) and explanations are reported by family. No v2 feature enters the "
        "probit core; anyone using full2 in a GLM must use ridge / L2 or drop the partner listed in §3.", "",
        "## 5. Shrunk default-rate encodings", "", md(te), "",
        "## 6. Data checks (hypotheses tested before building features)", "", md(checks, "{:.4f}"), "",
        "## 7. Missing / quality counts: default rate by value", "", md(cnt), "",
        "## 8. Inner LightGBM ablation (fit 2007-08, score 2009; monotone constraints from expected signs; "
        "fixed settings, no tuning; seed 0 unless stated)", "", md(abl, "{:.5f}"), "",
        "## 9. Seed-noise ablation (3 seeds; run of the 21 candidates, `python -m src.features.ablation_v2`)", "",
        "In this table `full2` = all 21 candidates with interest_to_sales constrained '+' (run before the two "
        "rejections and the sign change in D-092 / D-096). Seed s.d. of AUC is about 0.0005, so single-feature "
        "differences below ~0.001 are noise.", "",
        (md(pd.read_csv(os.path.join(OUT, "round2_ablation_seeds.csv")), "{:.5f}")
         if os.path.exists(os.path.join(OUT, "round2_ablation_seeds.csv")) else "(not run)"), "",
    ]
    with open(os.path.join(OUT, "round2_univariate.md"), "w") as fh:
        fh.write("\n".join(out_md))

    # ---------------- FEATURE_SPEC round-2 section --------------------------------------------
    spec_rows = []
    for f in B.V2_FEATURES:
        s = st.set_index("feature").loc[f["name"]]
        spec_rows.append({"name": f["name"] + (" (REJECTED, not in full2)" if f.get("rejected") else ""), "formula": f["formula"], "units": f.get("units", ""),
                          "family": f["family"], "expected sign vs PD": SIGN[f["sign"]],
                          "reason": f["reason"], "source": f["src"],
                          "zero / negative-denominator rule": f["zero_rule"],
                          "missing treatment": f["missing_rule"],
                          "winsor (train 2007-09)": (
                              "[{:.4g}, {:.4g}]".format(*params["v2"]["winsor"][f["name"]])
                              if f["name"] in params["v2"]["winsor"] else "n/a"),
                          "AUC 07-09": f"{s['auc_0709']:.3f}", "max |rho|": f"{max(s['max_rho_existing'], s['max_rho_new']):.2f}"})
    sec = "\n".join([
        MARK0, "", "## Round 2: feature set `full2` (feature-architect, 2026-11-02)", "",
        "`FULL2_FEATURES` = the 26 v1 features (unchanged) + the 19 adopted features below (45 in total; the 2 rows "
        "marked REJECTED are computed for evidence only). Opt-in: "
        "`params = B.fit(train, feature_set=\"full2\")`, `X = B.transform(rows, params, feature_set=\"full2\")` "
        "(add `oof=True` ONLY for training rows, needs column `id`). Decisions D-090..D-099 in "
        "`reports/features/round2_decisions.md`; evidence `reports/features/round2_univariate.md`.", "",
        pd.DataFrame(spec_rows).to_markdown(index=False), "", MARK1, ""])
    spec = open(SPEC).read()
    if MARK0 in spec:
        spec = spec[:spec.index(MARK0)] + sec + spec[spec.index(MARK1) + len(MARK1) + 1:]
    else:
        spec = spec.rstrip("\n") + "\n\n" + sec
    with open(SPEC, "w") as fh:
        fh.write(spec)
    print(md(show))
    print(md(te))
    print(md(checks, "{:.4f}"))
    print(md(abl, "{:.5f}"))
    print(md(pairs))


if __name__ == "__main__":
    main()
