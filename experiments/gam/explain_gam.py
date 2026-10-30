"""Explainability for the EBM challengers (fold-3 model: trained on fs_year 2007-2009, explained on 2010 test rows).
usage: python experiments/gam/explain_gam.py GAM-0x     -> experiments/gam/<exp>/explain.md, shape_*.png, explain_*.csv"""
import os, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gam_lib as G
from src.features import build as B
ROOT = G.ROOT
sys.path.insert(0, os.path.join(ROOT, "experiments/gbm"))
from explain import PLAIN  # noqa  (credit-officer wording shared with the GBM explainer)

exp = sys.argv[1]
od = os.path.join(HERE, exp)
arts = {k: G.load_model(os.path.join(od, f"model_fold{k}")) for k in (1, 2, 3)}
art = arts[3]; ebm = art["ebm"]; fp = art["feature_params"]; feats = art["features"]
signs = B.expected_signs()
panel = pd.read_parquet(os.path.join(ROOT, "data/interim/panel.parquet"))
ok = panel[panel.label_status == "ok"]
te = ok[ok.fs_year == 2010].reset_index(drop=True)
tr = ok[ok.fs_year.isin([2007, 2008, 2009])].sample(100000, random_state=0)
rng = np.random.default_rng(0)
sub = te.iloc[rng.choice(len(te), 40000, replace=False)].reset_index(drop=True)
X = B.transform(sub, fp)[feats]; Xtr = B.transform(tr, fp)[feats]
terms = ebm.term_names_
contrib = pd.DataFrame(ebm.eval_terms(X), columns=terms)
imp = contrib.abs().mean().sort_values(ascending=False)
main_terms = [t for t in terms if " & " not in t]
inter_terms = [t for t in terms if " & " in t]


def shape(model, f):
    """step function for a continuous feature: (left edges, right edges, scores) excluding the missing bin"""
    i = model.term_names_.index(f)
    d = model.explain_global().data(i)
    edges = np.asarray(d["names"], float); sc = np.asarray(d["scores"], float)
    miss = float(model.term_scores_[i][0])
    return edges[:-1], edges[1:], sc, miss


def cat_scores(model, f):
    i = model.term_names_.index(f)
    d = model.explain_global().data(i)
    return dict(zip(d["names"], np.asarray(d["scores"], float))), float(model.term_scores_[i][0])


def center(mid, sc, x):  # centre the shape at the data-weighted mean (so curves are comparable across folds)
    idx = np.clip(np.searchsorted(mid, x.dropna().to_numpy()), 0, len(mid) - 1)
    return sc - sc[idx].mean()

# ------------------------------------------------------------------ shape plots + face-validity table
ncol = 5; nrow = int(np.ceil(len(feats) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.35 * nrow))
rows = []
for ax, f in zip(axes.ravel(), feats):
    s = signs.get(f, 0); sname = {1: "+", -1: "-", 0: "none"}[s]
    if f in B.CAT_FEATURES:
        cs, miss = cat_scores(ebm, f)
        ser = pd.Series(cs).sort_values()
        ax.barh(range(len(ser)), ser.values, color="tab:blue"); ax.set_yticks(range(len(ser))); ax.set_yticklabels(ser.index, fontsize=6)
        rows.append(dict(feature=f, expected="none (categorical)", importance=imp[f], shape="levels: " + ", ".join(f"{k} {v:+.2f}" for k, v in ser.items()),
                         against_share=np.nan, verdict="n/a"))
    else:
        for k in (1, 2):
            lo, hi, sc, _ = shape(arts[k]["ebm"], f); mid = (lo + hi) / 2
            ax.step(mid, center(mid, sc, Xtr[f]), where="mid", color="0.7", lw=0.8)
        lo, hi, sc, miss = shape(ebm, f); mid = (lo + hi) / 2
        scc = center(mid, sc, Xtr[f])
        ax.step(mid, scc, where="mid", color="tab:red", lw=1.5)
        x = Xtr[f].dropna()
        a, b = np.quantile(x, [0.01, 0.99])
        if a < b: ax.set_xlim(a, b)
        if f in ("neg_equity", "debt_split_missing", "rev_nonpositive"): ax.set_xlim(-0.2, 1.2)
        d = np.diff(sc)
        # data-weighted movement: only bins containing data (Xtr) count
        q = np.quantile(x, [0.02, 0.5, 0.98]); idx = np.clip(np.searchsorted(mid, q), 0, len(mid) - 1)
        if s == 0:
            verdict, against = "no expectation", np.nan
        else:
            against = float(np.abs(d[(d * s) < -1e-9]).sum() / max(np.abs(d).sum(), 1e-12))
            tot = (sc[idx[2]] - sc[idx[0]]) * s
            verdict = "agrees (monotone)" if against < 1e-6 and tot >= 0 else ("mostly agrees" if tot > 0 else "CONTRADICTS")
        rows.append(dict(feature=f, expected=sname, importance=imp[f],
                         shape=f"2nd pct {sc[idx[0]]:+.2f} / median {sc[idx[1]]:+.2f} / 98th pct {sc[idx[2]]:+.2f} log-odds; missing-bin {miss:+.2f}",
                         against_share=against, verdict=verdict))
    ax.set_title(f"{f} (exp {sname})", fontsize=8); ax.tick_params(labelsize=6); ax.axhline(0, color="k", lw=0.3)
for ax in axes.ravel()[len(feats):]: ax.axis("off")
fig.suptitle(f"{exp}: shape functions (log-odds contribution; red = fold-3 model 2007-09, grey = fold-1/2 models), centred at the training mean", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(od, "shape_functions.png"), dpi=110); plt.close(fig)
fv = pd.DataFrame(rows).sort_values("importance", ascending=False); fv.to_csv(os.path.join(od, "explain_shapes.csv"), index=False)

# ------------------------------------------------------------------ interactions
L_int = []
if inter_terms:
    ii = imp[inter_terms].sort_values(ascending=False)
    fig, axes = plt.subplots(1, min(3, len(ii)), figsize=(4.4 * min(3, len(ii)), 3.6), squeeze=False)
    for ax, t in zip(axes[0], ii.index[:3]):
        i = terms.index(t); d = ebm.explain_global().data(i)
        z = np.asarray(d["scores"], float)
        ax.imshow(z.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-abs(z).max(), vmax=abs(z).max())
        a_, b_ = t.split(" & "); ax.set_xlabel(a_, fontsize=7); ax.set_ylabel(b_, fontsize=7); ax.set_title(t, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(od, "interaction_heatmaps.png"), dpi=110); plt.close(fig)
    # face validity of the combined effect: for each signed feature of the pair, vary it over its training
    # deciles with the partner at each of its training quartiles, other features at the median row; count
    # movement against the expected sign in the ADDITIVE total score (main + all terms)
    base = Xtr.median(numeric_only=True)
    for t in ii.index:
        a_, b_ = t.split(" & ")
        for fv_, other in ((a_, b_), (b_, a_)):
            s = signs.get(fv_, 0)
            if s == 0 or fv_ in B.CAT_FEATURES: continue
            grid = np.unique(np.quantile(Xtr[fv_].dropna(), np.linspace(0.02, 0.98, 25)))
            oth_vals = (np.unique(np.quantile(Xtr[other].dropna(), [0.1, 0.5, 0.9])) if other not in B.CAT_FEATURES
                        else list(Xtr[other].astype(str).value_counts().index[:3]))
            viol = []
            for ov in oth_vals:
                rows_ = Xtr.iloc[:1].copy()
                rows_ = pd.concat([rows_] * len(grid), ignore_index=True)
                for c in rows_.columns:
                    if c in B.CAT_FEATURES: rows_[c] = pd.Categorical([Xtr[c].astype(str).mode()[0]] * len(grid), categories=Xtr[c].cat.categories)
                    else: rows_[c] = base[c]
                rows_[fv_] = grid
                if other in B.CAT_FEATURES: rows_[other] = pd.Categorical([ov] * len(grid), categories=Xtr[other].cat.categories)
                else: rows_[other] = ov
                lo_ = ebm.decision_function(rows_)
                dd = np.diff(lo_)
                viol.append(float(np.abs(dd[(dd * s) < -1e-6]).sum() / max(np.abs(dd).sum(), 1e-12)))
            L_int.append(dict(term=t, importance=float(ii[t]), varied=fv_, partner=other, expected=s,
                              against_share_max_over_partner_values=max(viol), against_share_mean=float(np.mean(viol))))
    pd.DataFrame(L_int).to_csv(os.path.join(od, "explain_interaction_face_validity.csv"), index=False)

# ------------------------------------------------------------------ reason codes
p = G.predict_pd(art, sub); order = np.argsort(p)
reasons = []
for lab, qq in {"low PD (5th percentile)": 0.05, "middle PD (median)": 0.50, "high PD (99th percentile)": 0.99}.items():
    j = order[int(qq * (len(p) - 1))]
    c = contrib.loc[j]
    # attribute interaction terms back to the signed partner for the wording: keep the term name
    up = c.sort_values(ascending=False); up = [t for t in up.index if c[t] > 0][:3]
    dn = [t for t in c.sort_values().index if c[t] < 0][:2]
    def desc(t, raising=True):
        if " & " in t:
            a_, b_ = t.split(" & ")
            return f"combination {PLAIN[a_][0]} x {PLAIN[b_][0]} (values {_v(j, a_)}, {_v(j, b_)}; {c[t]:+.2f} log-odds)"
        val = X.loc[j, t]
        if t in B.CAT_FEATURES:
            return f"{PLAIN[t][0]} = {val} ({'raises' if raising else 'lowers'} PD, {c[t]:+.2f} log-odds)"
        pct = (Xtr[t] <= val).mean() * 100 if pd.notna(val) else np.nan
        txt = PLAIN[t][1] if raising else f"favourable {PLAIN[t][0]}"
        if pd.isna(val): return f"{txt}: {PLAIN[t][0]} not available ({c[t]:+.2f} log-odds)"
        return f"{txt}: {PLAIN[t][0]} = {_nat(t, val):.3g} (percentile {pct:.0f} of 2007-09 applicants; {c[t]:+.2f} log-odds)"
    def _nat(t, v): return np.sign(v) * np.expm1(abs(v)) if B.META[t].get("slog") else v
    def _v(j_, t): return str(X.loc[j_, t]) if t in B.CAT_FEATURES else f"{_nat(t, X.loc[j_, t]):.3g}"
    reasons.append(dict(case=lab, row_id=int(sub.loc[j, "row_id"]), pd=float(p[j]), y=int(sub.loc[j, "y"]),
                        ateco=str(X.loc[j, "ateco_group"]), up=[desc(t) for t in up], down=[desc(t, False) for t in dn],
                        intercept=float(ebm.intercept_[0]), total=float(c.sum())))

# ------------------------------------------------------------------ timing
big = te.sample(100000, random_state=1); t0 = time.time(); G.predict_pd(art, big); t_full = time.time() - t0
t0 = time.time(); B.transform(big, fp); t_tr = time.time() - t0
one = te.iloc[[0]]; t0 = time.time()
for i in range(20): G.predict_pd(art, te.iloc[[i]])
t_row = (time.time() - t0) / 20

# ------------------------------------------------------------------ markdown
rec = art["record"]
L = [f"# {exp} explainability (fold-3 model: trained on fs_year 2007-2009; explained on a 40,000-row random sample of 2010 test rows)\n",
     f"{G.EXPERIMENTS[exp]['desc']}. Chosen config {rec['chosen_cfg']}, {len(terms)} terms ({len(main_terms)} main, {len(inter_terms)} pairwise). "
     "An EBM is a sum of one lookup table per feature (plus pairs): PD = expit(intercept + sum of term scores) before the Platt/level step, so every "
     "firm's log-odds splits exactly into per-feature contributions (checked: the term contributions plus the intercept reproduce the model's log-odds). "
     "Method background (cyclic gradient boosting of shape functions, GA2M) is from Lou et al. 2012 and Nori et al. 2019 [UNVERIFIED: R-24/R-25 not verified]; "
     "the link to RiskCalc v3.1 as 'closely related to a GAM' is R-01 p. 14 fn. 11.\n",
     "## 1. Shape functions\n", "![shape functions](shape_functions.png)\n",
     "Red = the fold-3 model (what this section describes); grey = fold-1 and fold-2 models (shape stability across training windows). "
     "x-axes are cut at the training 1st-99th percentile; features marked slog are on a signed-log scale. Each curve is the log-odds "
     "contribution, centred at the training average. Missing values have their own score (printed in the table).\n",
     "Importance = mean |contribution| (log-odds) on the 2010 sample. `against_share` = share of the total variation of the step function that moves "
     "AGAINST the expected sign (0 = monotone as expected).\n",
     fv.to_markdown(index=False, floatfmt=".3f")]
fam = pd.Series({f: imp[f] for f in main_terms}).groupby(lambda f: B.META[f]["family"].replace(" (flag)", "")).sum().sort_values(ascending=False)
L += ["\nFamily importance (sum of mean |contribution| of main effects):\n", fam.to_frame("mean_abs_contribution").to_markdown(floatfmt=".3f")]
if inter_terms:
    L += ["\n## 2. Interaction terms\n", "![interactions](interaction_heatmaps.png)\n",
          imp[inter_terms].sort_values(ascending=False).to_frame("mean_abs_contribution").to_markdown(floatfmt=".4f"),
          "\nFace validity of the combined (additive total) effect: vary one signed feature over its 2nd-98th percentile, partner at its 10/50/90th percentile "
          "(or top-3 categories), other features at the training median; share of movement against the expected sign:\n",
          pd.DataFrame(L_int).to_markdown(index=False, floatfmt=".3f")]
L += ["\n## 3. Reason codes for three example 2010 applicants\n",
      "Top-3 terms pushing the PD UP and the 2 strongest pulling it DOWN (log-odds contributions of the EBM before the monotone Platt/level map; "
      "the ranking is unchanged by that map). Percentiles vs a 100,000-row sample of 2007-09 applicants. Values on the natural scale (signed-log back-transformed; "
      "winsorised at the training 1st/99th percentile).\n"]
for r in reasons:
    L += [f"### {r['case']}: row_id {r['row_id']}, PD {r['pd']*100:.2f}% (own calibration), industry section {r['ateco']}, observed default in window: {'yes' if r['y'] else 'no'}\n",
          "Raises PD:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(r["up"])),
          "\nLowers PD:\n" + "\n".join(f"- {s}" for s in r["down"]) + "\n"]
chk = np.abs(contrib.sum(axis=1).to_numpy() + ebm.intercept_[0] - ebm.decision_function(X)).max()
L += ["## 4. Cost\n", f"- Max |sum of term contributions + intercept - model log-odds| over the sample: {chk:.2e}.\n"
      f"- Prediction (feature transform + EBM + calibration), 100,000 rows: {t_full:.1f} s total, feature transform {t_tr:.1f} s; one row at a time: {t_row*1000:.0f} ms/row (dominated by the transform's fixed cost). "
      "Complexity O(n x (F + T)) lookups (T = terms), linear in n, no dependence on other rows; the harness needs the `interpret` package (0.7.8) or an export of the term tables.\n"]
open(os.path.join(od, "explain.md"), "w").write("\n".join(L) + "\n")
imp.to_csv(os.path.join(od, "explain_importance.csv"))
print("\n".join(L[:12])[:200]); print(fv.to_string(index=False)); print(pd.DataFrame(L_int).to_string() if L_int else "")
