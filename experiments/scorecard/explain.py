import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/kietnguyen/Downloads/pd-agent-kit")
import experiments.scorecard.scorecard_lib as L
import src.features.build as B
R = "/Users/kietnguyen/Downloads/pd-agent-kit/experiments/scorecard/"
out = []
# sign checks, all folds
viol = []
for eid in ["SC-01", "SC-02", "SC-03", "SC-04"]:
    for f in json.load(open(R + f"{eid}/config.json"))["folds"]:
        for k, v in f["coef"].items():
            if k == "const": continue
            if eid == "SC-03": exp = +1 if True else 0   # WoE oriented to risk: beta must be > 0
            else: exp = B.META[k]["sign"] if k in B.META else 0
            if exp != 0 and np.sign(v) != exp: viol.append((eid, f["fold"], k, round(v, 4), exp))
print("violations", viol)
# SC-03 explain
p = L.load_params(R + "SC-03/params_fold3.json")
cd = p["glm"]; se = dict(zip(cd["names"], cd["se"])); co = dict(zip(cd["names"], cd["coef"]))
tab = pd.DataFrame([{"feature": c, "family": B.META[c]["family"], "expected_effect_of_riskier_bin": "+ (beta>0)", "beta": co[c], "se": se[c], "z": co[c]/se[c],
      "sign_ok": co[c] > 0, "iv_like_woe_range": max(p["bins"][c]["woe"]) - min(p["bins"][c]["woe"])} for c in p["bins"]])
pts = L.scorecard_points(p)
# SC-02 table fold 3
p2 = L.load_params(R + "SC-02/params_fold3.json"); c2 = dict(zip(p2["glm"]["names"], p2["glm"]["coef"])); s2 = dict(zip(p2["glm"]["names"], p2["glm"]["se"]))
t2 = pd.DataFrame([{"term": k, "expected_sign": B.META[k]["sign"] if k in B.META else "n/a (categorical dummy)", "beta": v, "se": s2[k],
      "ok": ("n/a" if k not in B.META or B.META[k]["sign"] == 0 else str(np.sign(v) == B.META[k]["sign"]))} for k, v in c2.items()])
# reason codes
tr = pd.read_parquet("/Users/kietnguyen/Downloads/pd-agent-kit/data/splits/walkforward_test_rows.parquet")
panel = pd.read_parquet("/Users/kietnguyen/Downloads/pd-agent-kit/data/interim/panel.parquet").set_index("row_id")
pr = pd.read_parquet(R + "SC-03/oot_predictions.parquet"); pr = pr[pr.fold == 3].sort_values("pd")
picks = {"low": pr.iloc[int(len(pr)*0.10)], "mid": pr.iloc[int(len(pr)*0.50)], "high": pr.iloc[int(len(pr)*0.99)]}
s = p["scaling"]; X = None
rc = []
for nm, r in picks.items():
    df = panel.loc[[r.row_id]].reset_index()
    Xf = B.transform(df, p["feature_params"], impute=True)
    W = L._woe_matrix(Xf, p["bins"]).iloc[0]
    # contribution to logit relative to the training-average-woe-free baseline: beta*woe (woe=0 = average firm)
    contrib = {c: co[c] * W[c] for c in W.index}
    top = sorted(contrib.items(), key=lambda kv: -kv[1])
    rc.append((nm, r.pd, panel.loc[r.row_id, "y"], Xf.iloc[0], top))
def words(c, v):
    return f"{c} = {v:.3g}" if not isinstance(v, str) else f"{c} = {v}"
txt = ["# SC-03 WoE scorecard: explanation (model refit on fold 3 training window, fs_year 2007-2009)\n",
 "Best discriminator of the benchmark set (see metrics.md). Params: `params_fold3.json`; points table: `scorecard_points_fold3.csv`.\n",
 "## How to read it\nEach ratio is cut into bins learned on the training window; WoE = ln((share of defaults in bin)/(share of non-defaults in bin)), positive = riskier than average. Numeric WoE is forced monotone in the direction of the expected sign (pool-adjacent-violators), so a worse ratio can never earn a better score. A logit on the WoE columns gives PD. Points: score = offset - factor*ln(PD/(1-PD)) with factor = PDO/ln2; here PDO=20, base score 600 at good:bad odds 50:1 (an illustrative convention, not a bank policy). Higher points = safer.\n",
 "## Coefficients and sign check (WoE columns are oriented so every beta must be positive)\n", tab.round(4).to_markdown(index=False),
 "\nA beta near 1 means the WoE is used at face value; beta well below 1 means the feature overlaps with others (collinearity shrinks it). No negative betas in folds 1-3 (see sign check in `decisions.md`).\n",
 "## Same variables as plain logit (SC-02, fold 3), expected vs actual sign\n", t2.round(4).to_markdown(index=False),
 "\n## Driver curves: bin, default rate, WoE, points (the scorecard)\n", pts.round(4).to_markdown(index=False),
 "\n## Reason codes for three fold-3 (2010) test firms\nContribution = beta x WoE (log-odds vs an average-risk bin); positive pushes PD up. The three firms are the 10th, 50th and 99th percentile of predicted PD.\n"]
nice = {"log_assets":"firm size (log assets)","equity_ratio":"equity / assets","st_debt_ratio":"short-term debt / assets","cash_ratio":"cash / assets","roa_net":"net return on assets","interest_coverage":"interest coverage","interest_to_assets":"interest expense / assets","ar_days":"receivable days","tangible_ratio":"tangible fixed assets / assets","neg_equity":"negative book equity flag","ateco_group":"industry group","legal_group":"legal form","macro_area":"macro-area"}
for nm, pdv, yv, xr, top in rc:
    txt.append(f"### {nm.upper()} PD firm: predicted PD {pdv:.2%} (actual outcome in window: {'default' if yv==1 else 'no default'})\n")
    up = [t for t in top if t[1] > 0.02][:3]; dn = [t for t in top[::-1] if t[1] < -0.02][:3]
    txt.append("Raises risk: " + ("; ".join(f"{nice[c]} ({xr[c] if not isinstance(xr[c], str) else xr[c]!s:.8} -> +{v:.2f} log-odds)" if not isinstance(xr[c], (float,int,np.floating)) else f"{nice[c]} = {xr[c]:.3g} (+{v:.2f} log-odds)" for c, v in up) or "nothing material"))
    txt.append("\nLowers risk: " + ("; ".join(f"{nice[c]}" + (f" = {xr[c]:.3g}" if isinstance(xr[c], (float,int,np.floating)) else f" = {xr[c]}") + f" ({v:.2f} log-odds)" for c, v in dn) or "nothing material") + "\n")
    print(nm, pdv, up, dn)
open(R + "SC-03/explain.md", "w").write("\n".join(txt))
