"""SHAP for the FINAL model (unconstrained LightGBM, 210 trees, 75 features, fit on FY2007-2011).

Exact TreeSHAP via LightGBM pred_contrib (log-odds scale of the raw model; the level calibration is one constant
shift, so it does not change any contribution). Rows: a fixed random sample of 50,000 of the 835,016 training
rows (seed 11) + the per-firm example table.
Outputs (reports/final_model/): 2026-11-21_shap_global.csv, 2026-11-21_shap_direction.csv,
2026-11-21_shap_family.csv, 2026-11-21_shap_firm_examples.csv, 2026-11-21_shap_bar.png, 2026-11-21_shap_summary.md
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "reports/final_model"; OUT.mkdir(parents=True, exist_ok=True)
params = json.loads((HERE / "final_params.json").read_text())
feats = params["feature_columns"]
booster = lgb.Booster(model_file=str(HERE / "final_lightgbm.txt"))
df = pd.read_parquet(HERE / "final_train_features.parquet")
X = df.sample(50000, random_state=11)
contrib = booster.predict(X[feats], pred_contrib=True)
sh = pd.DataFrame(contrib[:, :-1], columns=feats, index=X.index)
base_value = float(contrib[0, -1])

# global importance
g = pd.DataFrame({"mean_abs_shap": sh.abs().mean(), "share_pct": 100 * sh.abs().mean() / sh.abs().mean().sum()})
g = g.sort_values("mean_abs_shap", ascending=False); g["rank"] = np.arange(1, len(g) + 1)
g.to_csv(OUT / "2026-11-21_shap_global.csv")

# direction vs expected sign (sign table D-157 candidates)
st = pd.read_csv(REPO / "experiments/gbm/2026-11-20_grid75/sign_table.csv").set_index("feature")
rows = []
for f in feats:
    x, s = X[f], sh[f]
    rho = spearmanr(x, s).statistic if x.nunique() > 2 else (s[x == 1].mean() - s[x == 0].mean() if x.nunique() == 2 else np.nan)
    exp = int(st.loc[f, "candidate"]) if f in st.index else 0
    rows.append(dict(feature=f, rank=int(g.loc[f, "rank"]), mean_abs_shap=round(g.loc[f, "mean_abs_shap"], 5),
                     expected_sign=exp, value_vs_shap=round(float(rho), 3) if pd.notna(rho) else np.nan,
                     agrees=(np.sign(rho) == exp) if exp != 0 and pd.notna(rho) else None))
d = pd.DataFrame(rows).sort_values("rank"); d.to_csv(OUT / "2026-11-21_shap_direction.csv", index=False)

# families of near-twin features (as in experiments/gbm/2026-11-20_grid75/2026-11-21_07_shap_sign_check.py)
FAM = {"cash_flow": (["operating_cash_flow_ratio", "operating_cash_flow_to_debt", "operating_cash_flow_margin", "cash_interest_coverage"], "operating_cash_flow_ratio"),
       "profitability": (["roa", "op_margin", "net_prof_margin", "ebitda_to_debt", "int_coverage"], "roa"),
       "liquidity": (["quick_ratio", "cur_debt_ratio", "cash_ratio", "wc_ratio"], "cur_debt_ratio")}
fam = pd.DataFrame([dict(family=k, anchor=a, spearman_anchor_vs_family_shap=round(spearmanr(X[a], sh[c].sum(axis=1)).statistic, 3),
                         mean_abs_family_shap=round(sh[c].sum(axis=1).abs().mean(), 4)) for k, (c, a) in FAM.items()])
fam.to_csv(OUT / "2026-11-21_shap_family.csv", index=False)

# per-firm reason codes: 6 example firms (3 highest-PD, 3 typical), top 5 drivers each
lo = base_value + sh.sum(axis=1)
c = params["calibration"]
pd_cal = np.clip(1 / (1 + np.exp(-(c["anchor_a"] + lo + c["delta"]))), c["floor"], c["cap"])
pick = list(pd_cal.sort_values(ascending=False).index[:3]) + list((pd_cal - pd_cal.median()).abs().sort_values().index[:3])
ex = []
for i in pick:
    top = sh.loc[i].abs().sort_values(ascending=False).index[:5]
    for r, f in enumerate(top, 1):
        ex.append(dict(id=int(X.loc[i, "id"]), fs_year=int(X.loc[i, "fs_year"]), defaulted=int(X.loc[i, "label"]),
                       pd=round(float(pd_cal.loc[i]), 5), driver_rank=r, feature=f, value=float(X.loc[i, f]),
                       shap_log_odds=round(float(sh.loc[i, f]), 4), effect="raises PD" if sh.loc[i, f] > 0 else "lowers PD"))
pd.DataFrame(ex).to_csv(OUT / "2026-11-21_shap_firm_examples.csv", index=False)

# bar chart of the top 20
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    top = g.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.mean_abs_shap, color="#4C72B0")
    ax.set_xlabel("mean |SHAP| (log-odds), 50,000 training rows"); ax.set_title("Final model: top 20 drivers")
    fig.tight_layout(); fig.savefig(OUT / "2026-11-21_shap_bar.png", dpi=150)
except Exception as e:  # plotting is optional
    print("plot skipped:", e)

dis = d[(d.agrees == False)]  # noqa: E712
md = ["# Final model SHAP summary (2026-11-21)\n",
      f"Model: unconstrained LightGBM, 210 trees, 75 features, fit on all FY2007-2011 rows. Exact TreeSHAP on 50,000 "
      f"training rows (log-odds; calibration is a constant shift). Base value {base_value:.4f}.\n",
      "\n## Top 15 drivers\n\n", g.head(15).round(4).to_markdown(), "\n",
      "\n## Direction vs expected sign (features with a candidate sign)\n\n",
      d[d.expected_sign != 0].to_markdown(index=False), "\n",
      f"\nFeatures whose SHAP direction is against the expected sign: {list(dis.feature)}\n",
      "\n## Families of near-twin features (summed SHAP vs anchor ratio)\n\n", fam.to_markdown(index=False), "\n",
      "\nPer-firm reason codes: `2026-11-21_shap_firm_examples.csv`. Chart: `2026-11-21_shap_bar.png`.\n"]
(OUT / "2026-11-21_shap_summary.md").write_text("".join(md))
print("".join(md))
