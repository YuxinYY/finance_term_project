"""Per-feature constraint table (final free model D-160 vs mono20 D-161) + a toy monotone-constraint demo.

1. For the 26 candidate-sign features: expected sign, constrained in mono20 or not, share of the curve moving
   against the sign in the development free model (sign_violation_check.csv), and value-vs-SHAP rank
   correlation in the final free model and in mono20 (exact TreeSHAP, same 100,000-row sample as script 09).
2. Toy example (synthetic data) showing that the constraint acts during training.
Outputs: quick_ratio_audit/constraint_table.csv, quick_ratio_audit/monotone_demo.txt
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIT = REPO / "experiments/final/2026-11-21_final_fit"
OUT = HERE / "quick_ratio_audit"
OUT.mkdir(exist_ok=True)

# ---- 1. constraint table
st = pd.read_csv(HERE / "sign_table.csv").set_index("feature")
cand = st[st.candidate != 0]
viol = pd.read_csv(HERE / "sign_violation_check.csv")
free_dev = viol[viol["mode"] == "free"].set_index("feature")
mono_params = json.loads((FIT / "variants/mono20/params.json").read_text())

df = pd.read_parquet(FIT / "final_train_features.parquet")
s = df.sample(100_000, random_state=0)
rho = {}
for name, mf, pf in [("free_final", FIT / "final_lightgbm.txt", FIT / "final_params.json"),
                     ("mono20", FIT / "variants/mono20/lightgbm.txt", FIT / "variants/mono20/params.json")]:
    feats = json.loads(pf.read_text())["feature_columns"]
    c = lgb.Booster(model_file=str(mf)).predict(s[feats], pred_contrib=True)
    for f in cand.index:
        sh = c[:, feats.index(f)]
        rho[(name, f)] = spearmanr(s[f], sh)[0] if np.std(sh) > 0 else np.nan

rows = []
for f, r in cand.iterrows():
    rows.append({"feature": f, "expected_sign": int(r.candidate), "constrained_in_mono20": int(r.constraint) != 0,
                 "dev_free_share_against": free_dev.loc[f, "share_movement_against"] if f in free_dev.index else np.nan,
                 "final_free_value_vs_shap": rho[("free_final", f)], "mono20_value_vs_shap": rho[("mono20", f)]})
t = pd.DataFrame(rows)
t["final_free_agrees"] = np.sign(t.final_free_value_vs_shap) == t.expected_sign
t["mono20_agrees"] = np.sign(t.mono20_value_vs_shap) == t.expected_sign
t = t.sort_values(["constrained_in_mono20", "dev_free_share_against"], ascending=[False, False])
t.to_csv(OUT / "constraint_table.csv", index=False)
print(t.round(3).to_string(index=False))
print("n constrained:", int(t.constrained_in_mono20.sum()))

# ---- 2. toy demo (synthetic): true PD falls with liquidity, with a bump near 0.5
rng = np.random.default_rng(0)
n = 20000
liq = rng.uniform(0, 3, n)
logit = -3 - 1.0 * liq + 1.5 * np.exp(-((liq - 0.5) / 0.2) ** 2)
y = rng.random(n) < 1 / (1 + np.exp(-logit))
X, grid = liq.reshape(-1, 1), np.array([[0.1], [0.5], [1.0], [2.0], [2.9]])
lines = ["liquidity grid: " + " ".join(f"{g:.1f}" for g in grid.ravel())]
for name, mc in [("free", [0]), ("monotone -1", [-1])]:
    m = lgb.train({"objective": "binary", "monotone_constraints": mc, "verbose": -1, "num_leaves": 8},
                  lgb.Dataset(X, y), 100)
    lines.append(f"{name:12s} PD %: " + " ".join(f"{v:.2f}" for v in m.predict(grid) * 100))
(OUT / "monotone_demo.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
