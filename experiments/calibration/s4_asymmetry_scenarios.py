"""Stage 4 (calibration-specialist): cost of being too low vs too high, and benign vs stressed
holdout scenarios.

(1) Asymmetry, empirical: for GB-01 and each test year, start from the ORACLE level (each year's
    own calibration-in-the-large intercept; diagnostic only) and move the level so that the mean PD
    is k x the observed rate (k = 0.5 .. 2.0). Report the extra log loss vs the oracle.
    Compared: 30% too low (k = 0.7), 30% too high (k = 1.3), and the log-symmetric pair 1/1.3 vs 1.3.
(2) Scenarios: for candidate holdout targets tau and possible holdout default rates pi, the extra
    log loss from the level alone (Bernoulli KL, which is what a pure logit shift of a model with
    slope ~ 1 adds) and the regret table; expected-loss and minimax-regret choices.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import metrics as M  # noqa: E402

OUT = ROOT / "reports/calibration"
EPS = M.EPS


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def solve_shift(z, target):
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if expit(z + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def kl(pi, q):
    return pi * np.log(pi / q) + (1 - pi) * np.log((1 - pi) / (1 - q))


testrows = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet")
panel = pd.read_parquet(ROOT / "data/interim/panel.parquet", columns=["row_id", "y"])
base = testrows.merge(panel, on="row_id")
o = pd.read_parquet(ROOT / "experiments/gbm/GB-01/oot_predictions.parquet")
base = base.merge(o, on=["row_id", "fold"])
rows = []
for t, g in base.groupby("test_year"):
    y = g.y.to_numpy().astype(int)
    z = logit(g.pd.to_numpy())
    pi = y.mean()
    ll_oracle = M.log_loss(y, expit(z + solve_shift(z, pi)))
    for k in [0.5, 0.6, 0.7, 1 / 1.3, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        p = expit(z + solve_shift(z, k * pi))
        rows.append({"test_year": t, "observed_rate": pi, "k_mean_pd_over_observed": round(k, 4),
                     "log_loss": M.log_loss(y, p), "extra_vs_correct_level": M.log_loss(y, p) - ll_oracle,
                     "level_only_kl": kl(pi, k * pi)})
asy = pd.DataFrame(rows)
asy.to_csv(OUT / "asymmetry_gb01.csv", index=False)

# ---------------------------------------------------------------- scenarios
taus = [0.0128, 0.0141, 0.0143, 0.0150, 0.0157, 0.0170, 0.0200, 0.0212]
pis = {"benign like fs2008 (1.28%)": 0.0128, "like fs2011 est. (1.43%)": 0.0143,
       "like fs2009 (1.62%)": 0.01615, "stressed like fs2010 (1.70%)": 0.01701,
       "worse: +25% on fs2010 (2.13%)": 0.01701 * 1.25, "benign: 1.0%": 0.010}
sc = []
for tau in taus:
    r = {"tau": tau}
    for name, pi in pis.items():
        r[name] = kl(pi, tau) * 1e4  # in 1e-4 log-loss units
    vals = [kl(pi, tau) for pi in pis.values()]
    core = [kl(pi, tau) for n, pi in pis.items() if n.startswith(("benign like", "like", "stressed"))]
    r["mean_regret_core4"] = np.mean(core) * 1e4
    r["max_regret_core4"] = np.max(core) * 1e4
    r["max_regret_all6"] = np.max(vals) * 1e4
    sc.append(r)
sc = pd.DataFrame(sc)
sc.to_csv(OUT / "scenario_regret.csv", index=False)
# continuous optimum over the core 4 scenarios
grid = np.linspace(0.010, 0.022, 1201)
core_pis = [0.0128, 0.0143, 0.01615, 0.01701]
mean_reg = np.array([np.mean([kl(p, q) for p in core_pis]) for q in grid])
max_reg = np.array([np.max([kl(p, q) for p in core_pis]) for q in grid])
opt = {"expected_loss_optimum_equal_weights": float(grid[mean_reg.argmin()]),
       "mean_of_core_rates": float(np.mean(core_pis)),
       "minimax_regret_optimum": float(grid[max_reg.argmin()]),
       "minimax_regret_value_1e4": float(max_reg.min() * 1e4)}
json.dump(opt, open(OUT / "scenario_optimum.json", "w"), indent=1)
print(asy.pivot(index="k_mean_pd_over_observed", columns="test_year", values="extra_vs_correct_level").mul(1e4).round(3))
print(sc.round(3).to_string())
print(opt)
