"""Stage 2 (calibration-specialist): the default-rate series and the central-tendency rules.

Uses panel labels only (no model). Writes reports/calibration/:
  rates_by_cohort.csv      : default rate per fs_year cohort for y (start 6), y_s0, y_s3
  monthly_hazard.csv       : calendar monthly default hazard from the start-6 cohorts (each cohort's
                             window covers Jul t+1 .. Jun t+2, so the cohorts tile calendar time)
  cohort2011_partial.csv   : Q6, what the 2011 cohort (11 of 12 months observed) says
  ct_targets.csv           : the central-tendency candidates per fold and label, from training-
                             window labels only, and their level-only cost (Bernoulli KL) in the
                             test year
  fig/monthly_hazard.png
Timing (what a bank knows): under start 6, the label of fs_year t is complete on 30 Jun t+2, and a
decision with an fs_year t+1 statement is made from about 30 Jun t+2. So at decision time for test
year t+1 the bank has seen every default up to 30 Jun t+2 and NOTHING of the test window
(30 Jun t+2 .. 30 Jun t+3). A "partially observed next-year rate" is therefore not available (Q1e).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/calibration"
(OUT / "fig").mkdir(parents=True, exist_ok=True)
P = pd.read_parquet(ROOT / "data/interim/panel.parquet",
                    columns=["row_id", "id", "fs_year", "stmt_date", "def_date", "y", "label_status",
                             "win_start", "win_end", "y_s0", "status_s0", "y_s3", "status_s3"])
P["fs_year"] = P["fs_year"].astype(int)
DATA_END = pd.Timestamp("2013-05-31")


def logit(p):
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def kl(pi, q):
    """Expected extra log loss per row from predicting level q when the true rate is pi (Bernoulli KL)."""
    return pi * np.log(pi / q) + (1 - pi) * np.log((1 - pi) / (1 - q))


# ---------------------------------------------------------------- 1. cohort rates
LABELS = {"s6": ("y", "label_status"), "s0": ("y_s0", "status_s0"), "s3": ("y_s3", "status_s3")}
rows = []
for lab, (yc, sc) in LABELS.items():
    ok = P[P[sc] == "ok"]
    for t, g in ok.groupby("fs_year"):
        rows.append({"label": lab, "fs_year": t, "n": len(g), "defaults": int(g[yc].sum()),
                     "rate": g[yc].mean()})
rates = pd.DataFrame(rows)

# ---------------------------------------------------------------- 2. monthly hazard (start-6 cohorts)
# at-risk rows of cohort t in month j of its window: rows alive at window start (ok or censored, i.e.
# not already_defaulted) and not yet defaulted before month j. Cohort 2011 is observed to 2013-05-31.
haz = []
for t in range(2007, 2012):
    c = P[(P.fs_year == t) & (P.label_status.isin(["ok", "censored"]))]
    ws = c["win_start"].iloc[0]
    for j in range(12):
        m0, m1 = ws + pd.DateOffset(months=j), ws + pd.DateOffset(months=j + 1)
        if m0 > DATA_END:
            break
        at_risk = (c.def_date.isna() | (c.def_date >= m0)).sum()
        d = ((c.def_date >= m0) & (c.def_date < min(m1, DATA_END + pd.Timedelta(days=1)))).sum()
        haz.append({"cohort": t, "month_in_window": j + 1, "month_start": m0.date(), "at_risk": int(at_risk),
                    "defaults": int(d), "hazard": d / at_risk,
                    "complete_month": bool(m1 <= DATA_END + pd.Timedelta(days=1))})
haz = pd.DataFrame(haz)
haz.to_csv(OUT / "monthly_hazard.csv", index=False)

# ---------------------------------------------------------------- 3. 2011 cohort (Q6)
c11 = P[(P.fs_year == 2011) & (P.label_status.isin(["ok", "censored"]))]
ws11, we11 = c11.win_start.iloc[0], c11.win_end.iloc[0]
d11 = ((c11.def_date >= ws11) & (c11.def_date <= DATA_END)).sum()
n11 = len(c11)
# share of a complete cohort's 12-month defaults that fall in the first 11 months (seasonality-aware)
share = []
for t in range(2007, 2011):
    c = P[(P.fs_year == t) & (P.label_status == "ok")]
    ws = c.win_start.iloc[0]
    cut = ws + pd.DateOffset(months=11)
    share.append({"cohort": t, "defaults_12m": int(c.y.sum()),
                  "defaults_first_11m": int(((c.def_date >= ws) & (c.def_date < cut)).sum())})
share = pd.DataFrame(share)
share["share_11_of_12"] = share.defaults_first_11m / share.defaults_12m
s_mean, s_min, s_max = share.share_11_of_12.mean(), share.share_11_of_12.min(), share.share_11_of_12.max()
# alternative scaling: 12/11
part = pd.DataFrame([
    {"item": "cohort 2011 rows alive at window start (ok+censored)", "value": n11},
    {"item": "window", "value": f"{ws11.date()} to {we11.date()} (observed to {DATA_END.date()})"},
    {"item": "defaults observed in first 11 months", "value": int(d11)},
    {"item": "11-month rate", "value": d11 / n11},
    {"item": "12-month rate, scaled 12/11", "value": d11 / n11 * 12 / 11},
    {"item": "12-month rate, scaled by mean 11-of-12 share of cohorts 2007-2010", "value": d11 / n11 / s_mean},
    {"item": "  range using min/max share", "value": f"{d11 / n11 / s_max:.5f} .. {d11 / n11 / s_min:.5f}"},
    {"item": "mean 11-of-12 share (2007-2010 cohorts)", "value": s_mean},
    {"item": "lower bound: treat unobserved month as zero defaults (keep-as-0)", "value": d11 / n11},
    {"item": "cohort 2010 12-month rate (complete)", "value": rates.query("label=='s6' and fs_year==2010").rate.iloc[0]},
])
# first-11-months comparison on the same months (like-for-like, no scaling)
r11 = []
for t in range(2007, 2012):
    c = P[(P.fs_year == t) & (P.label_status.isin(["ok", "censored"]))]
    ws = c.win_start.iloc[0]
    cut = ws + pd.DateOffset(months=11)
    r11.append({"cohort": t, "n": len(c), "defaults_first_11m": int(((c.def_date >= ws) & (c.def_date < cut)).sum())})
r11 = pd.DataFrame(r11)
r11["rate_first_11m"] = r11.defaults_first_11m / r11.n
part.to_csv(OUT / "cohort2011_partial.csv", index=False)
share.to_csv(OUT / "cohort_share_11_of_12.csv", index=False)
r11.to_csv(OUT / "cohort_rate_first_11_months.csv", index=False)

# ---------------------------------------------------------------- 4. central-tendency rules
def rules(yr_rates: pd.Series, pooled: float, half_year: float | None) -> dict:
    """yr_rates: index = training fs_years, values = default rate. Training-window information only."""
    yrs = np.array(yr_rates.index, float)
    z = logit(yr_rates.to_numpy())
    out = {"a_pooled": pooled, "b_last_year": yr_rates.iloc[-1],
           "d_ttc_equal_weight": float(yr_rates.mean())}
    if len(yrs) >= 2:
        b1, b0 = np.polyfit(yrs, z, 1)
        out["c_trend_logit"] = float(expit(b0 + b1 * (yrs[-1] + 1)))
        out["c2_half_trend"] = float(expit(z[-1] + 0.5 * (b0 + b1 * (yrs[-1] + 1) - z[-1])))
        out["f_mean_last2"] = float(yr_rates.iloc[-2:].mean())
    else:  # one training year: no trend can be estimated; fall back to the only rate
        out["c_trend_logit"] = out["c2_half_trend"] = out["f_mean_last2"] = out["b_last_year"]
    if half_year is not None:
        out["e_last_6m_annualised"] = half_year
    return out


ct = []
for lab, (yc, sc) in LABELS.items():
    ok = P[P[sc] == "ok"]
    yr = ok.groupby("fs_year")[yc].mean()
    test_years = [t for t in yr.index if t >= 2008]
    for t in test_years:
        tr = ok[ok.fs_year < t]
        hy = None
        if lab == "s6":  # last 6 months of the last training cohort's window: Jan..Jun of year t+1
            last = haz[haz.cohort == t - 1]
            h = last[last.month_in_window > 6]
            hy = float(1 - np.prod(1 - h.hazard.to_numpy()) ** 1) * 2 if len(h) == 6 else None
        r = rules(yr.loc[yr.index < t], float(tr[yc].mean()), hy)
        pi = float(yr.loc[t])
        for k, v in r.items():
            ct.append({"label": lab, "test_year": t, "rule": k, "target": v, "observed": pi,
                       "obs_over_target": pi / v, "level_kl": kl(pi, v)})
ct = pd.DataFrame(ct)
ct.to_csv(OUT / "ct_targets.csv", index=False)
summ = ct.groupby(["label", "rule"]).agg(mean_level_kl=("level_kl", "mean"), max_level_kl=("level_kl", "max"),
                                         mean_abs_log_ratio=("obs_over_target", lambda s: np.mean(np.abs(np.log(s))))).reset_index()
summ.to_csv(OUT / "ct_rules_summary.csv", index=False)

rates.to_csv(OUT / "rates_by_cohort.csv", index=False)

# ---------------------------------------------------------------- figure: calendar monthly hazard
fig, ax = plt.subplots(figsize=(9, 3.6))
h = haz.copy()
h["ms"] = pd.to_datetime(h.month_start)
for t, g in h.groupby("cohort"):
    ax.plot(g.ms, g.hazard * 1e4, marker="o", ms=3, label=f"fs_year {t} cohort")
ax.set_ylabel("monthly default hazard (bp)")
ax.set_title("Monthly default hazard in calendar time (start-6 label windows; 2011 cohort ends 2013-05)")
ax.legend(fontsize=7, ncol=5)
fig.tight_layout()
fig.savefig(OUT / "fig/monthly_hazard.png", dpi=130)

print(rates.pivot(index="fs_year", columns="label", values="rate").round(5))
print(part.to_string())
print(share.to_string()); print(r11.to_string())
print(ct.pivot_table(index=["label", "test_year"], columns="rule", values="target").round(5).to_string())
print(summ.to_string())
