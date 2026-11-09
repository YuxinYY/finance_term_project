"""Calibration in 60 lines: why ranking and level are separate problems.

Run:  ~/miniconda3/envs/datascience/bin/python docs/2026-11-08_study_calibration_example.py
      (any Python with numpy, scipy, scikit-learn; ~5 s)

Story: a bank fits a PD model on year 1 (default rate ~0.9%) and uses it in
year 2, when defaults rise (~1.4%). Same firms' risk ordering, higher level.
Expected output (seed 0): raw obs/pred 1.66; prior shift to 1.5% -> 0.98;
Platt on year 1 -> 1.66 (no help); isotonic AUC 0.8409 vs 0.8422.
"""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss

rng = np.random.default_rng(0)

def make_year(n, intercept):
    x = rng.normal(size=(n, 2))                       # two "ratios", e.g. leverage, ROA
    true_logit = intercept + 1.2 * x[:, 0] - 0.8 * x[:, 1]
    y = rng.binomial(1, expit(true_logit))
    return x, y

x1, y1 = make_year(200_000, -5.75)   # year 1: the training year
x2, y2 = make_year(200_000, -5.2)   # year 2: the "test" year (worse cycle)
print(f"default rate year 1 {y1.mean():.2%}  year 2 {y2.mean():.2%}")

# 1) Fit on year 1 only. Its PDs are calibrated FOR YEAR 1.
model = LogisticRegression(C=1e6, max_iter=1000).fit(x1, y1)
pd_raw = model.predict_proba(x2)[:, 1]

def report(name, p):
    obs_pred = y2.mean() / p.mean()
    # calibration slope: regress y on logit(p); 1.0 = spread is right
    slope = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y2).coef_[0, 0]
    print(f"{name:<28} AUC {roc_auc_score(y2, p):.4f}  log loss {log_loss(y2, p):.5f}  "
          f"mean PD {p.mean():.2%}  obs/pred {obs_pred:.2f}  slope {slope:.2f}")

report("raw (fit on year 1)", pd_raw)

# 2) PRIOR SHIFT (Saerens et al. 2002 eq. 4): add one number to every logit.
#    delta = logit(target rate) - logit(rate the model was fit on).
#    The target must be chosen WITHOUT year-2 labels. Here: pretend the bank's
#    best guess for next year is 1.5% (in our project this is tau = 1.49%).
r_w, tau = y1.mean(), 0.015
delta = logit(tau) - logit(r_w)
pd_shift = expit(logit(pd_raw) + delta)
print(f"\nprior shift: r_w {r_w:.3%} -> tau {tau:.3%}, delta = {delta:+.3f}")
report("prior shift to tau=1.5%", pd_shift)

# 3) ORACLE: shift to the true year-2 rate (cheating; shows the ceiling).
pd_oracle = expit(logit(pd_raw) + logit(y2.mean()) - logit(r_w))
report("oracle shift (uses y2!)", pd_oracle)

# 4) PLATT fitted on a slice of YEAR 1 (no year-2 data). Changes slope+intercept.
#    Platt fitted on the SAME year as training cannot see the year-2 rise.
half = len(y1) // 2
m_a = LogisticRegression(C=1e6, max_iter=1000).fit(x1[:half], y1[:half])
s_b = logit(m_a.predict_proba(x1[half:])[:, 1]).reshape(-1, 1)
platt = LogisticRegression(C=1e6).fit(s_b, y1[half:])
pd_platt = platt.predict_proba(logit(pd_raw).reshape(-1, 1))[:, 1]
report("Platt on year-1 slice", pd_platt)

# 5) ISOTONIC on the same slice: a step function -> ties -> AUC can drop.
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
iso.fit(m_a.predict_proba(x1[half:])[:, 1], y1[half:])
report("isotonic on year-1 slice", iso.predict(pd_raw))

print("\nReadings:")
print(" - AUC is identical for raw / shift / Platt: monotone maps never change the ranking.")
print(" - Raw under-predicts year 2 (obs/pred > 1): a LEVEL problem, slope is already ~1.")
print(" - Platt fitted on year 1 cannot fix it: it only knows year 1's level.")
print(" - Only an explicit target (tau) moves the level; how good it is depends on the guess.")
print(" - Isotonic gains nothing here and makes ties (AUC a bit lower).")
