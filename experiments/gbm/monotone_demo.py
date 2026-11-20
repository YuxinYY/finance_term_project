"""Monotone constraint demo on synthetic data. Run: python monotone_demo.py"""
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0); n = 60000
# True model: only equity matters, more equity = safer. wc is a noisy twin of equity.
equity = rng.normal(0, 1, n)
wc = 0.9 * equity + np.sqrt(1 - .9**2) * rng.normal(0, 1, n)
p = 1 / (1 + np.exp(4.5 + 0.9 * equity))
y = rng.binomial(1, p)
X = np.c_[equity, wc]; tr = np.arange(n) < n // 2; te = ~tr
base = dict(objective="binary", num_leaves=31, learning_rate=0.05, min_data_in_leaf=20,
            verbose=-1, seed=0, monotone_constraints_method="intermediate")
def fit(mono):
    return lgb.train({**base, "monotone_constraints": mono},
                     lgb.Dataset(X[tr], y[tr], free_raw_data=False), 300)
def pd_curve(m, j):  # partial dependence of feature j on 1,000 test rows
    Z = X[te][:1000].copy(); out = []
    for v in np.linspace(-2, 2, 21):
        Z[:, j] = v; out.append(m.predict(Z).mean())
    return np.array(out)
print(f"defaults: {y.sum()} of {n} ({y.mean():.2%})")
for name, mono in [("unconstrained", [0, 0]), ("constrained (-,-)", [-1, -1])]:
    m = fit(mono); a = roc_auc_score(y[te], m.predict(X[te])); c = pd_curve(m, 1)
    print(f"{name:18s} test AUC {a:.4f} | wc curve: upward steps {(np.diff(c) > 1e-9).sum()} of 20, "
          f"PD at wc=-2 {c[0]:.3%}, at wc=+2 {c[-1]:.3%}")
# The guarantee for one firm: raising wc can never raise its PD under the constraint.
m = fit([-1, -1]); Z = np.tile(X[te][0], (41, 1)); Z[:, 1] = np.linspace(-3, 3, 41)
print("one firm, constrained: any PD rise as wc rises?", bool((np.diff(m.predict(Z)) > 1e-12).any()))
