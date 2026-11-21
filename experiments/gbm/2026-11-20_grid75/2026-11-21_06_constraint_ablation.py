"""Where does the unconstrained model's gain come from?

Winner setting (num_leaves 31, min_child_samples 3000, feature_fraction 0.5, bagging 1.0, lambda_l2 10),
full-data binning, seeds 1-2, rounds 1-3, early stopping on validation AUC. Configurations:
  mono      the 20-constraint table (D-157)
  free      no constraints
  rel_<f>   mono with the constraint on feature f released (20 configs)
  lit13     constrain only features with a DIRECT verified literature sign and no documented
            non-monotone shape (see 2026-11-21 sign report): 13 features
  mono_m3   mono with quick_ratio, int_coverage, operating_cash_flow_margin released (data-driven; for
            information only, selected on validation)
Validation metrics only.
"""
import importlib, time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from common import load, feature_list, OUT_DIR, ROUNDS

grid = importlib.import_module("2026-11-20_02_grid")
feats = feature_list()
base = pd.read_csv(OUT_DIR / "sign_table.csv").set_index("feature").loc[feats, "constraint"].astype(int).to_dict()
S = dict(num_leaves=31, min_child_samples=3000, feature_fraction=0.5, bagging_fraction=1.0, lambda_l2=10.0)
LIT13 = ["equity_ratio", "cash_ratio", "wc_ratio", "roa", "op_margin", "net_prof_margin", "payables_ratio",
         "financing_expense_ratio", "ebitda_to_debt", "operating_cash_flow_ratio", "operating_cash_flow_to_debt",
         "operating_cash_flow_margin", "loss_flag"]
configs = {"mono": dict(base), "free": {f: 0 for f in feats}}
for f, s in base.items():
    if s != 0:
        configs[f"rel_{f}"] = {**base, f: 0}
configs["lit13"] = {f: (base[f] if f in LIT13 else 0) for f in feats}
configs["mono_m3"] = {**base, "quick_ratio": 0, "int_coverage": 0, "operating_cash_flow_margin": 0}
assert all(base[f] != 0 for f in LIT13)

OUT = OUT_DIR / "ablation_results.csv"
done = set()
if OUT.exists():
    d = pd.read_csv(OUT); done = set(zip(d.config, d["round"], d.seed))
t0 = time.time()
for rnd in ROUNDS:
    tr, va = load(rnd, "train"), load(rnd, "validation")
    dsp = {"max_bin": 255, "feature_pre_filter": False, "bin_construct_sample_cnt": 10**7}
    ds_tr = lgb.Dataset(tr[feats], tr.label, params=dsp, free_raw_data=False).construct()
    ds_va = lgb.Dataset(va[feats], va.label, reference=ds_tr, free_raw_data=False).construct()
    for seed in (1, 2):
        for name, cons in configs.items():
            if (name, rnd, seed) in done:
                continue
            p = dict(grid.FIXED, **S, seed=seed)
            mono = [cons[f] for f in feats]
            if any(mono):
                p.update(monotone_constraints=mono, monotone_constraints_method="intermediate")
            m = lgb.train(p, ds_tr, num_boost_round=grid.MAX_ROUNDS, valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(grid.PATIENCE, first_metric_only=True, verbose=False)])
            pv = m.predict(va[feats], num_iteration=m.best_iteration)
            row = dict(config=name, round=rnd, seed=seed, n_constrained=sum(1 for v in mono if v),
                       auc=roc_auc_score(va.label, pv), logloss=log_loss(va.label, pv), best_iter=m.best_iteration)
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"{name} {rnd} s{seed}: AUC {row['auc']:.5f} LL {row['logloss']:.5f} ({(time.time()-t0)/60:.0f} min)", flush=True)

d = pd.read_csv(OUT)
g = d.groupby("config").agg(auc=("auc", "mean"), logloss=("logloss", "mean"), n=("auc", "size"),
                            n_constrained=("n_constrained", "first"))
m = g.loc["mono"]
g["dAUC_vs_mono"] = g.auc - m.auc
g["dLL_vs_mono"] = g.logloss - m.logloss
g.sort_values("dAUC_vs_mono", ascending=False).to_csv(OUT_DIR / "ablation_summary.csv")
print(g.sort_values("dAUC_vs_mono", ascending=False).round(5).to_string())
