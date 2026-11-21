"""Estimation step 2b: alternative final models (same frozen preprocessing, data, setting and calibration recipe as
2026-11-21_02_train_final.py; only the monotone constraints and the tree count differ), each packaged as its own
harness bundle so any of them can be swapped in.

Variants (constraints from experiments/gbm/2026-11-20_grid75/sign_table.csv, D-157 / D-159):
  mono20      full monotone, 20 constraints (D-157)                           trees 240
  lit13       constrain only the 13 direct-literature features (D-159, lit13) trees 200
  mono19      mono20 with cur_debt_ratio released (optional tweak, D-159)     trees 310
Tree count rule (same as the unconstrained final model): mean of the round-3 best iterations of that configuration
over the refit and seed runs (mono20: 232, 214, 245, 260; lit13: 189, 205; mono19: 301, 322), rounded to 10.
Outputs: variants/<name>/{lightgbm.txt, params.json, insample_summary.json} and
harness/2026-11-21_variants/<name>/ + harness/2026-11-21_variants/2026-11-21_Turquoise_harness_<name>.zip
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor_7949207" / "src"))
from finance_project.data_processing import preprocess  # noqa: E402,F401

REPO = HERE.parents[2]
BASE_BUNDLE = REPO / "harness/2026-11-21_turquoise_submission"
OUT_BUNDLES = REPO / "harness/2026-11-21_variants"
SIGN = pd.read_csv(REPO / "experiments/gbm/2026-11-20_grid75/sign_table.csv").set_index("feature")
LIT13 = ["equity_ratio", "cash_ratio", "wc_ratio", "roa", "op_margin", "net_prof_margin", "payables_ratio",
         "financing_expense_ratio", "ebitda_to_debt", "operating_cash_flow_ratio", "operating_cash_flow_to_debt",
         "operating_cash_flow_margin", "loss_flag"]
base_params = json.loads((HERE / "final_params.json").read_text())
FEATS = base_params["feature_columns"]
mono20 = {f: int(SIGN.loc[f, "constraint"]) for f in FEATS}
VARIANTS = {
    "mono20": (mono20, 240, "full monotone, 20 constraints (D-157)"),
    "lit13": ({f: (mono20[f] if f in LIT13 else 0) for f in FEATS}, 200, "13 direct-literature constraints (lit13, D-159)"),
    "mono19": ({**mono20, "cur_debt_ratio": 0}, 310, "full monotone with cur_debt_ratio released (19 constraints)"),
}
RUNTIME = ["harness.py", "inference.py", "missing_rules.py", "anomaly_rules.py", "feature_rules.py",
           "bundle_paths.py", "README.md", "requirements.txt", "config/feature_schema.json",
           "data/interim/ateco_sector_english.csv", "artifacts/lightgbm.txt", "artifacts/preprocessing.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    df = pd.read_parquet(HERE / "final_train_features.parquet")
    y = df.label.to_numpy(); rate = float(y.mean())
    cal = base_params["calibration"]
    ds = lgb.Dataset(df[FEATS], y, params={"max_bin": 255, "feature_pre_filter": False, "bin_construct_sample_cnt": 10**7})
    rows = []
    for name, (cons, trees, desc) in VARIANTS.items():
        t0 = time.time()
        mono = [cons[f] for f in FEATS]
        p = dict(base_params["model"]["params"], monotone_constraints=mono, monotone_constraints_method="intermediate")
        b = lgb.train(p, ds, num_boost_round=trees)
        d = HERE / "variants" / name; d.mkdir(parents=True, exist_ok=True)
        b.save_model(str(d / "lightgbm.txt"), num_iteration=trees)
        p_raw = b.predict(df[FEATS], num_iteration=trees)
        lo = logit(np.clip(p_raw, 1e-6, 1 - 1e-6))
        a = brentq(lambda a: expit(a + lo).mean() - rate, -3, 3)
        pd_final = np.clip(expit(a + lo + cal["delta"]), cal["floor"], cal["cap"])
        params = json.loads(json.dumps(base_params))
        params["best_iteration"] = trees
        params["model_feature_names"] = b.feature_name()
        params["calibration"]["anchor_a"] = float(a)
        params["model"].update(params=p, n_trees=trees, variant=name, description=desc,
                               monotone_constraints={f: c for f, c in cons.items() if c != 0})
        (d / "params.json").write_text(json.dumps(params, indent=2, allow_nan=False) + "\n")
        summ = dict(variant=name, description=desc, n_constraints=int(sum(1 for c in mono if c)), trees=trees,
                    anchor_a=float(a), mean_raw_pd=float(p_raw.mean()), mean_final_pd=float(pd_final.mean()),
                    insample_auc=float(roc_auc_score(y, p_raw)), insample_logloss_raw=float(log_loss(y, p_raw)),
                    min_pd=float(pd_final.min()), max_pd=float(pd_final.max()), seconds=round(time.time() - t0, 1))
        (d / "insample_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
        rows.append(summ)
        # harness bundle: same runtime files as the main bundle, this variant's model + params
        bd = OUT_BUNDLES / name
        if bd.exists():
            shutil.rmtree(bd)
        shutil.copytree(BASE_BUNDLE, bd, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(d / "lightgbm.txt", bd / "artifacts/lightgbm.txt")
        (bd / "artifacts/preprocessing.json").write_text(json.dumps(params, indent=2, allow_nan=False) + "\n")
        readme = (bd / "README.md").read_text()
        readme = readme.replace("# Team Turquoise: PD harness (test bundle, 2026-11-21)",
                                f"# Team Turquoise: PD harness (test bundle, 2026-11-21) - VARIANT {name}\n\n"
                                f"**Model variant: {desc}, {trees} trees.** Everything else (preprocessing, calibration "
                                "recipe, runtime code, requirements) is identical to the main (unconstrained) bundle; "
                                "statements below about 'unconstrained' and '210 trees' refer to the main bundle.")
        (bd / "README.md").write_text(readme)
        man = json.loads((BASE_BUNDLE / "artifacts/manifest.json").read_text())
        man["model"] = f"LightGBM, {desc}, {trees} trees, 75 features, fit on all labelled FY2007-2011 rows (835,016)"
        man["bundle_sha256"] = {r: sha(bd / r) for r in RUNTIME}
        (bd / "artifacts/manifest.json").write_text(json.dumps(man, indent=2) + "\n")
        z = OUT_BUNDLES / f"2026-11-21_Turquoise_harness_{name}.zip"
        with ZipFile(z, "w", compression=ZIP_DEFLATED) as zf:
            for r in RUNTIME + ["artifacts/manifest.json"]:
                zf.write(bd / r, r)
        print(json.dumps(summ), flush=True)
    pd.DataFrame(rows).to_csv(HERE / "variants" / "variants_summary.csv", index=False)


if __name__ == "__main__":
    main()
