"""Estimation step 2c: mono20 without goodwill_ratio (74 features), D-164.

Why: the raw field `goodwill` behaves like value added (EBITDA + personnel cost), not accounting goodwill
(D-010, D-091), so goodwill_ratio cannot be explained to a credit officer under its name; KN decision 2026-11-21.
Cost in the walk-forward (experiments/gbm/2026-11-20_grid75/2026-11-21_11_drop_goodwill.py): validation AUC
-0.00054 (bootstrap intervals include 0 in every round), test AUC +0.00001.

Everything else identical to mono20 (2026-11-21_06_train_variants.py): same frozen preprocessing (the feature is
still computed by the team code but no longer selected), same data, setting, 20 constraints, calibration recipe.
Tree count rule (as for every final model): mean round-3 best iteration of this configuration (243, 216) -> 230.
The harness selects columns by params["feature_columns"], so only the model and params change.
Outputs: variants/mono20_nogw/ and harness/2026-11-21_variants/mono20_nogw/ + zip
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NAME, TREES, DROP = "mono20_nogw", 230, "goodwill_ratio"
DESC = "full monotone (20 constraints, D-157) without goodwill_ratio, 74 features (D-164)"
BASE_BUNDLE = REPO / "harness/2026-11-21_variants/mono20"
OUT_BUNDLE = REPO / "harness/2026-11-21_variants" / NAME
RUNTIME = ["harness.py", "inference.py", "missing_rules.py", "anomaly_rules.py", "feature_rules.py",
           "bundle_paths.py", "README.md", "requirements.txt", "config/feature_schema.json",
           "data/interim/ateco_sector_english.csv", "artifacts/lightgbm.txt", "artifacts/preprocessing.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    t0 = time.time()
    base = json.loads((HERE / "variants/mono20/params.json").read_text())
    feats = [f for f in base["feature_columns"] if f != DROP]
    assert len(feats) == 74
    old_mono = dict(zip(base["feature_columns"], base["model"]["params"]["monotone_constraints"]))
    mono = [old_mono[f] for f in feats]
    assert sum(1 for c in mono if c) == 20
    df = pd.read_parquet(HERE / "final_train_features.parquet")
    y = df.label.to_numpy(); rate = float(y.mean())
    ds = lgb.Dataset(df[feats], y, params={"max_bin": 255, "feature_pre_filter": False, "bin_construct_sample_cnt": 10**7})
    p = dict(base["model"]["params"], monotone_constraints=mono)
    b = lgb.train(p, ds, num_boost_round=TREES)
    d = HERE / "variants" / NAME; d.mkdir(parents=True, exist_ok=True)
    b.save_model(str(d / "lightgbm.txt"), num_iteration=TREES)
    p_raw = b.predict(df[feats], num_iteration=TREES)
    lo = logit(np.clip(p_raw, 1e-6, 1 - 1e-6))
    a = brentq(lambda a: expit(a + lo).mean() - rate, -3, 3)
    cal = base["calibration"]
    pd_final = np.clip(expit(a + lo + cal["delta"]), cal["floor"], cal["cap"])
    params = json.loads(json.dumps(base))
    params["feature_columns"] = feats
    params["model_feature_names"] = b.feature_name()
    params["best_iteration"] = TREES
    params["calibration"]["anchor_a"] = float(a)
    params["model"].update(params=p, n_trees=TREES, variant=NAME, description=DESC,
                           monotone_constraints={f: c for f, c in zip(feats, mono) if c != 0},
                           dropped_features=[DROP])
    (d / "params.json").write_text(json.dumps(params, indent=2, allow_nan=False) + "\n")
    summ = dict(variant=NAME, description=DESC, n_features=len(feats), n_constraints=20, trees=TREES, anchor_a=float(a),
                mean_raw_pd=float(p_raw.mean()), mean_final_pd=float(pd_final.mean()),
                insample_auc=float(roc_auc_score(y, p_raw)), insample_logloss_raw=float(log_loss(y, p_raw)),
                min_pd=float(pd_final.min()), max_pd=float(pd_final.max()), seconds=round(time.time() - t0, 1))
    (d / "insample_summary.json").write_text(json.dumps(summ, indent=2) + "\n")

    if OUT_BUNDLE.exists():
        shutil.rmtree(OUT_BUNDLE)
    shutil.copytree(BASE_BUNDLE, OUT_BUNDLE, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(d / "lightgbm.txt", OUT_BUNDLE / "artifacts/lightgbm.txt")
    (OUT_BUNDLE / "artifacts/preprocessing.json").write_text(json.dumps(params, indent=2, allow_nan=False) + "\n")
    readme = (OUT_BUNDLE / "README.md").read_text()
    readme = readme.replace("VARIANT mono20", f"VARIANT {NAME}")
    readme = readme.replace("**Model variant: full monotone, 20 constraints (D-157), 240 trees.**",
                            f"**Model variant: {DESC}, {TREES} trees.** goodwill_ratio is still computed by the team "
                            "feature code but is not a model input.")
    readme = readme.replace("p = 75 features and T = 210 trees", f"p = 74 model features and T = {TREES} trees")
    readme = readme.replace("the 75 model features", "the 74 model features")
    readme = readme.replace("ordered 75 features", "ordered 74 model features")
    (OUT_BUNDLE / "README.md").write_text(readme)
    man = json.loads((BASE_BUNDLE / "artifacts/manifest.json").read_text())
    man["model"] = f"LightGBM, {DESC}, {TREES} trees, fit on all labelled FY2007-2011 rows (835,016)"
    man["feature_count"] = 74
    man["bundle_sha256"] = {r: sha(OUT_BUNDLE / r) for r in RUNTIME}
    (OUT_BUNDLE / "artifacts/manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    z = OUT_BUNDLE.parent / f"2026-11-21_Turquoise_harness_{NAME}.zip"
    with ZipFile(z, "w", compression=ZIP_DEFLATED) as zf:
        for r in RUNTIME + ["artifacts/manifest.json"]:
            zf.write(OUT_BUNDLE / r, r)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
