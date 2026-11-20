"""Build the v3 walk-forward feature files (company_random validation + FY2011 lockbox) for the Colab
feature-importance notebook, using Chloe and Desmond's own preprocessing code.

Design (proposed in DECISION_LOG D-155):
  round 1: train FY2007 (80% of firms)      | early-stopping firms: 20% of FY2007 firms     | test FY2008
  round 2: train FY2007-08 (80% of firms)   | 20% of firms in FY2007-08                     | test FY2009
  round 3: train FY2007-09 (80% of firms)   | 20% of firms in FY2007-09                     | test FY2010
  round 4: train FY2007-10 (80% of firms)   | 20% of firms in FY2007-10                     | LOCKBOX FY2011
  final  : all labelled rows FY2007-11, no validation (tree count comes from round 4's early stopping)

Every learned statistic (sector medians, industry target encoding, legal/region levels) is fitted on the
round's TRAIN rows only (`fit_preprocessor`) and applied to the early-stopping firms, the test year and the
lockbox with `transform_with_preprocessor`. This is the fix for audit finding A1.

Imputation Method G (previous-year zero -> zero) is ON for training rows (as in the team's notebook) and OFF
for every transformed split, so each evaluation row uses only its own record, exactly like the harness
(harness/2026-11-16_turquoise_submission/missing_rules.py, D-152).

Labels: the s5 label from data/interim/2026-11-08_panel_s5.parquet (status 'ok' rows only). The audit
(reports/data/2026-11-16_audit, round 5) found it identical to Chloe's `label` column on 100% of rows.

Usage:
  python src/data/2026-11-18_build_colab_splits.py --team-repo <checkout of teammate/project-update @ 4e0ba3a> \
      --out ~/Downloads/2026-11-18_turquoise_colab
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TEAM_COMMIT = "4e0ba3a"
SEED = 42


def _load_harness_fill_missing():
    path = ROOT / "harness/2026-11-16_turquoise_submission/missing_rules.py"
    sys.path.insert(0, str(path.parent))  # missing_rules imports bundle_paths from its own folder
    spec = importlib.util.spec_from_file_location("harness_missing_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fill_missing


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-repo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--validation-size", type=float, default=0.2)
    args = ap.parse_args()
    team = args.team_repo.resolve()
    out = args.out.expanduser().resolve()
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data/LOCKBOX_do_not_open").mkdir(exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)

    commit = subprocess.run(["git", "-C", str(team), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    if commit != EXPECTED_TEAM_COMMIT:
        raise SystemExit(f"team repo is at {commit}, expected {EXPECTED_TEAM_COMMIT}")

    sys.path.insert(0, str(team / "src"))
    from finance_project.data_processing import preprocess as pp
    from finance_project.data_processing.make_walk_forward_splits import make_walk_forward_splits

    harness_fill = _load_harness_fill_missing()
    mode = {"use_firm_history": True}
    pp.fill_missing = lambda df, medians: harness_fill(df, medians, use_firm_history=mode["use_firm_history"])
    cfg = pp.PreprocessingConfig(feature_mode="firm_year")  # same config as the submitted harness

    # ---- labelled input ------------------------------------------------------------------------
    raw = pd.read_csv(ROOT / "data/raw/train-2025.csv", low_memory=False)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")] + ["def_date"])
    panel = pd.read_parquet(ROOT / "data/interim/2026-11-08_panel_s5.parquet",
                            columns=["id", "fs_year", "y_s5", "status_s5"])
    ok = panel[panel.status_s5 == "ok"][["id", "fs_year", "y_s5"]].rename(columns={"y_s5": "label"})
    labelled = raw.merge(ok, on=["id", "fs_year"], how="inner", validate="one_to_one")
    labelled["label"] = labelled["label"].astype(int)
    print(f"labelled rows {len(labelled):,}, default rate {labelled.label.mean():.5f}")

    splits = make_walk_forward_splits(labelled, validation_strategy="company_random", rounds=4,
                                      validation_size=args.validation_size, random_state=SEED,
                                      first_train_year=2007)

    manifest = {"created": "2026-11-18", "decision": "D-155 (OPEN)",
                "team_code": f"teammate/project-update @ {commit} (preprocess.py, make_walk_forward_splits.py)",
                "preprocess_sha256": _sha(team / "src/finance_project/data_processing/preprocess.py"),
                "label": "s5 (data/interim/2026-11-08_panel_s5.parquet, status ok) = Chloe's label (audit r5)",
                "feature_mode": "firm_year", "method_G": "on for train rows, off for transformed rows",
                "validation": f"company_random, {args.validation_size:.0%} of firm ids in the training window",
                "rounds": []}

    def save(frame: pd.DataFrame, name: str, lockbox: bool = False) -> str:
        sub = "data/LOCKBOX_do_not_open" if lockbox else "data"
        path = out / sub / f"{name}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        return str(path.relative_to(out))

    def te_check(frame: pd.DataFrame, art) -> dict:
        allowed = set(np.round(art.industry_mapping.values, 12)) | {round(art.industry_global_mean, 12)}
        return {"mean_te": float(frame.ateco_sector_te.mean()),
                "share_te_from_frozen_mapping": float(np.isin(np.round(frame.ateco_sector_te.values, 12),
                                                              list(allowed)).mean())}

    for s in splits:
        t0 = time.time()
        r = s.round_number
        lockbox = s.test_year == 2011
        mode["use_firm_history"] = True
        train_f, art = pp.fit_preprocessor(s.train, s.history, cfg)
        mode["use_firm_history"] = False
        val_f = pp.transform_with_preprocessor(s.validation, s.history, art)
        test_f = pp.transform_with_preprocessor(s.test, s.history, art)
        joblib.dump(art, out / f"artifacts/round_{r:02d}_preprocessor.joblib")
        test_name = f"round_{r:02d}_LOCKBOX_FY2011" if lockbox else f"round_{r:02d}_test"
        rec = {"round": r, "role": "lockbox (score once, D-087/D-119)" if lockbox else "model selection",
               "train_years": list(s.train_years), "test_year": s.test_year,
               "files": {"train": save(train_f, f"round_{r:02d}_train"),
                         "valid_es": save(val_f, f"round_{r:02d}_valid_es"),
                         "test": save(test_f, test_name, lockbox)}}
        for k, raw_df, f in [("train", s.train, train_f), ("valid_es", s.validation, val_f), ("test", s.test, test_f)]:
            rec[f"{k}_rows_in"] = int(len(raw_df))
            rec[f"{k}_rows_out"] = int(len(f))
            rec[f"{k}_default_rate"] = None if (lockbox and k == "test") else float(f.label.mean())
            rec[f"{k}_firms"] = int(f.id.nunique())
        rec["train_valid_firm_overlap"] = len(set(train_f.id) & set(val_f.id))
        rec["leak_check_valid_es"] = te_check(val_f, art)
        rec["leak_check_test"] = te_check(test_f, art)
        rec["train_rate_global_te_mean"] = float(art.industry_global_mean)
        rec["seconds"] = round(time.time() - t0, 1)
        manifest["rounds"].append(rec)
        print(json.dumps(rec, indent=1))

    # ---- final fit on all labelled rows FY2007-2011 ----------------------------------------------
    mode["use_firm_history"] = True
    final_f, final_art = pp.fit_preprocessor(labelled, labelled, cfg)
    joblib.dump(final_art, out / "artifacts/final_preprocessor.joblib")
    manifest["final"] = {"train_years": [2007, 2008, 2009, 2010, 2011], "rows_in": int(len(labelled)),
                         "rows_out": int(len(final_f)), "default_rate": float(final_f.label.mean()),
                         "file": save(final_f, "final_train_FY2007_2011")}
    feature_cols = [c for c in final_f.columns if c not in ("id", "fs_year", "label", "pred_start_dt")]
    manifest["feature_columns"] = feature_cols
    manifest["n_features"] = len(feature_cols)
    schema = json.loads((team / "config/feature_schema.json").read_text())
    manifest["feature_groups"] = schema["feature_groups"]
    (out / "data/manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    split_def = ROOT / "data/splits/2026-11-18_walkforward_v3_company_random.json"
    if not split_def.exists():  # frozen: written once
        ids = {f"round_{s.round_number:02d}_valid_es_firm_ids": sorted(map(int, s.validation.id.unique()))
               for s in splits}
        split_def.write_text(json.dumps({"manifest": manifest, **ids}, default=str))
    print("done:", out)


if __name__ == "__main__":
    main()
