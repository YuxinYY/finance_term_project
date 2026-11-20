"""Build the v4 walk-forward feature files: Chloe's temporal_year design + FY2011 lockbox (D-156).

    round | train      | validation (early stopping only) | test   | role
    1     | FY2007     | FY2008                           | FY2009 | model selection
    2     | FY2007-08  | FY2009                           | FY2010 | model selection
    3     | FY2007-09  | FY2010                           | FY2011 | LOCKBOX (written to LOCKBOX_do_not_open/)
    final | FY2007-11 (all labelled rows) | none            | -      | harness model

Starts from src/data/2026-11-18_build_colab_splits.py (agent kit, v3 / D-155) with these changes:
  * split = the team's make_walk_forward_splits(validation_strategy="temporal_year", rounds=3);
  * the team's preprocess.py is used unmodified (feature schema 1.3.0, commit 6bf455a). Method G no
    longer exists in its fill_missing (notebooks/features/02_missing_values.ipynb), so every transformed
    row uses only its own record; the v3 monkeypatch to the harness fill_missing is removed (D-157);
  * fit uses the round's TRAINING rows as both anchors and history (no validation/test row is passed
    to fit at all). transform is called with history = the split itself; in firm_year mode features
    are row-local, which the script verifies by re-transforming a random subset as a separate batch.

Every learned statistic (method B industry-group medians and the 3-smallest-group fallback, industry
target encoding, legal-structure and region categories) is fitted on the round's training years only.

Label: s5 (D-151), data/interim/2026-11-08_panel_s5.parquet, status_s5 == "ok" only.

Usage (from the repo root, python with pandas/sklearn/pyarrow/joblib):
  python src/data/2026-11-20_build_v4_temporal_year.py
Outputs:
  data/processed/2026-11-20_v4_temporal_year/{round_0r_{train,valid,test}.parquet, final_fit.parquet,
      LOCKBOX_do_not_open/round_03_test.parquet, artifacts/*.joblib}
  data/splits/2026-11-20_walkforward_v4_temporal_year.json   (written once, chmod 444; D-015)
  reports/data/2026-11-20_v4_checks.md
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from finance_project.data_processing import preprocess as pp  # noqa: E402
from finance_project.data_processing.make_walk_forward_splits import make_walk_forward_splits  # noqa: E402

TAG = "2026-11-20_v4_temporal_year"
OUT = ROOT / "data/processed" / TAG
LOCK = OUT / "LOCKBOX_do_not_open"
ART = OUT / "artifacts"
SPLIT_JSON = ROOT / "data/splits/2026-11-20_walkforward_v4_temporal_year.json"
CHECKS_MD = ROOT / "reports/data/2026-11-20_v4_checks.md"
RAW = ROOT / "data/raw/train-2025.csv"
PANEL = ROOT / "data/interim/2026-11-08_panel_s5.parquet"
LOCKBOX_YEAR = 2011
SEED = 42
META = ["id", "fs_year", "label"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def expected_static_features() -> list[str]:
    s = pp.FEATURE_SCHEMA
    return [*s["period_numeric_features"], *s["period_binary_features"], *s["mode_features"]["firm_year"],
            "ateco_sector_te"]


def te_check(frame: pd.DataFrame, art) -> dict:
    allowed = np.array(sorted(set(np.round(art.industry_mapping.values, 12)) | {round(art.industry_global_mean, 12)}))
    te = np.round(frame["ateco_sector_te"].to_numpy(), 12)
    return {"mean_te": float(te.mean()),
            "share_te_from_frozen_mapping": float(np.isin(te, allowed).mean()),
            "share_te_global_mean_fallback": float((te == round(art.industry_global_mean, 12)).mean())}


def main() -> None:
    if SPLIT_JSON.exists():
        raise SystemExit(f"{SPLIT_JSON} exists; split files are written once (D-015).")
    for d in (OUT, LOCK, ART, CHECKS_MD.parent):
        d.mkdir(parents=True, exist_ok=True)

    # ---- labelled input ------------------------------------------------------------------------
    raw = pd.read_csv(RAW, low_memory=False)
    raw_rows = len(raw)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")] + ["def_date"])
    panel = pd.read_parquet(PANEL, columns=["id", "fs_year", "y_s5", "status_s5"])
    ok = panel[panel.status_s5 == "ok"][["id", "fs_year", "y_s5"]].rename(columns={"y_s5": "label"})
    labelled = raw.merge(ok, on=["id", "fs_year"], how="inner", validate="one_to_one")
    labelled["label"] = labelled["label"].astype(int)
    print(f"raw rows {raw_rows:,}; labelled rows {len(labelled):,}; default rate {labelled.label.mean():.5f}")

    splits = make_walk_forward_splits(labelled, validation_strategy="temporal_year", rounds=3,
                                      first_train_year=2007)
    cfg = pp.PreprocessingConfig(feature_mode="firm_year")
    static = expected_static_features()

    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    manifest: dict = {
        "created": "2026-11-20", "decision": "D-156 (ADOPTED, KN 2026-11-20; supersedes D-155)",
        "builder": rel(Path(__file__).resolve()), "repo_commit": commit,
        "code_sha256": {p: sha(ROOT / p) for p in [
            "src/finance_project/data_processing/preprocess.py",
            "src/finance_project/data_processing/make_walk_forward_splits.py",
            "notebooks/features/01_flag_anomalies.ipynb", "notebooks/features/02_missing_values.ipynb",
            "config/feature_schema.json"]},
        "feature_schema": pp.feature_schema_metadata() | {"path": "config/feature_schema.json"},
        "label": "s5 (D-151): y=1 if def_date in [1 Jun t+1, 1 Jun t+2); panel 2026-11-08_panel_s5.parquet, "
                 "status_s5 == 'ok'",
        "input_sha256": {"panel_s5": sha(PANEL)},
        "feature_mode": "firm_year",
        "fit_rule": "fit_preprocessor(train, history=train); transform_with_preprocessor(split, history=split, "
                    "frozen artifact) for validation, test, lockbox",
        "validation_use": "early stopping only (D-156); level via prior shift (D-072), not Platt on the "
                          "validation year (D-150)",
        "lockbox": "round 3 test (FY2011) is in LOCKBOX_do_not_open/; score once, after model choice "
                   "(D-087/D-119). Its labels/default rate are not summarised here.",
        "rounds": [], "files": {},
    }
    checks: dict = {"rounds": []}

    def save(frame: pd.DataFrame, path: Path) -> None:
        frame.to_parquet(path, index=False, compression="zstd")
        manifest["files"][rel(path)] = {"sha256": sha(path), "rows": int(len(frame)),
                                        "columns": int(frame.shape[1])}

    feature_lists: dict[str, list[str]] = {}
    rng = np.random.default_rng(SEED)

    for s in splits:
        t0 = time.time()
        r = s.round_number
        lockbox = s.test_year == LOCKBOX_YEAR
        train_f, art = pp.fit_preprocessor(s.train, s.train, cfg)
        val_f = pp.transform_with_preprocessor(s.validation, s.validation, art)
        test_f = pp.transform_with_preprocessor(s.test, s.test, art)
        feats = list(art.feature_columns)
        train_f, val_f, test_f = (f[META + feats] for f in (train_f, val_f, test_f))
        art_path = ART / f"round_{r:02d}_preprocessor.joblib"
        joblib.dump(art, art_path)
        manifest["files"][rel(art_path)] = {"sha256": sha(art_path)}

        # row-independence: re-transform a random 2,000-row subset of the test year as its own batch
        pick = rng.choice(len(s.test), size=min(2000, len(s.test)), replace=False)
        sub = s.test.iloc[np.sort(pick)]
        sub_f = pp.transform_with_preprocessor(sub, sub, art)[META + feats]
        joined = sub_f.merge(test_f, on=["id", "fs_year"], suffixes=("", "_full"), validate="one_to_one")
        a = joined[["label", *feats]].to_numpy(float)
        b = joined[["label_full", *[f"{c}_full" for c in feats]]].to_numpy(float)
        row_indep_max_abs = float(np.nanmax(np.abs(a - b)))

        paths = {"train": OUT / f"round_{r:02d}_train.parquet", "valid": OUT / f"round_{r:02d}_valid.parquet",
                 "test": (LOCK if lockbox else OUT) / f"round_{r:02d}_test.parquet"}
        for k, f in [("train", train_f), ("valid", val_f), ("test", test_f)]:
            save(f, paths[k])
            feature_lists[rel(paths[k])] = list(f.columns[len(META):])

        train_ids = set(s.train.id)
        valid_ids = set(s.validation.id)
        test_ids = s.test.id
        rec = {"round": r, "role": "LOCKBOX (score once, D-087/D-119)" if lockbox else "model selection",
               "train_years": list(s.train_years), "validation_year": s.validation_year, "test_year": s.test_year,
               "files": {k: rel(p) for k, p in paths.items()}, "preprocessor": rel(art_path),
               "n_features": len(feats), "train_level_te_global_mean": float(art.industry_global_mean)}
        for k, raw_df, f in [("train", s.train, train_f), ("valid", s.validation, val_f), ("test", s.test, test_f)]:
            hide = lockbox and k == "test"
            rec[k] = {"rows_in": int(len(raw_df)), "rows_out": int(len(f)), "firms": int(f.id.nunique()),
                      "years": sorted(map(int, f.fs_year.unique())),
                      "defaults": None if hide else int(f.label.sum()),
                      "default_rate": None if hide else float(f.label.mean())}
        for k, ids in [("valid", s.validation.id), ("test", test_ids)]:
            seen = ids.isin(train_ids)
            rec[k]["rows_firm_seen_in_train"] = int(seen.sum())
            rec[k]["rows_firm_new_vs_train"] = int((~seen).sum())
        seen_tv = test_ids.isin(train_ids | valid_ids)
        rec["test"]["rows_firm_seen_in_train_or_valid"] = int(seen_tv.sum())
        rec["test"]["rows_firm_new_vs_train_and_valid"] = int((~seen_tv).sum())
        if not lockbox:  # seen/new default rates for selection rounds only
            seen = test_ids.isin(train_ids).to_numpy()
            rec["test"]["default_rate_seen_in_train"] = float(test_f.label.to_numpy()[seen].mean())
            rec["test"]["default_rate_new_vs_train"] = float(test_f.label.to_numpy()[~seen].mean())
        manifest["rounds"].append(rec)

        ck = {"round": r, "lockbox": lockbox,
              "year_overlap": {"train_valid": sorted(set(train_f.fs_year) & set(val_f.fs_year)),
                               "train_test": sorted(set(train_f.fs_year) & set(test_f.fs_year)),
                               "valid_test": sorted(set(val_f.fs_year) & set(test_f.fs_year))},
              "te_train_level": float(art.industry_global_mean),
              "te_valid": te_check(val_f, art), "te_test": te_check(test_f, art),
              "valid_rate": float(val_f.label.mean()),
              "test_rate": None if lockbox else float(test_f.label.mean()),
              "row_independence_max_abs_diff": row_indep_max_abs,
              "row_independence_rows": int(len(joined)),
              "key_match": {k: bool(f[["id", "fs_year"]].merge(raw_df[["id", "fs_year"]], how="outer",
                                                                indicator=True)["_merge"].eq("both").all())
                            for k, raw_df, f in [("train", s.train, train_f), ("valid", s.validation, val_f),
                                                 ("test", s.test, test_f)]},
              "label_match": {k: bool(f[["id", "fs_year", "label"]].merge(
                  raw_df[["id", "fs_year", "label"]], on=["id", "fs_year"], suffixes=("", "_in"))
                  .pipe(lambda m: (m.label == m.label_in).all()))
                  for k, raw_df, f in [("train", s.train, train_f), ("valid", s.validation, val_f),
                                       ("test", s.test, test_f)]},
              "legal_categories": art.legal_categories, "legal_reference": art.legal_reference_category,
              "city_categories": art.city_categories, "city_reference": art.city_reference_category,
              "industry_groups_in_median_table": [g for g in art.sector_medians.index if g != "ALL"]}
        checks["rounds"].append(ck)
        print(f"round {r} done in {time.time() - t0:.0f}s")

    # ---- final fit on all labelled rows FY2007-2011 ----------------------------------------------
    final_f, final_art = pp.fit_preprocessor(labelled, labelled, cfg)
    final_feats = list(final_art.feature_columns)
    final_f = final_f[META + final_feats]
    final_path = OUT / "final_fit.parquet"
    save(final_f, final_path)
    feature_lists[rel(final_path)] = final_feats
    final_art_path = ART / "final_preprocessor.joblib"
    joblib.dump(final_art, final_art_path)
    manifest["files"][rel(final_art_path)] = {"sha256": sha(final_art_path)}
    manifest["final"] = {"train_years": sorted(map(int, labelled.fs_year.unique())), "rows_in": int(len(labelled)),
                         "rows_out": int(len(final_f)), "firms": int(final_f.id.nunique()),
                         "defaults": int(final_f.label.sum()), "default_rate": float(final_f.label.mean()),
                         "file": rel(final_path), "preprocessor": rel(final_art_path),
                         "n_features": len(final_feats),
                         "note": "no validation set; tree count / early stopping must come from the rounds"}
    checks["final"] = {"legal_categories": final_art.legal_categories, "city_categories": final_art.city_categories,
                       "industry_groups_in_median_table": [g for g in final_art.sector_medians.index if g != "ALL"]}

    # ---- feature-name checks -----------------------------------------------------------------------
    distinct = {tuple(v) for v in feature_lists.values()}
    ref = final_feats
    dyn = [c for c in ref if c not in static]
    checks["features"] = {
        "n_distinct_feature_lists": len(distinct),
        "n_features": len(ref),
        "static_part_equals_schema_in_order": ref[:len(static)] == static,
        "dynamic_part_prefixes_ok": all(any(c.startswith(p) for p in pp.FEATURE_SCHEMA["dynamic_feature_prefixes"])
                                        for c in dyn),
        "dynamic_features": dyn,
        "differences_vs_final": {k: {"missing": sorted(set(ref) - set(v)), "extra": sorted(set(v) - set(ref))}
                                 for k, v in feature_lists.items() if v != ref},
    }
    manifest["feature_columns"] = ref
    manifest["n_features"] = len(ref)
    manifest["feature_groups"] = pp.FEATURE_SCHEMA["feature_groups"]
    manifest["checks"] = checks

    # ---- totals: rows in = rows out ---------------------------------------------------------------
    checks["totals"] = {"raw_rows": raw_rows, "labelled_rows": int(len(labelled)),
                        "labelled_by_year": {int(k): int(v) for k, v in labelled.fs_year.value_counts().sort_index().items()},
                        "final_rows_out": int(len(final_f))}

    # ---- split definition (frozen) -----------------------------------------------------------------
    for s in splits:
        manifest[f"round_{s.round_number:02d}_test_firm_seen_in_train"] = {
            "note": "list of test-year firm ids that also appear in the training years",
            "ids": sorted(map(int, set(s.test.id) & set(s.train.id)))}
    SPLIT_JSON.write_text(json.dumps(manifest, indent=1, default=str))
    os.chmod(SPLIT_JSON, 0o444)
    (OUT / "manifest.json").write_text(json.dumps({k: v for k, v in manifest.items()
                                                   if not k.endswith("_firm_seen_in_train")}, indent=1, default=str))
    write_checks_md(manifest, checks)
    print("done:", OUT)


def write_checks_md(m: dict, c: dict) -> None:
    L = ["# v4 temporal_year split: build checks (generated by "
         "`src/data/2026-11-20_build_v4_temporal_year.py`; do not edit by hand)", "",
         f"Repo commit {m['repo_commit']}; feature schema {m['feature_schema']['version']} "
         f"(sha256 {m['feature_schema']['sha256'][:12]}); label s5 (D-151). Decision D-156.", "",
         "## 1. Counts (rows in = rows out)", "",
         "| Round | Split | Years | Rows in | Rows out | Firms | Defaults | Default rate | Rows, firm seen in train | Rows, firm new vs train |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for rec in m["rounds"]:
        for k in ("train", "valid", "test"):
            d = rec[k]
            dr = "withheld (lockbox)" if d["default_rate"] is None else f"{d['default_rate']:.4%}"
            de = "withheld" if d["defaults"] is None else f"{d['defaults']:,}"
            seen = f"{d['rows_firm_seen_in_train']:,}" if k != "train" else "-"
            new = f"{d['rows_firm_new_vs_train']:,}" if k != "train" else "-"
            L.append(f"| {rec['round']} | {k}{' (LOCKBOX)' if rec['test_year'] == LOCKBOX_YEAR and k == 'test' else ''} "
                     f"| {','.join(map(str, d['years']))} | {d['rows_in']:,} | {d['rows_out']:,} | {d['firms']:,} | {de} | {dr} | {seen} | {new} |")
    f = m["final"]
    L.append(f"| final | final_fit | {','.join(map(str, f['train_years']))} | {f['rows_in']:,} | {f['rows_out']:,} | "
             f"{f['firms']:,} | {f['defaults']:,} | {f['default_rate']:.4%} | - | - |")
    t = c["totals"]
    L += ["", f"Raw rows {t['raw_rows']:,}; labelled (status ok) rows {t['labelled_rows']:,}; by year "
          f"{t['labelled_by_year']}.", ""]
    ok_rows = all(rec[k]["rows_in"] == rec[k]["rows_out"] for rec in m["rounds"] for k in ("train", "valid", "test")) \
        and f["rows_in"] == f["rows_out"]
    L.append(f"- Rows in = rows out in every file: **{ok_rows}**.")
    L.append("- (id, fs_year) keys identical to the input split and labels unchanged: "
             f"**{all(all(r['key_match'].values()) and all(r['label_match'].values()) for r in c['rounds'])}**.")
    L.append("")
    L += ["Seen/new vs train+valid years (test sets): " + "; ".join(
        f"round {rec['round']}: seen {rec['test']['rows_firm_seen_in_train_or_valid']:,} / new "
        f"{rec['test']['rows_firm_new_vs_train_and_valid']:,}" for rec in m["rounds"]) + ".", ""]
    sel = [rec for rec in m["rounds"] if "default_rate_seen_in_train" in rec["test"]]
    L += ["Test default rate, firm seen in train vs new (selection rounds): " + "; ".join(
        f"round {rec['round']}: {rec['test']['default_rate_seen_in_train']:.4%} vs "
        f"{rec['test']['default_rate_new_vs_train']:.4%}" for rec in sel) + ".", ""]
    L += ["## 2. No train/valid/test year overlap", ""]
    for r in c["rounds"]:
        L.append(f"- Round {r['round']}: overlaps {r['year_overlap']} -> "
                 f"**{all(len(v) == 0 for v in r['year_overlap'].values())}**")
    L += ["", "## 3. Target encoding is frozen at the TRAINING level (not the evaluation-year rate)", "",
          "| Round | Train level (TE global mean) | Valid mean TE | Valid year DR | Test mean TE | Test year DR | Share valid TE from frozen mapping | Share test TE from frozen mapping |",
          "|---|---|---|---|---|---|---|---|"]
    for r in c["rounds"]:
        tr = "withheld (lockbox)" if r["test_rate"] is None else f"{r['test_rate']:.4%}"
        L.append(f"| {r['round']} | {r['te_train_level']:.4%} | {r['te_valid']['mean_te']:.4%} | {r['valid_rate']:.4%} | "
                 f"{r['te_test']['mean_te']:.4%} | {tr} | {r['te_valid']['share_te_from_frozen_mapping']:.2%} | "
                 f"{r['te_test']['share_te_from_frozen_mapping']:.2%} |")
    L += ["", "## 4. Feature names", "",
          f"- Distinct feature lists across all {len(m['files']) - 4} feature files: **{c['features']['n_distinct_feature_lists']}** "
          f"(1 = identical). Features: {c['features']['n_features']}.",
          f"- Static part equals the schema (period_numeric + period_binary + firm_year mode features + ateco_sector_te, "
          f"same order): **{c['features']['static_part_equals_schema_in_order']}**.",
          f"- Dynamic one-hot columns all use the schema prefixes {pp.FEATURE_SCHEMA['dynamic_feature_prefixes']}: "
          f"**{c['features']['dynamic_part_prefixes_ok']}**. Columns: {', '.join(c['features']['dynamic_features'])}.",
          f"- Differences vs final_fit: {c['features']['differences_vs_final'] or 'none'}.",
          "- Files: `id, fs_year, label`, then the features in the order above.", "",
          "## 5. Each evaluation row uses only its own record", "",
          "Method G is not in the code (schema 1.3.0); fit receives only training rows. A random 2,000-row subset of each "
          "test year was re-transformed as its own batch and compared with the full-file transform:", ""]
    for r in c["rounds"]:
        L.append(f"- Round {r['round']}: {r['row_independence_rows']:,} rows, max abs difference "
                 f"{r['row_independence_max_abs_diff']:.3g}")
    L += ["", "## 6. Frozen categories per round (fitted on training years only)", ""]
    for r in c["rounds"]:
        L.append(f"- Round {r['round']}: legal reference `{r['legal_reference']}`, dummies {r['legal_categories']}; "
                 f"region reference `{r['city_reference']}`; method-B industry groups with own median "
                 f"{r['industry_groups_in_median_table']} (others use the overall median).")
    L.append(f"- Final: legal dummies {c['final']['legal_categories']}; method-B groups "
             f"{c['final']['industry_groups_in_median_table']}.")
    L += ["", "## 7. Files (sha256)", "", "| File | Rows | sha256 |", "|---|---|---|"]
    for p, d in m["files"].items():
        L.append(f"| `{p}` | {d.get('rows', '-')} | `{d['sha256']}` |")
    CHECKS_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
