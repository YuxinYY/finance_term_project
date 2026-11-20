"""Train-vs-harness feature parity test (feature-architect audit, 2026-11-20).

Question: does a firm-year get the SAME 80 model features from the training code path
(`src/finance_project/data_processing/preprocess.py` + notebook exports) as from the
submitted harness (`tests/harness_submission`, Chloe; `harness/2026-11-16_turquoise_submission`,
Kiet's D-152 patch), when both use the same frozen parameters?

All code is read from git (pinned commits), never from the working tree, because other
agents edit `src/finance_project/data_processing/` concurrently:
  * PIN    = 478b058  current consolidated code (schema 1.3.0, Desmond's 6bf455a included)
  * BUNDLE = 4e0ba3a  code the frozen bundle was built from (verified below against the
                      bundle manifest's `source_sha256`; it is 6bf455a~1)

Paths compared (all on the same FY2011 rows):
  T_bundle   training path @BUNDLE, frozen params from artifacts/preprocessing.json,
             history = the firm's FY2007-2011 rows (as make_walk_forward_splits builds it)
  T_bundle0  same, history = the anchor rows only (no prior years -> Method G cannot fire)
  T_pin      training path @PIN (schema 1.3.0); medians refitted on the round-03 training
             window (labelled FY2007-09) with PIN's own sector_medians; TE / categories frozen
  H_chloe    tests/harness_submission @PIN (Method G ON, reads other rows in the batch)
  H_kiet     harness/2026-11-16_turquoise_submission @PIN (Method G OFF, D-152)

Outputs (relative to the repo root):
  reports/features/2026-11-20_parity_by_feature.csv   per comparison x feature
  reports/features/2026-11-20_parity_summary.csv      provenance, row drops, PD diffs,
                                                      batch-vs-single invariance, smoke tests

Run:  python src/audit/2026-11-20_train_harness_parity.py [--n 20000]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PIN = "478b058"
BUNDLE = "4e0ba3a"
OUT_DIR = REPO / "reports/features"
TRAIN_RAW_DEFAULT = Path("/Users/kietnguyen/Downloads/pd-agent-kit/data/raw/train-2025.csv")
PANEL_DEFAULT = Path("/Users/kietnguyen/Downloads/pd-agent-kit/data/interim/2026-11-08_panel_s5.parquet")
SEED = 20261120
PREV_ZERO = ["liab_lt", "liab_lt_emp", "debt_lt"]

TRAIN_FILES = [
    "src/finance_project/__init__.py",
    "src/finance_project/common.py",
    "src/finance_project/data_processing/__init__.py",
    "src/finance_project/data_processing/preprocess.py",
    "config/feature_schema.json",
    "notebooks/features/01_flag_anomalies.ipynb",
    "notebooks/features/02_missing_values.ipynb",
]
HARNESS_FILES = [
    "anomaly_rules.py", "bundle_paths.py", "feature_rules.py", "inference.py", "missing_rules.py",
    "harness.py", "requirements.txt", "README.md", "artifacts/lightgbm.txt", "artifacts/manifest.json",
    "artifacts/preprocessing.json", "config/feature_schema.json", "data/interim/ateco_sector_english.csv",
]
SHARED_MODULES = ["finance_project", "anomaly_rules", "bundle_paths", "feature_rules", "inference",
                  "missing_rules"]

summary_rows: list[dict] = []


def note(check: str, value, detail: str = "") -> None:
    summary_rows.append({"check": check, "value": value, "detail": detail})
    print(f"[{check}] {value}  {detail}")


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"],
                          check=True, capture_output=True).stdout


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def materialise(commit: str, files: list[str], dest: Path, prefix: str = "") -> None:
    for f in files:
        target = dest / f
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git_bytes(commit, prefix + f))


def purge_modules() -> None:
    for name in list(sys.modules):
        if any(name == m or name.startswith(m + ".") for m in SHARED_MODULES):
            del sys.modules[name]


def load_training(root: Path):
    purge_modules()
    sys.path.insert(0, str(root / "src"))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mod = importlib.import_module("finance_project.data_processing.preprocess")
    finally:
        sys.path.remove(str(root / "src"))
        purge_modules()
    return mod


def load_harness(root: Path):
    purge_modules()
    sys.path.insert(0, str(root))
    try:
        inf = importlib.import_module("inference")
    finally:
        sys.path.remove(str(root))
        purge_modules()
    return inf


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def harness_features(inf, frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Official harness transform; returns finite rows indexed by input position."""
    return quiet(inf.transform, frame, params)


def training_features(mod, anchors, history, artifacts) -> pd.DataFrame:
    return quiet(mod.transform_with_preprocessor, anchors, history, artifacts)


def compare(name: str, left: pd.DataFrame, right: pd.DataFrame, features: list[str],
            rows: str) -> list[dict]:
    """left/right indexed by the same row key; compare every feature."""
    out = []
    common = left.index.intersection(right.index)
    for f in features:
        in_l, in_r = f in left.columns, f in right.columns
        rec = {"comparison": name, "rows": rows, "feature": f, "in_left": in_l, "in_right": in_r,
               "n_rows_compared": len(common)}
        if not (in_l and in_r):
            rec.update(n_differ=np.nan, share_differ=np.nan, n_differ_material=np.nan,
                       share_differ_material=np.nan, max_abs_diff=np.nan, n_nan_mismatch=np.nan)
            out.append(rec)
            continue
        a = pd.to_numeric(left.loc[common, f], errors="coerce").to_numpy(float)
        b = pd.to_numeric(right.loc[common, f], errors="coerce").to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        nan_mis = np.isnan(a) ^ np.isnan(b)
        diff = np.where(both_nan | nan_mis, 0.0, np.abs(a - b))
        differ = (diff > 0) | nan_mis
        scale = np.maximum(1.0, np.maximum(np.nan_to_num(np.abs(a)), np.nan_to_num(np.abs(b))))
        material = (diff > 1e-9 * scale) | nan_mis
        rec.update(n_differ=int(differ.sum()), share_differ=float(differ.mean()) if len(a) else np.nan,
                   n_differ_material=int(material.sum()),
                   share_differ_material=float(material.mean()) if len(a) else np.nan,
                   max_abs_diff=float(np.nanmax(diff)) if len(a) else np.nan,
                   n_nan_mismatch=int(nan_mis.sum()))
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n-single", type=int, default=500)
    ap.add_argument("--train-raw", type=Path, default=TRAIN_RAW_DEFAULT)
    ap.add_argument("--panel", type=Path, default=PANEL_DEFAULT)
    ap.add_argument("--workdir", type=Path, default=None)
    args = ap.parse_args()

    work = Path(args.workdir or tempfile.mkdtemp(prefix="parity_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    t_bundle, t_pin = work / "train_bundle", work / "train_pin"
    h_chloe, h_kiet = work / "h_chloe", work / "h_kiet"
    materialise(BUNDLE, TRAIN_FILES, t_bundle)
    materialise(PIN, TRAIN_FILES, t_pin)
    materialise(PIN, HARNESS_FILES, h_chloe, "tests/harness_submission/")
    materialise(PIN, HARNESS_FILES, h_kiet, "harness/2026-11-16_turquoise_submission/")

    # ---- 0. provenance ---------------------------------------------------------------
    manifest = json.loads((h_chloe / "artifacts/manifest.json").read_text())
    lookup_bytes = (h_chloe / "data/interim/ateco_sector_english.csv").read_bytes()
    for path, expected in manifest["source_sha256"].items():
        if path.startswith("models/") or path.startswith("model_artifacts/"):
            note(f"provenance:{path}", "not in repo", "joblib not committed (models/* gitignored)")
            continue
        if path == "data/interim/ateco_sector_english.csv":
            got_b = got_p = sha(lookup_bytes)
            src = "bundle copy (data/interim/* is gitignored)"
        else:
            got_b, got_p = sha(git_bytes(BUNDLE, path)), sha(git_bytes(PIN, path))
            src = ""
        note(f"provenance:{path}", f"{BUNDLE}={'MATCH' if got_b == expected else 'DIFF'}; "
             f"{PIN}={'MATCH' if got_p == expected else 'DIFF'}", src)
    ignored = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q",
                              "data/interim/ateco_sector_english.csv"]).returncode == 0
    note("repro:ateco_lookup_gitignored", ignored,
         "preprocess.py raises FileNotFoundError on import in a fresh clone unless the file is copied in")
    for root in (t_bundle, t_pin):
        (root / "data/interim").mkdir(parents=True, exist_ok=True)
        (root / "data/interim/ateco_sector_english.csv").write_bytes(lookup_bytes)

    # ---- load code -------------------------------------------------------------------
    mod_b = load_training(t_bundle)
    mod_p = load_training(t_pin)
    inf_c = load_harness(h_chloe)
    inf_k = load_harness(h_kiet)
    params, booster = quiet(inf_k.load_parameters)
    params_c, _ = quiet(inf_c.load_parameters)
    assert params == params_c, "the two harnesses ship different preprocessing.json"
    feats = list(params["feature_columns"])
    note("schema:train_bundle_version", mod_b.FEATURE_SCHEMA_VERSION)
    note("schema:train_pin_version", mod_p.FEATURE_SCHEMA_VERSION)
    note("schema:harness_version", inf_k.feature_rules.FEATURE_SCHEMA["schema_version"])
    note("schema:frozen_params_version", params["schema_version"])
    note("schema:frozen_hash_equals_bundle_code_hash", params["schema_hash"] == mod_b.FEATURE_SCHEMA_HASH)
    note("schema:frozen_hash_equals_pin_code_hash", params["schema_hash"] == mod_p.FEATURE_SCHEMA_HASH)
    note("model:n_features", len(feats))

    # ---- data ------------------------------------------------------------------------
    raw = pd.read_csv(args.train_raw, low_memory=False)
    panel = pd.read_parquet(args.panel, columns=["id", "fs_year", "y"])
    lab = raw.merge(panel, on=["id", "fs_year"], how="left", validate="one_to_one")
    train = lab[lab["fs_year"].between(2007, 2009) & lab["y"].notna()].rename(columns={"y": "label"})
    note("train_window:rows", len(train), "labelled (s5) FY2007-09 = round-03 training anchors")
    note("train_window:default_rate_equals_frozen_fallback_pd",
         bool(abs(train["label"].mean() - params["fallback_pd"]) < 1e-15),
         f"{train['label'].mean():.18f} vs {params['fallback_pd']:.18f}")

    # ---- 1. refit the frozen parameters with the bundle code (provenance of preprocessing.json)
    med_frozen = pd.DataFrame(**params["medians"])
    med_b = mod_b._missing["sector_medians"](train.drop(columns="label"))
    med_b = med_b.reindex(index=med_frozen.index, columns=med_frozen.columns)
    note("refit_bundle:medians_max_abs_diff", float(np.nanmax(np.abs(med_b.to_numpy() - med_frozen.to_numpy()))),
         f"groups={list(med_frozen.index)}")
    clean_tr = quiet(mod_b.clean_and_impute, train, med_frozen, mod_b.PreprocessingConfig(feature_mode="firm_year"))
    te_frame = clean_tr[["id", "label", mod_b.ATECO_INDUSTRY_COLUMN]].reset_index(drop=True)
    _, te_map, te_mean = mod_b._oof_target_encode(te_frame, mod_b.PreprocessingConfig(feature_mode="firm_year"))
    te_frozen = pd.Series(params["industry_mapping"])
    note("refit_bundle:te_mapping_max_abs_diff", float((te_map.reindex(te_frozen.index) - te_frozen).abs().max()),
         f"groups refit={len(te_map)} frozen={len(te_frozen)}")
    note("refit_bundle:te_global_mean_diff", float(te_mean - params["industry_global_mean"]))
    legal_all = sorted(clean_tr["legal_struct"].fillna("Unknown").astype(str).unique())
    city_all = sorted(mod_b._region(clean_tr["HQ_city"]).unique())
    note("refit_bundle:legal_categories_match", legal_all[1:] == params["legal_categories"],
         f"reference (all-zero) category = {legal_all[0]}")
    note("refit_bundle:city_categories_match", city_all[1:] == params["city_categories"],
         f"reference (all-zero) category = {city_all[0]}")
    legal_ref, city_ref = legal_all[0], city_all[0]

    # PIN medians (schema 1.3.0: no roe, 3 smallest groups dropped)
    med_p = mod_p._missing["sector_medians"](train.drop(columns="label"))
    firms = train.assign(g=mod_p._ateco_industry_group(train["ateco_sector"])).groupby("g")["id"].nunique()
    dropped = sorted(set(med_frozen.index) - set(med_p.index) - {"ALL"})
    note("refit_pin:groups_falling_back_to_overall_median", dropped,
         "; ".join(f"{g}:{int(firms.get(g, 0))} firms" for g in dropped))
    note("refit_pin:median_columns_removed", sorted(set(med_frozen.columns) - set(med_p.columns)))

    # ---- 2. sample -------------------------------------------------------------------
    fy11 = raw[raw["fs_year"] == 2011]
    sample = fy11.sample(n=min(args.n, len(fy11)), random_state=SEED)
    g_rows = fy11[fy11[PREV_ZERO].isna().any(axis=1)]
    prev = raw[raw["fs_year"] == 2010].set_index("id")[PREV_ZERO]
    g_fire = g_rows[[bool(((prev.loc[i] == 0) & g_rows.loc[k, PREV_ZERO].isna().to_numpy()).any())
                     if i in prev.index else False for k, i in zip(g_rows.index, g_rows["id"])]]
    note("sample:random_fy2011_rows", len(sample), f"seed={SEED}")
    note("sample:fy2011_rows_with_liab_lt/liab_lt_emp/debt_lt_missing", len(g_rows))
    note("sample:of_which_method_G_would_fire_from_FY2010", len(g_fire))
    anchors = pd.concat([sample, g_rows[~g_rows.index.isin(sample.index)]])
    anchors = anchors.sort_index()
    history_full = raw[raw["id"].isin(anchors["id"]) & raw["fs_year"].between(2007, 2011)]
    key = lambda d: pd.MultiIndex.from_frame(d[["id", "fs_year"]])  # noqa: E731

    # ---- 3. feature paths ---------------------------------------------------------------
    def bundle_artifacts(mod, medians, feature_columns):
        return mod.PreprocessingArtifacts(
            config=mod.PreprocessingConfig(**params["config"]),
            feature_schema_version=mod.FEATURE_SCHEMA_VERSION, feature_schema_hash=mod.FEATURE_SCHEMA_HASH,
            sector_medians=medians, industry_mapping=te_frozen, industry_global_mean=params["industry_global_mean"],
            legal_categories=params["legal_categories"], legal_reference_category=legal_ref,
            city_categories=params["city_categories"], city_reference_category=city_ref,
            feature_columns=feature_columns)

    art_b = bundle_artifacts(mod_b, med_frozen, feats)
    pin_feats = [*mod_p._period_numeric_features(), *mod_p._period_binary_features(),
                 *mod_p.FEATURE_SCHEMA["mode_features"]["firm_year"], "ateco_sector_te",
                 *[f"legal_{c}" for c in params["legal_categories"]],
                 *[f"city_{c}" for c in params["city_categories"]]]
    art_p = bundle_artifacts(mod_p, med_p, pin_feats)

    def by_key(df):
        return df.set_index(["id", "fs_year"])

    T_b = by_key(training_features(mod_b, anchors, history_full, art_b))
    T_b0 = by_key(training_features(mod_b, anchors, anchors, art_b))
    T_p = by_key(training_features(mod_p, anchors, history_full, art_p))
    a_idx = key(anchors.reset_index(drop=True))
    H_c = harness_features(inf_c, anchors.reset_index(drop=True), params)
    H_c.index = a_idx[H_c.index]
    H_k = harness_features(inf_k, anchors.reset_index(drop=True), params)
    H_k.index = a_idx[H_k.index]
    for nm, df in [("T_bundle", T_b), ("T_bundle0", T_b0), ("T_pin", T_p), ("H_chloe", H_c), ("H_kiet", H_k)]:
        note(f"rows_kept:{nm}", len(df), f"of {len(anchors)} anchors (rest dropped by training / fallback PD in harness)")

    rows_all = "random FY2011 sample + all FY2011 rows with liab_lt/liab_lt_emp/debt_lt missing"
    recs = []
    recs += compare("T_bundle0 vs H_kiet (arithmetic parity, no prior years)", T_b0, H_k, feats, rows_all)
    recs += compare("T_bundle vs H_kiet (training history incl. FY2010; G on in training, off in harness)",
                    T_b, H_k, feats, rows_all)
    recs += compare("T_bundle vs H_chloe (batch = FY2011 only)", T_b, H_c, feats, rows_all)
    recs += compare("H_chloe vs H_kiet (same batch)", H_c, H_k, feats, rows_all)
    recs += compare("T_pin (schema 1.3.0) vs H_kiet (frozen 1.2.1 bundle)", T_p, H_k,
                    list(dict.fromkeys(pin_feats + feats)), rows_all)

    # ---- 4. PD impact ---------------------------------------------------------------------
    def pds(df):
        X = df[feats].astype(float)
        return pd.Series(inf_k.calibrated_predictions(X, params, booster), index=df.index)

    pd_h = pds(H_k)
    for nm, df in [("T_bundle", T_b), ("T_bundle0", T_b0), ("H_chloe", H_c)]:
        common = df.index.intersection(pd_h.index)
        d = (pds(df.loc[common]) - pd_h.loc[common]).abs()
        note(f"pd_abs_diff_vs_H_kiet:{nm}:max", float(d.max()), f"rows={len(common)}")
        note(f"pd_abs_diff_vs_H_kiet:{nm}:n_rows_gt_1e-12", int((d > 1e-12).sum()))
    # T_pin cannot be scored by the frozen model (81 features incl. roe_negative; roe on a new scale)
    roe_cmp = pd.concat([T_p["roe"], H_k["roe"]], axis=1, keys=["pin", "harness"]).dropna()
    note("T_pin:roe_abs_value_gt_150_share", float((roe_cmp["pin"].abs() > 150).mean()),
         "frozen model saw roe only in [-150, 149.93] (lightgbm.txt feature_infos)")
    note("T_pin:roe_max_abs", float(roe_cmp["pin"].abs().max()))
    raw_roe_blank = anchors.set_index(["id", "fs_year"])["roe"].isna().reindex(roe_cmp.index)
    d_roe = (roe_cmp["pin"] - roe_cmp["harness"]).abs()
    note("T_pin:roe_raw_blank_share", float(raw_roe_blank.mean()),
         "harness fills these with the industry median; 1.3.0 recomputes them")
    note("T_pin:roe_absdiff_gt_0.01_share_when_raw_present", float((d_roe[~raw_roe_blank] > 0.01).mean()),
         "provider roe is rounded to 2 decimals; larger gaps = formula differences")
    note("T_pin:roe_absdiff_median_when_raw_blank", float(d_roe[raw_roe_blank].median()))
    note("T_pin:roe_negative_share", float(T_p["roe_negative"].mean()) if "roe_negative" in T_p else np.nan,
         "flag = profit<0 AND eqty_tot<0 (roe sign is positive in these rows)")

    # ---- 5. invariance: batch vs chunks vs single rows ---------------------------------
    rng = np.random.default_rng(SEED)
    base = anchors.reset_index(drop=True)
    perm = rng.permutation(len(base))
    for nm, inf in [("H_kiet", inf_k), ("H_chloe", inf_c)]:
        full = harness_features(inf, base, params)
        full.index = a_idx[full.index]
        parts = []
        for chunk in np.array_split(perm, 10):
            part = base.iloc[chunk].reset_index(drop=True)
            f = harness_features(inf, part, params)
            f.index = key(part)[f.index]
            parts.append(f)
        chunked = pd.concat(parts)
        r = compare(f"{nm}: full batch vs shuffled 10 chunks", full, chunked, feats, rows_all)
        note(f"invariance:{nm}:shuffle_chunks_cells_differ", int(np.nansum([x["n_differ"] for x in r])))
        # singles: random subset + every Method-G candidate row, alone vs in a batch with FY2010 rows
        single_idx = list(rng.choice(len(sample), size=min(args.n_single, len(sample)), replace=False))
        g_pos = [i for i, k in enumerate(base.index) if anchors.index[i] in g_rows.index]
        single_idx = sorted(set(single_idx) | set(g_pos))
        singles = []
        for i in single_idx:
            row = base.iloc[[i]].reset_index(drop=True)
            f = harness_features(inf, row, params)
            f.index = key(row)[f.index]
            singles.append(f)
        singles = pd.concat(singles)
        r = compare(f"{nm}: single row vs full batch", singles, full, feats, f"{len(single_idx)} rows")
        note(f"invariance:{nm}:single_vs_batch_cells_differ", int(np.nansum([x["n_differ"] for x in r])),
             f"{len(single_idx)} rows scored alone")
        # multi-year batch: G-candidate FY2011 rows together with the same firms' FY2010 rows
        gb = pd.concat([raw[(raw["fs_year"] == 2010) & raw["id"].isin(g_rows["id"])], g_rows]).reset_index(drop=True)
        fm = harness_features(inf, gb, params)
        fm.index = key(gb)[fm.index]
        fm = fm[fm.index.get_level_values(1) == 2011]
        alone = harness_features(inf, g_rows.reset_index(drop=True), params)
        alone.index = key(g_rows.reset_index(drop=True))[alone.index]
        r = compare(f"{nm}: FY2011 rows with vs without the firm's FY2010 row in the batch", fm, alone, feats,
                    f"{len(g_rows)} FY2011 rows with a PREV_ZERO field missing")
        recs += r
        cells = int(np.nansum([x["n_differ"] for x in r]))
        rows_changed = 0
        if len(fm):
            common = fm.index.intersection(alone.index)
            d = (pds(fm.loc[common]) - pds(alone.loc[common])).abs()
            rows_changed = int((d > 1e-12).sum())
            note(f"invariance:{nm}:multi_year_batch_pd_max_abs_diff", float(d.max()))
        note(f"invariance:{nm}:multi_year_batch_cells_differ", cells, f"rows with PD change={rows_changed}")

    # ---- 6. extra static-in-code checks ---------------------------------------------------
    # 6a. how often Method G fired in the round-03 TRAINING window (history FY2007-09)
    hist = raw[raw["fs_year"].between(2007, 2009)].sort_values(["id", "fs_year"])
    prev_tr = hist.groupby("id")[PREV_ZERO].shift(1)
    fires = (hist[PREV_ZERO].isna() & prev_tr.eq(0)).any(axis=1)
    fires = fires & key(hist).isin(key(train))
    note("method_G:train_window_anchor_rows_where_G_fired", int(fires.sum()),
         f"of {len(train)} training anchors ({fires.sum() / len(train):.6%})")
    # 6b. FY2011 rows affected by the 3-smallest-groups fallback (B-imputed field in O/P/U)
    bcols = [c for c in mod_p._missing["B_COLS"] if c in fy11.columns]
    grp11 = mod_p._ateco_industry_group(fy11["ateco_sector"])
    hit = grp11.isin(dropped) & fy11[bcols].isna().any(axis=1)
    note("small_group_fallback:fy2011_rows_affected", int(hit.sum()),
         f"rows in {dropped} with a Method-B field missing, of {len(fy11)} FY2011 rows")
    hit_tr = mod_p._ateco_industry_group(train["ateco_sector"]).isin(dropped) & train[bcols].isna().any(axis=1)
    note("small_group_fallback:train_window_rows_affected", int(hit_tr.sum()), f"of {len(train)}")
    # 6c. province -> region map: team (preprocess/feature_rules) vs Kiet's src/data/codes.py
    ns: dict = {}
    exec(git_bytes(PIN, "src/data/codes.py").decode(), ns)
    codes = [str(i) for i in range(0, 112)] + ["Unknown", "nan"]
    team = inf_k.feature_rules._region(pd.Series(codes)).tolist()
    kiet = [ns["province_region"](c) or "Unknown" for c in codes]
    mism = [c for c, t, k in zip(codes, team, kiet) if t != k]
    note("region_map:team_vs_kiet_codes_mismatches", len(mism), f"codes={mism[:20]}")
    same_tr = mod_b._region(pd.Series(codes)).tolist() == team
    note("region_map:preprocess_vs_harness_identical", same_tr)
    # 6d. messy / unseen-category row through the patched harness
    messy = anchors.head(1).copy().reset_index(drop=True).astype(object)
    messy.loc[0, ["legal_struct", "HQ_city", "ateco_sector"]] = ["XYZ", 999, np.nan]
    messy.loc[0, "eqty_tot"] = "n.a."
    fm = harness_features(inf_k, messy, params)
    if len(fm):
        r0 = fm.iloc[0]
        note("unseen:legal_dummies_all_zero(=reference SAA)", bool(sum(r0[f"legal_{c}"] for c in params["legal_categories"]) == 0))
        note("unseen:HQ_city_999_maps_to_city_Unknown", bool(r0["city_Unknown"] == 1))
        note("unseen:ateco_missing_te_equals_global_mean", bool(r0["ateco_sector_te"] == params["industry_global_mean"]))
    else:
        note("unseen:messy_row", "fallback PD (non-finite feature)")

    # 6e. frozen model: monotone constraints by name vs MONOTONE_DIRECTIONS in the training code,
    #     features constant in the training window, features never used in a split
    import re
    src = git_bytes(PIN, "src/finance_project/model_development/lightgbm_model.py").decode()
    block = re.search(r"MONOTONE_DIRECTIONS[^=]*=\s*\{(.*?)\}", src, re.S).group(1)
    code_dirs = {k: int(v) for k, v in re.findall(r'"([^"]+)":\s*(-?\d)', block)}
    txt = (h_kiet / "artifacts/lightgbm.txt").read_text()
    names = re.search(r"^feature_names=(.*)$", txt, re.M).group(1).split()
    cons = [int(x) for x in re.search(r"^monotone_constraints=(.*)$", txt, re.M).group(1).split()]
    infos = re.search(r"^feature_infos=(.*)$", txt, re.M).group(1).split()
    model_dirs = {n: c for n, c in zip(names, cons) if c != 0}
    note("model:monotone_constraints_match_code", model_dirs == code_dirs,
         "; ".join(f"{k}:{v:+d}" for k, v in sorted(model_dirs.items())))
    note("model:features_unusable_in_training(feature_infos=none)", [n for n, i in zip(names, infos) if i == "none"])
    used = booster.feature_importance("split")
    note("model:features_never_split_on", [n for n, u in zip(booster.feature_name(), used) if u == 0])
    for n, i in zip(names, infos):
        if n in ("roe", "AR_ratio", "cur_debt_ratio", "int_coverage", "lt_debt_to_capital"):
            note(f"model:training_range:{n}", i)

    # ---- 7. smoke tests on the PIN training code (finding a) --------------------------
    try:
        demo = train.sample(n=3000, random_state=SEED)
        tf, art = quiet(mod_p.fit_preprocessor, demo, demo, mod_p.PreprocessingConfig(feature_mode="firm_year"))
        note("smoke:pin_fit_preprocessor_runs", True,
             f"3,000-row demo fit: features={len(art.feature_columns)}; roe_negative in features={'roe_negative' in art.feature_columns}")
    except Exception as e:  # noqa: BLE001
        note("smoke:pin_fit_preprocessor_runs", False, repr(e)[:300])
    try:
        quiet(mod_p.transform_with_preprocessor, anchors.head(50), anchors.head(50), art_b)
        note("smoke:pin_transform_with_frozen_1.2.1_artifact", "runs")
    except Exception as e:  # noqa: BLE001
        note("smoke:pin_transform_with_frozen_1.2.1_artifact", "raises", repr(e)[:200])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(OUT_DIR / "2026-11-20_parity_by_feature.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "2026-11-20_parity_summary.csv", index=False)
    print("wrote", OUT_DIR / "2026-11-20_parity_by_feature.csv", OUT_DIR / "2026-11-20_parity_summary.csv")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
