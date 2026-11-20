"""Rebuild the default label under the s5 window (D-151) and write the v2 walk-forward splits.

Run (from anywhere):
    ~/miniconda3/envs/datascience/bin/python src/data/2026-11-08_build_labels_s5.py

Label s5 (D-151): for a statement dated stmt_date (31 Dec of fs_year t on every row),
    win_start = stmt_date + 1 day + 5 months   (= 1 Jun t+1)
    win_end   = win_start + 12 months          (= 1 Jun t+2, exclusive: window runs through 31 May t+2)
    status    = after_default      if def_date <= stmt_date        (0 rows expected)
                already_defaulted  if def_date <  win_start        (not a performing applicant: drop)
                censored           if win_end > data_end + 1 day   (window not fully observed: drop)
                ok                 otherwise;  y = 1 if win_start <= def_date < win_end else 0 (only for ok)
    PRIMARY (D-009): def_date as given, data_end = 2013-05-31 (so a window ending 31 May 2013 is complete).
    SENSITIVITY (D-009): def_date > 2012-12-31 blanked, data_end = 2012-12-31  -> y_s5_d2012 / status_s5_d2012.
Labels are built directly from the panel's stmt_date/def_date. src/common/labels.make_labels is NOT used
(D-011/D-068: it keeps observed defaults in incomplete windows as ok/y=1).

Reads (never modified): data/interim/panel.parquet (v1 panel).
Writes (new, date-prefixed; split files chmod 444):
    data/interim/2026-11-08_panel_s5.parquet
    data/splits/2026-11-08_walkforward_v2.json
    data/splits/2026-11-08_walkforward_v2_test_rows.parquet
    reports/data/2026-11-08_labels_s5.md
    reports/data/2026-11-08_walkforward_v2_folds.csv
Refuses to overwrite existing split files (frozen); delete nothing, re-run only on a clean checkout.
Stops with an error if the D-151 sanity numbers are not reproduced.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.mdutil import md  # noqa: E402

PANEL_V1 = ROOT / "data/interim/panel.parquet"
PANEL_S5 = ROOT / "data/interim/2026-11-08_panel_s5.parquet"
SPLITS = ROOT / "data/splits"
WF2 = SPLITS / "2026-11-08_walkforward_v2.json"
WF2_ROWS = SPLITS / "2026-11-08_walkforward_v2_test_rows.parquet"
REP = ROOT / "reports/data"
REP_MD = REP / "2026-11-08_labels_s5.md"
REP_FOLDS = REP / "2026-11-08_walkforward_v2_folds.csv"

DATA_END_PRIMARY = pd.Timestamp("2013-05-31")
DATA_END_D2012 = pd.Timestamp("2012-12-31")
START_DAYS, START_MONTHS, HORIZON = 1, 5, 12
SELECTION_TEST_YEARS = [2008, 2009, 2010]
LOCKBOX_TEST_YEAR = 2011
FINAL_FIT_YEARS = [2007, 2008, 2009, 2010, 2011]

# D-151 sanity numbers (must reproduce, else stop)
EXPECT = {"usable_2007_2011": 835_016, "defaults_2007_2011": 11_336, "already_2007_2011": 2_897,
          "already_by_month": {1: 558, 2: 518, 3: 559, 4: 568, 5: 694}}


def label_s5(stmt: pd.Series, dd: pd.Series, data_end: pd.Timestamp) -> pd.DataFrame:
    win_start = stmt + pd.Timedelta(days=START_DAYS) + pd.DateOffset(months=START_MONTHS)
    win_end = win_start + pd.DateOffset(months=HORIZON)
    after = dd.notna() & (dd <= stmt)
    already = dd.notna() & (dd < win_start)
    complete = win_end <= data_end + pd.Timedelta(days=1)
    status = pd.Series("ok", index=stmt.index, dtype=object)
    status[~complete] = "censored"
    status[already] = "already_defaulted"
    status[after] = "after_default"
    in_win = dd.notna() & (dd >= win_start) & (dd < win_end)
    y = in_win.astype(float).where(status == "ok")
    return pd.DataFrame({"y": y, "status": status, "win_start": win_start, "win_end": win_end,
                         "complete": complete, "default_in_incomplete_window": (~complete) & in_win & (status == "censored")})


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    for p in (WF2, WF2_ROWS):
        if p.exists():
            sys.exit(f"{p} exists and is frozen; refusing to overwrite.")
    pan = pd.read_parquet(PANEL_V1)
    n = len(pan)
    v1_cols = list(pan.columns)
    assert pan["row_id"].is_unique
    assert (pan["row_id"] == pan["id"].astype("int64") * 10000 + pan["fs_year"].astype("int64")).all()
    stmt, dd = pan["stmt_date"], pan["def_date"]
    stmt_dec31 = bool(((stmt.dt.year == pan["fs_year"]) & (stmt.dt.month == 12) & (stmt.dt.day == 31)).all())

    P_ = label_s5(stmt, dd, DATA_END_PRIMARY)
    dd12 = dd.where(~(dd > DATA_END_D2012))
    S_ = label_s5(stmt, dd12, DATA_END_D2012)
    # window dates equal the calendar definition 1 Jun t+1 / 1 Jun t+2
    exp_ws = pd.to_datetime((pan["fs_year"].astype(int) + 1).astype(str) + "-06-01")
    exp_we = pd.to_datetime((pan["fs_year"].astype(int) + 2).astype(str) + "-06-01")
    assert (P_["win_start"] == exp_ws).all() and (P_["win_end"] == exp_we).all()

    # keep v1 labels for comparison before overwriting the primary columns
    v1 = pan[["fs_year", "y", "label_status"]].copy()
    assert v1["label_status"].astype(str).equals(pan["status_s6"].astype(str))

    out = pan.copy()
    out["y_s5"] = P_["y"]
    out["status_s5"] = P_["status"].astype("string")
    out["win_start_s5"] = P_["win_start"]
    out["win_end_s5"] = P_["win_end"]
    out["y_s5_d2012"] = S_["y"]
    out["status_s5_d2012"] = S_["status"].astype("string")
    out["y"] = out["y_s5"]
    out["label_status"] = out["status_s5"]
    out["win_start"] = out["win_start_s5"]
    out["win_end"] = out["win_end_s5"]
    assert list(out.columns)[: len(v1_cols)] == v1_cols and len(out) == n

    # ---------------- sanity checks (D-151) ----------------
    fy = out["fs_year"].astype(int)
    m711 = fy.between(2007, 2011)
    ok = out["status_s5"] == "ok"
    got = {"usable_2007_2011": int((m711 & ok).sum()),
           "defaults_2007_2011": int(out.loc[m711 & ok, "y_s5"].sum()),
           "already_2007_2011": int((m711 & (out["status_s5"] == "already_defaulted")).sum())}
    am = out.loc[m711 & (out["status_s5"] == "already_defaulted"), "def_date"].dt.month.value_counts().sort_index()
    got["already_by_month"] = {int(k): int(v) for k, v in am.items()}
    st12 = out.loc[fy == 2012, "status_s5"].value_counts().to_dict()
    checks = {k: (got[k] == EXPECT[k]) for k in EXPECT}
    checks["fs2012_no_usable_rows"] = int((ok & (fy == 2012)).sum()) == 0
    if not all(checks.values()):
        sys.exit(f"SANITY CHECK FAILED: got {got}, expected {EXPECT}; fs2012 status {st12}")

    PANEL_S5.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PANEL_S5, index=False)

    # ---------------- folds ----------------
    years = sorted(int(y) for y in fy.unique())
    complete_years = sorted(int(y) for y in fy[P_["complete"]].unique())
    assert set(SELECTION_TEST_YEARS + [LOCKBOX_TEST_YEAR]) <= set(complete_years)

    def fold_stats(train_years, test_year=None, ycol="y", scol="label_status"):
        trm = fy.isin(train_years) & (out[scol] == "ok")
        d = {"train_years": train_years, "n_train": int(trm.sum()), "train_defaults": int(out.loc[trm, ycol].sum()),
             "train_default_rate": round(float(out.loc[trm, ycol].mean()), 6)}
        if test_year is None:
            return d, None
        tem = (fy == test_year) & (out[scol] == "ok")
        te = out[tem]
        seen_ids = set(out.loc[fy.isin(train_years), "id"].tolist())
        seen = te["id"].isin(seen_ids)
        d.update({"test_year": test_year, "n_test": int(len(te)), "test_defaults": int(te[ycol].sum()),
                  "test_default_rate": round(float(te[ycol].mean()), 6) if len(te) else None,
                  "n_test_seen_firm": int(seen.sum()), "n_test_new_firm": int((~seen).sum()),
                  "test_defaults_seen_firm": int(te.loc[seen, ycol].sum()),
                  "test_defaults_new_firm": int(te.loc[~seen, ycol].sum()),
                  "test_default_rate_seen_firm": round(float(te.loc[seen, ycol].mean()), 6) if seen.any() else None,
                  "test_default_rate_new_firm": round(float(te.loc[~seen, ycol].mean()), 6) if (~seen).any() else None})
        return d, (te["row_id"].values, seen.values)

    folds, rows = [], []
    for k, ty in enumerate(SELECTION_TEST_YEARS, start=1):
        d, (rid, seen) = fold_stats([y for y in years if y < ty], ty)
        folds.append({"fold": k, "role": "model_selection", **d})
        rows.append(pd.DataFrame({"row_id": rid, "fold": k, "test_year": ty, "firm_seen_in_train": seen,
                                  "role": "model_selection"}))
    lb, (rid, seen) = fold_stats([y for y in years if y < LOCKBOX_TEST_YEAR], LOCKBOX_TEST_YEAR)
    lockbox = {"fold": 4, "role": "lockbox",
               "WARNING": "one-time lockbox per D-087/D-119 - not for model selection. Score the pre-registered finalists "
                          "once (referee), report whatever it says, never tune on it.", **lb}
    rows.append(pd.DataFrame({"row_id": rid, "fold": 4, "test_year": LOCKBOX_TEST_YEAR, "firm_seen_in_train": seen,
                              "role": "lockbox"}))
    # D-009 sensitivity for the lockbox year
    sens = out.loc[fy == LOCKBOX_TEST_YEAR, "status_s5_d2012"].value_counts().to_dict()
    perf11 = (fy == LOCKBOX_TEST_YEAR) & (out["status_s5_d2012"] != "already_defaulted") & (out["status_s5_d2012"] != "after_default")
    d11_obs = int((perf11 & dd12.notna() & (dd12 >= P_["win_start"])).sum())
    d11_2013 = int(((fy == LOCKBOX_TEST_YEAR) & ok & (out["y_s5"] == 1) & (dd > DATA_END_D2012)).sum())
    lockbox["d009_sensitivity_d2012"] = {
        "label_columns": ["y_s5_d2012", "status_s5_d2012"], "data_end": str(DATA_END_D2012.date()),
        "fs2011_status_counts": {str(k): int(v) for k, v in sens.items()},
        "fs2011_usable_rows": int(((fy == LOCKBOX_TEST_YEAR) & (out["status_s5_d2012"] == "ok")).sum()),
        "fs2011_performing_rows_at_window_start": int(perf11.sum()),
        "fs2011_defaults_observed_by_2012-12-31_(7_of_12_months)": d11_obs,
        "fs2011_primary_defaults_dated_2013": d11_2013,
        "note": "If the 2013 def_dates are invalid (Appendix B literal reading), fs2011 has no complete s5 window: "
                "the lockbox and the 2011 rows of final_fit do not exist; folds 1-3 are identical under both treatments."}
    ff, _ = fold_stats(FINAL_FIT_YEARS)
    final_fit = {"fold": "final_fit", "role": "final_fit (refit of the chosen model; no test rows)", **ff}
    ff12, _ = fold_stats([2007, 2008, 2009, 2010], ycol="y_s5_d2012", scol="status_s5_d2012")
    final_fit["d009_sensitivity_d2012"] = {"train_years": [2007, 2008, 2009, 2010], "n_train": ff12["n_train"],
                                           "train_defaults": ff12["train_defaults"],
                                           "note": "under d2012 only fs2007-2010 are usable"}
    # folds 1-3 identical under d2012?
    same_sel = all(out.loc[fy == y, "status_s5"].astype(str).equals(out.loc[fy == y, "status_s5_d2012"].astype(str))
                   and out.loc[fy == y, "y_s5"].fillna(-1).equals(out.loc[fy == y, "y_s5_d2012"].fillna(-1))
                   for y in [2007, 2008, 2009, 2010])

    TR = pd.concat(rows, ignore_index=True)
    TR["fold"] = TR["fold"].astype("int64")
    TR["test_year"] = TR["test_year"].astype("int64")
    TR["row_id"] = TR["row_id"].astype("int64")
    SPLITS.mkdir(parents=True, exist_ok=True)
    TR.to_parquet(WF2_ROWS, index=False)

    spec = {
        "version": 2, "created": "2026-11-08", "decision": "D-151 (supersedes the start-6 primary label of D-002; v1 file kept)",
        "created_by": "data-auditor (src/data/2026-11-08_build_labels_s5.py)",
        "supersedes_for_new_work": "data/splits/walkforward.json (v1, start-6 label; frozen, unchanged)",
        "panel": "data/interim/2026-11-08_panel_s5.parquet",
        "panel_note": "Same rows and the same 64 columns as data/interim/panel.parquet plus y_s5, status_s5, win_start_s5, "
                      "win_end_s5, y_s5_d2012, status_s5_d2012. The PRIMARY columns y / label_status / win_start / win_end "
                      "are OVERWRITTEN with the s5 versions (equal to y_s5 / status_s5 / win_start_s5 / win_end_s5). "
                      "The v1 start-6 label is still available as y_s6 / status_s6.",
        "row_id": "id * 10000 + fs_year (int64; unique; reproducible from any file with id and fs_year)",
        "label": {"name": "s5", "column": "y", "status_column": "label_status",
                  "window": "y = 1 if def_date in [1 Jun fs_year+1, 1 Jun fs_year+2), i.e. through 31 May fs_year+2",
                  "computed_as": "win_start = stmt_date + 1 day + 5 months; win_end = win_start + 12 months (stmt_date = 31 Dec fs_year on every row)",
                  "horizon_months": HORIZON, "data_end": str(DATA_END_PRIMARY.date()),
                  "complete_rule": "win_end <= data_end + 1 day (a window ending 31 May 2013 inclusive is complete)",
                  "statuses": {"ok": "usable", "already_defaulted": "def_date < win_start: drop",
                               "censored": "window not fully observed: drop (whether or not a default is seen in it)",
                               "after_default": "def_date <= stmt_date: drop (0 rows)"},
                  "d009_treatment": "primary (def_date as given, incl. 2013-01..2013-05 dates)",
                  "d009_sensitivity_columns": {"y": "y_s5_d2012", "status": "status_s5_d2012",
                                               "rule": "def_date > 2012-12-31 blanked, data_end 2012-12-31"},
                  "built_without": "src/common/labels.make_labels (D-011/D-068 bug)"},
        "design": "expanding window, out-of-time: train on fs_year < test_year (i.e. <= t), test on fs_year == test_year (t+1)",
        "train_rows": "panel rows with fs_year in train_years AND label_status == 'ok'",
        "test_rows": "panel rows with fs_year == test_year AND label_status == 'ok' (listed in "
                     "2026-11-08_walkforward_v2_test_rows.parquet with columns row_id, fold, test_year, firm_seen_in_train, role)",
        "firm_seen_in_train": "True if the firm id has at least one panel row (any label_status) with fs_year in train_years",
        "folds": folds,
        "folds_note": "Model selection uses folds 1-3 ONLY (test 2008, 2009, 2010).",
        "lockbox": lockbox,
        "final_fit": final_fit,
        "excluded_test_years": {"2012": "s5 window ends 2014-06-01, after data_end: every fs2012 row is censored or already_defaulted"},
        "d009_folds_1_3_identical_under_d2012": bool(same_sel),
        "test_rows_file_sha256": sha256(WF2_ROWS),
        "panel_file_sha256": sha256(PANEL_S5),
    }
    WF2.write_text(json.dumps(spec, indent=2))
    for p in (WF2, WF2_ROWS):
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    FT = pd.DataFrame([{k: v for k, v in f.items() if k not in ("WARNING", "d009_sensitivity_d2012")} for f in folds + [lockbox, final_fit]])
    FT["train_years"] = FT["train_years"].map(lambda t: f"{t[0]}-{t[-1]}" if len(t) > 1 else str(t[0]))
    FT["test_year"] = FT["test_year"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    for c in ["n_test", "test_defaults", "n_test_seen_firm", "n_test_new_firm", "test_defaults_seen_firm", "test_defaults_new_firm"]:
        FT[c] = FT[c].astype("Int64")
    FT.to_csv(REP_FOLDS, index=False)

    # ---------------- report ----------------
    L: list[str] = []
    W = L.append
    W("# Labels s5 (D-151) and walk-forward splits v2\n")
    W("Generated by `src/data/2026-11-08_build_labels_s5.py` (data-auditor, 2026-11-08) from `data/interim/panel.parquet` "
      "(v1, unchanged). Every number below is computed by that script.\n")
    W("## 1. Definition\n")
    W(f"- `stmt_date` == 31 Dec of `fs_year` on every row: **{stmt_dec31}** ({n:,} rows).")
    W("- s5: win_start = 1 Jun t+1, win_end = 1 Jun t+2 (exclusive). y = 1 if win_start <= def_date < win_end. "
      "Statuses: `after_default` (def_date <= stmt_date), `already_defaulted` (def_date < win_start), `censored` "
      "(win_end > data_end + 1 day, regardless of outcome), `ok`. Built directly from the panel; `make_labels` not used (D-011/D-068).")
    W("- PRIMARY (D-009): def_date as given, data_end 2013-05-31. SENSITIVITY: 2013 def_dates blanked, data_end 2012-12-31 "
      "(`y_s5_d2012`, `status_s5_d2012`).")
    W("- In `2026-11-08_panel_s5.parquet` the primary columns `y`, `label_status`, `win_start`, `win_end` hold the s5 values; "
      "v1 start-6 values remain in `y_s6`/`status_s6`.\n")
    W("## 2. Sanity check against D-151\n")
    sc = pd.DataFrame({"expected": {k: str(EXPECT[k]) for k in EXPECT}, "computed": {k: str(got[k]) for k in EXPECT},
                       "match": {k: checks[k] for k in EXPECT}})
    W(md(sc))
    W(f"\nfs_year 2012 status counts: " + ", ".join(f"{k}={v:,}" for k, v in st12.items())
      + f". Usable fs2012 rows: {int((ok & (fy == 2012)).sum())}. (The fs2012 `already_defaulted` rows are firms with a "
      "2013 def_date before 1 Jun 2013; every other fs2012 row is censored.)\n")
    W("## 3. Per fs_year, s5 (primary and D-009 sensitivity)\n")
    def per_year(ycol, scol):
        g = pd.DataFrame({"fs_year": fy, "y": out[ycol], "s": out[scol].astype(str)})
        t = g.groupby("fs_year").agg(rows=("y", "size"), usable=("y", lambda v: int(v.notna().sum())),
                                     defaults=("y", lambda v: int(np.nansum(v))), default_rate=("y", "mean"),
                                     already_defaulted=("s", lambda v: int((v == "already_defaulted").sum())),
                                     censored=("s", lambda v: int((v == "censored").sum())),
                                     after_default=("s", lambda v: int((v == "after_default").sum())))
        return t
    W("Primary (`y_s5`, `status_s5`):\n")
    W(md(per_year("y_s5", "status_s5"), ".4f"))
    W("\nD-009 sensitivity (`y_s5_d2012`, `status_s5_d2012`):\n")
    W(md(per_year("y_s5_d2012", "status_s5_d2012"), ".4f"))
    W(f"\nfs2007-2010 labels identical under both D-009 treatments: **{same_sel}**. Under d2012 fs2011 is entirely "
      f"censored/already_defaulted ({', '.join(f'{k}={v:,}' for k, v in sens.items())}); "
      f"of its {int(perf11.sum()):,} performing rows, {d11_obs:,} default between 1 Jun and 31 Dec 2012 (7 of 12 months). "
      f"{d11_2013:,} of the {got['defaults_2007_2011'] - int(out.loc[fy.between(2007, 2010) & ok, 'y_s5'].sum()):,} primary fs2011 "
      "defaults are dated Jan-May 2013 and exist only if the 2013 dates are valid.\n")
    W("## 4. v2 (s5) vs v1 (s6) per fs_year\n")
    a = v1.assign(ok=v1["label_status"].astype(str) == "ok").groupby("fs_year").agg(
        v1_usable=("ok", "sum"), v1_defaults=("y", lambda v: int(np.nansum(v))), v1_rate=("y", "mean"))
    b = out.assign(ok=ok).groupby("fs_year").agg(v2_usable=("ok", "sum"), v2_defaults=("y_s5", lambda v: int(np.nansum(v))),
                                                 v2_rate=("y_s5", "mean"))
    C = a.join(b)
    C["d_usable"] = C["v2_usable"] - C["v1_usable"]
    C["d_defaults"] = C["v2_defaults"] - C["v1_defaults"]
    C["rate_ratio_v2_v1"] = C["v2_rate"] / C["v1_rate"]
    W(md(C, ".4f"))
    s8 = fy.between(2008, 2010)
    W(f"\nSelection test years 2008-2010 pooled: v1 {int(v1.loc[s8, 'y'].sum()):,} defaults / {int((s8 & (v1['label_status'].astype(str) == 'ok')).sum()):,} rows "
      f"({v1.loc[s8, 'y'].mean():.4%}); v2 {int(out.loc[s8 & ok, 'y_s5'].sum()):,} / {int((s8 & ok).sum()):,} ({out.loc[s8 & ok, 'y_s5'].mean():.4%}).\n")
    W("## 5. Rows whose label changes between s6 (v1) and s5 (v2)\n")
    def cls(y, s):
        s = s.astype(str)
        return np.where(s == "ok", np.where(y == 1, "ok_y1", "ok_y0"), s)
    trans = pd.DataFrame({"fs_year": fy, "v1": cls(v1["y"], v1["label_status"]), "v2": cls(out["y_s5"], out["status_s5"])})
    # reason by def_date relative to the two windows
    ws6 = stmt + pd.DateOffset(months=6)
    we6 = ws6 + pd.DateOffset(months=12)
    reason = pd.Series("", index=out.index, dtype=object)
    ch = trans["v1"] != trans["v2"]
    r = reason.copy()
    r[ch & (fy == 2011) & (trans["v1"] == "censored")] = "fs2011: s6 window (to 2013-06-30) incomplete, s5 window (to 2013-05-31) complete"
    r[ch & (r == "") & dd.notna() & (dd >= P_["win_start"]) & (dd < ws6)] = "default 1-29 Jun t+1: before s6 start (already_defaulted), inside s5 window"
    r[ch & (r == "") & dd.notna() & (dd >= P_["win_end"]) & (dd < we6)] = "default 1-29 Jun t+2: inside s6 window, after s5 window end"
    r[ch & (r == "")] = "other"
    trans["reason"] = r
    X = trans[ch].groupby(["reason", "v1", "v2"]).size().rename("rows").reset_index()
    W(f"Rows with a different (status, y) class: **{int(ch.sum()):,}** of {n:,}.\n")
    W(md(X.set_index(["reason", "v1", "v2"]), ".0f"))
    Xy = trans[ch].groupby(["fs_year", "v1", "v2"]).size().rename("rows")
    W("\nBy fs_year:\n")
    W(md(Xy.reset_index().set_index(["fs_year", "v1", "v2"]), ".0f"))
    ch_ex11 = ch & (fy != 2011)
    W(f"\nExcluding the fs2011 cohort (which moves from censored to usable): {int(ch_ex11.sum()):,} rows change; "
      f"{int((ch_ex11 & (trans['v1'] == 'already_defaulted') & (trans['v2'] == 'ok_y1')).sum()):,} June-of-t+1 defaults become y=1 "
      f"(gain {int((ch_ex11 & (trans['v1'] == 'already_defaulted')).sum()):,} usable rows) and "
      f"{int((ch_ex11 & (trans['v1'] == 'ok_y1') & (trans['v2'] == 'ok_y0')).sum()):,} June-of-t+2 defaults become y=0. "
      "A firm defaulting on 30 Jun t+1 is y=1 under both (s6 start = 30 Jun) and one defaulting on 30 Jun t+2 is y=0 under both.\n")
    W("## 6. Walk-forward folds v2 (`data/splits/2026-11-08_walkforward_v2.json`)\n")
    W("Folds 1-3 are the only model-selection folds. Fold 4 is the **one-time lockbox per D-087/D-119 - not for model selection**. "
      "`final_fit` is the training set for the final refit (no test rows).\n")
    show = FT[["fold", "role", "train_years", "test_year", "n_train", "train_defaults", "train_default_rate", "n_test", "test_defaults",
               "test_default_rate", "n_test_seen_firm", "n_test_new_firm", "test_defaults_seen_firm", "test_defaults_new_firm",
               "test_default_rate_seen_firm", "test_default_rate_new_firm"]]
    W(md(show.set_index("fold"), ".4f"))
    W(f"\nD-009 sensitivity: lockbox fs2011 usable rows under d2012 = {lockbox['d009_sensitivity_d2012']['fs2011_usable_rows']:,}; "
      f"final_fit under d2012 = fs2007-2010 only, {ff12['n_train']:,} rows / {ff12['train_defaults']:,} defaults. Folds 1-3 identical "
      f"under both treatments: {same_sel}.\n")
    W(f"Files: test rows `{WF2_ROWS.relative_to(ROOT)}` ({len(TR):,} rows; sha256 {spec['test_rows_file_sha256'][:16]}...), "
      f"panel `{PANEL_S5.relative_to(ROOT)}` ({len(out):,} x {out.shape[1]}), fold table `{REP_FOLDS.relative_to(ROOT)}`.")
    REP_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
