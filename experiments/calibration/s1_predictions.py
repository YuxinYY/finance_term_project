"""Stage 1 (calibration-specialist): cache every prediction the calibration study needs.

All fits use only the fold's training rows (fs_year <= t, label_status == 'ok').
Outputs in experiments/calibration/cache/:
  gb_train_k.parquet  : GB-01 final model (refit with experiments/gbm/gbm_lib.fit_pd_model, identical
                        code/seed) -> raw and anchored PD on every TRAINING row of fold k (in-sample)
  gb_oot_check.csv    : refit reproduces the submitted OOT PDs (max |diff|)
  rc_train_k.parquet  : RC-02 saved params_fold{k}.json -> PD on every training row of fold k
  rc_innerB_k.parquet : RC-02 inner model (fit on the same inner-train rows GB-01 used) -> PD on
                        GB-01's in-window held-out slice B (same rows as inner_B_predictions)
  final_gb.parquet / final_rc.parquet : "fold 4" = models fit on fs_year 2007-2010 (the rows a
                        final harness refit would use under the primary label) -> PD on fs_year
                        2010, 2011 and 2012 rows (any label status). Used for the 2011 study (Q6)
                        and the final level recommendation. 2011/2012 labels are NOT used in any fit.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/gbm"))
sys.path.insert(0, str(ROOT / "experiments/riskcalc"))
import gbm_lib as G  # noqa: E402
import riskcalc as RC  # noqa: E402

OUT = ROOT / "experiments/calibration/cache"
OUT.mkdir(parents=True, exist_ok=True)
panel = pd.read_parquet(ROOT / "data/interim/panel.parquet")
splits = json.load(open(ROOT / "data/splits/walkforward.json"))
testrows = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet")
gb_oot = pd.read_parquet(ROOT / "experiments/gbm/GB-01/oot_predictions.parquet")
ok = panel[panel.label_status == "ok"]

checks = []
folds = [dict(fold=f["fold"], train_years=f["train_years"]) for f in splits["folds"]]
folds.append(dict(fold=4, train_years=[2007, 2008, 2009, 2010]))
for f in folds:
    k, yrs = f["fold"], f["train_years"]
    tr = ok[ok.fs_year.isin(yrs)].reset_index(drop=True)
    t0 = time.time()
    # ---------------- GB-01 refit (same code, same seed) ----------------
    art = G.fit_pd_model(tr, exp="GB-01", log=lambda *a: None)
    rec = art["record"]
    if k <= 3:
        g = pd.DataFrame({"row_id": tr.row_id.values, "fs_year": tr.fs_year.values, "id": tr.id.values,
                          "y": tr.y.values, "pd_raw": G.predict_pd(art, tr, raw=True),
                          "pd": G.predict_pd(art, tr)})
        g.to_parquet(OUT / f"gb_train_{k}.parquet", index=False)
        te = panel.set_index("row_id").loc[testrows.loc[testrows.fold == k, "row_id"]].reset_index()
        p = G.predict_pd(art, te)
        sub = gb_oot[gb_oot.fold == k].set_index("row_id").loc[te.row_id, "pd"].to_numpy()
        checks.append({"model": "GB-01", "fold": k, "max_abs_diff_vs_submitted": float(np.max(np.abs(p - sub))),
                       "calibrator": json.dumps(art["calibrator"])})
    # ---------------- RC-02 ----------------
    cfg = {**RC.DEFAULT_CONFIG, **{"ratios": list(RC.DEFAULT_CONFIG["ratios"]) + ["bank_debt_ratio", "ap_days"]}}
    if k <= 3:
        prm = RC.load_params(ROOT / f"experiments/riskcalc/RC-02/params_fold{k}.json")
        pd.DataFrame({"row_id": tr.row_id.values, "fs_year": tr.fs_year.values, "id": tr.id.values,
                      "y": tr.y.values, "pd": RC.predict_riskcalc(tr, prm)}).to_parquet(OUT / f"rc_train_{k}.parquet", index=False)
        # inner model on the SAME inner-train rows GB-01 used; score slice B
        u = G._firm_u(tr["id"].to_numpy())
        if len(yrs) >= 2:
            m_tr = (tr.fs_year < max(yrs)).to_numpy()
            m_b = (~m_tr) & (u >= 0.5)
        else:
            m_tr, m_b = u < 0.70, u >= 0.85
        pin = RC.fit_riskcalc(tr[m_tr], cfg)
        b = tr[m_b]
        pd.DataFrame({"row_id": b.row_id.values, "y": b.y.values, "pd_inner": RC.predict_riskcalc(b, pin),
                      "inner_base_rate": pin["base_rate"]}).to_parquet(OUT / f"rc_innerB_{k}.parquet", index=False)
    else:
        prm = RC.fit_riskcalc(tr, cfg)
        RC.save_params(prm, OUT / "final_rc_params.json")
        G.save_model(art, str(OUT / "final_gb_model"))
        sc = panel[panel.fs_year.isin([2010, 2011, 2012])].reset_index(drop=True)
        cols = ["row_id", "id", "fs_year", "y", "label_status", "y_s0", "status_s0", "y_s3", "status_s3",
                "def_date", "stmt_date"]
        o = sc[cols].copy()
        o["pd_gb"] = G.predict_pd(art, sc)
        o["pd_gb_raw"] = G.predict_pd(art, sc, raw=True)
        o["pd_rc"] = RC.predict_riskcalc(sc, prm)
        o.to_parquet(OUT / "final_scored_2010_2012.parquet", index=False)
        checks.append({"model": "fold4 (2007-2010)", "fold": 4, "max_abs_diff_vs_submitted": np.nan,
                       "calibrator": json.dumps(art["calibrator"])})
    print(f"fold {k} done in {time.time() - t0:.0f}s", flush=True)
pd.DataFrame(checks).to_csv(OUT / "gb_oot_check.csv", index=False)
print(pd.DataFrame(checks))
