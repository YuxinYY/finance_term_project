"""Why do some firms' PDs fall when only interest expense rises? Test: negative EBITDA makes coverage rise with interest."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments/gbm"))
import gbm_lib as G
from src.features import build as B
OUT = ROOT / "reports/tournament"
p = pd.read_parquet(ROOT / "data/interim/panel.parquet", filters=[("fs_year", "==", 2010)])
te = p[p.label_status == "ok"].drop(columns=["y", "def_date", "label_status", "win_start", "win_end"]).reset_index(drop=True)
raw = te[pd.to_numeric(te.exp_financing, errors="coerce") > 0].sample(3000, random_state=3).reset_index(drop=True)   # same rows as r2_audit? see note
F = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4]
out = []
for e, fs in (("GB-25", "full2"), ("GB-20", "full2"), ("GB-01", "full")):
    a = G.load_model(str(ROOT / f"experiments/gbm/{e}/model_fold3"))
    C = []
    for f in F:
        r = raw.copy(); r["exp_financing"] = pd.to_numeric(r.exp_financing) * f
        C.append(G._logit(a["booster"].predict(B.transform(r, a["feature_params"], feature_set=fs)[a["features"]])))
    C = np.vstack(C); down = (np.diff(C, axis=0) < -1e-9).any(axis=0)
    neg = pd.to_numeric(raw.ebitda, errors="coerce").to_numpy() < 0
    out.append({"exp": e, "firms": len(raw), "share_ebitda_negative": float(neg.mean()), "step_down_share_all": float(down.mean()),
                "step_down_share_ebitda_negative": float(down[neg].mean()), "step_down_share_ebitda_nonneg": float(down[~neg].mean()),
                "worst_step_ebitda_nonneg": float(np.diff(C, axis=0)[:, ~neg].min())})
pd.DataFrame(out).to_csv(OUT / "r2_raw_interest_by_ebitda.csv", index=False)
print(pd.DataFrame(out).T)
