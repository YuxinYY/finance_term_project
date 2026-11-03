"""One bootstrap worker: r2_boot_worker.py <tag> <seed> <n_boot> <comma models>. Same estimator as score_r1.bootstrap_draws."""
import ast, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.common import metrics as M
OUT = ROOT / "reports/tournament"
tag, seed, n, models = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(",")
ns = {"np": np, "M": M, "N_BOOT": n, "SEED": seed}
tree = ast.parse(open(OUT / "score_r1.py").read())
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in {"AUCEngine", "row_loss", "bootstrap_draws"}], type_ignores=[]), "s", "exec"), ns)
tr = pd.read_parquet(ROOT / "data/splits/walkforward_test_rows.parquet")
pn = pd.read_parquet(ROOT / "data/interim/panel.parquet", columns=["row_id", "id", "y", "fs_year"], filters=[("fs_year", "<", 2011)])
assert pn.fs_year.max() <= 2010
b = tr.merge(pn, on="row_id", validate="one_to_one")
c = pd.read_csv(OUT / "r2_contract.csv").set_index("exp")["file"]
P = {e: b[["row_id", "fold"]].merge(pd.read_parquet(ROOT / c[e], columns=["row_id", "fold", "pd"]), on=["row_id", "fold"], how="left")["pd"].to_numpy(float) for e in models}
y, firm, year, seen = b.y.astype(int).to_numpy(), b.id.to_numpy(), b.test_year.to_numpy(), b.firm_seen_in_train.to_numpy(bool)
subs = {"ALL": np.ones(len(y), bool), "2008": year == 2008, "2009": year == 2009, "2010": year == 2010, "seen": seen, "new": ~seen}
d = ns["bootstrap_draws"](y, firm, P, subs, n_boot=n, seed=seed)
np.savez_compressed(OUT / f"r2_boot_{tag}.npz", **{"|".join(k): v for k, v in d.items()})
print("done", tag, flush=True)
