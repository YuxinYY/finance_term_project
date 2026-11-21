"""Check that the estimation code reproduces the submitted model.

Run after 2026-11-21_11_estimate_end_to_end.py:
    python experiments/final/2026-11-21_final_fit/2026-11-21_12_verify_rebuild.py   # rebuild/ vs the mono20_nogw bundle
    ... --rebuilt other/dir   (the PD comparison in step 2 needs the 50k sample parquet, see SAMPLE)

1. Byte comparison of lightgbm.txt and preprocessing.json. Kiet's rebuild (Apple M1, estimation/requirements.txt)
   is byte-identical to the submitted bundle.
2. If the bytes differ (another CPU, thread count or library version can change the last float digits),
   it compares every numeric parameter and the PDs of the 50k sample scored by the submitted bundle and by a
   copy of the bundle with the rebuilt files, then reports the largest differences.
Uses only the estimation environment; the harness code itself runs in a subprocess with this Python.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUNDLE = REPO / "harness/2026-11-21_variants/mono20_nogw"
SAMPLE = Path.home() / "Downloads/2026-11-21_Turquoise_full_codebase/data/2026-11-21_sample_50k.parquet"
FILES = ["lightgbm.txt", "preprocessing.json"]


def flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def score(bundle: Path, sample: Path, out: Path):
    code = ("import sys, numpy as np, pandas as pd; sys.path.insert(0, sys.argv[1]); from harness import predict; "
            "pd.DataFrame(np.asarray(predict(pd.read_parquet(sys.argv[2]))).ravel()).to_csv(sys.argv[3], header=False, index=False, float_format='%.17g')")
    subprocess.run([sys.executable, "-c", code, str(bundle), str(sample), str(out)], check=True, stderr=subprocess.DEVNULL)
    return pd.read_csv(out, header=None)[0].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuilt", type=Path, default=HERE / "rebuild")
    a = ap.parse_args()
    submitted = BUNDLE / "artifacts"
    same = {f: (a.rebuilt / f).read_bytes() == (submitted / f).read_bytes() for f in FILES}
    for f, s in same.items():
        print(f"  [{'PASS' if s else 'DIFF'}] {f} byte-identical to the submitted bundle")
    if all(same.values()):
        print("Rebuild reproduces the submitted model exactly.")
        return

    new = dict(flatten(json.loads((a.rebuilt / "preprocessing.json").read_text())))
    old = dict(flatten(json.loads((submitted / "preprocessing.json").read_text())))
    diffs = []
    for k in sorted(set(new) | set(old)):
        u, v = new.get(k), old.get(k)
        if isinstance(u, (int, float)) and isinstance(v, (int, float)) and not isinstance(u, bool):
            if u != v:
                diffs.append((abs(u - v), k, u, v))
        elif u != v:
            diffs.append((np.inf, k, u, v))
    print(f"preprocessing.json: {len(diffs)} differing entries; largest:")
    for d, k, u, v in sorted(diffs, key=lambda t: -t[0] if np.isfinite(t[0]) else -1e300)[:10]:
        print(f"    {k}: rebuilt {u!r} vs submitted {v!r}")

    sample = SAMPLE
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(BUNDLE, tmp / "bundle", ignore=shutil.ignore_patterns("__pycache__"))
        for f in FILES:
            shutil.copyfile(a.rebuilt / f, tmp / "bundle" / "artifacts" / f)
        p_old = score(BUNDLE, sample, tmp / "old.csv")
        p_new = score(tmp / "bundle", sample, tmp / "new.csv")
    d = np.abs(p_new - p_old)
    print(f"50k sample PDs, rebuilt vs submitted: max abs diff {d.max():.2e}, mean abs diff {d.mean():.2e}, "
          f"rank correlation {pd.Series(p_new).corr(pd.Series(p_old), method='spearman'):.6f}")


if __name__ == "__main__":
    main()
