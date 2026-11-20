"""Single-call harness (skeleton). The harness-engineer agent completes predict.py.

Contract (assignment Appendix A):
  - one call:  harness("holdout.csv")  ->  list of PDs, same order as input rows
  - a PD in (0,1) for EVERY row, never re-estimates, never uses statistics of the input batch
CLI:  python harness/harness.py holdout.csv predictions.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NA_VALUES = ["", " ", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", "-", "."]
PD_FLOOR, PD_CAP = 3e-4, 0.999  # floor = Basel corporate PD floor 0.03% (verify R-15 before citing)


def _load(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(data, na_values=NA_VALUES, keep_default_na=True, low_memory=False)


def harness(data) -> list[float]:
    df = _load(data)
    from predict import predict_pd  # noqa: E402  (implemented by harness-engineer)

    pd_hat = np.asarray(predict_pd(df), dtype=float)
    fallback = float(np.load(HERE / "params" / "central_tendency.npy")) if (HERE / "params" / "central_tendency.npy").exists() else 0.02
    pd_hat = np.where(np.isfinite(pd_hat), pd_hat, fallback)
    pd_hat = np.clip(pd_hat, PD_FLOOR, PD_CAP)
    assert len(pd_hat) == len(df), "harness must return one PD per input row"
    return pd_hat.tolist()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    src, dst = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "predictions.csv")
    out = harness(src)
    pd.DataFrame({"pd": out}).to_csv(dst, index=False)
    print(f"wrote {len(out)} PDs to {dst}")
