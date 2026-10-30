"""Markdown table helper: integers with thousands separators, floats with a fixed format."""
from __future__ import annotations

import numpy as np
import pandas as pd


def md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(str)
        elif pd.api.types.is_integer_dtype(s):
            out[c] = s.map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
        elif pd.api.types.is_float_dtype(s):
            whole = s.dropna()
            if len(whole) and np.all(np.mod(whole, 1) == 0) and whole.abs().max() > 1:
                out[c] = s.map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
            else:
                out[c] = s.map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    return out.astype(str).to_markdown(disable_numparse=True)
