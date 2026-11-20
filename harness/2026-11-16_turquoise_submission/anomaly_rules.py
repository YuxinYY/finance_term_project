# Frozen notebook export; no fitting is performed by the harness.
import numpy as np
import pandas as pd
from bundle_paths import project_path

# Flags for economically implausible values; see docs/project-update/features/01-financial-anomalies.md
# Original values are never changed. Run on raw data, before fill_missing (02_missing_values.ipynb), so flags reflect reported values, not imputed ones.
# Non-negative by definition (Riva 10.2, 10.3.2). AR, AP_st, AP_lt are left out: offsetting can make them negative in practice.
NEG_BS_COLS = [
    "asst_intang_fixed", "asst_tang_fixed", "asst_fixed_fin", "liab_lt", "liab_lt_emp",
    "debt_bank_st", "debt_bank_lt", "debt_fin_st", "debt_fin_lt", "debt_st", "debt_lt",
]


def flag_anomalies(df):
    """Take raw data, return a copy with the 0/1 column neg_bs_flag added. Missing inputs never trigger a flag."""
    x = df[NEG_BS_COLS].apply(pd.to_numeric, errors="coerce")  # blanks, "NA" etc. become NaN; NaN comparisons are False
    df = df.copy()
    df["neg_bs_flag"] = (x < 0).any(axis=1).astype(int)
    return df

