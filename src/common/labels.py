"""Default-label construction for the Banca Massiccia PD project.

Shared, frozen code: challengers must not edit. The data-auditor chooses the parameters and
logs the choice in docs/DECISION_LOG.md (D-002).

Label for a firm-year row with statement date s:
    y = 1  if  s + start_months  <=  def_date  <  s + start_months + horizon_months
    y = 0  otherwise
Rows are EXCLUDED (status column explains why) when:
    - "already_defaulted": def_date < s + start_months  (firm not a performing applicant)
    - "censored":          window end > data_end and no default observed inside the window
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_labels(
    df: pd.DataFrame,
    stmt_col: str | None = "stmt_date",
    year_col: str = "fs_year",
    def_col: str = "def_date",
    start_months: int = 6,
    horizon_months: int = 12,
    data_end: str = "2012-12-31",
) -> pd.DataFrame:
    """Return a copy of df with columns: stmt_dt, win_start, win_end, y, label_status.

    If stmt_col is missing/None, the statement date is assumed to be Dec 31 of fs_year
    (log this assumption). label_status in {"ok", "already_defaulted", "censored"};
    y is NaN for excluded rows.
    """
    out = df.copy()
    if stmt_col and stmt_col in out.columns:
        stmt = pd.to_datetime(out[stmt_col], errors="coerce")
        fallback = pd.to_datetime(out[year_col].astype("Int64").astype(str) + "-12-31", errors="coerce")
        stmt = stmt.fillna(fallback)
    else:
        stmt = pd.to_datetime(out[year_col].astype("Int64").astype(str) + "-12-31", errors="coerce")

    win_start = stmt + pd.DateOffset(months=start_months)
    win_end = win_start + pd.DateOffset(months=horizon_months)
    d = pd.to_datetime(out[def_col], errors="coerce") if def_col in out.columns else pd.Series(pd.NaT, index=out.index)
    end = pd.Timestamp(data_end)

    in_window = d.notna() & (d >= win_start) & (d < win_end)
    already = d.notna() & (d < win_start)
    censored = (~in_window) & (~already) & (win_end > end + pd.Timedelta(days=1))

    status = np.where(already, "already_defaulted", np.where(censored, "censored", "ok"))
    y = np.where(status == "ok", in_window.astype(float), np.nan)

    out["stmt_dt"], out["win_start"], out["win_end"] = stmt, win_start, win_end
    out["y"], out["label_status"] = y, status
    return out


def label_sensitivity(df: pd.DataFrame, starts=(0, 3, 6), **kw) -> pd.DataFrame:
    """Counts of usable rows, defaults and exclusions under different window starts."""
    rows = []
    for s in starts:
        lab = make_labels(df, start_months=s, **kw)
        ok = lab["label_status"] == "ok"
        rows.append({
            "start_months": s,
            "usable_rows": int(ok.sum()),
            "defaults": int(lab.loc[ok, "y"].sum()),
            "default_rate": float(lab.loc[ok, "y"].mean()) if ok.any() else np.nan,
            "already_defaulted": int((lab["label_status"] == "already_defaulted").sum()),
            "censored": int((lab["label_status"] == "censored").sum()),
        })
    return pd.DataFrame(rows)
