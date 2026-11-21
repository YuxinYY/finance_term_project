"""Exact examples of the D-154 feature-construction issues, for the team.

Joins Chloe's 11.20 round-03 feature files (all labelled FY2007-2011 rows: train + validation + test) with the
raw file on (id, fs_year). For each issue: affected column, rule that causes it, row count, default rate,
where the affected rows sit in the feature's distribution, and 3 example firms whose raw inputs are all
present (no imputation), with the arithmetic recomputed from raw and checked against the file value.
Output: printed tables + construction_issue_examples.csv + construction_issue_summary.csv
"""
import numpy as np
import pandas as pd
from common import load, OUT_DIR

RAW = "/Users/kietnguyen/Downloads/pd-agent-kit/data/raw/train-2025.csv"
RAWCOLS = ["id", "fs_year", "rev_operating", "prof_operations", "profit", "cf_operations", "AR", "ebitda",
           "exp_financing", "cash_and_equiv", "debt_st", "debt_lt", "eqty_tot", "asst_current"]


def safe_div(n, d):  # team rule (feature_rules._safe_divide): a zero denominator becomes 1
    return n / d.mask(d == 0, d + 1)


ISSUES = [
    # (issue id, feature, numerator, denominator, mask on raw, description)
    ("1a zero denominator", "int_coverage", "ebitda", "exp_financing", lambda r: r.exp_financing == 0,
     "exp_financing = 0, so the ratio becomes EBITDA in euros"),
    ("1b zero denominator", "cash_interest_coverage", "cf_operations", "exp_financing", lambda r: r.exp_financing == 0,
     "exp_financing = 0, so the ratio becomes cf_operations in euros"),
    ("1c zero denominator", "cash_to_financing_exp_ratio", "cash_and_equiv", "exp_financing", lambda r: r.exp_financing == 0,
     "exp_financing = 0, so the ratio becomes cash in euros"),
    ("1d zero denominator", "op_margin", "prof_operations", "rev_operating", lambda r: r.rev_operating == 0,
     "rev_operating = 0, so the ratio becomes operating profit in euros"),
    ("1e zero denominator", "AR_ratio", "AR", "rev_operating", lambda r: r.rev_operating == 0,
     "rev_operating = 0, so the ratio becomes receivables in euros"),
    ("1f zero denominator", "quick_ratio", "cash_and_equiv+AR", "debt_st", lambda r: r.debt_st == 0,
     "debt_st = 0, so the ratio becomes cash + AR in euros"),
    ("2a negative revenue", "op_margin", "prof_operations", "rev_operating", lambda r: (r.rev_operating < 0) & (r.prof_operations < 0),
     "rev_operating < 0 and operating loss: two negatives give a positive margin"),
    ("2b negative revenue", "net_prof_margin", "profit", "rev_operating", lambda r: (r.rev_operating < 0) & (r.profit < 0),
     "rev_operating < 0 and net loss: two negatives give a positive margin"),
    ("2c negative revenue", "operating_cash_flow_margin", "cf_operations", "rev_operating", lambda r: (r.rev_operating < 0) & (r.cf_operations < 0),
     "rev_operating < 0 and negative cash earnings: positive margin"),
    ("3a negative equity", "lt_debt_to_capital", "debt_lt", "debt_lt+eqty_tot", lambda r: (r.debt_lt + r.eqty_tot) < 0,
     "debt_lt + eqty_tot < 0 (equity more negative than LT debt): ratio <= 0, next to firms with no LT debt"),
    ("3b negative equity", "debt_to_eqty", "debt_st+debt_lt", "eqty_tot", lambda r: r.eqty_tot < 0,
     "eqty_tot < 0: debt/equity turns negative, i.e. 'less levered' than a debt-free firm"),
]


def ev(df, expr):
    out = 0
    for p in expr.split("+"):
        out = out + df[p]
    return out


def main():
    f = pd.concat([load("round 03", s) for s in ("train", "validation", "test")], ignore_index=True)
    raw = pd.read_csv(RAW, usecols=RAWCOLS)
    d = f.merge(raw, on=["id", "fs_year"], how="left", suffixes=("", "_raw"))
    base_dr = d.label.mean()
    summ, ex = [], []
    for iid, feat, num, den, mask, desc in ISSUES:
        n_, d_ = ev(d, num), ev(d, den)
        m = mask(d).fillna(False)
        sub = d[m]
        pct = (d[feat].rank(pct=True))[m]
        summ.append(dict(issue=iid, column=feat, formula=f"{num} / {den}", what_happens=desc, rows=int(m.sum()),
                         share_pct=round(100 * m.mean(), 2), default_rate_pct=round(100 * sub.label.mean(), 2),
                         vs_average=round(sub.label.mean() / base_dr, 1),
                         median_percentile_of_value=round(100 * pct.median(), 1),
                         feature_median_value=float(sub[feat].median())))
        # examples: raw inputs present, training years, recomputed value matches the file
        need = [c for c in set(num.split("+") + den.split("+")) if c in raw.columns]
        cand = sub[sub[need].notna().all(axis=1) & (sub.fs_year <= 2009)].copy()
        cand["recomputed"] = safe_div(ev(cand, num), ev(cand, den))
        cand = cand[np.isclose(cand.recomputed, cand[feat], rtol=1e-4, atol=1e-6)]
        # prefer a defaulter among the examples if there is one
        pick = pd.concat([cand[cand.label == 1].sort_values("id").head(1), cand[cand.label == 0].sort_values("id").head(2)])
        for _, r in pick.iterrows():
            ex.append(dict(issue=iid, column=feat, id=int(r.id), fs_year=int(r.fs_year), defaulted_next_year=int(r.label),
                           **{c: r[c] for c in need}, value_in_file=r[feat], recomputed_from_raw=r.recomputed,
                           percentile_in_column=round(100 * d[feat].rank(pct=True)[r.name], 1)))
    S = pd.DataFrame(summ); E = pd.DataFrame(ex)
    S.to_csv(OUT_DIR / "construction_issue_summary.csv", index=False)
    E.to_csv(OUT_DIR / "construction_issue_examples.csv", index=False)
    print(f"all labelled rows FY2007-2011: {len(d):,}; average default rate {100*base_dr:.2f}%\n")
    print(S.drop(columns=["what_happens"]).to_string(index=False))
    print()
    pd.set_option("display.width", 250)
    for iid in E.issue.unique():
        e = E[E.issue == iid].dropna(axis=1, how="all")
        print(f"--- {iid}: {e.column.iloc[0]}")
        print(e.drop(columns=["issue", "column"]).to_string(index=False))


if __name__ == "__main__":
    main()
