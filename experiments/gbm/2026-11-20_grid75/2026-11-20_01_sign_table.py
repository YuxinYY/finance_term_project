"""Monotone sign table for the 75 features.

Rule (KN 2026-11-20: "if the signs are confused keep it unconstrained"):
  constrain feature f with sign s only if
    (a) s comes from economics: the team's old LightGBM bundle (round-03, 80 features) or FEATURE_SPEC, AND
    (b) FY2007 training data agree (round-01 train = FY2007, inside every round's training window, so no leakage):
        continuous: Spearman(decile index, decile default rate) has sign s and |rho| >= 0.7
        0/1 flag:   default rate gap in direction s with |z| >= 2
  otherwise 0 (unconstrained).
Output: sign_table.csv, sign_table.md
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import load, feature_list, OUT_DIR

OLD_BUNDLE = {  # from ~/Downloads/turquoise/artifacts/lightgbm.txt monotone_constraints x feature_names
    "equity_ratio": -1, "st_debt_share": 1, "lt_debt_to_capital": 1, "cash_ratio": -1,
    "cash_to_financing_exp_ratio": -1, "AR_ratio": 1, "payables_ratio": 1, "wc_ratio": -1,
    "cur_debt_ratio": -1, "quick_ratio": -1, "roa": -1, "op_margin": -1, "net_prof_margin": -1,
    "int_coverage": -1, "financing_expense_ratio": 1, "ebitda_to_debt": -1,
    "operating_cash_flow_ratio": -1, "loss_flag": 1, "ateco_sector_te": 1,
}
SPEC = {  # FEATURE_SPEC expected signs, mapped to the closest 75-feature definition
    "size": (-1, "log_assets"), "tangible_asset_ratio": (-1, "tangible_ratio"),
    "goodwill_ratio": (1, "intangible_ratio"), "operating_cash_flow_to_debt": (-1, "cfo_to_debt"),
    "operating_cash_flow_margin": (-1, "ebitda_margin (cash analogue)"),
    "cash_interest_coverage": (-1, "interest_coverage (cash analogue)"),
    "neg_bs_flag": (1, "neg_equity"), "debt_detail_missing": (1, "debt_split_missing"),
}


def check(x, y, s):
    x = pd.Series(x); y = pd.Series(y)
    if set(x.dropna().unique()) <= {0.0, 1.0}:
        n1, n0 = (x == 1).sum(), (x == 0).sum()
        if n1 < 100 or n0 < 100:
            return False, f"flag too rare (n1={n1})"
        p1, p0 = y[x == 1].mean(), y[x == 0].mean()
        p = y.mean(); z = (p1 - p0) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
        return (np.sign(z) == s and abs(z) >= 2), f"DR 1:{p1:.4f} vs 0:{p0:.4f}, z={z:+.1f}"
    dec = pd.qcut(x.rank(method="first"), 10, labels=False)
    dr = y.groupby(dec).mean()
    rho = spearmanr(dr.index, dr.values).statistic
    return (np.sign(rho) == s and abs(rho) >= 0.7), f"decile DR rho={rho:+.2f}; DR d1..d10 " + \
        " ".join(f"{v*100:.1f}" for v in dr.values)


def main():
    tr = load("round 01", "train")
    rows = []
    for f in feature_list():
        if f in OLD_BUNDLE:
            s, src = OLD_BUNDLE[f], "old LightGBM bundle (team)"
        elif f in SPEC:
            s, src = SPEC[f][0], f"FEATURE_SPEC {SPEC[f][1]}"
        else:
            rows.append(dict(feature=f, candidate=0, source="no economic sign (level, dummy, zero-flag, ambiguous)",
                             fy2007_check="", agrees="", constraint=0))
            continue
        ok, msg = check(tr[f].to_numpy(), tr["label"].to_numpy(), s)
        rows.append(dict(feature=f, candidate=s, source=src, fy2007_check=msg, agrees=ok, constraint=s if ok else 0))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_DIR / "sign_table.csv", index=False)
    with open(OUT_DIR / "sign_table.md", "w") as fh:
        fh.write("# Monotone sign table, 75 features (2026-11-20)\n\n"
                 "Rule: constrain only if economic sign AND FY2007 training data agree (see script docstring). "
                 f"Constrained: {int((t.constraint != 0).sum())} of {len(t)}. "
                 f"Candidates dropped for disagreement: {int(((t.candidate != 0) & (t.constraint == 0)).sum())}.\n\n")
        fh.write(t.to_markdown(index=False))
    print(t[t.candidate != 0].to_string(index=False))
    print("constrained:", int((t.constraint != 0).sum()))


if __name__ == "__main__":
    main()
