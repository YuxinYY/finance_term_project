"""Evidence for each candidate-sign feature (27): univariate stability by training year, zero/negative-denominator
states, unconstrained vs monotone partial dependence, nearest twin feature.

Data: round-3 TRAIN file (FY2007-2009) only; raw file joined on (id, fs_year) to know denominator states.
Outputs: sign_evidence.csv, sign_evidence_curves.csv, sign_evidence_states.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from common import load, feature_list, OUT_DIR

RAW = "/Users/kietnguyen/Downloads/pd-agent-kit/data/raw/train-2025.csv"
# numerator, denominator in raw fields (definitions from feature_rules.make_period_features)
DEN = {
    "quick_ratio": ("cash_and_equiv+AR", "debt_st"), "cur_debt_ratio": ("asst_current", "debt_st"),
    "int_coverage": ("ebitda", "exp_financing"), "cash_interest_coverage": ("cf_operations", "exp_financing"),
    "cash_to_financing_exp_ratio": ("cash_and_equiv", "exp_financing"),
    "operating_cash_flow_margin": ("cf_operations", "rev_operating"), "op_margin": ("prof_operations", "rev_operating"),
    "net_prof_margin": ("profit", "rev_operating"), "AR_ratio": ("AR", "rev_operating"),
    "lt_debt_to_capital": ("debt_lt", "debt_lt+eqty_tot"), "st_debt_share": ("debt_st", "debt_st+debt_lt"),
    "ebitda_to_debt": ("ebitda", "debt_st+debt_lt"), "operating_cash_flow_to_debt": ("cf_operations", "debt_st+debt_lt"),
}


def ev(raw, expr):
    out = 0
    for part in expr.split("+"):
        out = out + raw[part]
    return out


def main():
    feats = feature_list()
    st = pd.read_csv(OUT_DIR / "sign_table.csv").set_index("feature")
    cand = st[st.candidate != 0]
    tr = load("round 03", "train")
    cols = ["id", "fs_year", "cash_and_equiv", "AR", "debt_st", "debt_lt", "asst_current", "ebitda", "exp_financing",
            "cf_operations", "rev_operating", "prof_operations", "profit", "eqty_tot"]
    raw = pd.read_csv(RAW, usecols=cols)
    tr = tr.merge(raw, on=["id", "fs_year"], how="left", suffixes=("", "_raw"))
    models = {m: lgb.Booster(model_file=str(OUT_DIR / f"model_{m}_round_03.txt")) for m in ("free", "mono")}
    X = tr[feats].sample(5000, random_state=1).to_numpy(copy=True)
    spear = tr[feats].sample(100000, random_state=2).rank().corr()

    rows, curves, states = [], [], []
    for f, r in cand.iterrows():
        s = int(r.candidate)
        row = dict(feature=f, candidate=s, constraint=int(r.constraint))
        for y in (2007, 2008, 2009):
            d = tr[tr.fs_year == y]
            dec = pd.qcut(d[f].rank(method="first"), 10, labels=False) if d[f].nunique() > 2 else d[f]
            dr = d.label.groupby(dec).mean()
            row[f"rho_{y}"] = round(spearmanr(dr.index, dr.values).statistic, 2) if len(dr) > 2 else np.nan
            row[f"dr_{y}"] = " ".join(f"{v*100:.1f}" for v in dr.values)
        tw = spear[f].drop(f).abs().sort_values(ascending=False)
        row["twin"], row["twin_rho"] = tw.index[0], round(spear.loc[f, tw.index[0]], 2)
        # partial dependence on the round-3 models over train quantiles 1..99
        qs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        j = feats.index(f)
        for q in qs:
            v = float(np.quantile(tr[f], q))
            Xv = X.copy(); Xv[:, j] = v
            curves.append(dict(feature=f, q=q, value=v, **{f"pd_{m}": float(b.predict(Xv).mean()) for m, b in models.items()}))
        # denominator states
        if f in DEN:
            num, den = ev(tr, DEN[f][0]), ev(tr, DEN[f][1])
            masks = {"den = 0 (ratio = numerator in EUR)": den == 0, "den < 0": den < 0,
                     "num < 0 & den > 0": (num < 0) & (den > 0), "normal (num >= 0, den > 0)": (num >= 0) & (den > 0)}
            for name, mk in masks.items():
                mk = mk.fillna(False)
                if mk.sum() == 0:
                    continue
                fv = tr.loc[mk, f]
                states.append(dict(feature=f, state=name, rows=int(mk.sum()), share=round(mk.mean(), 4),
                                   dr=round(tr.loc[mk, "label"].mean(), 4),
                                   feature_p05=float(fv.quantile(0.05)), feature_median=float(fv.median()),
                                   feature_p95=float(fv.quantile(0.95)),
                                   pct_rank_median=round(float((tr[f] < fv.median()).mean()), 3)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "sign_evidence.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT_DIR / "sign_evidence_curves.csv", index=False)
    pd.DataFrame(states).to_csv(OUT_DIR / "sign_evidence_states.csv", index=False)
    print(pd.DataFrame(rows)[["feature", "candidate", "constraint", "rho_2007", "rho_2008", "rho_2009", "twin", "twin_rho"]].to_string(index=False))
    print(pd.DataFrame(states).to_string(index=False))


if __name__ == "__main__":
    main()
