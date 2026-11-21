"""How often does a firm get a counter-intuitive reason code? (mono20, mono20_nogw (D-164) and the unconstrained D-160 model)

Definition of a CONFLICT for firm i and feature f with an intended direction s_f (+1: higher value -> higher PD):
  - the firm is in the WORST decile of f (highest decile if s_f = +1, lowest if s_f = -1) and f LOWERS its PD
    (SHAP < 0), or
  - the firm is in the BEST decile of f and f RAISES its PD (SHAP > 0).
  Binary flags with s_f = +1: flag = 1 and SHAP < 0.
A conflict is VISIBLE when f is among the firm's top-5 drivers by |SHAP| (what a reason-code report would show).

Intended directions:
  - sign_table.csv candidate signs (D-157; literature or feature spec, checked against FY2007), constrained or not;
  - raw amounts ebitda, profit (-) and debt_to_eqty (+): ASSUMPTION (economic intuition, no verified source for an
    amount's sign; these are free in every model).
Data: all 835,016 training rows (in-sample), exact TreeSHAP.
Outputs: reports/final_model/2026-11-21_reason_code_conflicts_by_feature.csv, _summary.json, _examples.csv,
_diagnostics.csv (mono20_nogw: construction-artefact and size checks on the visible conflicts)
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "reports/final_model"
sign = pd.read_csv(REPO / "experiments/gbm/2026-11-20_grid75/sign_table.csv").set_index("feature")
df = pd.read_parquet(HERE / "final_train_features.parquet")

directions = {f: (int(r.candidate), "sign table (D-157)", int(r.constraint) != 0) for f, r in sign.iterrows() if r.candidate != 0}
for f, s in {"ebitda": -1, "profit": -1, "debt_to_eqty": 1}.items():
    directions[f] = (s, "assumption (amount / no verified sign)", False)

# Ratios whose sign flips when the denominator (equity, LT debt + equity) is negative: a negative value there
# means INSOLVENCY, so "low value raises PD" is correct economics, not a model error (D-154 #1).
ARTEFACT_RATIOS = {"debt_to_eqty", "lt_debt_to_capital"}
models = {"mono20": (HERE / "variants/mono20/lightgbm.txt", HERE / "variants/mono20/params.json"),
          "mono20_nogw": (HERE / "variants/mono20_nogw/lightgbm.txt", HERE / "variants/mono20_nogw/params.json"),
          "free (D-160)": (HERE / "final_lightgbm.txt", HERE / "final_params.json")}
rows, summ, examples, diag = [], {}, [], []
for mname, (mf, pf) in models.items():
    feats = json.loads(pf.read_text())["feature_columns"]
    sh = lgb.Booster(model_file=str(mf)).predict(df[feats], pred_contrib=True)[:, :-1]
    sh = pd.DataFrame(sh, columns=feats)
    absv = sh.abs().to_numpy()
    rank = (-absv).argsort(axis=1).argsort(axis=1)  # 0 = largest |SHAP|
    rank = pd.DataFrame(rank, columns=feats)
    any_visible = np.zeros(len(df), bool)
    any_visible3 = np.zeros(len(df), bool)
    any_visible_constrained = np.zeros(len(df), bool)
    any_visible_genuine = np.zeros(len(df), bool)
    for f, (s, src, constrained) in directions.items():
        if f not in feats:
            continue
        x, v = df[f], sh[f]
        if set(x.dropna().unique()) <= {0, 1}:
            worst, best = x == (1 if s > 0 else 0), pd.Series(False, index=x.index)
        else:
            dec = pd.qcut(x.rank(method="first"), 10, labels=False)
            worst = dec == (9 if s > 0 else 0)
            best = dec == (0 if s > 0 else 9)
        conflict = (worst & (v < 0)) | (best & (v > 0))
        visible = conflict & (rank[f] < 5)
        any_visible |= visible.to_numpy()
        artefact = (x < 0) if f in ARTEFACT_RATIOS else pd.Series(False, index=x.index)
        any_visible_genuine |= (visible & ~artefact).to_numpy()
        any_visible3 |= (conflict & (rank[f] < 3)).to_numpy()
        if constrained:
            any_visible_constrained |= visible.to_numpy()
        rows.append(dict(model=mname, feature=f, intended_sign=s, sign_source=src, constrained_in_mono20=constrained,
                         rows_in_extreme_deciles=int((worst | best).sum()), conflicts=int(conflict.sum()),
                         conflict_share_of_extreme_rows=float(conflict.sum() / max((worst | best).sum(), 1)),
                         visible_top5=int(visible.sum()),
                         visible_top5_artefact_negative_denominator=int((visible & artefact).sum()), visible_top5_share_of_all_firms=float(visible.mean()),
                         mean_abs_shap_when_visible=float(v[visible].abs().mean()) if visible.any() else 0.0))
        if mname == "mono20_nogw" and visible.any():
            # Diagnostics: is the conflict a construction artefact (ratio sign flipped by a negative denominator /
            # negative amount) or a size effect (raw amounts scale with firm size)?
            diag.append(dict(feature=f, visible_conflicts=int(visible.sum()),
                             share_negative_value=float((x[visible] < 0).mean()),
                             share_negative_equity=float((df.equity_ratio[visible] < 0).mean()),
                             median_size_conflicts=float(df["size"][visible].median()),
                             median_size_all=float(df["size"].median()),
                             default_rate_conflicts=float(df.label[visible].mean())))
            i = v[visible].abs().idxmax()
            examples.append(dict(feature=f, row=int(i), id=int(df.id[i]), fs_year=int(df.fs_year[i]), value=float(x[i]),
                                 shap=float(v[i]), rank=int(rank[f][i]) + 1, label=int(df.label[i])))
    summ[mname] = dict(firms=len(df), firms_with_visible_conflict_top5=int(any_visible.sum()),
                       share_top5=float(any_visible.mean()), firms_with_visible_conflict_top3=int(any_visible3.sum()),
                       share_top3=float(any_visible3.mean()),
                       firms_with_genuine_visible_conflict_top5=int(any_visible_genuine.sum()),
                       share_genuine_top5=float(any_visible_genuine.mean()),
                       firms_with_visible_conflict_on_mono20_constrained_features=int(any_visible_constrained.sum()))

t = pd.DataFrame(rows).sort_values(["model", "visible_top5"], ascending=[True, False])
t.to_csv(OUT / "2026-11-21_reason_code_conflicts_by_feature.csv", index=False)
(OUT / "2026-11-21_reason_code_conflicts_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
pd.DataFrame(examples).to_csv(OUT / "2026-11-21_reason_code_conflicts_examples.csv", index=False)
pd.DataFrame(diag).to_csv(OUT / "2026-11-21_reason_code_conflicts_diagnostics.csv", index=False)
print(pd.DataFrame(diag).round(4).to_string(index=False))
print(json.dumps(summ, indent=2))
pd.set_option("display.width", 250)
print(t[["model", "feature", "intended_sign", "constrained_in_mono20", "conflict_share_of_extreme_rows", "visible_top5",
         "visible_top5_share_of_all_firms", "mean_abs_shap_when_visible"]].round(4).to_string(index=False))
print(pd.DataFrame(examples).round(4).to_string(index=False))
