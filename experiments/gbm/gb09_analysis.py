"""GB-09: does selection on inner slice A transfer to the untouched inner slice B? (inner data only)
Writes GB-09/search_trials.csv and GB-09/selection_check.csv."""
import json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
c = json.load(open(os.path.join(HERE, "GB-09", "config.json")))
rows, chk = [], []
for f in c["folds"]:
    t = pd.DataFrame(f["trials"]); t["fold"] = f["fold"]
    t = pd.concat([t.drop(columns="cfg"), pd.json_normalize(t["cfg"])], axis=1)
    rows.append(t)
    inc = t.iloc[0] if f["fold"] == 1 else t.iloc[1:5].sort_values("A_log_loss").iloc[0]   # GB-01's own pick
    best = t.sort_values("A_log_loss").iloc[0]
    rho = spearmanr(t.A_log_loss, t.B_log_loss).correlation
    chk.append(dict(fold=f["fold"], n_cfgs=len(t), spearman_A_vs_B_logloss=rho,
                    A_gain_vs_GB01=inc.A_log_loss - best.A_log_loss, B_gain_vs_GB01=inc.B_log_loss - best.B_log_loss,
                    A_auc_gain=best.A_auc - inc.A_auc, B_auc_gain=best.B_auc - inc.B_auc,
                    A_ll_spread_all=t.A_log_loss.max() - t.A_log_loss.min(),
                    A_ll_spread_top10=t.A_log_loss.nsmallest(10).max() - t.A_log_loss.min(),
                    chosen=str(f["chosen_cfg"]), oot_auc=f["oot_auc"]))
pd.concat(rows).to_csv(os.path.join(HERE, "GB-09", "search_trials.csv"), index=False)
ck = pd.DataFrame(chk); ck.to_csv(os.path.join(HERE, "GB-09", "selection_check.csv"), index=False)
print(ck.drop(columns="chosen").round(5).to_string())
