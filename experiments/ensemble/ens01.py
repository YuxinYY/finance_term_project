"""ENS-01 (D-086): log-odds averages of GB-01, GAM-01, RC-02 vs GB-01 alone; firm-clustered bootstrap. Run from repo root."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score as A
lg=lambda p: np.log(p/(1-p))
P=pd.read_parquet('data/interim/panel.parquet',columns=['row_id','id','fs_year','y'])
d=P
for k,f in [('gb','experiments/gbm/GB-01/oot_predictions.parquet'),('gam','experiments/gam/GAM-01/oot_predictions_calibrated.parquet'),('rc','experiments/riskcalc/RC-02/oot_predictions.parquet')]:
    x=pd.read_parquet(f)[['row_id','pd']].rename(columns={'pd':k}); d=d.merge(x,on='row_id')
d=d.dropna(subset=['y']); print(len(d), d.y.sum())
for k in ['gb','gam','rc']: d['z'+k]=lg(d[k].clip(1e-6,1-1e-6))
d['e2']=(d.zgb+d.zgam)/2; d['e3']=(d.zgb+d.zgam+d.zrc)/3; d['e7']=0.7*d.zgb+0.3*d.zgam
for c in ['zgb','e2','e7','e3']:
    print(c, round(A(d.y,d[c]),4), [round(A(g.y,g[c]),4) for _,g in d.groupby('fs_year')])
# firm-clustered bootstrap of AUC diff e2 - gb, 2010 and pooled
rng=np.random.default_rng(0); firms=d.id.unique(); idx=d.groupby('id').indices
for name,sub in [('pooled',d),('2010',d[d.fs_year==2010])]:
    fi=sub.id.unique(); gi=sub.reset_index(drop=True).groupby('id').indices; s=sub.reset_index(drop=True)
    for c in ['e2','e7']:
        ds=[]
        for b in range(100):
            pick=rng.choice(fi,len(fi)); rows=np.concatenate([gi[f] for f in pick]); t=s.iloc[rows]
            ds.append(A(t.y,t[c])-A(t.y,t.zgb))
        print(name,c,'dAUC',round(np.mean(ds),4),np.round(np.percentile(ds,[2.5,97.5]),4))
