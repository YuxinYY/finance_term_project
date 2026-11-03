"""Numbers for the Project Update table, recomputed from saved OOT predictions (own level). Run from repo root."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score as A, log_loss as L
M={'GB-25':'experiments/gbm/GB-25','GB-20':'experiments/gbm/GB-20','GB-01':'experiments/gbm/GB-01','GB-11':'experiments/gbm/GB-11','GB-02':'experiments/gbm/GB-02',
   'GAM-01':'experiments/gam/GAM-01','RC-02':'experiments/riskcalc/RC-02','SC-03':'experiments/scorecard/SC-03'}
P=pd.read_parquet('data/interim/panel.parquet',columns=['row_id','id','fs_year','y']); d=P[P.fs_year<2011]
for k,p in M.items():
    d=d.merge(pd.read_parquet(f'{p}/oot_predictions.parquet')[['row_id','pd']].rename(columns={'pd':k}),on='row_id')
d=d.dropna(subset=['y']).reset_index(drop=True); rows=[]
for k in M:
    a=A(d.y,d[k]); rows.append(dict(model=k,n=len(d),defaults=int(d.y.sum()),auc=round(a,4),ar=round(2*a-1,3),log_loss=round(L(d.y,d[k]),5),**{f'auc_{y}':round(A(g.y,g[k]),4) for y,g in d.groupby('fs_year')}))
t=pd.DataFrame(rows); t.to_csv('reports/final/update_table.csv',index=False); print(t.to_string(index=False))
rng=np.random.default_rng(3)
for name,s in [('pooled',d),('2010',d[d.fs_year==2010].reset_index(drop=True))]:
    gi=s.groupby('id').indices; fi=np.array(list(gi)); v=[]
    for b in range(100):
        r=np.concatenate([gi[f] for f in rng.choice(fi,len(fi))]); u=s.iloc[r]; v.append(A(u.y,u['GB-25'])-A(u.y,u['GB-11']))
    print(name,'GB-25 minus GB-11',round(np.mean(v),4),np.round(np.percentile(v,[2.5,97.5]),4))
