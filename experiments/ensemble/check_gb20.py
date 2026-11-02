"""Orchestrator re-check: GB-20/GB-23 metrics from saved predictions; paired firm bootstrap vs GB-01, GB-02, GB-11. Run from repo root."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score as A, log_loss as L
P=pd.read_parquet('data/interim/panel.parquet',columns=['row_id','id','fs_year','y'])
d=P[P.fs_year<2011]
for k in ['GB-01','GB-02','GB-11','GB-20','GB-23']:
    d=d.merge(pd.read_parquet(f'experiments/gbm/{k}/oot_predictions.parquet')[['row_id','pd']].rename(columns={'pd':k}),on='row_id')
d=d.dropna(subset=['y']).reset_index(drop=True)
for k in ['GB-01','GB-02','GB-11','GB-20','GB-23']:
    print(k,len(d),round(A(d.y,d[k]),4),round(L(d.y,d[k]),5),[round(A(g.y,g[k]),4) for _,g in d.groupby('fs_year')])
rng=np.random.default_rng(1)
for name,s in [('pooled',d),('2010',d[d.fs_year==2010].reset_index(drop=True))]:
    gi=s.groupby('id').indices; fi=np.array(list(gi))
    res={o:[] for o in ['GB-01','GB-02','GB-11']}
    for b in range(100):
        rows=np.concatenate([gi[f] for f in rng.choice(fi,len(fi))]); t=s.iloc[rows]; a=A(t.y,t['GB-20'])
        for o in res: res[o].append(a-A(t.y,t[o]))
    for o,v in res.items(): print(name,'GB-20 minus',o,round(np.mean(v),4),np.round(np.percentile(v,[2.5,97.5]),4))
