"""Core-set selection evidence (feature-architect, 2026-10-30).
Inner time split INSIDE the fold-3 training window: fit params + logit on fs_year 2007-2008,
score fs_year 2009. Never reads fs_year 2010. Run: python -m src.features.core_selection
Writes reports/features/core_selection.csv."""
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from src.common.metrics import auc, log_loss
from src.features import build as B
p=pd.read_parquet('data/interim/panel.parquet')
p=p[(p.label_status=='ok')&(p.fs_year<=2009)].reset_index(drop=True)
a=p[p.fs_year<=2008]; b=p[p.fs_year==2009]
P=B.fit(a); Xa=B.transform(a,P,impute=True); Xb=B.transform(b,P,impute=True)
cats=['ateco_group','legal_group','macro_area']
base=['log_assets','equity_ratio','st_debt_ratio','cash_ratio','roa_net','ar_days','tangible_ratio','neg_equity']
sets={
 'A_cov':base+['interest_coverage'],
 'B_cfo':base+['cfo_to_debt'],
 'C_int_ta':base+['interest_to_assets'],
 'D_cov_noroa':[f for f in base if f!='roa_net']+['interest_coverage'],
 'E_cov+int_ta':base+['interest_coverage','interest_to_assets'],
 'F_cfo_noroa':[f for f in base if f!='roa_net']+['cfo_to_debt'],
 'G_A+flag':base+['interest_coverage','debt_split_missing'],
 'H_A_wc':[f if f!='cash_ratio' else 'wc_ratio' for f in base]+['interest_coverage'],
 'I_A_oper':[f if f!='roa_net' else 'roa_oper' for f in base]+['interest_coverage'],
 'J_E+bank':base+['interest_coverage','interest_to_assets','bank_debt_ratio','debt_split_missing'],
}
rows=[]
for k,fs in sets.items():
    Za=B.design_matrix(Xa,P,fs+cats); Zb=B.design_matrix(Xb,P,fs+cats)
    mu,sd=Za.mean(),Za.std().replace(0,1)
    lr=LogisticRegression(max_iter=3000,C=1.0).fit((Za-mu)/sd,a.y)
    pb=lr.predict_proba((Zb-mu)/sd)[:,1]
    co=pd.Series(lr.coef_[0],index=Za.columns)[fs]
    sg=B.expected_signs(fs)
    bad=[f"{f}({co[f]:+.3f})" for f in fs if sg[f]!=0 and np.sign(co[f])!=sg[f]]
    rows.append(dict(set=k, features=" ".join(fs), auc_2009=auc(b.y,pb), logloss_2009=log_loss(b.y,pb), sign_flips=" ".join(bad)))
    print(f"{k:14s} AUC2009={auc(b.y,pb):.4f} LL={log_loss(b.y,pb):.5f} flips={bad}")
pd.DataFrame(rows).to_csv('reports/features/core_selection.csv', index=False)
