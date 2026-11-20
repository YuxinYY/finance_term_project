"""Worked examples for docs/GB25_EXPLAINED_2026-11-03.md, using the saved GB-25 fold-3 model (trained on 2007-09).
Output: doc_examples_output.txt, shap_all_features_2010.csv. Run from repo root. Reads fs_year < 2011 only."""
import sys, numpy as np, pandas as pd
sys.path[:0] = ['.', 'experiments/gbm']
import gbm_lib as G
from src.features import build as B
from sklearn.metrics import log_loss as L, roc_auc_score as A
art = G.load_model('experiments/gbm/GB-25/model_fold3'); cal = art['calibrator']; F = art['features']; bst = art['booster']; fp = art['feature_params']
P = pd.read_parquet('data/interim/panel.parquet'); P = P[P.fs_year < 2011]
print('legal_struct most common:', P.legal_struct.value_counts().head(3).to_dict())
alfa = dict(id=1, fs_year=2010, asst_intang_fixed=.4e6, asst_tang_fixed=3e6, asst_fixed_fin=.6e6, asst_current=6e6, AR=2.5e6, cash_and_equiv=.5e6,
  asst_tot=10e6, eqty_tot=2e6, liab_lt=0., liab_lt_emp=.2e6, debt_bank_st=2e6, debt_bank_lt=1.5e6, debt_fin_st=0., debt_fin_lt=0., AP_st=1.8e6, AP_lt=0.,
  debt_st=5e6, debt_lt=2.8e6, rev_operating=12e6, COGS=11.4e6, prof_operations=.6e6, goodwill=2.7e6, inc_financing=-.3e6, exp_financing=.3e6, prof_financing=0.,
  inc_extraord=-.05e6, taxes=.25e6, profit=.2e6, days_rec=76., ebitda=1.2e6, roa=6., roe=10., wc_net=1e6, margin_fin=-2e6, cf_operations=.8e6,
  HQ_city=15, legal_struct=P.legal_struct.value_counts().index[0], ateco_sector=25)
def score(row):
    df = pd.DataFrame([row]); X = B.transform(df, fp, feature_set='full2')[F]
    raw = float(bst.predict(X)[0]); z = G._logit(raw); own = float(G._sigmoid(cal['a'] + cal['b'] * z))
    return X, raw, z, own
X, raw, z, own = score(alfa)
print('\nALFA features\n', X.T.to_string(header=False))
delta = 0.21749563462808386  # fold-3 common_cal delta, config.json
print(f"\nraw p {raw:.5f}; log-odds {z:.4f}; own-calibrated PD {own:.5f}; after fold-3 prior shift (+{delta:.4f}) {float(G._sigmoid(G._logit(own)+delta)):.5f}")
c = pd.Series(bst.predict(X, pred_contrib=True)[0], index=F + ['_bias'])
print('bias (average log-odds)', round(c['_bias'], 4), 'sum check', round(c.sum(), 4), 'vs', round(z, 4))
print(c.drop('_bias').sort_values(key=np.abs, ascending=False).head(10).round(3).to_string())
for name, ch in [('interest doubled, EBITDA unchanged', dict(exp_financing=.6e6)), ('equity 0.5M (debt_st up 1.5M)', dict(eqty_tot=.5e6, debt_st=6.5e6, wc_net=-.5e6, margin_fin=-3.5e6)),
                 ('net loss -0.5M', dict(profit=-.5e6, cf_operations=.1e6)), ('tangible 0 (moved to current)', dict(asst_tang_fixed=0., asst_current=9e6, wc_net=4e6))]:
    r = {**alfa, **ch}; print(f"{name}: own PD {score(r)[3]:.7f}")
print('\nmissing test: all debt split blank', f"{score({**alfa, **dict(debt_bank_st=np.nan, debt_bank_lt=np.nan, AP_st=np.nan, debt_fin_st=np.nan, debt_fin_lt=np.nan, AP_lt=np.nan)})[3]:.5f}")
t = P[(P.fs_year == 2010) & (P.label_status == 'ok')]
s = t.sample(30000, random_state=0); Xs = B.transform(s, fp, feature_set='full2')[F]
sh = pd.DataFrame(bst.predict(Xs, pred_contrib=True)[:, :-1], columns=F)
tab = pd.DataFrame(dict(mean_abs_shap=sh.abs().mean(), splits=bst.feature_importance('split'))).sort_values('mean_abs_shap', ascending=False)
tab['share'] = tab.mean_abs_shap / tab.mean_abs_shap.sum(); tab.round(4).to_csv('experiments/gbm/GB-25/shap_all_features_2010.csv')
print('\n', tab.round(4).to_string())
print('\nbooster: trees', bst.num_trees())
pp = pd.read_parquet('experiments/gbm/GB-25/oot_predictions.parquet')[['row_id', 'pd']].merge(t[['row_id', 'y']], on='row_id')
rate = fp.get('p0', None); tr = cal['target_rate']
print('2010 log loss: GB-25', round(L(pp.y, pp.pd), 5), '; constant at training rate', round(L(pp.y, np.full(len(pp), tr)), 5), '; constant at actual 2010 rate (hindsight)', round(L(pp.y, np.full(len(pp), pp.y.mean())), 5))
