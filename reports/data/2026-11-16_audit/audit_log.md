# Audit run
raw rows 1,023,552; labelled (s5) rows 835,016; default rate 1.3576%

## Round 1: re-verify documented raw-data claims (01-financial-anomalies.md, 02-missing-value.md)
                                                                identity  match_rate  rows_computable
                               D: prof_operations = rev_operating - COGS    99.9897%          1023375
                                      E: wc_net = asst_current - debt_st    99.9823%          1023546
                                 E: margin_fin = eqty_tot - fixed assets    99.9561%           982500
                          E: roa = prof_operations/asst_tot*100 (+-0.01)    99.9944%          1023525
                                   I: roe = profit/eqty_tot*100 (+-0.01)    99.9886%           950615
                     cf_operations = profit + (ebitda - prof_operations)    99.8572%          1023464
balance: eqty+debt_st+debt_lt+liab_lt+liab_lt_emp = asst_tot (within 1%)    80.0531%          1023170
                            assets: fixed+current = asst_tot (within 1%)    79.3386%          1023238
                                    bank_st+fin_st+AP_st = debt_st (+-1)     1.3185%           900735
                                    bank_lt+fin_lt+AP_lt = debt_lt (+-1)    59.7072%           931261

Missing shares (>0): eqty_corp_family_tot 100.000%, def_date 95.630%, days_rec 72.318%, debt_fin_st 11.995%, debt_bank_st 11.854%, AP_st 11.786%, AP_lt 9.012%, debt_fin_lt 8.995%, debt_bank_lt 8.888%, roe 7.126%, margin_fin 3.982%, prof_financing 0.064%, exp_financing 0.036%, inc_extraord 0.030%, liab_lt 0.029%, HQ_city 0.027%, asst_fixed_fin 0.023%, rev_operating 0.017%, asst_intang_fixed 0.016%, liab_lt_emp 0.016%, debt_lt 0.015%, asst_tang_fixed 0.010%, cf_operations 0.009%, ebitda 0.008%, taxes 0.003%, roa 0.003%, inc_financing 0.003%, COGS 0.002%, prof_operations 0.002%, goodwill 0.002%, profit 0.002%, cash_and_equiv 0.002%, AR 0.001%, wc_net 0.001%, debt_st 0.000%, asst_current 0.000%, eqty_tot 0.000%
neg_bs rows 5,117 (doc: 5,117); default rate among labelled 1.54%
asst_tot min 1,500,000; rows asst_tot<1.5M: 0
eqty_tot<0 share 3.98%; goodwill<0 share 10.52%; rev_operating==0 share 3.05%
def_date years: {2008.0: 881, 2009.0: 3397, 2010.0: 8238, 2011.0: 12197, 2012.0: 14442, 2013.0: 5570}

## Round 2: imputation accuracy (mask known values in FY2010 rows, impute with frozen round-3 medians, compare to truth)
         column  n_masked  median_abs_err_(x asst_tot; roe in pts)  p90_abs_err  spearman_est_vs_truth  share_truth_exactly_0  share_est_0
   asst_current      3000                                   0.1799       0.6178                 0.3876                 0.0007       0.0000
             AR      3000                                   0.1223       0.4015                 0.5024                 0.0027       0.0000
 cash_and_equiv      3000                                   0.0170       0.1415                 0.2343                 0.0130       0.0000
       eqty_tot      3000                                   0.1608       0.5331                 0.1738                 0.0003       0.0000
        debt_st      3000                                   0.1887       0.4710                 0.3446                 0.0020       0.0000
        debt_lt      3000                                   0.1241       0.4653                 0.2546                 0.2773       0.0000
   debt_bank_st      3000                                   0.0639       0.2759                 0.2073                 0.2753       0.0020
   debt_bank_lt      3000                                   0.0220       0.3922                 0.4482                 0.4803       0.5003
          AP_st      3000                                   0.0775       0.2921                 0.5642                 0.0223       0.0023
  rev_operating      3000                                   0.0000       0.0000                 1.0000                 0.0270       0.0270
           COGS      3000                                   0.0000       0.0000                 0.9997                 0.0013       0.0010
         ebitda      3000                                   0.0290       0.1087                 0.3119                 0.0007       0.0000
         profit      3000                                   0.0117       0.0863                 0.1594                 0.0017       0.0000
  exp_financing      3000                                   0.0085       0.0148                 0.1426                 0.0420       0.0000
  cf_operations      3000                                   0.0210       0.0946                 0.3060                 0.0010       0.0000
        liab_lt      3000                                   0.0000       0.0315                 0.2022                 0.6323       0.7617
asst_tang_fixed      3000                                   0.0000       0.0000                 0.9528                 0.0940       0.1207
            roe      3000                                   6.7750      45.5490                 0.1479                 0.0030       0.0167

## Round 3: independent recompute of features from raw (my own formulas, plain division) on complete-input rows
                    feature  rows_compared  exact_match_rate  rows_mine_undefined(denominator 0/NaN)
                       size         170549           1.00000                                       0
               equity_ratio         170549           1.00000                                       0
               debt_to_eqty         170399           1.00000                                     150
              st_debt_share         170366           1.00000                                     183
         lt_debt_to_capital         170403           1.00000                                     146
             bank_debt_conc         139354           1.00000                                   31195
       tangible_asset_ratio         170498           1.00000                                      51
             goodwill_ratio         170547           1.00000                                       2
                 cash_ratio         170546           1.00000                                       3
cash_to_financing_exp_ratio         164537           1.00000                                    6012
             cur_asst_ratio         170549           1.00000                                       0
                   AR_ratio         165187           1.00000                                    5362
             payables_ratio         141019           1.00000                                   29530
                   wc_ratio         170546           1.00000                                       3
             cur_debt_ratio         170279           1.00000                                     270
                quick_ratio         170276           1.00000                                     273
             asset_turnover         170474           1.00000                                      75
                        roa         170547           1.00000                                       2
                        roe         158258           1.00000                                   12291
                  op_margin         165187           1.00000                                    5362
            net_prof_margin         165187           1.00000                                    5362
               int_coverage         164511           1.00000                                    6038
    financing_expense_ratio         170453           1.00000                                      96
             ebitda_to_debt         170336           1.00000                                     213
 operating_cash_flow_margin         165160           1.00000                                    5389
     cash_interest_coverage         164510           1.00000                                    6039
  operating_cash_flow_ratio         170507           1.00000                                      42
operating_cash_flow_to_debt         170335           1.00000                                     214
                  loss_flag         170549           1.00000                                       0
neg_pl_flag: schema definition (incl. ebitda<prof_operations) flags 1801 FY2009 rows; code flags 1796
cash_interest_coverage: schema says NaN when exp_financing<=0; FY2009 rows with exp_financing<=0: 6,047; code instead divides by (0+1) or by the negative value

## Round 4: zero/negative denominators and monotone constraints (pooled labelled FY2007-2011)
                    feature  constraint  rows_denom_0  DR_denom_0  rows_denom_neg  DR_denom_neg  DR_all
               debt_to_eqty           0            11      0.1818           31269        0.1105  0.0136
              st_debt_share           1           233      0.0000              81        0.0123  0.0136
         lt_debt_to_capital           1             5      0.4000           11636        0.1833  0.0136
             bank_debt_conc           0           233      0.0000              81        0.0123  0.0136
cash_to_financing_exp_ratio          -1         28790      0.0134             557        0.0036  0.0136
                   AR_ratio           1         25613      0.0188            2341        0.0756  0.0136
             cur_debt_ratio          -1          1384      0.0217             319        0.0094  0.0136
                quick_ratio          -1          1384      0.0217             319        0.0094  0.0136
                  op_margin          -1         25613      0.0188            2341        0.0756  0.0136
            net_prof_margin          -1         25613      0.0188            2341        0.0756  0.0136
               int_coverage          -1         28790      0.0134             557        0.0036  0.0136
             ebitda_to_debt          -1           233      0.0000              81        0.0123  0.0136
 operating_cash_flow_margin           0         25613      0.0188            2341        0.0756  0.0136
     cash_interest_coverage           0         28790      0.0134             557        0.0036  0.0136
operating_cash_flow_to_debt           0           233      0.0000              81        0.0123  0.0136

Negative-denominator rows vs monotone constraint:
                    feature  constraint  rows_neg_denom  DR_neg_denom  median_percentile_of_those_rows constraint_implies
              st_debt_share           1              81        0.0123                           0.8610            high PD
         lt_debt_to_capital           1           11636        0.1833                           0.1429             low PD
cash_to_financing_exp_ratio          -1             557        0.0036                           0.0005            high PD
                   AR_ratio           1            2341        0.0756                           0.0015             low PD
             cur_debt_ratio          -1             319        0.0094                           0.0002            high PD
                quick_ratio          -1             319        0.0094                           0.0002            high PD
                  op_margin          -1            2341        0.0756                           0.9925             low PD
            net_prof_margin          -1            2341        0.0756                           0.9804             low PD
               int_coverage          -1             557        0.0036                           0.0442            high PD
             ebitda_to_debt          -1              81        0.0123                           0.0177            high PD

## Round 5: rows dropped by Chloe's pipeline, labels, target encoding
FY2007 features_train.csv: labelled raw 144,251, Chloe rows 144,251, dropped 0; DR dropped nan% vs kept 0.87%
   label agreement with our s5 label: 100.0000%; pred_start_dt values: ['2008-06-01']
   ateco_sector_te distinct values per industry: median 5, max 5 (OOF (expected on train))
FY2008 features_validation.csv: labelled raw 161,383, Chloe rows 161,383, dropped 0; DR dropped nan% vs kept 1.21%
   label agreement with our s5 label: 100.0000%; pred_start_dt values: ['2009-06-01']
   ateco_sector_te distinct values per industry: median 5, max 5 (should be 1 on val/test)
FY2009 features_test.csv: labelled raw 170,549, Chloe rows 170,549, dropped 0; DR dropped nan% vs kept 1.58%
   label agreement with our s5 label: 100.0000%; pred_start_dt values: ['2010-06-01']
   ateco_sector_te distinct values per industry: median 5, max 5 (should be 1 on val/test)
duplicate (id, fs_year): 0

## Round 6: distributions and empirical default rates (frozen round-3 features, labelled FY2007-2011)
                    feature    median  share_zero  share_neg  univariate_AUC  constraint constraint_vs_data
operating_cash_flow_to_debt    0.0332    0.001044     0.2188          0.2357           0                   
  operating_cash_flow_ratio   0.02212    0.001044     0.2187          0.2414          -1             agrees
             ebitda_to_debt   0.06578   0.0007329     0.1579          0.2455          -1             agrees
                     profit 1.164e+04    0.001892     0.3461           0.248           0                   
               equity_ratio    0.1868   1.317e-05    0.03745          0.2597          -1             agrees
     cash_interest_coverage     1.864    0.001044     0.2192          0.2642           0                   
                     ebitda 1.559e+05   0.0007329     0.1579          0.2654           0                   
                        roa      2.53    0.002163     0.2267          0.2671          -1             agrees
            prof_operations 8.762e+04   0.0006874     0.2276          0.2679           0                   
               int_coverage     3.238   0.0007329     0.1583          0.2716          -1             agrees
 operating_cash_flow_margin    0.0357    0.001044     0.2177          0.2742           0                   
            net_prof_margin  0.005315    0.001892     0.3448          0.2874          -1             agrees
                  op_margin   0.04183   0.0006874     0.2266          0.2976          -1             agrees
                  loss_flag         0      0.6539          0          0.6754           1             agrees
             goodwill_ratio    0.1135   0.0007413     0.1012          0.3619           0                   
    financing_expense_ratio   0.01081     0.03448  0.0006671          0.6268           1             agrees
       tangible_asset_ratio      0.15     0.09679  0.0002922           0.384           0                   
                   wc_ratio    0.1015   2.156e-05     0.3019          0.3856          -1             agrees
             cur_debt_ratio     1.234   0.0001701   0.000382          0.3871          -1             agrees
             payables_ratio    0.1293     0.02954   0.001146           0.595           1             agrees
             cur_asst_ratio     0.704   0.0001701          0          0.5929           0                   
cash_to_financing_exp_ratio     1.388     0.01271   0.002776          0.4081          -1             agrees
                quick_ratio    0.7891   0.0002371  0.0008982          0.4132          -1             agrees
              exp_financing 3.959e+04     0.03448  0.0006671          0.5838           0                   
                   AR_ratio    0.4166    0.001939   0.003324          0.5824           1             agrees
                 cash_ratio   0.01243     0.01271   0.002115          0.4348          -1             agrees
              st_debt_share    0.8408    0.001657  0.0002946          0.5646           1             agrees
            ateco_sector_te   0.01362           0          0          0.5624           1             agrees
                        roe      2.99    0.004408      0.284          0.4485           0                   
                  legal_SRU         0      0.8529          0          0.5337           0                   
                       COGS 1.935e+06   0.0006144   0.007674          0.5312           0                   
               debt_to_eqty     3.354    0.000279    0.03753          0.4689           0                   
                       size     15.06           0          0          0.4734           0                   
               zero_debt_lt         0      0.7226          0          0.5228           0                   
             zero_bank_debt         0      0.8199          0          0.4807           0                   
             city_Lombardia         0      0.7311          0          0.4825           0                   
                  legal_SPA         0      0.8946          0          0.4835           0                   
         lt_debt_to_capital    0.2806      0.2774   0.004225          0.4837           1           CONFLICT
                  legal_SRL         1      0.2781          0          0.4872           0                   
              rev_operating  2.04e+06     0.03086   0.002804          0.4873           0                   
               city_Toscana         0      0.9266          0          0.5103           0                   
              city_Campania         0      0.9415          0          0.5089           0                   
             bank_debt_conc    0.2505      0.1801   0.000406          0.5064           0                   
         zero_rev_operating         0      0.9691          0          0.5061           0                   
             asset_turnover    0.5919     0.03086   0.002804           0.494           0                   
        debt_detail_missing         0      0.8586          0          0.5058           0                   
                city_Marche         0      0.9718          0          0.5049           0                   
                neg_pl_flag         0        0.99          0          0.5044           0                   
                  legal_SAU         0      0.9747          0          0.4957           0                   
   city_Trentino-Alto Adige         0      0.9788          0          0.4957           0                   
        city_Emilia-Romagna         0      0.8995          0          0.4961           0                   
                 city_Lazio         0      0.8992          0          0.5027           0                   
                city_Veneto         0      0.8898          0          0.5022           0                   
              city_Sardegna         0      0.9848          0          0.4984           0                   
                city_Puglia         0      0.9662          0          0.5012           0                   
              city_Calabria         0      0.9871          0          0.4989           0                   
               city_Sicilia         0      0.9653          0           0.501           0                   
              city_Piemonte         0      0.9349          0          0.5009           0                   
                zero_profit         0      0.9981          0          0.5009           0                   
                  zero_cash         0      0.9873          0          0.5009           0                   
            city_Basilicata         0      0.9955          0          0.4994           0                   
         city_Valle d'Aosta         0      0.9982          0          0.4994           0                   
                zero_ebitda         0      0.9993          0          0.5006           0                   
                city_Umbria         0      0.9873          0          0.5005           0                   
               zero_debt_st         0      0.9983          0          0.5005           0                   
       zero_prof_operations         0      0.9993          0          0.5005           0                   
 city_Friuli-Venezia Giulia         0      0.9798          0          0.4995           0                   
                 zero_AP_st         0      0.9705          0          0.4996           0                   
               city_Unknown         0      0.9997          0          0.5004           0                   
         zero_cf_operations         0       0.999          0          0.5004           0                   
               city_Liguria         0      0.9816          0          0.5004           0                   
                city_Molise         0      0.9975          0          0.4996           0                   
                neg_bs_flag         0      0.9951          0          0.5003           0                   
                    zero_ar         0      0.9981          0          0.4997           0                   
         zero_exp_financing         0      0.9655          0          0.4998           0                   
            zero_total_debt         0      0.9997          0          0.4999           0                   
                zero_wc_net         0           1          0             0.5           0                   
                  legal_SRS         0           1          0             0.5           0                   
          zero_asst_current         0      0.9998          0             0.5           0                   
              zero_asst_tot         0           1          0             0.5           0                   

Decile default-rate pattern for constrained features (D1 = lowest values):
  equity_ratio                 (10 bins) 0.054 0.018 0.015 0.012 0.010 0.007 0.006 0.004 0.004 0.005
  st_debt_share                (9 bins) 0.006 0.008 0.011 0.014 0.016 0.017 0.017 0.016 0.018
  lt_debt_to_capital           (9 bins) 0.020 0.004 0.005 0.006 0.008 0.010 0.011 0.014 0.025
  cash_ratio                   (10 bins) 0.017 0.017 0.017 0.017 0.015 0.014 0.012 0.010 0.008 0.009
  cash_to_financing_exp_ratio  (10 bins) 0.020 0.019 0.019 0.017 0.014 0.012 0.010 0.008 0.008 0.010
  AR_ratio                     (10 bins) 0.011 0.010 0.010 0.010 0.010 0.012 0.015 0.020 0.019 0.021
  payables_ratio               (10 bins) 0.008 0.008 0.010 0.011 0.013 0.013 0.015 0.016 0.016 0.025
  wc_ratio                     (10 bins) 0.033 0.018 0.014 0.013 0.011 0.010 0.009 0.007 0.010 0.011
  cur_debt_ratio               (10 bins) 0.015 0.027 0.022 0.018 0.012 0.010 0.008 0.008 0.008 0.008
  quick_ratio                  (10 bins) 0.013 0.018 0.021 0.021 0.016 0.013 0.012 0.007 0.006 0.008
  roa                          (10 bins) 0.060 0.015 0.011 0.009 0.008 0.008 0.007 0.007 0.005 0.005
  op_margin                    (10 bins) 0.042 0.030 0.014 0.009 0.009 0.008 0.006 0.005 0.005 0.007
  net_prof_margin              (10 bins) 0.043 0.027 0.016 0.014 0.009 0.007 0.005 0.004 0.003 0.008
  int_coverage                 (10 bins) 0.049 0.028 0.013 0.012 0.009 0.006 0.005 0.004 0.005 0.006
  financing_expense_ratio      (10 bins) 0.011 0.009 0.008 0.007 0.009 0.010 0.012 0.014 0.019 0.035
  operating_cash_flow_ratio    (10 bins) 0.064 0.013 0.013 0.011 0.009 0.007 0.005 0.005 0.004 0.005
  ebitda_to_debt               (10 bins) 0.058 0.018 0.014 0.012 0.010 0.008 0.006 0.004 0.003 0.003
  ateco_sector_te              (6 bins) 0.008 0.014 0.015 0.016 0.016 0.018

Univariate AUC by year (top 15 features):
  feature                      2007 2008 2009 2010 2011
  operating_cash_flow_to_debt  0.259 0.232 0.233 0.244 0.224
  operating_cash_flow_ratio    0.266 0.237 0.239 0.250 0.231
  ebitda_to_debt               0.274 0.244 0.243 0.252 0.240
  profit                       0.272 0.237 0.252 0.259 0.236
  equity_ratio                 0.282 0.255 0.247 0.265 0.253
  cash_interest_coverage       0.281 0.256 0.264 0.274 0.251
  ebitda                       0.292 0.260 0.262 0.272 0.264
  roa                          0.296 0.260 0.267 0.278 0.263
  prof_operations              0.297 0.257 0.268 0.279 0.265
  int_coverage                 0.289 0.268 0.270 0.279 0.260
  operating_cash_flow_margin   0.294 0.264 0.272 0.282 0.269
  net_prof_margin              0.303 0.278 0.292 0.295 0.282
  op_margin                    0.324 0.280 0.299 0.305 0.298
  loss_flag                    0.659 0.682 0.666 0.672 0.683
  goodwill_ratio               0.377 0.388 0.350 0.367 0.355
