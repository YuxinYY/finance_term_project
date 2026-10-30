# SC-03 WoE scorecard: explanation (model refit on fold 3 training window, fs_year 2007-2009)

Best discriminator of the benchmark set (see metrics.md). Params: `params_fold3.json`; points table: `scorecard_points_fold3.csv`.

## How to read it
Each ratio is cut into bins learned on the training window; WoE = ln((share of defaults in bin)/(share of non-defaults in bin)), positive = riskier than average. Numeric WoE is forced monotone in the direction of the expected sign (pool-adjacent-violators), so a worse ratio can never earn a better score. A logit on the WoE columns gives PD. Points: score = offset - factor*ln(PD/(1-PD)) with factor = PDO/ln2; here PDO=20, base score 600 at good:bad odds 50:1 (an illustrative convention, not a bank policy). Higher points = safer.

## Coefficients and sign check (WoE columns are oriented so every beta must be positive)

| feature            | family          | expected_effect_of_riskier_bin   |   beta |     se |       z | sign_ok   |   iv_like_woe_range |
|:-------------------|:----------------|:---------------------------------|-------:|-------:|--------:|:----------|--------------------:|
| log_assets         | size            | + (beta>0)                       | 0.4268 | 0.1352 |  3.1569 | True      |              0.5271 |
| equity_ratio       | leverage        | + (beta>0)                       | 0.2461 | 0.028  |  8.7887 | True      |              2.9507 |
| st_debt_ratio      | leverage        | + (beta>0)                       | 0.4222 | 0.0191 | 22.1126 | True      |              2.6542 |
| cash_ratio         | liquidity       | + (beta>0)                       | 0.28   | 0.0634 |  4.4136 | True      |              0.6576 |
| roa_net            | profitability   | + (beta>0)                       | 0.6032 | 0.018  | 33.4225 | True      |              3.1659 |
| interest_coverage  | coverage        | + (beta>0)                       | 0.2815 | 0.0233 | 12.0839 | True      |              2.3601 |
| interest_to_assets | coverage        | + (beta>0)                       | 0.4718 | 0.0227 | 20.7622 | True      |              1.7832 |
| ar_days            | activity        | + (beta>0)                       | 0.534  | 0.0458 | 11.6703 | True      |              0.7903 |
| tangible_ratio     | asset structure | + (beta>0)                       | 0.563  | 0.0358 | 15.7067 | True      |              1.7532 |
| neg_equity         | leverage (flag) | + (beta>0)                       | 0.1359 | 0.0254 |  5.3493 | True      |              2.4003 |
| ateco_group        | industry        | + (beta>0)                       | 0.9852 | 0.0579 | 17.0052 | True      |              0.9752 |
| legal_group        | firm type       | + (beta>0)                       | 0.1669 | 0.0667 |  2.5021 | True      |              0.7347 |
| macro_area         | geography       | + (beta>0)                       | 0.5377 | 0.2051 |  2.6211 | True      |              1.2897 |

A beta near 1 means the WoE is used at face value; beta well below 1 means the feature overlaps with others (collinearity shrinks it). No negative betas in folds 1-3 (see sign check in `decisions.md`).

## Same variables as plain logit (SC-02, fold 3), expected vs actual sign

| term                  | expected_sign           |    beta |     se | ok    |
|:----------------------|:------------------------|--------:|-------:|:------|
| const                 | n/a (categorical dummy) | -5.5957 | 0.269  | n/a   |
| log_assets            | -1                      |  0.0124 | 0.017  | False |
| equity_ratio          | -1                      | -0.352  | 0.1002 | True  |
| st_debt_ratio         | 1                       |  1.1799 | 0.0611 | True  |
| cash_ratio            | -1                      | -0.3173 | 0.1809 | True  |
| roa_net               | -1                      | -9.2162 | 0.2243 | True  |
| interest_coverage     | -1                      | -0.065  | 0.0083 | True  |
| interest_to_assets    | 1                       | 15.078  | 0.8013 | True  |
| ar_days               | 1                       |  0.0818 | 0.0079 | True  |
| tangible_ratio        | -1                      | -1.0059 | 0.0648 | True  |
| neg_equity            | 1                       |  0.8349 | 0.0413 | True  |
| ateco_group=A         | n/a (categorical dummy) | -0.5345 | 0.1433 | n/a   |
| ateco_group=E         | n/a (categorical dummy) | -0.213  | 0.152  | n/a   |
| ateco_group=F         | n/a (categorical dummy) | -0.369  | 0.0417 | n/a   |
| ateco_group=G         | n/a (categorical dummy) | -0.2618 | 0.0392 | n/a   |
| ateco_group=H         | n/a (categorical dummy) | -0.2399 | 0.0844 | n/a   |
| ateco_group=I         | n/a (categorical dummy) | -0.6875 | 0.1092 | n/a   |
| ateco_group=J         | n/a (categorical dummy) | -0.2449 | 0.0865 | n/a   |
| ateco_group=L         | n/a (categorical dummy) | -0.5156 | 0.0466 | n/a   |
| ateco_group=M         | n/a (categorical dummy) | -0.296  | 0.0694 | n/a   |
| ateco_group=N         | n/a (categorical dummy) | -0.1366 | 0.0887 | n/a   |
| ateco_group=R         | n/a (categorical dummy) | -0.2856 | 0.1386 | n/a   |
| ateco_group=OTHER     | n/a (categorical dummy) | -0.5851 | 0.1037 | n/a   |
| legal_group=SAU       | n/a (categorical dummy) | -0.6568 | 0.1041 | n/a   |
| legal_group=SPA       | n/a (categorical dummy) | -0.2073 | 0.0546 | n/a   |
| legal_group=SRU       | n/a (categorical dummy) |  0.0472 | 0.0349 | n/a   |
| legal_group=OTHER     | n/a (categorical dummy) |  0.2068 | 1.0093 | n/a   |
| macro_area=Centre     | n/a (categorical dummy) |  0.0545 | 0.036  | n/a   |
| macro_area=Islands    | n/a (categorical dummy) |  0.1554 | 0.0645 | n/a   |
| macro_area=North-East | n/a (categorical dummy) | -0.0596 | 0.0356 | n/a   |
| macro_area=South      | n/a (categorical dummy) |  0.1088 | 0.0442 | n/a   |
| macro_area=OTHER      | n/a (categorical dummy) |  0.8187 | 0.4615 | n/a   |

## Driver curves: bin, default rate, WoE, points (the scorecard)

| feature            | bin                   |      n |   defaults |   bad_rate |     woe |   points |
|:-------------------|:----------------------|-------:|-----------:|-----------:|--------:|---------:|
| log_assets         | <= 14.29              |  23793 |        389 |     0.0163 |  0.2324 |  44.1788 |
| log_assets         | (14.29, 14.36]        |  23793 |        360 |     0.0151 |  0.1538 |  45.1469 |
| log_assets         | (14.36, 14.43]        |  23793 |        329 |     0.0138 |  0.0625 |  46.2705 |
| log_assets         | (14.43, 14.76]        |  95172 |       1302 |     0.0137 |  0.0506 |  46.4179 |
| log_assets         | (14.76, 14.95]        |  47586 |        610 |     0.0128 | -0.0149 |  47.2244 |
| log_assets         | (14.95, 16.57]        | 214137 |       2693 |     0.0126 | -0.0349 |  47.4707 |
| log_assets         | (16.57, 17.14]        |  23793 |        281 |     0.0118 | -0.097  |  48.2347 |
| log_assets         | > 17.14               |  23794 |        231 |     0.0097 | -0.2947 |  50.6697 |
| equity_ratio       | <= 0.002802           |  23793 |       1860 |     0.0782 |  1.861  |  33.8259 |
| equity_ratio       | (0.002802, 0.01176]   |  23793 |        492 |     0.0207 |  0.4714 |  43.6932 |
| equity_ratio       | (0.01176, 0.02373]    |  23793 |        443 |     0.0186 |  0.3645 |  44.4522 |
| equity_ratio       | (0.02373, 0.03781]    |  23793 |        401 |     0.0169 |  0.2632 |  45.1715 |
| equity_ratio       | (0.03781, 0.05401]    |  23793 |        368 |     0.0155 |  0.1761 |  45.7905 |
| equity_ratio       | (0.05401, 0.07285]    |  23794 |        334 |     0.014  |  0.0778 |  46.4885 |
| equity_ratio       | (0.07285, 0.09504]    |  23793 |        308 |     0.0129 | -0.0042 |  47.0706 |
| equity_ratio       | (0.09504, 0.1205]     |  23792 |        277 |     0.0116 | -0.1114 |  47.8316 |
| equity_ratio       | (0.1205, 0.1492]      |  23793 |        249 |     0.0105 | -0.219  |  48.5956 |
| equity_ratio       | (0.1492, 0.1813]      |  23793 |        204 |     0.0086 | -0.4198 |  50.0215 |
| equity_ratio       | (0.1813, 0.2171]      |  23794 |        189 |     0.0079 | -0.4966 |  50.5672 |
| equity_ratio       | (0.2171, 0.3023]      |  47585 |        311 |     0.0065 | -0.6941 |  51.9695 |
| equity_ratio       | (0.3023, 0.3541]      |  23794 |        121 |     0.0051 | -0.944  |  53.7437 |
| equity_ratio       | (0.3541, 0.4142]      |  23792 |        113 |     0.0047 | -1.0124 |  54.2292 |
| equity_ratio       | > 0.4142              | 118966 |        525 |     0.0044 | -1.0896 |  54.7778 |
| st_debt_ratio      | <= 0.05731            |  47586 |        213 |     0.0045 | -1.074  |  60.1247 |
| st_debt_ratio      | (0.05731, 0.1062]     |  23794 |        119 |     0.005  | -0.9607 |  58.7444 |
| st_debt_ratio      | (0.1062, 0.1626]      |  23792 |        127 |     0.0053 | -0.8955 |  57.9498 |
| st_debt_ratio      | (0.1626, 0.3269]      |  71380 |        444 |     0.0062 | -0.7444 |  56.1094 |
| st_debt_ratio      | (0.3269, 0.3796]      |  23792 |        173 |     0.0073 | -0.5855 |  54.173  |
| st_debt_ratio      | (0.3796, 0.4298]      |  23793 |        181 |     0.0076 | -0.5401 |  53.6203 |
| st_debt_ratio      | (0.4298, 0.4796]      |  23793 |        201 |     0.0084 | -0.4347 |  52.3364 |
| st_debt_ratio      | (0.4796, 0.5299]      |  23793 |        239 |     0.01   | -0.2603 |  50.212  |
| st_debt_ratio      | (0.5299, 0.5789]      |  23794 |        247 |     0.0104 | -0.2272 |  49.8081 |
| st_debt_ratio      | (0.5789, 0.6278]      |  23792 |        267 |     0.0112 | -0.1485 |  48.85   |
| st_debt_ratio      | (0.6278, 0.6772]      |  23794 |        302 |     0.0127 | -0.0242 |  47.3349 |
| st_debt_ratio      | (0.6772, 0.7273]      |  23792 |        309 |     0.013  | -0.0009 |  47.0516 |
| st_debt_ratio      | (0.7273, 0.7788]      |  23793 |        362 |     0.0152 |  0.1594 |  45.0989 |
| st_debt_ratio      | (0.7788, 0.8324]      |  23793 |        416 |     0.0175 |  0.3006 |  43.379  |
| st_debt_ratio      | (0.8324, 0.8893]      |  23793 |        534 |     0.0224 |  0.5551 |  40.2785 |
| st_debt_ratio      | (0.8893, 0.9519]      |  23793 |        629 |     0.0264 |  0.7227 |  38.2356 |
| st_debt_ratio      | > 0.9519              |  23794 |       1432 |     0.0602 |  1.5802 |  27.789  |
| cash_ratio         | <= 0.001006           |  95172 |       1500 |     0.0158 |  0.1942 |  45.4718 |
| cash_ratio         | (0.001006, 0.006448]  |  95172 |       1460 |     0.0153 |  0.1667 |  45.6935 |
| cash_ratio         | (0.006448, 0.009308]  |  23793 |        352 |     0.0148 |  0.131  |  45.9824 |
| cash_ratio         | (0.009308, 0.01821]   |  47587 |        682 |     0.0143 |  0.0981 |  46.2483 |
| cash_ratio         | (0.01821, 0.03375]    |  47585 |        635 |     0.0133 |  0.0258 |  46.8325 |
| cash_ratio         | (0.03375, 0.04556]    |  23794 |        276 |     0.0116 | -0.1151 |  47.9708 |
| cash_ratio         | (0.04556, 0.06152]    |  23792 |        256 |     0.0108 | -0.191  |  48.5835 |
| cash_ratio         | (0.06152, 0.08407]    |  23793 |        235 |     0.0099 | -0.2773 |  49.2812 |
| cash_ratio         | (0.08407, 0.1168]     |  23794 |        212 |     0.0089 | -0.3811 |  50.1196 |
| cash_ratio         | > 0.1168              |  71379 |        587 |     0.0082 | -0.4634 |  50.7846 |
| roa_net            | <= -0.06701           |  23793 |       2229 |     0.0937 |  2.0589 |  11.2033 |
| roa_net            | (-0.06701, -0.03218]  |  23793 |        578 |     0.0243 |  0.6361 |  35.9694 |
| roa_net            | (-0.03218, -0.01786]  |  23794 |        379 |     0.0159 |  0.2059 |  43.4568 |
| roa_net            | (-0.01786, -0.005465] |  47585 |        530 |     0.0111 | -0.1571 |  49.7745 |
| roa_net            | (-0.005465, 0.001912] |  95172 |        977 |     0.0103 | -0.2399 |  51.2168 |
| roa_net            | (0.001912, 0.00339]   |  23793 |        237 |     0.01   | -0.2688 |  51.7192 |
| roa_net            | (0.00339, 0.007895]   |  47587 |        365 |     0.0077 | -0.5332 |  56.3207 |
| roa_net            | (0.007895, 0.01126]   |  23792 |        143 |     0.006  | -0.7766 |  60.5574 |
| roa_net            | (0.01126, 0.0158]     |  23794 |        133 |     0.0056 | -0.8493 |  61.8235 |
| roa_net            | (0.0158, 0.02193]     |  23792 |        108 |     0.0045 | -1.0576 |  65.4495 |
| roa_net            | > 0.02193             | 118966 |        516 |     0.0043 | -1.107  |  66.3085 |
| interest_coverage  | <= -0.6855            |  47586 |       2310 |     0.0485 |  1.3529 |  36.0537 |
| interest_coverage  | (-0.6855, 0]          |  19123 |        536 |     0.028  |  0.783  |  40.6816 |
| interest_coverage  | (0, 0.519]            |  28463 |        521 |     0.0183 |  0.347  |  44.2226 |
| interest_coverage  | (0.519, 0.992]        |  71380 |        901 |     0.0126 | -0.0308 |  47.2909 |
| interest_coverage  | (0.992, 1.113]        |  23792 |        288 |     0.0121 | -0.0721 |  47.6258 |
| interest_coverage  | (1.113, 1.363]        |  47586 |        409 |     0.0086 | -0.4185 |  50.4396 |
| interest_coverage  | (1.363, 1.507]        |  23794 |        171 |     0.0072 | -0.5972 |  51.8907 |
| interest_coverage  | (1.507, 2.125]        |  71378 |        375 |     0.0053 | -0.914  |  54.4636 |
| interest_coverage  | > 2.125               | 142759 |        684 |     0.0048 | -1.0072 |  55.2205 |
| interest_to_assets | <= 0.008982           | 190344 |       1591 |     0.0084 | -0.4476 |  53.1338 |
| interest_to_assets | (0.008982, 0.01111]   |  23793 |        212 |     0.0089 | -0.3811 |  52.2286 |
| interest_to_assets | (0.01111, 0.01552]    |  47587 |        462 |     0.0097 | -0.2957 |  51.0666 |
| interest_to_assets | (0.01552, 0.02034]    |  47585 |        481 |     0.0101 | -0.255  |  50.5125 |
| interest_to_assets | (0.02034, 0.02586]    |  47586 |        582 |     0.0122 | -0.0625 |  47.8911 |
| interest_to_assets | (0.02586, 0.02918]    |  23793 |        357 |     0.015  |  0.1453 |  45.0627 |
| interest_to_assets | (0.02918, 0.03313]    |  23793 |        388 |     0.0163 |  0.2298 |  43.9126 |
| interest_to_assets | (0.03313, 0.03821]    |  23793 |        440 |     0.0185 |  0.3576 |  42.1721 |
| interest_to_assets | (0.03821, 0.04628]    |  23794 |        546 |     0.0229 |  0.5777 |  39.1753 |
| interest_to_assets | > 0.04628             |  23793 |       1136 |     0.0477 |  1.3357 |  28.8569 |
| ar_days            | <= 2.951              |  23793 |        204 |     0.0086 | -0.4198 |  53.5092 |
| ar_days            | (2.951, 3.569]        |  23793 |        209 |     0.0088 | -0.3954 |  53.1337 |
| ar_days            | (3.569, 3.964]        |  23794 |        210 |     0.0088 | -0.3907 |  53.0604 |
| ar_days            | (3.964, 4.245]        |  23792 |        217 |     0.0091 | -0.3576 |  52.5504 |
| ar_days            | (4.245, 4.597]        |  47586 |        472 |     0.0099 | -0.2741 |  51.2643 |
| ar_days            | (4.597, 4.915]        |  71379 |        709 |     0.0099 | -0.273  |  51.2477 |
| ar_days            | (4.915, 5]            |  23793 |        266 |     0.0112 | -0.1524 |  49.3882 |
| ar_days            | (5, 5.085]            |  23793 |        292 |     0.0123 | -0.0582 |  47.9368 |
| ar_days            | (5.085, 5.175]        |  23793 |        303 |     0.0127 | -0.0208 |  47.3607 |
| ar_days            | (5.175, 5.281]        |  23793 |        312 |     0.0131 |  0.0088 |  46.9045 |
| ar_days            | (5.281, 5.41]         |  23793 |        346 |     0.0145 |  0.1136 |  45.2907 |
| ar_days            | (5.41, 5.575]         |  23793 |        425 |     0.0179 |  0.3223 |  42.074  |
| ar_days            | > 5.575               | 118966 |       2230 |     0.0187 |  0.3705 |  41.3315 |
| tangible_ratio     | <= 0.01982            | 118965 |       2342 |     0.0197 |  0.4205 |  40.2102 |
| tangible_ratio     | (0.01982, 0.03191]    |  23794 |        438 |     0.0184 |  0.3529 |  41.3073 |
| tangible_ratio     | (0.03191, 0.04856]    |  23793 |        400 |     0.0168 |  0.2607 |  42.8056 |
| tangible_ratio     | (0.04856, 0.07189]    |  23792 |        361 |     0.0152 |  0.1566 |  44.4963 |
| tangible_ratio     | (0.07189, 0.1037]     |  23793 |        349 |     0.0147 |  0.1223 |  45.0537 |
| tangible_ratio     | (0.1037, 0.1441]      |  23793 |        315 |     0.0132 |  0.0185 |  46.7398 |
| tangible_ratio     | (0.1441, 0.1933]      |  23794 |        274 |     0.0115 | -0.1225 |  49.0303 |
| tangible_ratio     | (0.1933, 0.2512]      |  23793 |        264 |     0.0111 | -0.16   |  49.6393 |
| tangible_ratio     | (0.2512, 0.317]       |  23792 |        258 |     0.0108 | -0.1831 |  50.0155 |
| tangible_ratio     | (0.317, 0.4813]       |  47586 |        473 |     0.0099 | -0.272  |  51.4587 |
| tangible_ratio     | (0.4813, 0.5882]      |  23793 |        207 |     0.0087 | -0.4051 |  53.6214 |
| tangible_ratio     | (0.5882, 0.7171]      |  23793 |        185 |     0.0078 | -0.5181 |  55.4573 |
| tangible_ratio     | (0.7171, 0.8441]      |  23793 |        151 |     0.0063 | -0.722  |  58.7697 |
| tangible_ratio     | (0.8441, 0.9366]      |  23794 |         96 |     0.004  | -1.1754 |  66.1353 |
| tangible_ratio     | > 0.9366              |  23793 |         82 |     0.0034 | -1.3327 |  68.6905 |
| neg_equity         | neg_equity=0          | 457850 |       4440 |     0.0097 | -0.2978 |  48.2087 |
| neg_equity         | neg_equity=1          |  18011 |       1755 |     0.0974 |  2.1025 |  38.795  |
| ateco_group        | A                     |   7222 |         54 |     0.0075 | -0.5511 |  62.7057 |
| ateco_group        | C                     | 120816 |       1940 |     0.0161 |  0.2131 |  40.9839 |
| ateco_group        | E                     |   4369 |         48 |     0.011  | -0.1616 |  51.6344 |
| ateco_group        | F                     |  83883 |       1083 |     0.0129 | -0.008  |  47.2691 |
| ateco_group        | G                     |  85690 |       1251 |     0.0146 |  0.1165 |  43.7286 |
| ateco_group        | H                     |  12930 |        170 |     0.0131 |  0.0128 |  46.6761 |
| ateco_group        | I                     |  14032 |         97 |     0.0069 | -0.6341 |  65.0678 |
| ateco_group        | J                     |  10420 |        169 |     0.0162 |  0.2259 |  40.6196 |
| ateco_group        | L                     |  91783 |        764 |     0.0083 | -0.4514 |  59.8731 |
| ateco_group        | M                     |  20244 |        288 |     0.0142 |  0.0916 |  44.4371 |
| ateco_group        | N                     |   9000 |        160 |     0.0178 |  0.3194 |  37.961  |
| ateco_group        | R                     |   3540 |         64 |     0.0181 |  0.3411 |  37.3445 |
| ateco_group        | OTHER                 |  11932 |        107 |     0.009  | -0.3723 |  57.6249 |
| legal_group        | SAU                   |  12499 |        113 |     0.009  | -0.3644 |  48.7958 |
| legal_group        | SPA                   |  52553 |        476 |     0.0091 | -0.3658 |  48.8029 |
| legal_group        | SRL                   | 342710 |       4333 |     0.0126 | -0.0296 |  47.1832 |
| legal_group        | SRU                   |  67977 |       1272 |     0.0187 |  0.3689 |  45.2636 |
| legal_group        | OTHER                 |    122 |          1 |     0.0082 | -0.0662 |  47.3598 |
| macro_area         | Centre                | 102080 |       1453 |     0.0142 |  0.0907 |  45.6328 |
| macro_area         | Islands               |  23640 |        305 |     0.0129 | -0.0076 |  47.1583 |
| macro_area         | North-East            | 121167 |       1482 |     0.0122 | -0.0629 |  48.017  |
| macro_area         | North-West            | 168413 |       2155 |     0.0128 | -0.0173 |  47.3094 |
| macro_area         | South                 |  60411 |        794 |     0.0131 |  0.0102 |  46.8823 |
| macro_area         | OTHER                 |    150 |          6 |     0.04   |  1.2267 |  28.0099 |

## Reason codes for three fold-3 (2010) test firms
Contribution = beta x WoE (log-odds vs an average-risk bin); positive pushes PD up. The three firms are the 10th, 50th and 99th percentile of predicted PD.

### LOW PD firm: predicted PD 0.16% (actual outcome in window: no default)

Raises risk: receivable days = 5.86 (+0.20 log-odds); equity / assets = 0.0485 (+0.04 log-odds); firm size (log assets) = 14.4 (+0.03 log-odds)

Lowers risk: net return on assets = 0.0428 (-0.67 log-odds); industry group = L (-0.44 log-odds); short-term debt / assets = 0.157 (-0.38 log-odds)

### MID PD firm: predicted PD 0.52% (actual outcome in window: no default)

Raises risk: industry group (C -> +0.21 log-odds); tangible fixed assets / assets = 0.0595 (+0.09 log-odds); firm size (log assets) = 14.7 (+0.02 log-odds)

Lowers risk: net return on assets = 0.00548 (-0.32 log-odds); interest coverage = 3.32 (-0.28 log-odds); interest expense / assets = 0.00123 (-0.21 log-odds)

### HIGH PD firm: predicted PD 12.70% (actual outcome in window: no default)

Raises risk: net return on assets = -0.232 (+1.24 log-odds); equity / assets = -0.0809 (+0.46 log-odds); interest coverage = -2.29 (+0.38 log-odds)

Lowers risk: short-term debt / assets = 0.431 (-0.18 log-odds); receivable days = 5.04 (-0.03 log-odds); interest expense / assets = 0.0239 (-0.03 log-odds)
