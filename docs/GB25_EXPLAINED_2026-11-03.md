# GB-25, explained from zero

Written 3 November 2026, for Kiet, so that you can write the progress update yourself and defend it.

**How to use this.** Read Parts 1 to 4 in order; they build on each other. Everything else can be looked up. Each code block runs by itself (`numpy`, `scikit-learn`), so run them and change the numbers.

**How to trust it.** Each number comes from a file named next to it. The labels below matter, because they tell you how hard you can lean on a claim:

- **Measured:** a script printed it from our data.
- **VERIFIED source:** an agent read the source and checked the claim (`docs/REFERENCES.md`). A human has not checked these yet.
- **Assumption / hypothesis:** our reasoning, with no source or test behind it. Say so out loud.

**Nothing here has been checked by a human.** The whole project was AI-produced and still needs your verification, which is what this document is for. "Referee" below means an AI agent, not an independent person.

---

## Part 0. The one-page version

**What it is.** GB-25 is a gradient-boosted tree model built with LightGBM. It takes 45 inputs from one firm's latest annual statement and outputs one number: the probability of default (PD) over a 12-month window.

**Its three defining choices.**
1. **45 inputs:** the 26 from round 1 plus 19 new ones.
2. **Monotone constraints on 36 of them:** for example, more equity can never raise the PD.
3. **Shallow trees:** 3 levels deep, so each tree can only see small combinations of inputs.

**The result.** On fiscal 2010, the cleanest test year: **AUC 0.8348 [0.8272, 0.8415]**, AR 0.670, log loss 0.07098. That is +0.0042 AUC [+0.0022, +0.0059] over round 1's GB-01 and about 0.023 over our scorecard (0.8115).

**The honest limits.**
- The gain is modest. The riskiest 10% of firms catch the same share of 2010 defaults as GB-01 (51.7% vs 51.8%).
- The PD **level** is still wrong in 2010: 19% more defaults than predicted, even after correction.
- New firms did not improve.
- There is a known defect for firms with negative EBITDA (Part 11).
- The model that will score the holdout **does not exist yet**. GB-25 is a recipe. So far it has been fitted three times, on data ending 2007, 2008 and 2009. The final fit on 2007-2011 has not been done.

**Was it tuned?** Lightly. 4 settings were tried per fold (only 1 in fold 1), with automatic tree-count selection. A bigger search (37 settings) was only run on the old features. It added almost nothing, so we didn't adopt it (Part 7).

**Did we handle overfitting?** Yes, in several ways, and the evidence is mixed in an understandable way. The model fits its training years noticeably harder than the next year, but its out-of-time PDs are not over-spread (Part 8).

**Why does testing stop at 2010?** Because 2011's 12-month window is one month short and 2012 has no outcomes at all. 2011 is also being kept back as a final check (Part 3).

---

## Part 1. The basics you need

### 1.1 What a PD is, and why log-odds keep appearing

A PD of 2% means: among many firms like this one, we expect about 2 in 100 to default in the window.

Models work on **log-odds** rather than probabilities, because a probability is stuck between 0 and 1 while log-odds can be any number, so you can add effects together.

- odds = PD / (1 − PD)
- log-odds = ln(odds)
- going back: PD = 1 / (1 + e^(−log-odds))

The key fact: **adding a constant to log-odds multiplies the odds.** Adding +0.69 doubles the odds, so a low PD roughly doubles.

```python
import numpy as np
def logit(p): return np.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + np.exp(-z))

for pd_ in [0.001, 0.01, 0.02, 0.10, 0.50]:
    print(f"PD {pd_:6.1%}  log-odds {logit(pd_):6.2f}  after +0.69: {sigmoid(logit(pd_) + 0.69):6.1%}")
```

Notice that +0.69 doubles a 1% PD but only takes 50% to about 67%. A step in log-odds is the same step in *odds* everywhere, not the same step in PD.

### 1.2 Two different jobs: ranking and level

The bank uses one number for two jobs (CLAUDE.md §1):

| Job | Needs | Example of failure |
|---|---|---|
| Approve or decline | Riskier firms get higher PDs (**ranking**) | Lending to the wrong firms |
| Price the loan | PD equals the true default rate (**level**) | Every PD is 30% too low, so every loan is underpriced |

**A model can be good at one and bad at the other.** Add the same constant to every firm's log-odds and the ranking is unchanged while the level moves. This is GB-25's actual situation: decent ranking, shaky level.

### 1.3 AUC and AR: measuring ranking

**AUC:** pick one firm that defaulted and one that didn't, at random. AUC is the chance the defaulter got the higher PD. 0.5 is a coin flip and 1.0 is perfect.

**AR** (accuracy ratio) = 2 × AUC − 1. Same information on a different scale, and it's the one Moody's reports.

**What +0.0042 AUC means:** out of every 10,000 defaulter/survivor pairs, 42 more are ordered correctly. Real, and small.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

y  = np.array([1, 1, 0, 0, 0, 0])              # two defaulted, four did not
pd_ = np.array([.30, .05, .20, .04, .03, .01])

pairs = [(a, b) for a in pd_[y == 1] for b in pd_[y == 0]]
right = sum(a > b for a, b in pairs)
print(f"{right} of {len(pairs)} pairs ranked correctly = {right / len(pairs):.3f}")
print("sklearn says:", roc_auc_score(y, pd_))
print("AR =", 2 * roc_auc_score(y, pd_) - 1)
```

**AUC ignores level.** Divide every PD by 10 and AUC doesn't change at all.

### 1.4 Log loss: measuring ranking and level together

For each firm, take the probability the model gave to what actually happened, take −ln of it, and average. Lower is better. It punishes confident mistakes hard.

```python
import numpy as np
def one_firm(pd_, defaulted): return -np.log(pd_ if defaulted else 1 - pd_)

print("Survivor, PD 1%:  ", round(one_firm(.01, False), 4))
print("Defaulter, PD 1%: ", round(one_firm(.01, True), 4))
print("Defaulter, PD 10%:", round(one_firm(.10, True), 4))
```

A log loss number means little alone. **Compare it to a dumb benchmark.** On 2010 (measured, `experiments/gbm/GB-25/doc_examples_output.txt`):

| Prediction on 2010 | Log loss |
|---|---|
| Same PD for everyone, at the training default rate | 0.08672 |
| Same PD for everyone, at the actual 2010 rate (impossible hindsight) | 0.08615 |
| **GB-25** | **0.07098** |
| GB-25 with the level correction | 0.07010 |

Log loss also changes with the default rate, so don't compare it across years: 2010 is higher than 2008 partly because 2010 had more defaults.

### 1.5 Observed/predicted and calibration slope: diagnosing level and spread

- **Observed/predicted** = actual default rate ÷ average PD. 1.0 is right. **1.19 means 19% more defaults than we predicted.**
- **Calibration slope:** how much actual log-odds move per unit of predicted log-odds. 1.0 is right. **Below 1 means PDs are too spread out** (too extreme, the classic sign of overfitting). Above 1 means too compressed.

**Brier score** is reported as *overall probability accuracy*, not as calibration (R-16 p. 32, VERIFIED).

### 1.6 Confidence intervals and "paired" tests

Our data are one sample. Rerun history and the numbers would differ. Square brackets are 95% intervals: a plausible range for the true value.

How we get them: **firm-clustered bootstrap.** Resample whole firms with replacement, because a firm's rows across years aren't independent, recompute the metric, and repeat many times.

**Paired** means both models are scored on the same resample and we keep the *difference*. This matters a lot. Random halves of the same year can differ by up to 0.017 AUC (`GB-07`, fold 2), which is bigger than every model gap we're measuring. Pairing cancels that shared luck. So **only quote CIs on differences when comparing models.**

GB-25's own AUC interval on 2010 is 0.0143 wide, but the interval on its *paired* gain over GB-01 is only 0.0037 wide.

---

## Part 2. The problem and the data

### 2.1 What one row is

One row is **one firm's annual statement** (a firm-year). There are 1,023,552 rows from 237,711 firms, fiscal 2007-2012. The file has no loans or loan applications (D-001), so we model firms, not loans.

### 2.2 The label: the most important decision in the project

Every statement is dated 31 December of year *t*. We define:

> **Default = 1 if the firm defaults between 30 June *t*+1 and 30 June *t*+2.**

Why not start at year-end? Because **the bank can't see the statement on 31 December.** Italian accounts must be approved within 120 days of year-end (180 if the statute allows; Civil Code art. 2364, R-28) and filed within 30 days of approval (art. 2435, R-29). So the statement is available around mid-*t*+1.

If we trained on defaults in the first 6 months after year-end, we'd be predicting defaults that have already happened by the time the bank could use the model. 4,237 rows have a default in that gap (`reports/data/00_labels.md`). They're excluded rather than counted as defaults.

**Three things to know:**
- Those two articles are verified in their **current** text only. We did not find the 2007-2012 wording.
- Real filing dates aren't in the data, so the 6-month start is an **assumption**, a reasoned one.
- It moves the level a lot: 1.41% with this window, 1.07% from year-end (D-002a). This question is still open with Roger/TA.

### 2.3 Default rates by year

| Fiscal year | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 |
|---|---|---|---|---|---|---|
| Rows with a usable label | 144,214 | 161,261 | 170,386 | 174,512 | 183,953 (partial) | none |
| Defaults | 1,379 | 2,064 | 2,752 | 2,968 | 2,357 in 11 months | – |
| Default rate | 0.96% | 1.28% | 1.62% | 1.70% | 1.28% in 11 months | – |

(`data/splits/walkforward.json`, `reports/data/00_labels.md`, D-077.)

**Rates rose every year through 2010.** A model trained on the past always sees a lower default rate than the next year's. That is the root of the level problem.

**Only about 1.3% of rows are defaults.** The signal lives in 7,784 test defaults, not in 506,159 test rows. That's why we report default counts everywhere.

---

## Part 3. Why testing stops at 2010, and what 2011 is for

This confuses people, so here is the timeline. A year is usable as a test only if its full 12-month window has been observed. **The data end on 31 May 2013.**

```
Statement   Window starts   Window ends    Observed by 31 May 2013?
FY 2009     30 Jun 2010     30 Jun 2011    Yes, complete
FY 2010     30 Jun 2011     30 Jun 2012    Yes, complete   <- last complete year
FY 2011     30 Jun 2012     30 Jun 2013    11 of 12 months <- one month short
FY 2012     30 Jun 2013     30 Jun 2014    0 months        <- no outcomes at all
```

**So:**
- **FY 2012 can't be used for training or testing.** There's no outcome to learn from.
- **FY 2011 is incomplete.** About 9% of a cohort's 12-month defaults arrive in month 12 (for the 2010 cohort, 2,697 of 2,968 arrive in the first 11 months; `reports/calibration/cohort_share_11_of_12.csv`). Treating it as a full year would make it look safer than it is.
- **The last complete test year is FY 2010.** Hence three folds.

### The three folds (`data/splits/walkforward.json`)

| Fold | Train on | Test on | Train rows / defaults | Test rows / defaults |
|---|---|---|---|---|
| 1 | 2007 | 2008 | 144,214 / 1,379 | 161,261 / 2,064 |
| 2 | 2007-08 | 2009 | 305,475 / 3,443 | 170,386 / 2,752 |
| 3 | 2007-09 | 2010 | 475,861 / 6,195 | 174,512 / 2,968 |

This is **walk-forward** testing: fit on the past, test on the next year, add a year, repeat. Moody's does the same (R-33 Sec. 4.8 p. 23; R-07 Sec. 3, VERIFIED).

**Why not random cross-validation?** It would mix future years into training and flatter the model. A lender can't do that.

**There's no timing leak.** FY 2009's window closes 30 June 2011, exactly when decisions on FY 2010 statements begin (D-070). So fold 3's labels were all knowable at decision time.

### What 2011 is for

2011 has three separate roles. Keep them apart:

1. **A one-time final check (planned, not done).** We've looked at 2008-2010 **113 times** (D-142), so those scores are flattered by selection. The plan (D-087, D-119, D-148) is to freeze the finalists, score them once on 2011, report the result whatever it says, and use it only to choose the model. **Proposed finalists:** GB-25, GB-01 (to beat), GAM-01 (transparent fallback). **This needs your sign-off before anyone opens 2011.**
2. **Training data for the final model.** After that check, we refit on 2007-2011 using the 11-month label (D-077). Including the truncated cohort beat dropping it in a fold-3 test (log loss 0.07063 vs 0.07086; `reports/calibration/q6_partial_cohort_test.csv`, GB-01).
3. **An input to the level.** The target PD of 1.49% is anchored on 2011's estimated rate (D-073).

**2011 isn't clean, and you must say so.** The calibration round already scored GB-01 and RC-02 on it (GB-01 AUC 0.8497, `reports/calibration/q6_2011_pseudo_test.csv`), and the level rules were tested there. No model was chosen from those numbers, so it still works for the model choice. It **cannot** test the level, because the level was built from it (D-118). GB-25 has **never** been scored on 2011: the code filters `fs_year < 2011`.

**What we don't know: the holdout period.** Roger hasn't told us which years the holdout covers. It's on the list of open questions, and it affects how far ahead the level must reach.

---

## Part 4. The model structure

### 4.1 One tree

A decision tree asks yes/no questions and ends at a **leaf** holding a number.

```
              tangible_ratio < 0.15 ?
              /                    \
            yes                     no
             |                       |
     roa_net < 0.01 ?          equity_ratio < 0.10 ?
       /        \                 /          \
    +0.08     +0.02            +0.01       -0.04      <- leaf values, added to log-odds
```

*(Illustration only; these aren't real splits.)*

Each path combines conditions, which is how trees capture interactions like "low profit matters more if the firm also has few tangible assets". They also handle odd distributions naturally, since a split only cares about order.

### 4.2 Boosting: many small corrections

**Gradient boosting** (R-38 Sec. 2.1, VERIFIED) builds trees in sequence:

1. Start with the average log-odds.
2. Find where the model is currently most wrong (the gradient of the log loss).
3. Fit a small tree to that error.
4. Add a small fraction of it (the **learning rate**, 0.03 for us).
5. Repeat hundreds of times.

The final score is the sum over all trees. GB-25's fold-3 model has **548 trees**, each with at most 8 leaves.

```python
# A toy of the boosting idea: each round corrects a small share of the remaining error.
target, pred, lr = -3.0, -5.0, 0.3
for i in range(1, 9):
    pred += lr * (target - pred)
    print(f"round {i}: prediction {pred:.3f}")
```

**Why LightGBM and not XGBoost?** We didn't compare them. XGBoost isn't installed, and you chose LightGBM. Same family; **do not claim LightGBM was shown to be better.**

### 4.3 Monotone constraints: the key choice

A monotone constraint forces a direction. If `equity_ratio` is constrained to −1, then holding everything else fixed, more equity can **never raise** the PD, anywhere (R-39 LightGBM docs).

**Why accept lower accuracy for this?**
- **The bank must explain decisions.** With the constraints dropped, a firm with cash flow around twice its debt comes out as risky as one burning cash (log-odds −4.66 vs −4.64; `GB-08/`, D-103). No credit officer accepts that.
- **It's a form of regularisation.** Odd reversals in the tail may be noise or a sample quirk. This is partly an assumption: we can't prove that a given reversal is noise.

**What it costs (measured).** Unconstrained GB-02 beats GB-01 by +0.0065 AUC [+0.0052, +0.0079]. So the constraints have a real, known price. We pay it deliberately.

**A pattern we can't explain.** Actual default rates do rise among the very most profitable firms (D-102). We don't have a verified reason. Say "we observe it and constrain it".

**36 of 45 inputs are constrained.** The other 9 have no clear expected direction: `asset_turnover`, `extraord_to_assets`, `rev_nonpositive`, `bs_gap`, `fin_fixed_ratio`, `tfr_ratio`, and the 3 categoricals. LightGBM can't constrain categoricals (R-39).

### 4.4 GB-25's exact settings

From `experiments/gbm/GB-25/config.json`.

| Setting | Value | What it does | Why |
|---|---|---|---|
| `objective` | binary | Minimises log loss | We need probabilities |
| `learning_rate` | 0.03 | Each tree's contribution is small | Slower, more stable learning |
| `num_leaves` / `max_depth` | 8 / 3 (folds 2-3 chosen; fold 1 pre-set) | Tree size | Limits interaction depth |
| `min_data_in_leaf` | 2000 (fold 1: 1000) | Smallest allowed leaf | With a 1.3% default rate, 2,000 rows means about 26 defaults per leaf |
| `feature_fraction` | 0.8 | Each tree sees a random 80% of inputs | Reduces reliance on one input |
| `bagging_fraction` / `freq` | 0.8 / 1 | Each tree sees a random 80% of rows | Same idea, for rows |
| `lambda_l2` | 10 | Shrinks leaf values | Stops extreme leaves |
| `monotone_constraints_method` | intermediate | How strictly constraints limit splits | "advanced" tied (GB-05, D-057) |
| `cat_smooth`, `cat_l2`, `min_data_per_group` | 50, 10, 500 | Regularise categorical splits | Rare levels are noisy |
| Early stopping | 150 rounds, max 4000 | Stop when validation stops improving | Sets tree count |
| Final tree count | 1.2 × best round | More rows support slightly more trees | **Heuristic, not tuned** (D-052) |
| Class weighting | none | Trains on the natural default rate | Reweighting would destroy the level (D-054) |
| Missing values | Native NaN | The tree learns which way "missing" goes | A tie with imputing on GB-01 (D-056); **not retested on the new features** |

**Trees per fold:** 384→461, 480→576, 457→548 (best round → final). Each fold is a separately trained model.

### 4.5 From tree output to PD: three layers

**Layer 1: raw output.** The sum of the trees gives raw log-odds.

**Layer 2: own calibration, fitted inside each training window (D-053).**
`log-odds = a + b × raw log-odds`. The slope `b` comes from slice B, a held-out part of the last training year. The intercept `a` is then set so the average PD over the training window equals the training default rate. For fold 3: a = −0.080, b = 0.977.

**Layer 3: the level shift (prior shift, D-072).** Models learn the past default rate, but the next year's differs, so we add one constant to every firm's log-odds (R-22 eq. 4, VERIFIED):

`shift = logit(target rate) − logit(training rate)`

Because it's the same constant for every firm, **ranking doesn't change.** In the walk-forward the target comes from "rule e", the annualised default speed in the last 6 training months (D-071). Shifts: +0.200, +0.291, +0.217 (config.json).

For the holdout, the proposed target is **1.49%** and the shift is +0.079 if the fit includes 2011 (`experiments/calibration/harness_level_params.json`; recompute it on the final rows). PDs are then floored at 0.03% and capped at 99% (D-075).

**The 1.49% target has judgment in it.** It's the 2011 estimate of 1.43% adjusted under an *assumed* random walk (D-073). Rule e was also picked after seeing 2008-2010, so its measured gain is optimistic (D-071).

---

## Part 5. Every feature

### 5.1 How a feature gets built

Code: `src/features/build.py`. Every step has a reason.

1. **Clean:** turn junk values into NaN so messy holdout rows can't crash the model.
2. **Rebuild from accounting identities:** if a value is missing but an identity gives it, use the identity. For example, short-term debt = current assets − net working capital.
3. **Divide safely:** each ratio has a written rule for zero or negative denominators. A firm with no short-term debt has "infinite" liquidity, which is good; that gets clipped to the upper bound.
4. **Winsorise:** clip to the 1st and 99th percentile of the **training** window. Our choice: Moody's used p2/p98 in RiskCalc v1 (R-05 p. 55 fn. 42).
5. **Signed log** on some ratios: `slog(x) = sign(x) × ln(1 + |x|)`. It tames tails while keeping order. Trees ignore this, since they only care about order.
6. **Flags and categories.**
7. **Leave NaN in place** for LightGBM.

**The golden rule:** all bounds, percentiles and encodings are fitted on training rows and frozen. Features for a test row use only that row and the frozen numbers (CLAUDE.md §2.2). Single-row and batch scoring gave identical features on 999 rows (D-141).

### 5.2 How to read the tables

- **Sign:** − means more of this should lower PD, + should raise it, "free" means unconstrained.
- **AUC alone:** using only this feature, on 2007-09 training rows. **Below 0.5 means PD falls as it rises.** Further from 0.5 means more power alone.
- **SHAP rank:** importance in the fold-3 model on 2010 rows, of 45 (`experiments/gbm/GB-25/shap_all_features_2010.csv`).
- **Source:** where the idea came from. "Family" means R-01 Sec. 3.1 p. 13 lists the ratio family, not this exact ratio.

**Raw fields:** `asst_tot` total assets; `eqty_tot` book equity; `debt_st` / `debt_lt` liabilities due within / after 12 months; `rev_operating` sales; `COGS` all operating costs (it behaves this way in this file); `prof_operations` operating profit; `ebitda` operating profit before depreciation; `exp_financing` interest; `profit` net profit; `cf_operations` net profit + depreciation.

### 5.3 The 26 from round 1

**Size**

| Feature | Formula | What it means for a lender | Sign | AUC alone | SHAP rank | Source |
|---|---|---|---|---|---|---|
| `log_assets` | ln(asst_tot) | Bigger firms are more diversified, with better funding access | − | 0.477 | 32 | Family (R-01); RiskCalc deflates by CPI, we don't (external data) |

**Leverage: how much is owed**

| Feature | Formula | What it means | Sign | AUC alone | SHAP rank | Source |
|---|---|---|---|---|---|---|
| `equity_ratio` | eqty_tot / asst_tot | The cushion: the share of assets that can be lost before creditors are hit | − | 0.267 | 13 | Family; also in R-33 Table 3 |
| `st_debt_ratio` | debt_st / asst_tot | Claims that must be paid or refinanced within the PD horizon | + | 0.706 | 3 | Family; the story is an assumption |
| `bank_debt_ratio` | (debt_bank_st + debt_bank_lt) / asst_tot | Borrowing from banks, whose claims can trigger a missed payment | + | 0.579 | 37 | Assumption |
| `st_debt_share` | debt_st / (debt_st + debt_lt) | Maturity: more short-term means more refinancing risk this year | + | 0.570 | 24 | Assumption |
| `neg_equity` | 1 if eqty_tot < 0 | Liabilities exceed assets | + | 0.624 | **0 splits** | Assumption |

**Liquidity: can it pay soon?**

| Feature | Formula | What it means | Sign | AUC alone | SHAP rank | Source |
|---|---|---|---|---|---|---|
| `wc_ratio` | wc_net / asst_tot | Current assets left after short-term debts | − | 0.384 | 41 | R-32 (Altman), page not recorded |
| `current_ratio` | slog(asst_current / debt_st) | Current assets per euro of short-term debt | − | 0.383 | 36 | Assumption (classic ratio) |
| `cash_ratio` | cash_and_equiv / asst_tot | Immediate liquidity | − | 0.441 | 29 | Family; R-33 uses cash + securities |

**Profitability: does it make money?**

| Feature | Formula | What it means | Sign | AUC alone | SHAP rank | Source |
|---|---|---|---|---|---|---|
| `roa_net` | profit / asst_tot | Net return; losses erode equity and capacity to pay | − | 0.256 | 6 | Family |
| `roa_oper` | prof_operations / asst_tot | Operating return, before financing and one-offs | − | 0.283 | 30 | Family |
| `ebitda_margin` | slog(ebitda / rev_operating) | Operating cash margin per euro of sales | − | 0.279 | 38 | Family |
| `extraord_to_assets` | inc_extraord / asst_tot | One-offs. Both big losses and forced asset sales can precede default | free | 0.462 | 11 | Assumption, labelled hypothesis |

**Debt coverage: can earnings carry the debt?**

| Feature | Formula | What it means | Sign | AUC alone | SHAP rank | Source |
|---|---|---|---|---|---|---|
| `interest_coverage` | slog(ebitda / exp_financing) | Times earnings cover interest | − | 0.276 | 26 | Family. **Has a defect**, Part 11 |
| `cfo_to_debt` | cf_operations / (debt_st + debt_lt) | Cash earnings per euro of debt: the strongest single ratio | − | 0.248 | 21 | Family; similar to R-33's debt coverage |
| `interest_to_assets` | exp_financing / asst_tot | Interest burden (cost of debt × leverage) | + | 0.634 | 22 | Assumption; R-33 uses interest/sales |

**Activity: how the business runs**

| Feature | Formula | What it means | Sign | AUC alone | SHAP rank | Source |
|---|---|---|---|---|---|---|
| `asset_turnover` | rev_operating / asst_tot | Sales per euro of assets. Inverted U in our data, hence no constraint | free | 0.504 | 2 | Family; the story is an assumption |
| `ar_days` | slog(365 × AR / rev_operating) | Days to collect from customers. Slow collection ties up cash | + | 0.583 | 7 | Family; specific ratio is an assumption |
| `ap_days` | slog(365 × AP_st / COGS) | Days to pay suppliers. Stretching them signals stress | + | 0.591 | 19 | R-33 uses payables/sales |
| `rev_nonpositive` | 1 if rev_operating ≤ 0 | Holding, start-up or wind-down firm | free | 0.507 | **0 splits** | Assumption |

**Asset structure**

| Feature | Formula | What it means | Sign | AUC alone | SHAP rank | Source |
|---|---|---|---|---|---|---|
| `tangible_ratio` | asst_tang_fixed / asst_tot | **Top driver.** Fixed physical assets | − | 0.384 | **1** | See below |
| `intangible_ratio` | asst_intang_fixed / asst_tot | Capitalised costs and brands lose value in distress | + | 0.550 | 31 | Assumption |

**`tangible_ratio` is your biggest exposure.** It's the #1 driver, and **no verified source says it lowers PD.** R-48 (Almeida & Campello, VERIFIED) says tangible assets raise what creditors recover in default and support borrowing. That's about **recovery** in US public firms, and their tangibility measure differs from ours. Lower PD is **our data finding**, in every industry with enough defaults (within-industry AUC 0.56-0.73, `scorecard_r1.md` T16). Don't say "the literature shows".

**Missing-data flag and categories**

| Feature | Definition | What it means | Sign | SHAP rank | Source |
|---|---|---|---|---|---|
| `debt_split_missing` | 1 if any of debt_bank_st, debt_bank_lt, AP_st is missing | Debt breakdown absent (mostly small firms, D-016) | + | **0 splits** | R-34 (flag next to an imputed value) |
| `ateco_group` | ATECO section; under 30 training defaults → OTHER | Industry | free | 15 | R-01 p. 14; R-54 for the codes |
| `legal_group` | Legal form (SRL, SRU, SPA, SAU, OTHER) | Governance and capital rules | free | 27 | Assumption; code meanings unverified |
| `macro_area` | Province → five macro-areas | Regional conditions | free | 40 | Assumption; R-53 for the codes |

### 5.4 The 19 added in round 2

**F1. Cost-structure ratios (from RiskCalc 4.0 France, R-33 Table 3 p. 11, VERIFIED).** Dropping this group costs AUC in both inner windows: −0.0025 [−0.0041, −0.0006] and −0.0016 [−0.0030, −0.0006] (`GB-21`).

| Feature | Formula | What it means | Sign | AUC alone | SHAP rank |
|---|---|---|---|---|---|
| `operating_margin` | slog(prof_operations / rev_operating) | Buffer before costs exceed revenue | − | 0.291 | 35 |
| `interest_to_sales` | slog(exp_financing / rev_operating) | Share of each sales euro going to interest. **This is the feature that separates GB-25 from GB-20** | + (forced) | 0.580 | 16 |
| `payables_to_sales` | slog(AP_st / rev_operating) | Suppliers stretched relative to sales | + | 0.636 | 9 |
| `personnel_to_sales` | slog((goodwill − ebitda) / rev_operating) | Wage bill per euro of sales. Wages are hard to cut when sales fall | + | 0.570 | 28 |

**About `personnel_to_sales`:** the file has no wages field. The field named `goodwill` behaves like value added, so value added − EBITDA ≈ wages. Four checks support this (D-091): negative on only 0.012% of rows; zero mostly for firms with no staff fund, half of them property firms; Spearman 0.93 with the staff fund; median 9.3% of sales. **It's still an inference.** If Roger says the field means something else, drop the feature.

**F2. Sector percentiles (R-01 Sec. 3.3 pp. 17-18, VERIFIED).** "Where does this firm rank among its own sector's peers?" 40% equity means one thing in real estate and another in retail.

| Feature | Base ratio | Sign | AUC alone | SHAP rank |
|---|---|---|---|---|
| `interest_coverage_sect_pct` | interest_coverage | − | 0.268 | 4 |
| `equity_ratio_sect_pct` | equity_ratio | − | 0.262 | 5 |
| `roa_net_sect_pct` | roa_net | − | 0.252 | 10 |
| `cash_ratio_sect_pct` | cash_ratio | − | 0.426 | 17 |
| `tangible_ratio_sect_pct` | tangible_ratio | − | 0.392 | 20 |
| `st_debt_ratio_sect_pct` | st_debt_ratio | + | 0.703 | 23 |

**A trap here.** These rank high on SHAP, but **dropping the whole group changes AUC insignificantly** (+0.0004 and −0.0008, both CIs include zero). They correlate 0.88-0.98 with their base ratios, so they mostly take credit from them. **High SHAP does not mean high value.** Removing them might simplify the model at no cost; we haven't tested that as a final configuration.

**F3. Default-rate encodings (method R-46, R-47, VERIFIED; the economics are an assumption).** 83 divisions and 110 provinces are too many for a tree to use well. So each is replaced by one number: its historical default rate in log-odds versus the base rate, **shrunk** toward the average when the group is small.

| Feature | Levels | m (fitted) | Sign | AUC alone | SHAP rank |
|---|---|---|---|---|---|
| `province_te` | 110 | 1,834 | + | 0.547 | 12 |
| `ateco_div_te` | 83 | 555 | + | 0.580 | 14 |
| `region_te` | 20 | 3,879 | + | 0.530 | 18 |

Formula: `logit((defaults + m × p0) / (rows + m)) − logit(p0)`, where p0 is the base rate.

```python
import numpy as np
def logit(p): return np.log(p / (1 - p))
p0, m = 0.013, 1834                       # base rate; fitted province prior
for rows, defaults in [(200, 10), (2000, 100), (20000, 1000)]:   # each province's raw rate is 5%
    enc = logit((defaults + m * p0) / (rows + m)) - logit(p0)
    print(f"rows {rows:6d}: keeps {rows / (rows + m):4.0%} of its own rate; encoding {enc:+.3f}")
```

Same raw rate, but a small province gets pulled strongly toward average, because 10 defaults in 200 rows could easily be luck.

**Why "out-of-fold"?** If a firm's own default helped compute its province's rate, the feature would contain its own answer. So on training rows, each firm's encoding comes from a table built **without that firm** (grouped by firm id; R-47). Scoring rows use the full table. We tested the naive version: leaking made results **worse** out of time (0.8330 vs 0.8356 on 2009, D-094).

**The evidence is weaker than it sounds.**
- The group only pays with two training years (−0.0028 [−0.0044, −0.0013]), not with one (−0.0003, not significant).
- Province alone isn't significant.
- Province effects only partly persist into the next year (slope 0.50 / 0.78 / 0.48); region is erratic (0.06 / 1.31 / 0.95); division holds best (1.22 / 0.87 / 0.85) (D-124).
- Their upward shapes are **imposed** by the constraint, so they aren't evidence of anything.

**F4. Data-quality signals (R-34, R-05 pp. 60-61 for imputation).** Dropping the group: +0.0001 (not significant) and −0.0011 [−0.0018, −0.0003].

| Feature | Formula | What it means | Sign | SHAP rank |
|---|---|---|---|---|
| `bs_gap` | (asst_tot − eqty_tot − debt_st − debt_lt − liab_lt − liab_lt_emp) / asst_tot | Liabilities the file doesn't classify (possibly accruals: **hypothesis**) | free | 25 |
| `n_core_imputed` | Count of core inputs missing after rebuild | Mostly negative-revenue firms (1,078 rows at 6.8% default) | + | **0 splits** |

**F5. Previously unused fields (assumption, no source).** Dropping the group: −0.0020 [−0.0041, −0.0003] and −0.0022 [−0.0033, −0.0011].

| Feature | Formula | What it means | Sign | AUC alone | SHAP rank |
|---|---|---|---|---|---|
| `tfr_ratio` | liab_lt_emp / asst_tot | Staff-severance fund (below) | free | 0.510 | 8 |
| `nonbank_fin_debt` | (debt_fin_st + debt_fin_lt) / asst_tot | Fixed claims from non-bank lenders | + | 0.523 | 33 |
| `fin_fixed_ratio` | asst_fixed_fin / asst_tot | Stakes in other firms. Group support may help; stakes may be illiquid | free | 0.503 | 34 |
| `ap_lt_ratio` | AP_lt / asst_tot | Suppliers who became long-term creditors | + | 0.501 | 39 |

**`tfr_ratio`: your hardest question.** TFR (*trattamento di fine rapporto*) is the severance money Italian firms owe staff. It's the most robust new feature (dropping it alone costs −0.0023 and −0.0020, both CIs exclude zero), despite an alone-AUC of just 0.51.

Why it looks useless alone:
- **Sector mix:** half of zero-fund firms are real estate, which has low default rates.
- **Suppression:** the logit coefficient is −0.043 alone but −0.197 beside `personnel_to_sales` (+0.325). A high wage bill is risky; a large fund *for a given wage bill* is safer.

**Why does it work? We don't know.** Our hypothesis is that it reflects staff who stayed (a proxy for firm maturity, since we have no firm age) and patient funding. That's **unverified**, and the feature's reading depends on the `goodwill` inference. Firms with low turnover and a fund above 5% of assets actually default a lot (5.7-11.5%, 880 rows). Say: "It's a data finding with a hypothesis, not an established result."

### 5.5 Four of the 45 do nothing

`neg_equity`, `debt_split_missing`, `rev_nonpositive` and `n_core_imputed` have **0 splits** in the fold-3 model. So that model really uses 41 inputs. Probably other features already carry the same information (for example, `equity_ratio` for negative equity), though we didn't test that. Don't describe them as active drivers.

Two consequences:
- **"Zero monotone violations" means nothing for the three constrained ones.** A feature that's never used can't violate anything.
- The earlier claim that `n_core_imputed` "adds back risk" (D-095) doesn't happen in this model.

### 5.6 Considered and rejected

| Idea | Why not | Evidence |
|---|---|---|
| Net debt coverage (R-33's top ratio) | Duplicates `cfo_to_debt` | Spearman 0.990 (D-092) |
| Gross margin | Identical to operating margin in this file | Equal on 99.993% of rows |
| Rounding flag | Too rare | 503 rows, 2 defaults |
| Count of missing fields | Duplicates `debt_split_missing` | Spearman 0.994 |
| Growth features | The harness can't use a firm's other holdout rows; +0.002 AUC on seen firms only | D-020. A frozen training-data lookup is a question for Roger |
| Macro data | External data not cleared | D-004 |
| Text / sentiment | The data contain no text | D-003 |

---

## Part 6. The chain of decisions that produced GB-25

Each step: what we did, why, the evidence.

| # | Decision | Why | Evidence | ID |
|---|---|---|---|---|
| 1 | Model firm-years, with one PD per statement | No loan records exist | Data audit | D-001 |
| 2 | Label window starts 6 months after year-end | The statement isn't public earlier | Civil Code; assumption on timing | D-002 |
| 3 | Walk-forward folds, frozen once | Mimics real use; same test for everyone | Three folds | D-015 |
| 4 | Benchmarks first (scorecard, RiskCalc-style, then trees) | Must beat something simple and transparent | SC-03 0.8166 | D-031 |
| 5 | LightGBM with monotone constraints | Ranks well and drivers make sense | GB-01 0.8375 | D-050, D-051 |
| 6 | Reject unconstrained | +0.0065 AUC, but fails face validity | GB-02 | D-051, D-064 |
| 7 | Full 26 inputs over core 13 | Trees handle correlated inputs | GB-03 −0.0077 | D-055 |
| 8 | Native NaN, "intermediate" monotone | Statistical ties, so keep the simpler | GB-04, GB-05 | D-056, D-057 |
| 9 | No class reweighting | Would distort the level | Slope 0.976 without it | D-054 |
| 10 | Keep the EBM as a challenger | Fully readable | 0.012 AUC lower | D-112 |
| 11 | No blending | Worse in every year | −0.0032 [−0.0042, −0.0024]; R-10 p. 22 agrees | D-086 |
| 12 | Add 19 features (`full2`) | Averaging and tuning weren't adding information | Inner +0.0096; GB-20 +0.0061 | D-090, D-120 |
| 13 | Constrain new features with a clear direction | Same rule as before | 36 constrained | D-125 |
| 14 | Reject tuned GB-23 | Worse than GB-20 in 2009 | Better pooled and in 2010, fails "all years" | D-127 |
| 15 | **Force `interest_to_sales` to + (GB-25)** | Under GB-20, doubling interest lowers PD for 20% of firms | Under GB-25, 1.7%. Cost: −0.0021 AUC on 2010 | D-130, D-144 |

### Step 15 in detail, because you'll be asked

**GB-20 (free sign)** had the feature's top decile lowering risk (SHAP −0.149) although that decile's default rate is high (2.4%). It correlates 0.64 with `interest_to_assets`, which is constrained +, so a free near-twin could partly undo a constraint we impose.

**The sign was left free for a weak reason:** the feature-architect saw a dip in the top two deciles of 2007-09 data and dropped the sign after looking (D-099), which is post hoc. R-33 p. 12 shows an upward slope.

**Forcing + (GB-25):**
- SHAP at the top decile moves to +0.059. Part of the effect moved to other features (`ar_days`, `asset_turnover`, `ateco_group`) rather than disappearing.
- Cost: −0.0011 pooled and **−0.0021 on 2010**, about double the inner estimate of −0.0010.
- The overfitting gap shrinks by about 45-65% of the widening.

**Be accurate about how it was chosen.** GB-25's own config says "pre-specified", and that overstates it. It had no test score beforehand, but 2010 default rates by decile were on file (D-142). GB-25 is the lowest of the three new-feature models on 2010, so the choice didn't flatter it. **This is a judgment you can overrule.** GB-20 is statistically better on ranking.

---

## Part 7. Has tuning been done?

**Short answer: lightly, on purpose, and more tuning didn't help.**

**What GB-25 itself got (`GB-25/train_log.md`):**
- A pre-declared grid of **4 settings** for folds 2-3: `num_leaves/depth` {4/2, 8/3} × `min_data_in_leaf` {500, 2000}.
- Fold 1 has only one training year, so it used one pre-set configuration with no grid.
- Early stopping picked the tree count, then ×1.2.

**How choices avoid peeking.** Inside each training window, the last year is split by firm:

```
Fold 3 (train 2007-09, test 2010):
  2007-2008 ............ fit candidate trees and features
  2009, firm group A ... early stopping and choosing settings
  2009, firm group B ... calibration slope only
  Then refit on all of 2007-2009 with 1.2 x best trees. 2010 never touched.
```

**The grid barely mattered.** In fold 3, the four settings' validation log loss spans 0.06673-0.06695. Both folds chose 8 leaves / 2000 minimum.

**What a bigger search did.** GB-09 tried 37 random settings on the **old 26 features**: +0.0009 AUC [+0.0004, +0.0014], with log loss worse in 2008 and 2009. Chosen learning rates swung by fold (0.024 / 0.050 / 0.016), which looks like noise (D-104). Tuning can't add information about firms.

**Why not simply use GB-23 (tuned, 0.8389 on 2010)?** It beat GB-20 pooled and on 2010, but was worse in 2009, so it failed the all-years rule. **Left open:** nobody has run the tuned settings with the `interest_to_sales` fix. Fold 3's inner slices did prefer them, so it can be re-tested on inner slices only before the final fit (D-127).

**Other things tried.** Recency weights: +0.0002, no effect (D-105), though fold 3 picked the edge of the grid and folds 1-2 couldn't test it.

**The honest summary.** Don't say "fully optimised". Say: "Settings were pre-declared and chosen on inner splits. A wider search on the earlier feature set gave little, and it hasn't been repeated on the final configuration."

---

## Part 8. Overfitting

### 8.1 What it is

Overfitting means learning noise specific to your sample as though it were real. It looks great on data it has seen and worse on new data. It matters because the holdout is unseen, and because too-extreme PDs misprice loans.

There are four kinds in our setting.

### 8.2 Kind 1: the model memorises training data

**What we did:** shallow trees, 2,000-row leaves, L2 penalty, small learning rate, 80% row and feature sampling, early stopping, monotone constraints, shrunken encodings.

**What we measured:**

| Check | GB-01 | GB-25 | Reading |
|---|---|---|---|
| Same-year AUC, in-sample minus out-of-time: 2008 (fold 2) | 0.0262 | 0.0292 | Fits harder than it generalises |
| 2008 (fold 3) | 0.0173 | 0.0220 | |
| 2009 (fold 3) | 0.0146 | 0.0172 | |
| Calibration slope, out of time (2008 / 09 / 10) | 0.969 / 1.005 / 0.957 | 1.005 / 1.014 / 0.957 | Not over-spread, slightly in 2010 |
| Platt slope on held-out slice B | 1.017 / 0.943 / 0.998 | 0.951 / 0.931 / 0.977 | More features, slightly more overconfident |

(`GB-25/gb25_same_year_gap.csv`, `r2_metrics_long.csv`, `GB-25/config.json`, D-053.)

**How to read this.** The in-sample gap is real, and bigger than Moody's about 0.5 AR points (R-33 pp. 22-23). But it isn't like for like: it's a different country and model, and ours compares models trained on different years. **The test that matters is out of time, and there GB-25 is better than GB-01** while PDs aren't over-spread overall. So some of the extra fit is noise, and it doesn't cost out-of-time performance. Quote out-of-time numbers only, and show the gap.

**Exceptions:** new firms' slope is 0.884, and it's similar across models. The smallest firms' is 0.90. Sector R's is 0.72 on 73 defaults.

**Learning curve.** Validation log loss has a clean U shape and is flat near the minimum, so the 1.2× trees cost almost nothing (+0.00003 to +0.00009; `GB-07`). **This was only run for GB-01, not GB-25.**

### 8.3 Kind 2: we overfit by picking the best of many (the main risk now)

Try enough things on the same test years and something will look good by luck. **We've looked at 2008-2010 113 times** (D-142).

```python
# 20 "models" with identical true skill; only noise differs. Keep the one that looks best.
import numpy as np
from sklearn.metrics import roc_auc_score as auc
rng = np.random.default_rng(0)
n, signal = 8000, rng.normal(size=8000)
y = (signal + rng.logistic(size=n) * 1.2 > 3).astype(int)
test, fresh = np.arange(n) < n // 2, np.arange(n) >= n // 2
models = [signal + rng.normal(scale=2.0, size=n) for _ in range(20)]
scores = [auc(y[test], m[test]) for m in models]
best = int(np.argmax(scores))
fresh_scores = [auc(y[fresh], m[fresh]) for m in models]
print(f"on test:  average {np.mean(scores):.4f}, the one we picked {scores[best]:.4f}")
print(f"on fresh: average {np.mean(fresh_scores):.4f}, the one we picked {fresh_scores[best]:.4f}")
print("One run; the learning ledger's version averages 200 repeats: picked +0.0167 on test, +0.0009 on fresh.")
```

**What we did:**
- Choices on inner splits, with outer folds scored once per final configuration.
- Paired CIs, and a rule requiring a win in all three years.
- Quoting 2010 rather than the most flattering number.
- Planning the one-time 2011 check.

**Where we fell short (disclosed in D-100, D-129, D-142):**
- GB-08's constraint set was informed by later folds.
- GB-08b was conceived after seeing GB-08's score.
- The forest version was switched after a test score.
- Features were designed on 2007-09, so 2008-09 scores are mildly optimistic for all models.
- The win rule has no multiple-comparison correction. A tighter one (D-149) applies from the lockbox on, not to this promotion.

### 8.4 Kind 3: time drift

The world changes. A pattern from 2007-09 may fade by 2012.

**Measured:** GB-25's AUC falls from 0.8468 to 0.8441 to 0.8348. Part of that may be drift; we can't separate it from a changing default mix. Recency weighting didn't help. The final model trains on more years, up to 2011.

### 8.5 Kind 4: calibration overfitting

Level parameters fitted on test data would flatter us. **What we did:** all level parameters use training rows or slice B (D-070). **Weak spot:** rule e was chosen among seven rules after seeing 2008-2010, so its gain is optimistic (D-071).

### 8.6 Leakage is different

Overfitting is learning noise. **Leakage** is using information you wouldn't have. The audit found none (D-141):
- GB-25 refits from training rows reproduce the submitted PDs exactly, in all 3 folds.
- Flipping one firm-hash fold's labels leaves that fold's encodings unchanged.
- Single-row and batch features are identical.

This was done by an AI referee. Your own rerun would strengthen it.

---

## Part 9. A real prediction, step by step

We score a made-up firm, **Alfa Srl**, with the actual GB-25 fold-3 model (`experiments/gbm/GB-25/doc_examples.py`, output in `doc_examples_output.txt`).

**The firm (euros):** assets 10.0M (tangible 3.0M, intangible 0.4M, financial 0.6M, current 6.0M including receivables 2.5M and cash 0.5M). Equity 2.0M, short-term debt 5.0M, long-term debt 2.8M, staff fund 0.2M. Sales 12.0M, operating profit 0.6M, EBITDA 1.2M, interest 0.3M, net profit 0.2M, value added 2.7M. Manufacturing, Milan, SRL.

*(Long-term debt differs slightly from the Alfa in the learning ledger, so the staff fund fits the balance sheet.)*

**Step 1: features.** Some examples:

| Feature | Value | Calculation |
|---|---|---|
| `equity_ratio` | 0.20 | 2.0 / 10.0 |
| `tangible_ratio` | 0.30 | 3.0 / 10.0 |
| `interest_coverage` | 1.609 | slog(1.2 / 0.3) = ln(1 + 4) |
| `interest_to_sales` | 0.0247 | slog(0.3 / 12) |
| `personnel_to_sales` | 0.118 | slog((2.7 − 1.2) / 12) |
| `equity_ratio_sect_pct` | 0.427 | Below the median of its section |
| `province_te` | +0.184 | Milan's shrunk default-rate gap vs the average |

**Step 2: add up the trees.** The 548 trees give log-odds of **−5.884**.

**Step 3: see why (SHAP).** SHAP splits that score among the inputs (R-41, VERIFIED), and the parts add up exactly:

```
Baseline, the average firm's log-odds   -5.245
tangible_ratio        -0.304  (30% tangible: safer)
roa_net               -0.181  (profitable)
interest_coverage_sect_pct  -0.145
province_te           +0.133  (Milan: riskier than average)
payables_to_sales     -0.120
asset_turnover        +0.119
extraord_to_assets    +0.095
roa_net_sect_pct      -0.089
equity_ratio          -0.087
cash_ratio_sect_pct   -0.084
... other inputs ...
Total                 -5.884
```

Correlated ratios share credit, so report reasons **by family**: 42% of firms' top-3 reasons repeat a family (D-125).

**Step 4: calibrate.** Raw PD = 0.278%. Own calibration (a = −0.080, b = 0.977) gives **0.294%**.

**Step 5: level shift.** Add the fold-3 shift of +0.2175 to the log-odds, which gives **0.365%**. That's the PD the bank would use in fold 3's setting.

**What-if tests (own-calibrated PD, starting from 0.294%):**

| Change | New PD | Comment |
|---|---|---|
| Interest doubles to 0.6M | 0.842% | Moves three interest features together |
| Net loss of 0.5M | 0.842% | Coincidentally close (0.8418% vs 0.8423%) |
| Equity falls to 0.5M, short-term debt rises 1.5M | 0.559% | |
| Tangible assets 0, moved to current | 1.174% | About 4×, and working capital improves too. This is how strong the top driver is |
| Debt breakdown blank | 0.386% | The model copes with missing data |

These are illustrations of how the model behaves, not causal claims.

---

## Part 10. How to read the results

**Table** (own level, out of time; `reports/final/update_table.csv`, `r2_paired_bootstrap.csv`):

| Model | AUC pooled | AUC 2008 | AUC 2009 | AUC 2010 | Log loss pooled |
|---|---|---|---|---|---|
| **GB-25** | 0.8424 | 0.8468 | 0.8441 | **0.8348** | 0.06380 |
| GB-20 | 0.8435 | 0.8476 | 0.8445 | 0.8369 | 0.06367 |
| GB-02 unconstrained | 0.8440 | 0.8485 | 0.8419 | 0.8402 | 0.06332 |
| GB-11 random forest | 0.8395 | 0.8449 | 0.8396 | 0.8335 | 0.06392 |
| GB-01 | 0.8375 | 0.8441 | 0.8359 | 0.8307 | 0.06415 |
| GAM-01 EBM | 0.8254 | 0.8348 | 0.8233 | 0.8177 | 0.06517 |
| RC-02 RiskCalc-style | 0.8194 | 0.8328 | 0.8174 | 0.8115 | 0.06580 |
| SC-03 scorecard | 0.8166 | 0.8278 | 0.8117 | 0.8115 | 0.06656 |

**Paired comparisons for GB-25:**

| vs | Pooled ΔAUC | 2010 ΔAUC | Reading |
|---|---|---|---|
| GB-01 | +0.0049 [0.0037, 0.0061] | +0.0042 [0.0022, 0.0059] | Real gain. 2008 is barely above zero |
| GB-20 | −0.0011 [−0.0017, −0.0006] | −0.0021 [−0.0029, −0.0012] | We paid this on purpose |
| GB-11 forest | +0.0029 [0.0005, 0.0050] | +0.0013 [−0.0026, 0.0051] | Pooled win, **tied on 2010** |

**Why quote 2010?** Features were designed on 2007-09, so 2008-09 scores are slightly flattered (D-062). 2010 is cleanest, though not perfectly clean (D-142). Quote it first, with pooled beside it, labelled.

**How well does it separate firms? GB-25 on 2010, deciles by PD (`r2_deciles.csv`):**

| Decile | Mean PD | Actual default rate | Defaults |
|---|---|---|---|
| Safest 10% | 0.08% | 0.06% | 10 |
| 5th | 0.42% | 0.65% | 113 |
| 8th | 1.10% | 1.71% | 299 |
| 9th | 1.64% | 2.83% | 493 |
| Riskiest 10% | 6.52% | 8.79% | 1,534 |

The riskiest 10% has about 150 times the default rate of the safest, and holds 51.7% of all defaults. **Before the level shift, PDs are too low from decile 5 up.** That's the level problem, seen directly.

**Where it's weak (AR, GB-01 → GB-25; D-145, `r2_segments.csv`):**

| Segment | GB-01 | GB-25 | Comment |
|---|---|---|---|
| Smallest firms | 0.626 | 0.638 | Real gain |
| Largest firms | 0.735 | 0.741 | Strongest group |
| Construction | 0.559 | 0.568 | **+0.001 on 2010: a tie** |
| New firms | 0.573 | 0.569 | No gain. 2008 is worse: −0.020 [−0.034, −0.006] |

**The level (own → shifted; `r2_metrics_long.csv`):**

| | 2008 | 2009 | 2010 |
|---|---|---|---|
| Observed/predicted, own calibration | 1.23 | 1.37 | 1.44 |
| After the level shift | 1.03 | 1.06 | **1.19** |

On the 16-grade scale in 2010, grades 6-14 still sit at observed/predicted 1.33 (1.32 for GB-01), roughly 518 extra defaults (D-146). **New features don't fix this, since it's a problem of the year's default rate, not of ranking.** Mid-grade loans in a 2010-like year would be priced too low.

**What the gain means for the bank:** slightly better ordering and probabilities, not a new view of risk. In 2010, the riskiest 10% captures 51.7% against GB-01's 51.8%. Pooled it's 54.3% against 53.7%.

---

## Part 11. Known problems

1. **The final model doesn't exist yet.** No fit on 2007-2011, no harness, no stress tests.
2. **Negative-EBITDA defect (D-147).** For a firm with negative EBITDA, `interest_coverage` = EBITDA / interest. More interest makes that fraction *less negative*, so it looks "better", and PD can fall. This affects 40% of those firms in GB-25 and 30% in GB-01. Fixing it needs its own inner-split test.
3. **Level rests on the sample alone.** Moody's sets it above the sample rate using outside data, because samples miss defaults (R-33 p. 9). A year more stressed than 2010 would be under-predicted.
4. **About 40% of the new features' inner gain recovers constraint cost (D-126),** rather than adding purely new information. Constraint cost on inner splits drops from +0.0052/+0.0088 to +0.0032/+0.0048.
5. **Several economic readings have no source:** `tangible_ratio` for PD, `tfr_ratio`, `bs_gap`, `nonbank_fin_debt`, `ap_lt_ratio`, `fin_fixed_ratio`, and the economics of the encodings.
6. **`goodwill` is inferred to be value added,** which `personnel_to_sales` and part of the `tfr_ratio` reading rely on.
7. **Thin sectors:** several have 30-66 test defaults. Don't rank or correct them.
8. **Only three test years,** one cycle, one country.
9. **Human verification is behind.** Most of `docs/AI_USE_LOG.md` is pending.

---

## Part 12. Writing the update yourself

**Roger's spec** (Term Project Instructions, item 3): preliminary results, validation, hypotheses, and any issues.

A suggested skeleton:

1. **Problem and label (short).** One PD, two uses. Why the window starts 6 months after year-end. The assumption in that.
2. **Data.** Rows and firms, rising default rates, 2-3 findings. Why FY 2010 is the last complete test year.
3. **Features.** 26 + 19, by family; which groups held up and which didn't.
4. **Results.** The Part 10 table. Headline 2010, pooled beside it. What did not work.
5. **Validation.** Walk-forward, paired bootstrap, segments, level, overfitting.
6. **Hypotheses.** Supported or not, with numbers.
7. **Issues.** Part 11, plus open questions.
8. **Next steps.** Lockbox, final fit, harness, deck.
9. **AI use.** Honest, and in your own words.

Our version is at `reports/final/2026-11-03_Turquoise_Project_Update.md`. Use it as a check on yours, not as a template.

### Say this, not that

| Don't say | Say |
|---|---|
| "GB-25 is our final model" | "GB-25 is our leading candidate, pending a one-time check on 2011" |
| "AUC 0.84" alone | "AUC 0.835 on 2010; 0.842 pooled across 2008-2010" |
| "2010 is an untouched test" | "2010 is the cleanest year, though we've looked at it repeatedly" |
| "2011 is sealed" | "2011 has been partly used; it can settle the model choice, not test the level" |
| "The interest sign was pre-specified" | "We restored Moody's direction before scoring it, although 2010 default rates by decile were already on file" |
| "An independent referee confirmed" | "An AI referee agent audited it; we're re-running the key checks" |
| "The literature shows tangible assets lower PD" | "In our data they do, in every well-populated industry. Verified literature supports the recovery mechanism only" |
| "The staff fund proxies firm maturity" | "Our unverified hypothesis is…" |
| "As in RiskCalc" for bins, priors or transforms | "RiskCalc-style"; we didn't read the v3.1 technical document (R-02) or any Italy edition |
| "LightGBM beat XGBoost" | We never compared them |
| "The model is fully tuned" | "Settings were chosen on inner splits; a wider search gave little" |
| Any Basel II "one-sentence link" | Leave it out (D-017). The 0.03% floor may cite R-15 by analogy |

### Questions to be able to answer without notes

1. Why does the label window start 6 months after year-end?
2. Why is 2010 the last test year? What's wrong with 2011 and 2012?
3. What's the difference between ranking and level? Which is GB-25 bad at?
4. What does +0.0042 AUC mean in plain words?
5. Why accept lower accuracy for monotone constraints? How much did it cost?
6. Why GB-25 rather than GB-20, which scores higher?
7. What's your top driver, and what can you honestly say about why it works?
8. What is `tfr_ratio`, and why does it look useless alone?
9. What are out-of-fold encodings for?
10. How do you know you didn't overfit? Where might you have?
11. Was it tuned? What happened when you tuned more?
12. What's your target level, and which part of it is judgment?
13. Name one thing that's wrong with the model right now.
14. What did you try that failed?

### Three checks you can run yourself

```bash
# 1. Recompute the headline table from saved predictions
/Users/kietnguyen/miniconda3/envs/datascience/bin/python reports/final/update_table.py

# 2. Run the feature tests
/Users/kietnguyen/miniconda3/envs/datascience/bin/python -m pytest -q src/features

# 3. Reproduce Alfa Srl and the SHAP table
/Users/kietnguyen/miniconda3/envs/datascience/bin/python experiments/gbm/GB-25/doc_examples.py
```

Also open R-33 at pp. 11, 12 and 23, and R-48 at p. 1, and check that we've described them fairly. Then add your initials to `docs/AI_USE_LOG.md`.

---

## Where to go deeper

| Topic | File |
|---|---|
| Every decision, in full | `docs/DECISION_LOG.md` |
| Every decision, in plain language | `docs/LEARNING_LEDGER_2026-11-02.md` |
| Foundations of all model types | `docs/MODEL_FOUNDATIONS_2026-10-30.md` |
| What the papers say | `docs/STUDY_GUIDE_literature.md`, `docs/LIT_moodys_feature_cleaning_2026-10-30.md` |
| Sources and their status | `docs/REFERENCES.md` |
| Referee verdict for round 2 | `reports/tournament/r2_scorecard.md` |
| GBM results | `experiments/gbm/comparison.md`, `experiments/gbm/comparison_r2b.md` |
| Feature evidence | `reports/features/round2_decisions.md`, `reports/features/round2_univariate.md` |
