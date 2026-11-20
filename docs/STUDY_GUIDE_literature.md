# Study guide: what the literature tells us, in plain words

For Kiet, Desmond and Chloe. Written Oct 30, 2026 from `docs/REFERENCES.md` (the checked notes,
with page numbers). AI-generated (Claude Code); a team member must read the core papers
themselves (Section 7). Every claim here points to an R-number in `REFERENCES.md`.

---

## 0. The one thing to understand first

**Roger Stein, our professor, is the "Stein" in the reading list.** He co-wrote the RiskCalc v3.1
paper (R-01), the paper linking default models to lending profit (R-08), and the paper on how to
validate default models (R-07). Much of what he wrote on our proposal ("How do you know this?",
"Are there more robust metrics?", "How do these two differ?") is answered in his own papers.

So the safest strategy for the whole project is:

> **Do what Stein's papers say a careful modeller should do, cite them correctly, and show
> that we understood them, not just name-dropped them.**

He will notice immediately if we misquote his work. For example, our proposal implied Stein
(2005) supports cut-offs, but he actually argues that pricing is better than cut-offs (Section 3).

---

## 1. The big picture in six sentences

1. The bank needs one number per borrower: the **PD**, the chance the firm misses a payment in
   the next 12 months.
2. The bank uses that number in two ways: to **rank** firms (who gets a loan) and to **price**
   loans (what interest rate covers the risk).
3. Ranking needs **power**: defaulters should get higher PDs than non-defaulters.
   Pricing needs **calibration**: if we say 3%, about 3% should default.
4. These are **two different qualities**, and a model can have one without the other. Every
   serious paper measures them separately (R-01, R-07, R-16).
5. Private firms have no stock price, so we can't use market models. We must use
   **accounting ratios** from the financial statements. This tradition runs from Altman (1968)
   to RiskCalc.
6. We must test the model the way it will be used: train on the past, test on a **later year**
   it has never seen ("walk-forward"), and report how uncertain the results are.

If you can explain these six sentences with an example each, you can survive most of Roger's
questions.

---

## 2. Key concepts, explained simply (with runnable examples)

Run any example with:
`/Users/kietnguyen/miniconda3/envs/datascience/bin/python example.py`

### 2.1 Power vs calibration (the most important idea in the project)

- **Power** (also called discrimination or ranking): do bad firms get higher scores than good
  firms? We measure it with **AUC**.
- **Calibration** (level): are the PD numbers the right size? We measure it with
  **calibration-in-the-large** (average predicted PD vs actual default rate), a **binned
  observed-vs-predicted table**, and **log loss**.

A model can rank perfectly and still have wrong levels. Example: it says every firm is 10× riskier
than it really is. The ordering is still right, so AUC is unchanged, but every loan would be
overpriced and the bank would lose good customers.

```python
# example 1: same ranking, different levels -> same AUC, very different calibration
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

rng = np.random.default_rng(0)
n = 100_000
true_pd = 1 / (1 + np.exp(-(-4.5 + rng.normal(size=n))))      # true PDs, average about 1.5%
y = rng.binomial(1, true_pd)                                   # who actually defaults

good_model = true_pd                                           # right ranking, right level
scaled_model = np.clip(true_pd * 5, 0, 0.99)                   # right ranking, 5x too high

for name, p in [("good", good_model), ("5x too high", scaled_model)]:
    print(f"{name:12s} AUC={roc_auc_score(y, p):.3f}  "
          f"avg PD={p.mean():.3%} vs actual={y.mean():.3%}  log loss={log_loss(y, p):.4f}")
# AUC is identical; only the level (avg PD, log loss) reveals the problem.
```

**What to say:** "Ranking serves approval, level serves pricing, so we evaluate both separately
(Stein 2007 Sec. 1; RiskCalc v3.1 Sec. 4.1)."

### 2.2 AUC and AR are the same thing (Roger's margin note: "How do these two differ?")

- **AUC** = the probability that a randomly chosen defaulter gets a higher PD than a randomly
  chosen non-defaulter. 0.5 means a coin flip; 1.0 is perfect.
- **AR (Accuracy Ratio)** comes from the CAP curve, which banks like.
- **AR = 2 × AUC − 1.** They carry exactly the same information (R-14 eq. 17; Stein 2007 p. 83).
  Report one and mention the identity. Don't present them as two separate pieces of evidence.

```python
# example 2: AR from the CAP curve equals 2*AUC - 1
import numpy as np
from scipy.integrate import trapezoid
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(1)
score = rng.normal(size=20_000)
y = rng.binomial(1, 1 / (1 + np.exp(-(-4 + 1.2 * score))))

order = np.argsort(-score)                                      # riskiest first
cum_def = np.cumsum(y[order]) / y.sum()                         # CAP curve: share of defaults caught
x = np.arange(1, len(y) + 1) / len(y)                           # share of firms rejected
area_model = trapezoid(np.r_[0, cum_def], np.r_[0, x]) - 0.5     # area between model CAP and random
pi = y.mean()
area_perfect = (1 - pi) / 2                                     # area for a perfect model
ar = area_model / area_perfect
auc = roc_auc_score(y, score)
print(f"AR from CAP = {ar:.4f}   2*AUC-1 = {2*auc-1:.4f}")
```

### 2.3 Why log loss, and why Brier is NOT a calibration measure

- **Log loss** punishes confident wrong answers. Saying 0.1% for a firm that defaults is
  punished hard, which matches pricing: a firm priced as nearly riskless that then defaults is
  costly.
- **Brier score** (mean squared error of the PD). The Basel validation paper says it mostly
  reflects ranking power, **not** calibration (R-16 p. 32). In the deck, call Brier "overall
  probability accuracy", not calibration.
- Stein (2007, p. 89): log likelihood **compares** models with each other; it does not prove a
  model is well calibrated on its own. So also show the plain observed-vs-predicted table.
- Basel's validation paper also says there are **no really powerful calibration tests**, and
  standard tests are too optimistic when defaults are correlated, as in a recession (R-16
  pp. 33–34). So present calibration as a diagnostic picture, not a pass/fail test.

### 2.4 Uncertainty depends on the number of defaults, not the number of rows

We have about a million rows, which sounds like a lot. But metric noise is driven by the number of
**defaults** (Stein 2007 Sec. 4.1, pp. 97–99). Each of our test years has only 2,000–3,000
defaults. So:

- Always show the **number of defaults** next to every AUC.
- Always show a **confidence interval** (we bootstrap it).
- Our bootstrap resamples **firms**, not rows, because the same firm appears in several years.
  Treating its rows as independent would make us look more certain than we are.

```python
# example 3: AUC uncertainty shrinks with defaults, not with rows
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(2)
def auc_spread(n_rows, pd_level, reps=200):
    aucs = []
    for _ in range(reps):
        s = rng.normal(size=n_rows)
        y = rng.binomial(1, 1 / (1 + np.exp(-(np.log(pd_level / (1 - pd_level)) + s))))
        aucs.append(roc_auc_score(y, s))
    return np.std(aucs), int(n_rows * pd_level)

for n, p in [(10_000, 0.20), (100_000, 0.02), (1_000_000, 0.002)]:
    sd, d = auc_spread(n, p)
    print(f"rows={n:>9,}  ~defaults={d:>5}  AUC std={sd:.4f}")
# Same ~2,000 defaults -> similar noise, even with 100x more rows.
```

### 2.5 Walk-forward testing (Roger: "How, specifically, will you implement this?")

Train on years ≤ t, test on year t+1, then move forward one year (Stein 2007 Sec. 3, six steps on
p. 94; RiskCalc v3.1 Sec. 4.2–4.3). Our version (`data/splits/walkforward.json`):

| Fold | Train on | Test on | Test defaults |
|---|---|---|---|
| 1 | 2007 | 2008 | 2,064 |
| 2 | 2007–2008 | 2009 | 2,752 |
| 3 | 2007–2009 | 2010 | 2,968 |

Two rules from Stein (2007, p. 95) that we must follow:
- Tuning happens **inside** the training years only.
- If we keep tweaking the model while looking at the test years, those years stop being a
  fair test. The referee enforces this.

### 2.6 RiskCalc's design (the model Roger built at Moody's)

RiskCalc v3.1 (R-01 Sec. 3.1, pp. 13–14), in three steps:

1. **Transform each ratio separately.** For every ratio, learn a curve "ratio value → default
   rate" from the data, with no assumed shape. This handles weird values and makes each ratio
   speak the same language (risk).
2. **Combine with a probit.** A probit is like a logit with a different S-curve. It adds the
   transformed ratios plus industry dummies.
3. **Final calibration map.** Convert the probit score to a PD at the right level.

The paper itself says this is closely related to a GAM (generalised additive model). It uses
**one ratio from each of seven families**: profitability, leverage, debt coverage, growth,
liquidity, activity, size. Too many ratios overfit (pp. 12–13).

```python
# example 4: RiskCalc-style step 1, a nonparametric "ratio -> default rate" transform
import numpy as np, pandas as pd

rng = np.random.default_rng(3)
n = 50_000
leverage = rng.lognormal(mean=-0.5, sigma=0.6, size=n)          # debt / assets, skewed
true_pd = 1 / (1 + np.exp(-(-5 + 3 * np.minimum(leverage, 1.5))))  # risk rises, then flattens
y = rng.binomial(1, true_pd)

bins = pd.qcut(leverage, 20, duplicates="drop")                 # fit on TRAINING data only
table = pd.DataFrame({"y": y, "bin": bins}).groupby("bin", observed=True)["y"].mean()
centers = np.array([b.mid for b in table.index])
rates = table.values

def transform(x):                                               # frozen curve -> usable in harness
    return np.interp(x, centers, rates)                         # flat beyond the ends (robust)

print(pd.DataFrame({"leverage": [0.2, 0.6, 1.0, 3.0, 50.0],
                    "T(leverage) = default rate": transform(np.array([0.2, 0.6, 1.0, 3.0, 50.0]))}))
# Extreme 50.0 gets the same risk as the top bin: one crazy number can't blow up the PD.
```

Why this matters to a credit officer: each ratio's curve can be **plotted and explained**
("above 80% leverage, default rates climb steeply"). That's exactly what Roger said our proposal
lacked: a way to explain the model to bankers.

### 2.7 Cut-off vs pricing, and adverse selection (Stein 2005, R-08)

- **Cut-off:** lend if PD < threshold, otherwise decline. Stein shows the best cut-off depends on
  the default rate, how much is lost per default, the spread and the fees (pp. 1215–1226).
- **Pricing:** lend to everyone at a rate that covers their risk. The same ROC machinery gives a
  break-even price for every score (p. 1227).
- **His conclusion:** cut-offs are common but **sub-optimal**, and pricing is better (p. 1233).
- **Adverse selection:** if a competitor ranks borrowers better, it offers safe firms lower
  rates. Those firms leave, and our bank keeps the risky ones it under-priced. A better model
  means more profit, more so under pricing (Sec. 6, a simulation on US data).
- **Careful:** those dollar and basis-point numbers are from a US simulation. Don't present them
  as what Banca Massiccia would earn.

```python
# example 5: why calibration matters for pricing (stylised; LGD, margin and competitor are assumptions)
lgd, margin = 0.45, 0.005                           # assumed loss-given-default and profit margin
true_pd = 0.02
fair_rate = true_pd * lgd + margin                  # a competitor with an accurate model offers this
for model_pd in [0.02, 0.005, 0.06]:
    our_rate = model_pd * lgd + margin              # rate we quote from OUR model's PD
    accepts = our_rate <= fair_rate                 # borrower takes the cheaper offer
    profit = (our_rate - true_pd * lgd) if accepts else 0.0
    print(f"model PD {model_pd:.1%}: quote {our_rate:.2%} vs fair {fair_rate:.2%} -> "
          f"{'borrower accepts' if accepts else 'borrower goes to competitor'}, "
          f"expected profit {profit:+.2%} per euro")
# Too low: we win the loan and lose money on it. Too high: we lose the customer.
```

### 2.8 Central tendency, the cycle, TTC vs PIT

- **Central tendency** = the average PD level the model is calibrated to. Our default rate moves
  from about 1.0% (2007) to 1.7% (2010), so **which level** we calibrate the holdout to matters a
  lot for log loss.
- **TTC (through-the-cycle):** a stable, long-run average level. **PIT (point-in-time):** follows
  today's conditions. Basel's validation paper says underwriting-oriented rating systems tend
  toward TTC and pricing-oriented ones toward PIT (R-16 p. 11). Our bank does both, so we must
  choose and justify a level.
- **Prior-shift correction** (Saerens et al., R-22): if only the overall default rate changes, you
  can re-scale PDs to a new average without refitting. **Catch:** it assumes the ratios of
  defaulters and non-defaulters keep the same distribution, which is doubtful in 2008–12. Use it
  as a sensitivity check, not a fact.

```python
# example 6: prior-shift adjustment, moving PDs from a 1.0% world to a 1.7% world
import numpy as np
p_old = np.array([0.002, 0.01, 0.05, 0.20])
old, new = 0.010, 0.017
odds = p_old / (1 - p_old) * (new / (1 - new)) / (old / (1 - old))
print(np.round(odds / (1 + odds), 4))   # every PD moves up; the ranking is unchanged
```

### 2.9 Gradient-boosted trees need recalibration

Boosted trees push probabilities in a distorted way (Niculescu-Mizil & Caruana 2005, R-19,
abstract). Fix: fit a small calibration step (Platt or isotonic) **on an inner part of the
training years only**. Stein (2007, pp. 89–91) adds that recalibration fixes the **level** but
**cannot add ranking power**.

### 2.10 SHAP reason codes (per-firm explanations)

SHAP splits one firm's prediction into contributions from each feature, and they add up exactly
to the prediction (R-23 Thm. 1). Example of a reason code a loan officer can read: "High
short-term bank debt relative to assets added X percentage points to this firm's PD."
**Careful:** the fast tree version ("TreeSHAP") is a later paper we haven't verified. Don't cite
R-23 for it.

### 2.11 Italian law and the label timing (Roger: "How do you know this?")

- Shareholders must approve the accounts within **120 days** after year-end (up to 180 if the
  company's statute allows) (art. 2364, R-28).
- Approved accounts must be filed within **30 days** (art. 2435, R-29).
- So a December statement may legally not be public until around July. That's why our label
  window starts **6 months after** the statement date. These are **legal maximums**, not proof of
  actual filing dates, so present them that way.
- Small firms may file **abbreviated accounts** with fewer line items (art. 2435-bis, R-27). That
  is a plausible reason why the debt breakdown is missing for about 12% of rows, mostly small
  firms. Say "plausible", because we haven't verified the thresholds in force before 2008.

### 2.12 Basel, only if asked

Don't put a Basel sentence in the deck (see D-017). If someone asks: our PD is *conceptually
aligned* with the one-year PD in Basel II's internal-ratings approach, but our label is "missed
payment" from the assignment, **not** the Basel default definition (90 days past due / unlikely
to pay, para. 452). The 0.03% floor (para. 285) can justify the minimum PD in our harness.

---

## 3. Paper by paper: what each says and what we do with it

| Paper | In one line | What we take | What we must NOT say |
|---|---|---|---|
| **RiskCalc v3.1** (Dwyer, Kocagil & **Stein** 2004, R-01) ✅ | Private-firm PD model: per-ratio transforms → probit + industry → final calibration; validated on power and calibration separately, walk-forward, AR per year | Our main "finance-grounded" challenger design; seven ratio families; per-year AR table | "RiskCalc Italy uses ratios X, Y" (we haven't read the Italy doc) |
| **RiskCalc for Private Companies** (Falkenstein et al. 2000, R-05) ✅ | Private firms behave differently from public ones; no widely used benchmark existed then | Why private-firm-specific models; model supports, doesn't replace, the credit officer | "RiskCalc is the **industry standard**" ❌ (p. 10 says the opposite) |
| **Stein 2007** (validation pitfalls, R-07) ✅ | How to validate fairly: power ≠ calibration, walk-forward, noise depends on #defaults, don't reuse test years | Our whole evaluation design | "Likelihood proves calibration" ❌ |
| **Stein 2005** (ROC and lending profit, R-08) ✅ | Better ranking = more profit; cut-offs are a special case of pricing; weak models attract bad borrowers | The business case slide | "The bank uses a cut-off" as fact ❌ (it's our assumption); US numbers as our bank's numbers ❌ |
| **Engelmann, Hayden & Tasche 2003** (R-14) ✅ | AR = 2·AUC − 1; CIs for AUC | Answer to "How do AUC and CAP differ?" | Cite the Bundesbank paper, not the *Risk* article |
| **BCBS WP14** (Basel validation studies, R-16) ✅ | Calibration is hard to test; Brier ≈ ranking; TTC vs PIT | How we word calibration evidence | "Brier measures calibration" ❌ |
| **Altman 2000** (Z-score revisited, R-32) ✅ | 1968 Z-score: 5 ratios, 66 firms; needs market equity → public firms; Z' for private firms | Ratio-based lineage; possible benchmark | Details of Altman 1968 beyond what R-32 restates |
| **Lessmann et al. 2015** (R-17) ✅ | Big ML benchmark: random forests/ensembles often beat logit; beating logit alone isn't impressive | Why we include a GBM challenger | Their data is **retail**, not corporate |
| **Niculescu-Mizil & Caruana 2005** (R-19) ✅ abstract | Boosted trees need calibration | Why GBM gets a calibration step | Anything from the body (not read) |
| **Saerens et al. 2002** (R-22) ✅ | Re-scale PDs to a new default rate | Central-tendency sensitivity | That it's exact across the cycle |
| **Lundberg & Lee 2017** (R-23) ✅ | SHAP = the unique fair additive explanation | Reason codes | That this paper introduces TreeSHAP |
| **Italian Civil Code** 2364, 2435, 2435-bis ✅ current text | Approval ≤120/180 days, filing ≤30 days, abbreviated accounts | Label window; missingness story | Exact pre-2008 thresholds; "published March–June" |
| Ohlson 1980, Shumway 2001, XGBoost, Platt, isotonic, EBM | Not read | Nothing yet | **Don't describe their content** |

✅ = checked against the actual text, page numbers in `REFERENCES.md`.

---

## 4. What Roger criticised, and exactly how we fix it

| Roger said | What it really means | What we do (and where) |
|---|---|---|
| Too much "data in → predictions out", not enough formulation | Every choice needs a banking reason | Each decision log entry has a "Business reason" line |
| No real missing-data strategy | Missing values have causes; use accounting identities | Missingness study: debt breakdown missing as a block (~12%, small firms); `roe`/`margin_fin` blank when equity is negative → recompute, don't impute (`reports/data/02_missingness.md`) |
| NLP sentiment: "How?" | There is no text in the data | Dropped, with the reason logged (D-003) |
| Economic conditions: "How?" | Show it, don't promise it | Per-year results 2008/2009/2010 + central-tendency study |
| Not connected to explaining the model to bankers | Credit officers must understand each PD | Driver curves, 3 worked example firms with reason codes, PD-to-rating-grade table |
| No AI statement → assumed no AI | Be honest and specific | `docs/AI_USE_LOG.md` → AI-use slide |
| Standard DS proposal, no banking literature | Use finance and banking sources | This guide; 15 checked references |
| Statements without foundation ("How do you know this?") | Cite it or call it an assumption | "Claims we must NOT make" list in `REFERENCES.md` |
| AUC and CAP: "How do these differ?" | They're the same information | AR = 2·AUC − 1 (R-14) |
| "Are there more robust metrics?" | Calibration and uncertainty | Log loss, calibration-in-the-large, binned table, firm-clustered bootstrap CIs |
| "How do you know the bank uses a cut-off?" | We don't | "We assume…" + Stein 2005: cut-off is a special case of pricing |
| PD vs EL: "Why are you telling me this?" | Don't mention things you don't use | One sentence: we estimate PD only; LGD/EAD aren't in the data |
| Unit of analysis = application: "What does this mean?" | The data has statements, not loans | One row = one firm-year statement → one PD |
| ATECO grouping: "How?" | Show the method | Map to ATECO 2007 sections; pool sections with <30 defaults |
| VIF: "How?" | Show the table and the rule | VIF table in `docs/FEATURE_SPEC.md` |
| Tuning: "How, specifically?" | Inner splits inside the training years | Referee checks this |
| "Each year's model has different variables?" | Fixed feature set | One spec; final model refit on all labelled years |
| Publication lag: "How do you know?" | Need a source | Civil Code arts. 2364 + 2435 (legal maximums) |
| Selection bias: "How do you know?" | We can't confirm the data are only booked loans | State it as a limitation and an assumption |
| Team name missing | Admin, but he notices | "Team Turquoise" on the cover and in the file name |

---

## 5. The must-do checklist (if we do only these, we're safe)

**Formulation (15 pts)**
- [ ] Business problem slide: one PD, two uses (rank for approval, level for pricing), and why
      both qualities matter. Cite Stein 2005 for the money link, worded carefully.
- [ ] Unit of analysis = firm-year statement. Label = default in the 12 months starting 6 months
      after the statement, justified by Italian filing law. Show the sensitivity: the 0-month
      start gives 1.07% vs 1.41%.
- [ ] Features organised by RiskCalc's ratio families, with an expected-sign table ("more
      leverage → higher PD").
- [ ] Missing data handled with accounting logic, not blind imputation.
- [ ] Say clearly what we can't do: no growth features (we only use the current record), no
      macro data, no text.

**Out-of-sample (5 pts)**
- [ ] The harness never crashes and returns a PD for every row (test it on messy files).
- [ ] The PD level is chosen deliberately (central tendency) and that's explained.

**Documentation (10 pts)**
- [ ] Power and calibration reported **separately**, per year, with default counts and CIs.
- [ ] AUC only, with "AR = 2·AUC − 1" noted. Brier labelled "overall accuracy", not calibration.
- [ ] Benchmarks: a simple logit and the RiskCalc-style model, then the GBM must beat them with
      CIs to win.
- [ ] Explainability: driver curves, 3 example firms with reason codes, PD → rating-grade table.
- [ ] "What didn't work" slide (from the decision log's REJECTED entries).
- [ ] Limitations: selection bias (assumed), cycle sensitivity, new firms vs seen firms, label
      assumption.
- [ ] Bibliography with only papers we actually read.
- [ ] An honest, specific AI-use statement + an explicit per-person contributions appendix.
- [ ] **Every member can explain every slide.** Roger grades individually.
- [ ] No Basel sentence (D-017).

---

## 6. Phrases you can safely use (copy-paste ready)

- "We evaluate discriminatory power and calibration separately because the bank uses the PD
  both to rank applicants and to price loans (Stein, 2007; Dwyer, Kocagil & Stein, 2004)."
- "We report AUC; the accuracy ratio is a linear function of it, AR = 2·AUC − 1, so it adds no
  separate information (Engelmann, Hayden & Tasche, 2003)."
- "Metric uncertainty is driven by the number of defaults, so every metric is reported with its
  default count and a firm-clustered bootstrap confidence interval (Stein, 2007)."
- "Following RiskCalc v3.1, we select ratios across profitability, leverage, debt coverage,
  liquidity, activity and size, and transform each ratio non-parametrically before combining."
- "We assume the bank approves applicants against a threshold, as the assignment's transaction
  flow implies; Stein (2005) shows a cut-off is a coarse special case of risk-based pricing."
- "Italian law requires approval within 120 days of year-end (up to 180) and filing within 30
  days of approval, so a statement may not be available until mid-year; our label window starts
  six months after the statement date."

## Phrases that will cost us points

- "RiskCalc is the industry standard."
- "Brier score measures calibration."
- "The bank uses a cut-off" (stated as fact).
- "Financial statements are published between March and June."
- "The data includes only firms the bank lent to" (stated as fact).
- Anything about Ohlson, Shumway, XGBoost internals or EBM that we haven't read.
- Stein (2005)'s dollar figures presented as gains for Banca Massiccia.

---

## 7. What each of us must read personally (about 3 hours total)

Roger: "Don't refer to research you have not read." Split it up, then explain it to each other.

| Who | Read | Pages to focus on | Time |
|---|---|---|---|
| Everyone | RiskCalc v3.1 (R-01) | Sec. 3.1 (pp. 12–14), Sec. 4 (pp. 25–30) | 45 min |
| Everyone | Stein 2007 (R-07) | Sec. 1, 2.2–2.3 (pp. 77–91), Sec. 3 p. 94, Sec. 4.1 | 60 min |
| One person | Stein 2005 (R-08) | Sec. 1, 4, 6–7 | 30 min |
| One person | BCBS WP14 (R-16) | pp. 11–12, 32–34 | 20 min |
| One person | Engelmann et al. (R-14) | Abstract, Sec. III.2 | 15 min |
| One person | Civil Code arts. 2364, 2435, 2435-bis | the three articles | 10 min |

Links are in `docs/REFERENCES.md`. While reading, check one page reference each against our
notes. That's also what makes the AI-use statement honest ("AI located and summarised; [name]
verified pages X–Y").

---

## 8. Five questions Roger might ask, with short answers

1. **"Why not just maximise AUC?"** Because pricing needs the right level. A model can have a
   high AUC and badly wrong PDs; recalibration fixes the level but can't add ranking power
   (Stein 2007 Sec. 2.3).
2. **"Why does your label start 6 months after the statement?"** The bank can't use a statement
   before it exists publicly; Italian law allows up to 120/180 days plus 30 for filing. We tested
   0 and 3 months as sensitivity.
3. **"Is your GBM's improvement real?"** Show the firm-clustered bootstrap CI for the AUC
   difference. If it includes zero, we prefer the simpler model.
4. **"How would a loan officer use this?"** PD → rating grade → price band. Each PD comes with its
   top three reasons in words.
5. **"What could go wrong in deployment?"** Selection bias (we only see past borrowers, assumed),
   the cycle shifting the PD level, new firms, and abbreviated statements with missing items.
