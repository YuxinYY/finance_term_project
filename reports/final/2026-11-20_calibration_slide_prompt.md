# Prompt for Claude in the browser: Calibration slides (Turquoise deck)

Paste everything below the line into Claude in Chrome while "Presentation - turquoise" is open.

---

You are editing my team's Google Slides deck "Presentation - turquoise" (the open tab). Add a
**Calibration** section: one main slide plus one appendix slide. Match the existing slides exactly:
same fonts, sizes, colours and positions. Do not restyle, move or edit any other slide.

## 1. Where to put the slides

- **Main slide:** insert right after the "Modeling" slide and before the "Evaluation" slide.
- **Appendix slide:** insert after the last appendix slide ("Appendix: Outlier, Collinearity, Standardization").
- **Easiest way to match the theme:** duplicate the "Modeling" slide (Slide > Duplicate slide), then replace
  its title and fill its empty body box. That keeps the title box, the purple underline bar and the body box
  in the right places. Do the same for the appendix slide (duplicate an existing "Appendix: ..." slide and
  delete its table or body content first).
- Do not add the slide to the Contents list (the list stays: Problem Context, Data Context, Data Preparation,
  Modeling, Evaluation, Deployment). Calibration belongs under Modeling.

## 2. Theme spec (read from the deck; use these values if you build anything from scratch)

| Element | Spec |
|---|---|
| Slide | 16:9, 10 in × 5.625 in, white background (theme default, do not set a fill) |
| Title | Text box at x 0.30 in, y 0.27 in. **Montserrat Bold 24 pt**, black (theme colour Dark 1), left-aligned |
| Accent bar under the title | Straight line, **purple #8900E1**, weight **6 pt**, solid, from x 0.40 in, y 0.86 in, length 0.68 in |
| Body text | Text box at x 0.48 in, y about 1.2 in, width about 9.3 in. **Montserrat 12 pt**, black (Dark 1). Sub-headings in **Montserrat Bold 12 pt**. Line spacing and paragraph spacing as on the "Problem Context" slide |
| Footnote / source line | Montserrat 10 pt, black, bottom of the slide (about y 4.95 in), like the Civil Code footnote on "Dependent & Independent Variables" |
| Tables | Montserrat, 10-11 pt, black text, thin grey borders, header row bold. Copy the look of the tables on "Data Preparation" and "Appendix: Missing Value" |
| Emphasis colour | Only the purple #8900E1 (for one key number or a thin highlight). No other new colours, no shadows, no gradients, no emoji, no icons |
| In-text citations | APA style in brackets, like the deck already uses: (Saerens, Latinne & Decaestecker, 2002) |

Keep the text density similar to the "Problem Context" and "Deployment" slides. If the body overflows,
shorten wording; never go below 10 pt and never let text run past the bottom edge.

## 3. Main slide content

**Title:** `Calibration: Getting the PD Level Right`

**Body (left ~55% of the width):**

**Why a second step?**
The bank uses one PD two ways: to rank applicants against a cutoff (needs the right *order*) and to price
loans (needs the right *level*). A model can rank well and still sit at the wrong level, because default
rates move with the economy: 0.87% (FY2007) → 1.21% → 1.58% → 1.65% → 1.39% (FY2011).

**The problem we found**
A model trained on calmer years prices the level of those years. Test on FY2009 (model trained on FY2007):
raw mean PD 0.97% vs 1.58% observed, i.e. 1.62× too low. A Platt map fitted on FY2008 still leaves it
1.33× too low. We assume that under-predicted PDs mean loans priced below expected loss.

**Our fix: a prior shift (one frozen number)**
Every firm's log-odds moves by the same δ = logit(τ) − logit(training default rate)
(Saerens, Latinne & Decaestecker, 2002). Nobody changes places, so AUC is unchanged; only the level moves.

**Right side (~40% of the width): a small table, title "FY2009 test, same model"**

| Calibration step | Mean PD | Obs / pred | AUC | Log loss |
|---|---|---|---|---|
| Raw model | 0.97% | 1.62 | 0.843 | 0.0646 |
| Platt (fitted on FY2008) | 1.19% | 1.33 | 0.843 | 0.0635 |
| **Prior shift (ours)** | **1.58%** | **1.00** | **0.843** | **0.0628** |

Observed default rate FY2009: 1.58%. Bold the last row; you may colour its "1.00" purple #8900E1.

**Bottom line of the body, one sentence in bold:**
**Final harness level: τ = 1.46% (δ = +0.077 on the log-odds); every PD is kept between 0.03% and 99%.**

**Footnote (10 pt):**
`FY = financial-statement year; default = missed payment between 1 Jun FY+1 and 31 May FY+2. Log loss
improvement of the prior shift over Platt: 0.00066 (95% CI 0.00049–0.00085, firm-clustered bootstrap).`

## 4. Appendix slide content

**Title:** `Appendix: Choosing the Level τ`

**Intro line (12 pt):**
The holdout period is unknown, so τ is a forecast of next period's default rate made only from data the bank
has at decision time.

**Table (left, ~55% width), header "Candidate rules for τ":**

| Rule | τ | Verdict |
|---|---|---|
| Pooled average FY2007–11 | 1.36% | Gives calm 2007 the same weight as the crisis years |
| Last year (FY2011) | 1.39% | Up to date, but assumes next year = last year |
| Last 6 months, annualised | 1.60% | Won the 2008–10 back-test, but over-shot 2011 by 43% (turning point) |
| **Random-walk hedge, 2 years ahead** | **1.46%** | **Chosen** |

**Right side, sub-heading "How 1.46% is built" (bulleted, 12 pt):**
- Anchor on the latest complete year: FY2011 = 1.39%.
- Default-rate log-odds treated as a random walk; typical yearly move 0.23 (from the FY2007–11 changes).
- Averaging over possible paths 2 years ahead gives 1.46%: a little above the anchor, because a level that is
  too low costs about 1.5× more log loss than one equally too high.

**What we rejected (one short line each, 12 pt):**
- Isotonic regression: same log loss as the prior shift, but creates ties (AUC −0.0004) and jagged deciles.
- Estimating the new rate from the holdout batch itself (EM): uses other holdout rows, which the rules forbid.

**Footnote (10 pt):**
`Assumptions: random-walk persistence and the 2-year horizon are judgments. If the holdout period is
disclosed, the horizon is reset; the model's ranking is unaffected.`

## 5. Checks before you finish

1. Both slides use only Montserrat, black text, and the purple #8900E1 accent; title 24 pt bold, body 12 pt.
2. Title, purple bar and body box sit at the same positions as on the "Modeling" slide (compare by flipping
   between slides).
3. No text overflows any box or the slide edge; tables fit inside the slide margins (about 0.4 in each side).
4. Numbers are copied exactly as given above. Do not round, recompute or add any number or citation.
5. Do not change any other slide. Tell me when done and list anything you could not match.
