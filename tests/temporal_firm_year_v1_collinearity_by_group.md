# By-group collinearity review: current round 1 selection

## Current selection: reconstructed diagnostics

The current `vif_selected_features.json` selection was reconstructed on all **144,251 round 1 training rows**, using the saved full transformed training features, the feature-group definitions recorded in the selection file, and the current iterative selector. The reconstructed exclusions match the original saved selection. Results below are recomputed diagnostics, since the original elimination CSV is not available.

Within each group, the selector removes one feature with the highest VIF and recalculates until all retained VIFs are below 10. Constant or entirely missing columns are removed in a precheck. The process selects on round 1 training data only; the same selected feature list is applied to subsequent rounds.

| Stage | Group | Feature removed | VIF at removal | Reason |
|---|---|---|---:|---|
| Precheck | All features | `zero_asst_tot` | — | Constant (all zeros) |
| 1 | Business and structural | `legal_SRL` | 756.48 | VIF ≥ 10 |
| 2 | Business and structural | `city_Lombardia` | 13.17 | VIF ≥ 10 after first removal |
| 1 | Profitability and growth | `rev_operating` | 10,459,140,000 (approx.) | VIF ≥ 10 |
| 2 | Profitability and growth | `prof_operations` | 11.36 | VIF ≥ 10 after first removal |

| Feature group | Highest VIF among survivors |
|---|---:|
| Business and structural | 1.31 |
| Leverage and capital structure | 3.30 |
| Liquidity and activity | 2.67 |
| Profitability and growth | 1.94 |
| Coverage and cash flow | 2.22 |

These maxima are for the **original VIF-selected set**, before the subsequent manual exclusion of `goodwill_ratio`. The original selection retained 75 of 80 features; the manual exclusion leaves **74**. The diagnostic is within-group, so it does not guarantee every cross-group relationship or later-round VIF is below 10. Some flags are ungrouped and were not assessed against a group. VIF selection is not refitted on rounds 2 or 3.

## Historical v1 review

The following section is preserved for context only. It used an earlier feature set and method and does **not** describe the current selection or its final VIF values.

## Scope and method

This review summarizes the available `*_vif.csv` outputs for the
`temporal_firm_year` experiment, excluding the requested Business and
Structural group.  Diagnostics use a reproducible sample of 100,000 transformed
training rows from round 1.  VIFs near 1 indicate little linear overlap; values
above 5 merit review, and values above 10 are high.

## Leverage and Capital Structure

| Feature | VIF | Assessment |
| --- | ---: | --- |
| Debt ratio | 14.49 | High |
| Equity ratio | 13.85 | High |
| Current assets ratio | 3.34 | Moderate but acceptable |
| Tangible asset ratio | 3.15 | Moderate but acceptable |
| Goodwill ratio | 1.33 | Low |
| Short-term debt share | 1.33 | Low |
| Long-term debt to capital | 1.05 | Low |
| Bank debt concentration | 1.00 | Low |
| Negative balance-sheet flag | 1.00 | Low |

`debt_ratio` and `equity_ratio` are the only high-collinearity pair in this
group.  They share total assets as their denominator and both describe capital
structure, so this is economically expected.  For a linear model, retaining
one of them is the cleaner specification unless there is a reason to preserve
both.  `equity_ratio` is often the more directly interpretable capital-buffer
measure; tree models are less sensitive to this issue.

## Liquidity and Activity

| Feature | VIF | Assessment |
| --- | ---: | --- |
| Cash ratio | 1.08 | Low |
| Working-capital ratio | 1.07 | Low |
| Payables ratio | 1.04 | Low |
| Cash to debt | 1.01 | Low |
| Current over debt ratio | 1.00 | Low |
| Receivables ratio | 1.00 | Low |
| Change in AR turnover | Not estimable | All missing |

There is no material linear collinearity among the estimable liquidity and
activity features.  `change_in_ar_turnover` is all missing in round 1 because
the training anchors are FY2007 and no earlier company statement is available
in that round's history to form a prior-year turnover value.  This is a
first-window data-availability issue, not a VIF problem.  It should become
estimable in later expanding-window rounds that contain firms with consecutive
statements.

## Profitability and Growth

| Feature | VIF | Assessment |
| --- | ---: | --- |
| COGS ratio | 72,503,878,711,405 | Effectively perfect collinearity |
| Operating margin | 72,503,878,710,487 | Effectively perfect collinearity |
| ROA | 8.51 | Elevated; review |
| EBITDA ratio | 7.41 | Elevated; review |
| Tax ratio | 2.25 | Low |
| Negative P&L flag | 1.01 | Low |
| Net profit margin | 1.01 | Low |
| ROE | 1.00 | Low |
| Annualized change in ROA | Not estimable | All missing |
| Annualized sales growth | Not estimable | All missing |

The extremely large VIFs are expected: `op_margin = prof_operations /
rev_operating`, and the accounting identity `prof_operations = rev_operating −
COGS` means that, for the observed rows, `op_margin` is essentially `1 −
cogs_ratio`.  A linear model should use **one, not both**, of these features.
`op_margin` is generally the more intuitive profitability measure; `cogs_ratio`
is useful when direct cost intensity is the intended interpretation.

ROA and EBITDA ratio have elevated but sub-10 VIFs.  Both measure operating
performance scaled by assets, so their overlap is plausible.  They can remain
for now, but should be monitored in later rounds and in any final
WoE/logistic feature-selection step.  The two annualized change features are
unavailable in round 1 for the same lack-of-prior-record reason described
above.

## Coverage and Cash Flow

| Feature | VIF | Assessment |
| --- | ---: | --- |
| Interest coverage | 1.30 | Low |
| Cash to financing expense ratio | 1.30 | Low |
| Financing income ratio | 1.14 | Low |
| Operating cash-flow ratio | 1.11 | Low |
| Financing expense ratio | 1.04 | Low |
| Operating cash flow to debt | 1.02 | Low |
| Debt / EBITDA | 1.00 | Low |

This group has no material linear collinearity.  Its features measure distinct
relationships among financing cost, cash, operating cash flow, and debt.

## Recommended follow-up

1. Remove one of `op_margin` and `cogs_ratio` before the next linear-model run.
2. Evaluate whether to retain both `equity_ratio` and `debt_ratio` for the
   WoE/logistic model; leave both available to LightGBM if desired.
3. Run the same diagnostics for rounds 2 and 3 before making a final decision,
   because the lagged features are not informative in the FY2007-only first
   training window.
