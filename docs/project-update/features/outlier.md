# Outliers

Outliers are values that are valid but extreme, as opposed to values that should not exist by definition (see [financial-anomalies.md](financial-anomalies.md)).

## Position in the pipeline

```
flag_anomalies(raw)  →  fill_missing  →  outlier handling  →  feature engineering
```

- Anomaly flags are computed on the reported values before this stage, so outlier handling may change values in flagged rows without losing the anomaly information.
- Unlike the anomaly stage, this stage may change values (e.g. winsorizing), so the "keep original values" principle does not necessarily apply.

## Items deferred from the anomaly stage

| Issue | Scale | Note |
|---|---|---|
| `COGS > rev_operating` | 23.8% | Operating losses |
| `profit > rev_operating` | 2.0% | Financial income of holding companies |
| `rev_operating == 0` | 3.05% | Ratios with revenue as the denominator break |
| `roa` ranges from −964 to 908 | | The formula matches exactly; these are extreme values |

## Decision

**No handling for now.** Exploration results are in `notebooks/05_outliers.ipynb`:

- The extreme values are real and carry a strong default signal (e.g. the bottom 1% of `eqty_tot / asst_tot` has a default rate of 40.8%, versus 4.37% overall), so they must not be dropped.
- We expect to deliver a tree model. Trees only use the ordering of values, so they are insensitive to monotonic transformations, and values beyond the training range fall into the last leaf instead of being extrapolated. Winsorizing is therefore not needed.

If we switch to a linear model such as logistic regression, we need to come back: build ratios with `asst_tot` as the denominator, winsorize at the training set's p1/p99 (store the bounds and apply the same ones to the holdout), and use binning for variables like `roa` whose default rate is high at both ends.

Left for feature engineering: ratios with `rev_operating` as the denominator become inf or NaN when revenue is 0 (31,370 rows), so they must be avoided or handled separately; see [feature-engineering-zh.md](feature-engineering-zh.md).
