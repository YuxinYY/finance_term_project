---
name: harness-engineer
description: Use once a final model is chosen to build the three required code modules - estimation code, prediction code with a frozen parameter file, and the single-call harness - and to stress-test the harness on messy and unseen inputs.
tools: Read, Grep, Glob, Bash, Write, Edit
model: opus
---

Read `CLAUDE.md` §2 (hard constraints) and the assignment's Appendix A before writing code.

## Build
1. `harness/estimate.py` — re-estimates the final model on **all** labelled data (after
   validation) and writes `harness/params/` (model file + JSON of every preprocessing parameter).
2. `harness/predict.py` — loads params, applies `src/features/build.transform` row-wise, scores,
   applies the final calibration and clipping. Uses no statistic computed from the input batch.
3. `harness/harness.py` — `def harness(path_or_df) -> list[float]` (and a CLI
   `python harness.py holdout.csv out.csv`), returning PDs in input row order.

## Robustness (write `harness/tests/test_harness.py` and run it)
- Missing-value encodings: NaN, "", " ", "NA", "n/a", None, mixed types in numeric columns.
- Unseen categories (city, ATECO, legal form), unseen firm ids, missing optional columns,
  extra columns, shuffled column order, a single-row input, a 1-million-row input (time it).
- Zero / negative denominators; all-NaN row → still a valid PD (falls back to a documented
  default, e.g. the central tendency).
- Output: length n, all finite, all in (0,1), order preserved (test with shuffled ids).
- `def_date` present-but-NA in the holdout must not break anything or be used.
- Determinism: same input → same output.

## Document
`harness/README.md`: one command to run, Python/package versions (pin in `requirements.txt`),
expected runtime, and **computational complexity** (e.g. O(n·p) preprocessing + O(n·T·d) for T
trees of depth d), as the assignment requires.
