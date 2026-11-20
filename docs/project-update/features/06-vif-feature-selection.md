# Iterative VIF feature selection

Run `finance_project.data_processing.select_features_vif` after transforming the
three splits and before training a model, independently for each temporal round.
This replaces the former `preprocess collinearity` command. The existing
collinearity reports are historical results for earlier feature definitions.

```bash
PYTHONPATH=src /opt/miniconda3/envs/daily/bin/python \
  -m finance_project.data_processing.select_features_vif \
  --input-dir data/processed/walk_forward_temporal/round_01 \
  --output-dir data/processed/walk_forward_temporal_vif/round_01
```

Repeat with `round_02` and `round_03`, then point model training at the new root:

```bash
PYTHONPATH=src /opt/miniconda3/envs/daily/bin/python \
  -m finance_project.model_development.run_walk_forward_models \
  --data-dir data/processed/walk_forward_temporal_vif \
  --rounds 1,2,3 --models lightgbm --run-name temporal_firm_year_vif_v1
```

## Selection rule

Only training data determine feature selection. The script resolves groups and
wildcards from `config/feature_schema.json`, removes entirely missing/nonfinite
or constant training columns, then computes each group's VIF. If the maximum
is at least 10, exactly one feature with that VIF is removed, and VIF is
recomputed. Iteration ends only when every remaining group VIF is below 10.
Ties, including infinite VIF, are broken alphabetically by feature name.

Exact dependencies are handled using individual regression residuals, rather
than assigning infinite VIF to every variable in a singular matrix. If a feature
belongs to multiple groups, removing it removes it globally; final VIFs are
recomputed for every group. Nonconstant features outside configured groups are
retained and listed in the artifact. Metadata and targets never enter VIF.

All training rows are used by default. `--max-rows 100000` requests a fixed
reproducible diagnostic sample. For diagnostics only, infinity is treated as
missing and missing values receive training medians. Saved feature values and
row counts are preserved. Constants in the diagnostic sample are logged as such.
This is VIF on the transformed financial inputs; it is not VIF after WoE, and
it does not eliminate dependencies spanning separate groups.

## Outputs

- `features_train.csv`, `features_validation.csv`, `features_test.csv`: identical
  selected feature order across splits, with original metadata and rows.
- `selected_features.json`: ordered survivors, removed/ungrouped features,
  selection settings and schema identity. Retain this with the trained model
  and apply its feature list at inference time.
- `vif_elimination.csv`: precheck removals and one-feature-at-a-time VIF removals
  with group, iteration, VIF and reason.
- `final_vif.csv`: final per-group VIFs for the survivors.

Validation/test never refit selection. Missing selected columns raise an error;
extra columns are excluded. The full transformed feature files are preserved in
the input directory. Each temporal round can have a different selected list.
The submission/harness builders have not been updated to apply selection yet.
