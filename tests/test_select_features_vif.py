import json

import numpy as np
import pandas as pd
import pytest

from finance_project.data_processing.select_features_vif import (
    apply_selection,
    calculate_vif,
    run_selection,
    select_features,
)


def test_singular_predictors_do_not_give_unrelated_features_infinite_vif():
    x = np.tile([-1, -1, 1, 1], 10)
    y = np.tile([-1, 1, -1, 1], 10)
    frame = pd.DataFrame({"x": x, "duplicate": 2 * x, "independent": y})
    vif = calculate_vif(frame.corr())
    assert np.isinf(vif[["x", "duplicate"]]).all()
    assert vif["independent"] == pytest.approx(1)


def test_iterative_removal_recomputes_and_preserves_row_values():
    rng = np.random.default_rng(42)
    x = rng.normal(size=500)
    frame = pd.DataFrame({"id": range(500), "label": 0, "a": x, "b": 2*x,
                          "c": -3*x, "independent": rng.normal(size=500),
                          "constant": 5, "missing": np.nan})
    original = frame.copy()
    artifact, log, final = select_features(frame, {"group": ["*"]})
    drops = log.loc[log.reason.eq("vif_at_or_above_threshold")]
    assert drops.feature.tolist() == ["a", "b"]
    assert drops.iteration.tolist() == [1, 2]
    assert artifact["selected_features"] == ["c", "independent"]
    assert set(log.loc[log.iteration.eq(0), "reason"]) == {"constant", "all_missing_or_nonfinite"}
    assert (final.vif < 10).all()
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(apply_selection(frame, artifact), frame[["id", "label", "c", "independent"]])


def test_group_overlap_has_a_single_global_feature_list():
    x = np.arange(30)
    frame = pd.DataFrame({"a": x, "b": x * 2, "c": x * 3, "ungrouped_flag": x % 2})
    artifact, _, final = select_features(frame, {"first": ["a", "b"], "second": ["b", "c"]})
    assert artifact["selected_features"] == ["c", "ungrouped_flag"]
    assert artifact["ungrouped_features_retained"] == ["ungrouped_flag"]
    assert (final.vif < 10).all()


def test_finite_high_vif_drops_only_the_highest_then_stops():
    rng = np.random.default_rng(13)
    x = rng.normal(size=1000)
    frame = pd.DataFrame({"a": x, "b": x + .03*rng.normal(size=1000),
                          "c": rng.normal(size=1000)})
    artifact, log, final = select_features(frame, {"group": ["*"]})
    assert len(log) == 1 and np.isfinite(log.vif.iloc[0]) and log.vif.iloc[0] >= 10
    assert len(artifact["selected_features"]) == 2
    assert (final.vif < 10).all()


def test_selection_does_not_apply_diagnostic_imputation_to_outputs():
    frame = pd.DataFrame({"a": [0, 1, np.nan, 2, np.inf], "b": [2, 0, 1, 3, -1]})
    artifact, _, _ = select_features(frame, {"a_group": ["a"], "b_group": ["b"]})
    pd.testing.assert_frame_equal(apply_selection(frame, artifact), frame)
    with pytest.raises(ValueError, match="lacks selected"):
        apply_selection(frame.drop(columns="a"), artifact)


def test_round_output_uses_training_only_and_is_readable_by_model_runner(tmp_path):
    from finance_project.model_development.run_walk_forward_models import load_feature_splits

    input_dir = tmp_path / "raw_features"
    output_dir = tmp_path / "selected"
    input_dir.mkdir()
    x = np.arange(20)
    train = pd.DataFrame({"id": x, "fs_year": 2007, "label": x % 2, "a": x, "b": x * 2})
    train.to_csv(input_dir / "features_train.csv", index=False)
    # Validation/test contradict the training relation and include an unseen column.
    for split in ("validation", "test"):
        other = train.assign(a=x % 3, b=-x, extra=7)
        other.to_csv(input_dir / f"features_{split}.csv", index=False)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"schema_version": "test", "feature_groups": {"group": ["a", "b"]}}))
    artifact = run_selection(input_dir, output_dir, schema_path=schema)
    assert artifact["selected_features"] == ["b"]
    assert json.loads((output_dir / "selected_features.json").read_text())["selected_features"] == ["b"]
    result_train, validation, test, columns = load_feature_splits(output_dir)
    assert columns == ["b"]
    assert len(result_train) == len(validation) == len(test) == len(train)
    np.testing.assert_array_equal(validation.b, -x)
    assert (output_dir / "vif_elimination.csv").exists()
    assert (output_dir / "final_vif.csv").exists()
    with pytest.raises(ValueError, match="separate output"):
        run_selection(input_dir, input_dir, schema_path=schema)
