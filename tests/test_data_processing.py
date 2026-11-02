# Tests CSV cleanup and leakage-safe, ID-disjoint walk-forward dataset splits.

import pandas as pd

from finance_project.data_processing.make_dataset import (
    clean_dataframe,
    make_walk_forward_splits,
    write_walk_forward_splits,
)


def test_clean_dataframe_removes_unnamed_index_columns_and_duplicates():
    source = pd.DataFrame({"Unnamed: 0": [0, 1, 1], "profit": [5, 6, 6]})
    cleaned, dropped, duplicates = clean_dataframe(source)

    assert list(cleaned.columns) == ["profit"]
    assert dropped == ["Unnamed: 0"]
    assert duplicates == 1


def test_walk_forward_splits_expand_by_year_and_keep_validation_ids_out_of_train(tmp_path):
    source = pd.DataFrame(
        [
            {"id": company_id, "fs_year": year, "label": int(company_id % 2 == 0)}
            for year in range(2007, 2012)
            for company_id in range(1, 21)
        ]
    )

    splits = make_walk_forward_splits(
        source,
        first_train_year=2007,
        rounds=4,
        validation_size=0.2,
        random_state=7,
    )

    assert [split.test_year for split in splits] == [2008, 2009, 2010, 2011]
    assert [split.train_years for split in splits] == [
        (2007,),
        (2007, 2008),
        (2007, 2008, 2009),
        (2007, 2008, 2009, 2010),
    ]
    for split in splits:
        train_ids = set(split.train["id"])
        validation_ids = set(split.validation["id"])
        assert train_ids.isdisjoint(validation_ids)
        assert set(split.train["fs_year"]) == set(split.train_years)
        assert set(split.validation["fs_year"]) == set(split.train_years)
        assert set(split.test["fs_year"]) == {split.test_year}

    manifest = write_walk_forward_splits(
        splits,
        tmp_path,
        id_column="id",
        target_column="label",
    )
    assert len(manifest) == 4
    assert all(round_info["train_validation_id_overlap"] == 0 for round_info in manifest)
    for round_number in range(1, 5):
        round_dir = tmp_path / f"round_{round_number:02d}"
        assert (round_dir / "train.csv").is_file()
        assert (round_dir / "validation.csv").is_file()
        assert (round_dir / "test.csv").is_file()
