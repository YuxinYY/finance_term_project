import pandas as pd

from finance_project.data_processing.make_dataset import clean_dataframe


def test_clean_dataframe_removes_unnamed_index_columns_and_duplicates():
    source = pd.DataFrame({"Unnamed: 0": [0, 1, 1], "profit": [5, 6, 6]})
    cleaned, dropped, duplicates = clean_dataframe(source)

    assert list(cleaned.columns) == ["profit"]
    assert dropped == ["Unnamed: 0"]
    assert duplicates == 1
