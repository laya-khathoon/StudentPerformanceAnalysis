import pandas as pd

from src.data.preprocessing import build_preprocessor


def _frame():
    return pd.DataFrame({
        "age": [15, 16, None],
        "school": ["GP", "MS", None],
    })


def test_numeric_and_text_columns_go_to_separate_pipelines():
    preprocessor = build_preprocessor(_frame())
    names = [name for name, _, _ in preprocessor.transformers]
    assert names == ["numerical", "categorical"]


def test_missing_values_are_filled_after_fit():
    out = build_preprocessor(_frame()).fit_transform(_frame())
    dense = out.toarray() if hasattr(out, "toarray") else out
    assert not pd.isna(dense).any()


def test_unseen_category_is_ignored_not_an_error():
    preprocessor = build_preprocessor(_frame())
    preprocessor.fit(_frame())
    preprocessor.transform(pd.DataFrame({"age": [17], "school": ["NEW"]}))
