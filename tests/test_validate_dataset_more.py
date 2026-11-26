import pandas as pd
import pytest

from src.data.loader import validate_dataset


def _frame():
    return pd.DataFrame({
        "school": ["GP"], "age": [15], "studytime": [2], "failures": [0],
        "absences": [2], "G1": [10], "G2": [12], "G3": [14],
    })


def test_valid_dataset_passes():
    validate_dataset(_frame(), "math")


def test_missing_column_error_names_the_dataset_and_column():
    with pytest.raises(ValueError, match="math.*G3"):
        validate_dataset(_frame().drop(columns=["G3"]), "math")


def test_several_missing_columns_are_all_reported():
    with pytest.raises(ValueError) as error:
        validate_dataset(_frame().drop(columns=["G1", "age"]), "portuguese")
    assert "G1" in str(error.value) and "age" in str(error.value)


def test_empty_dataset_is_refused():
    with pytest.raises(ValueError, match="empty"):
        validate_dataset(_frame().iloc[0:0], "math")
