import pandas as pd
import pytest

from src.data import loader


@pytest.fixture
def sample_dataset():
    return pd.DataFrame({
        "school": ["GP", "MS"],
        "age": [15, 16],
        "studytime": [2, 3],
        "failures": [0, 1],
        "absences": [2, 4],
        "G1": [10, 12],
        "G2": [12, 14],
        "G3": [14, 16]
    })


def test_load_school_datasets(sample_dataset, tmp_path, monkeypatch):
    sample_dataset.to_csv(
        tmp_path / "student-mat.csv", sep=";", index=False
    )
    sample_dataset.to_csv(
        tmp_path / "student-por.csv", sep=";", index=False
    )

    monkeypatch.setattr(loader, "DATA_DIR", tmp_path)

    math_df, portuguese_df = loader.load_school_datasets()

    pd.testing.assert_frame_equal(math_df, sample_dataset)
    pd.testing.assert_frame_equal(portuguese_df, sample_dataset)


def test_missing_dataset_raises_error(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        loader.load_school_datasets()


def test_validate_dataset_accepts_valid_data(sample_dataset):
    assert loader.validate_dataset(sample_dataset, "Test Dataset") is True


def test_validate_dataset_detects_missing_columns():
    invalid_df = pd.DataFrame({
        "age": [15, 16],
        "G3": [14, 16]
    })

    with pytest.raises(ValueError, match="missing columns"):
        loader.validate_dataset(invalid_df, "Invalid Dataset")


def test_validate_dataset_rejects_empty_data(sample_dataset):
    empty_df = sample_dataset.iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        loader.validate_dataset(empty_df, "Empty Dataset")
