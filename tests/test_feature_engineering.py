import pandas as pd
import pytest

from src.features.feature_engineering import create_features


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Medu": [4, 2],
        "Fedu": [2, 4],
        "Dalc": [1, 2],
        "Walc": [3, 4],
        "G1": [10, 12],
        "G2": [12, 14],
        "G3": [14, 16]
    })


def test_parent_education_average(sample_data):
    result = create_features(sample_data, prediction_mode="later")

    assert result["parent_education_avg"].tolist() == [3.0, 3.0]


def test_total_alcohol_calculation(sample_data):
    result = create_features(sample_data, prediction_mode="later")

    assert result["total_alcohol"].tolist() == [4, 6]


def test_early_prediction_removes_grades(sample_data):
    result = create_features(sample_data, prediction_mode="early")

    assert "G1" not in result.columns
    assert "G2" not in result.columns


def test_later_prediction_creates_grade_features(sample_data):
    result = create_features(sample_data, prediction_mode="later")

    assert result["total_grade"].tolist() == [22, 26]
    assert result["grade_progress"].tolist() == [2, 2]


def test_invalid_prediction_mode_raises_error(sample_data):
    with pytest.raises(ValueError):
        create_features(sample_data, prediction_mode="invalid")


def test_original_dataframe_remains_unchanged(sample_data):
    original = sample_data.copy()

    create_features(sample_data, prediction_mode="later")

    pd.testing.assert_frame_equal(sample_data, original)
