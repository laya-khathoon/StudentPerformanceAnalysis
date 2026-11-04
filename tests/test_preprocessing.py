import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import build_preprocessor, prepare_data


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "school": ["GP", "MS"] * 10,
        "sex": ["F", "M"] * 10,
        "age": [15, 16, 17, 18, 15] * 4,
        "Medu": [4, 2, 3, 1, 4] * 4,
        "Fedu": [2, 4, 3, 2, 3] * 4,
        "Dalc": [1, 2, 1, 3, 2] * 4,
        "Walc": [3, 4, 2, 5, 1] * 4,
        "G1": [10, 12, 14, 8, 15] * 4,
        "G2": [12, 14, 15, 9, 16] * 4,
        "G3": [14, 16, 17, 10, 18] * 4
    })


def to_dense(data):
    return data.toarray() if hasattr(data, "toarray") else data


def test_prepare_data_returns_correct_split(sample_data):
    X_train, X_test, y_train, y_test, _ = prepare_data(
        sample_data, prediction_mode="later"
    )

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert len(y_train) + len(y_test) == len(sample_data)


def test_preprocessed_data_has_no_missing_values(sample_data):
    X_train, X_test, _, _, _ = prepare_data(
        sample_data, prediction_mode="later"
    )

    assert np.isfinite(to_dense(X_train)).all()
    assert np.isfinite(to_dense(X_test)).all()


def test_early_mode_excludes_previous_grades(sample_data):
    _, _, _, _, preprocessor = prepare_data(
        sample_data, prediction_mode="early"
    )

    features = list(preprocessor.feature_names_in_)

    assert "G1" not in features
    assert "G2" not in features


def test_later_mode_includes_engineered_grade_features(sample_data):
    _, _, _, _, preprocessor = prepare_data(
        sample_data, prediction_mode="later"
    )

    features = list(preprocessor.feature_names_in_)

    assert "total_grade" in features
    assert "grade_progress" in features


def test_preprocessor_handles_unseen_categories():
    train = pd.DataFrame({
        "age": [15, 16, 17],
        "school": ["GP", "MS", "GP"]
    })

    test = pd.DataFrame({
        "age": [18],
        "school": ["UNKNOWN"]
    })

    preprocessor = build_preprocessor(train)
    preprocessor.fit(train)
    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1
    assert np.isfinite(to_dense(transformed)).all()
