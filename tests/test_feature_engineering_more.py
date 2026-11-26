import pandas as pd

from src.features.feature_engineering import create_features


def _frame():
    return pd.DataFrame({
        "Medu": [3], "Fedu": [3], "Dalc": [1], "Walc": [3],
        "G1": [10], "G2": [14],
    })


def test_grade_progress_is_g2_minus_g1():
    assert create_features(_frame(), "later")["grade_progress"].iloc[0] == 4


def test_total_grade_is_g1_plus_g2():
    assert create_features(_frame(), "later")["total_grade"].iloc[0] == 24


def test_early_mode_does_not_add_grade_features():
    result = create_features(_frame(), "early")
    assert "total_grade" not in result.columns
    assert "grade_progress" not in result.columns


def test_early_mode_works_when_grades_are_already_missing():
    frame = _frame().drop(columns=["G1", "G2"])
    assert "parent_education_avg" in create_features(frame, "early").columns


def test_default_mode_is_later():
    assert "total_grade" in create_features(_frame()).columns
