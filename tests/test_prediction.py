import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression, LogisticRegression

from src.data.preprocessing import build_preprocessor, prepare_data
from src.features.feature_engineering import create_features
from src.models.prediction import predict_at_risk, predict_final_grade


SUBJECTS = ["Mathematics", "Portuguese"]
MODES = ["early", "later"]


def _synthetic_students(n=60, seed=7):
    """Small synthetic dataset with the full school dataset schema."""
    rng = np.random.default_rng(seed)

    categorical_values = {
        "school": ["GP", "MS"],
        "sex": ["F", "M"],
        "address": ["U", "R"],
        "famsize": ["LE3", "GT3"],
        "Pstatus": ["T", "A"],
        "Mjob": ["teacher", "services", "at_home"],
        "Fjob": ["teacher", "services", "other"],
        "reason": ["course", "home", "reputation"],
        "guardian": ["mother", "father", "other"],
        "schoolsup": ["yes", "no"],
        "famsup": ["yes", "no"],
        "paid": ["yes", "no"],
        "activities": ["yes", "no"],
        "nursery": ["yes", "no"],
        "higher": ["yes", "no"],
        "internet": ["yes", "no"],
        "romantic": ["yes", "no"],
    }

    data = {
        column: rng.choice(values, size=n)
        for column, values in categorical_values.items()
    }

    data.update({
        "age": rng.integers(15, 22, size=n),
        "Medu": rng.integers(0, 5, size=n),
        "Fedu": rng.integers(0, 5, size=n),
        "traveltime": rng.integers(1, 5, size=n),
        "studytime": rng.integers(1, 5, size=n),
        "failures": rng.integers(0, 4, size=n),
        "famrel": rng.integers(1, 6, size=n),
        "freetime": rng.integers(1, 6, size=n),
        "goout": rng.integers(1, 6, size=n),
        "Dalc": rng.integers(1, 6, size=n),
        "Walc": rng.integers(1, 6, size=n),
        "health": rng.integers(1, 6, size=n),
        "absences": rng.integers(0, 20, size=n),
        "G1": rng.integers(0, 21, size=n),
        "G2": rng.integers(0, 21, size=n),
        "G3": rng.integers(0, 21, size=n),
    })

    return pd.DataFrame(data)


def _individual_student():
    """One realistic individual student record (dataset schema)."""
    return {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 3,
        "Fedu": 3,
        "Mjob": "services",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "yes",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 1,
        "health": 5,
        "absences": 2,
        "G1": 12,
        "G2": 13,
    }


def _train_and_save_regression(tmp_path, subject, mode):
    df = _synthetic_students()

    X_train, _, y_train, _, preprocessor = prepare_data(
        df, prediction_mode=mode, random_state=0
    )

    model = LinearRegression().fit(X_train, y_train)

    prefix = f"{subject.lower()}_{mode}"
    joblib.dump(model, tmp_path / f"{prefix}_best_model.joblib")
    joblib.dump(preprocessor, tmp_path / f"{prefix}_preprocessor.joblib")


def _train_and_save_classification(tmp_path, subject, mode):
    df = _synthetic_students()

    X_train, _, y_train, _, preprocessor = prepare_data(
        df, prediction_mode=mode, random_state=0
    )

    # Positive class (1) means the student is at risk, mirroring
    # src/models/train_classification.py.
    y_train_risk = (y_train < 10).astype(int)

    model = LogisticRegression(max_iter=1000).fit(X_train, y_train_risk)

    prefix = f"{subject.lower()}_{mode}"
    joblib.dump(model, tmp_path / f"{prefix}_best_model.joblib")
    joblib.dump(preprocessor, tmp_path / f"{prefix}_preprocessor.joblib")


@pytest.fixture(scope="module")
def regression_model_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("regression_models")
    for subject in SUBJECTS:
        for mode in MODES:
            _train_and_save_regression(model_dir, subject, mode)
    return model_dir


@pytest.fixture(scope="module")
def classification_model_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("classification_models")
    for subject in SUBJECTS:
        for mode in MODES:
            _train_and_save_classification(model_dir, subject, mode)
    return model_dir


# --- Individual student grade predictions (regression) ---

@pytest.mark.parametrize("subject", SUBJECTS)
@pytest.mark.parametrize("mode", MODES)
def test_predict_final_grade_all_subject_mode_combinations(
    subject, mode, regression_model_dir
):
    prediction = predict_final_grade(
        _individual_student(), subject, mode,
        model_dir=regression_model_dir
    )

    assert isinstance(prediction, float)
    assert np.isfinite(prediction)


def test_predict_final_grade_uses_provided_features(regression_model_dir):
    student = _individual_student()

    strong = dict(student, G1=18, G2=19, studytime=4, failures=0)
    weak = dict(student, G1=5, G2=6, studytime=1, failures=3)

    strong_prediction = predict_final_grade(
        strong, "Mathematics", "later",
        model_dir=regression_model_dir
    )
    weak_prediction = predict_final_grade(
        weak, "Mathematics", "later",
        model_dir=regression_model_dir
    )

    assert strong_prediction != weak_prediction


def test_predict_final_grade_early_mode_ignores_g1_g2(regression_model_dir):
    student = _individual_student()

    base = predict_final_grade(
        student, "Mathematics", "early",
        model_dir=regression_model_dir
    )
    changed_grades = predict_final_grade(
        dict(student, G1=1, G2=2), "Mathematics", "early",
        model_dir=regression_model_dir
    )

    assert base == changed_grades


# --- Classification-based at-risk predictions ---

@pytest.mark.parametrize("subject", SUBJECTS)
@pytest.mark.parametrize("mode", MODES)
def test_predict_at_risk_all_subject_mode_combinations(
    subject, mode, classification_model_dir
):
    prediction, probability = predict_at_risk(
        _individual_student(), subject, mode,
        model_dir=classification_model_dir
    )

    assert prediction in (0, 1)
    assert 0.0 <= probability <= 1.0


def test_predict_at_risk_consistent_label_and_probability(
    classification_model_dir
):
    prediction, probability = predict_at_risk(
        _individual_student(), "Mathematics", "later",
        model_dir=classification_model_dir
    )

    # Logistic Regression decides at the 0.5 threshold.
    assert prediction == int(probability >= 0.5)


# --- Input validation ---

def test_predict_final_grade_rejects_invalid_subject(regression_model_dir):
    with pytest.raises(ValueError, match="Subject"):
        predict_final_grade(
            _individual_student(), "Science", "later",
            model_dir=regression_model_dir
        )


def test_predict_final_grade_rejects_invalid_mode(regression_model_dir):
    with pytest.raises(ValueError, match="mode"):
        predict_final_grade(
            _individual_student(), "Mathematics", "final",
            model_dir=regression_model_dir
        )


def test_predict_final_grade_rejects_empty_student_data(
    regression_model_dir
):
    with pytest.raises(ValueError, match="non-empty"):
        predict_final_grade(
            {}, "Mathematics", "later",
            model_dir=regression_model_dir
        )


def test_predict_final_grade_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_final_grade(
            _individual_student(), "Mathematics", "later",
            model_dir=tmp_path
        )


def test_predict_at_risk_rejects_invalid_subject(classification_model_dir):
    with pytest.raises(ValueError, match="Subject"):
        predict_at_risk(
            _individual_student(), "History", "early",
            model_dir=classification_model_dir
        )


def test_predict_at_risk_missing_preprocessor_raises(tmp_path):
    joblib.dump(
        LogisticRegression(),
        tmp_path / "mathematics_early_best_model.joblib"
    )

    with pytest.raises(FileNotFoundError):
        predict_at_risk(
            _individual_student(), "Mathematics", "early",
            model_dir=tmp_path
        )
