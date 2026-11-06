from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_engineering import create_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REGRESSION_MODEL_DIR = (
    PROJECT_ROOT / "models" / "school" / "regression"
)

DEFAULT_CLASSIFICATION_MODEL_DIR = (
    PROJECT_ROOT / "models" / "school" / "classification"
)

VALID_SUBJECTS = ["Mathematics", "Portuguese"]
VALID_MODES = ["early", "later"]


def _validate_inputs(student_data, subject, mode):
    if subject not in VALID_SUBJECTS:
        raise ValueError(
            "Subject must be Mathematics or Portuguese."
        )

    if mode not in VALID_MODES:
        raise ValueError(
            "Prediction mode must be early or later."
        )

    if not isinstance(student_data, dict) or not student_data:
        raise ValueError(
            "Student data must be a non-empty dictionary."
        )


def _load_model_and_preprocessor(model_directory, subject, mode):
    model_path = (
        model_directory /
        f"{subject.lower()}_{mode}_best_model.joblib"
    )

    preprocessor_path = (
        model_directory /
        f"{subject.lower()}_{mode}_preprocessor.joblib"
    )

    if not model_path.exists() or not preprocessor_path.exists():
        # Fresh deployment (e.g. Streamlit Cloud): the repo does not
        # store the dataset or trained models, so build them once here.
        from src.models.ensure_artifacts import ensure_artifacts

        ensure_artifacts()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {preprocessor_path}"
        )

    return joblib.load(model_path), joblib.load(preprocessor_path)


def _prepare_input(student_data, mode, preprocessor):
    input_df = pd.DataFrame([student_data])

    engineered_df = create_features(
        input_df,
        prediction_mode=mode
    )

    return preprocessor.transform(engineered_df)


def predict_final_grade(student_data, subject, mode, model_dir=None):
    """
    Predict a student's final grade using a saved regression model.

    Parameters:
        student_data: Dictionary containing student input features.
        subject: Mathematics or Portuguese.
        mode: early or later.
        model_dir: Optional directory containing saved models.

    Returns:
        Predicted final grade as a float.
    """

    _validate_inputs(student_data, subject, mode)

    model_directory = (
        Path(model_dir) if model_dir else DEFAULT_REGRESSION_MODEL_DIR
    )

    model, preprocessor = _load_model_and_preprocessor(
        model_directory, subject, mode
    )

    processed_input = _prepare_input(student_data, mode, preprocessor)

    prediction = float(model.predict(processed_input)[0])

    return prediction


def predict_at_risk(student_data, subject, mode, model_dir=None):
    """
    Predict whether a student is at risk of failing (final grade
    below 10) using a saved classification model.

    Parameters:
        student_data: Dictionary containing student input features.
        subject: Mathematics or Portuguese.
        mode: early or later.
        model_dir: Optional directory containing saved models.

    Returns:
        Tuple of (prediction, probability):
            prediction: 1 when the student is classified at risk,
                otherwise 0.
            probability: Model-estimated probability of being
                at risk, between 0 and 1.
    """

    _validate_inputs(student_data, subject, mode)

    model_directory = (
        Path(model_dir) if model_dir else DEFAULT_CLASSIFICATION_MODEL_DIR
    )

    model, preprocessor = _load_model_and_preprocessor(
        model_directory, subject, mode
    )

    processed_input = _prepare_input(student_data, mode, preprocessor)

    prediction = int(model.predict(processed_input)[0])
    probability = float(model.predict_proba(processed_input)[0, 1])

    return prediction, probability
