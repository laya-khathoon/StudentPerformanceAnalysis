from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_engineering import create_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_DIR = (
    PROJECT_ROOT / "models" / "school" / "regression"
)


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

    if subject not in ["Mathematics", "Portuguese"]:
        raise ValueError(
            "Subject must be Mathematics or Portuguese."
        )

    if mode not in ["early", "later"]:
        raise ValueError(
            "Prediction mode must be early or later."
        )

    if not isinstance(student_data, dict) or not student_data:
        raise ValueError(
            "Student data must be a non-empty dictionary."
        )

    model_directory = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

    model_path = (
        model_directory /
        f"{subject.lower()}_{mode}_best_model.joblib"
    )

    preprocessor_path = (
        model_directory /
        f"{subject.lower()}_{mode}_preprocessor.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {preprocessor_path}"
        )

    input_df = pd.DataFrame([student_data])

    engineered_df = create_features(
        input_df,
        prediction_mode=mode
    )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    processed_input = preprocessor.transform(engineered_df)

    prediction = float(model.predict(processed_input)[0])

    return prediction
