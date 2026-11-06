from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from theme import apply_theme, render_hero
from src.models.prediction import predict_final_grade


st.set_page_config(
    page_title="Performance Prediction",
    page_icon="🎯",
    layout="wide"
)

apply_theme()

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "school"

math_df = pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")
por_df = pd.read_csv(DATA_DIR / "student-por.csv", sep=";")


render_hero(
    "Academic Performance Prediction",
    "Estimate a student's final grade using machine learning models trained on school student data.",
    "PREDICTIVE ANALYTICS"
)

# Subject and prediction mode
col1, col2 = st.columns(2)

with col1:
    subject = st.selectbox(
        "Select Subject",
        ["Mathematics", "Portuguese"]
    )

with col2:
    mode = st.selectbox(
        "Prediction Stage",
        ["early", "later"],
        format_func=lambda x: "Early Prediction" if x == "early"
        else "Later Prediction"
    )

df = math_df if subject == "Mathematics" else por_df

if mode == "early":
    st.info(
        "Early prediction uses student and family characteristics. "
        "Previous grades G1 and G2 are excluded."
    )
else:
    st.info(
        "Later prediction includes previous period grades G1 and G2 "
        "to estimate the final grade."
    )

st.divider()

st.subheader("Student Information")

st.write(
    "Enter the student's information below. "
    "The available input ranges are based on the training dataset."
)

# Exclude target variable from prediction inputs.
input_columns = [
    col for col in df.columns
    if col != "G3"
]

if mode == "early":
    input_columns = [
        col for col in input_columns
        if col not in ["G1", "G2"]
    ]

student_data = {}

with st.form("student_prediction_form"):

    col1, col2 = st.columns(2)

    for index, column in enumerate(input_columns):

        series = df[column]

        with (col1 if index % 2 == 0 else col2):

            if pd.api.types.is_numeric_dtype(series):

                minimum = float(series.min())
                maximum = float(series.max())
                default = float(series.median())

                student_data[column] = st.number_input(
                    column.replace("_", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=1.0,
                    key=f"input_{column}"
                )

            else:

                options = sorted(
                    series.dropna().unique().tolist()
                )

                default = series.mode().iloc[0]

                student_data[column] = st.selectbox(
                    column.replace("_", " ").title(),
                    options=options,
                    index=options.index(default),
                    key=f"input_{column}"
                )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        use_container_width=True
    )


if submitted:

    try:

        # Keep the submitted values for the review expander below.
        input_df = pd.DataFrame([student_data])

        try:
            prediction = predict_final_grade(student_data, subject, mode)
        except FileNotFoundError:
            st.error("The trained model or preprocessor file was not found.")
            st.stop()

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Predicted Final Grade",
                f"{prediction:.2f} / 20"
            )

        with result_col2:

            if 0 <= prediction <= 20:
                st.metric(
                    "Predicted Grade Percentage",
                    f"{prediction / 20 * 100:.1f}%"
                )
            else:
                st.warning(
                    "The prediction is outside the dataset's "
                    "0–20 grade range. Treat it cautiously."
                )

        if 0 <= prediction <= 20:

            if prediction >= 10:
                st.success(
                    "The predicted grade is at or above the "
                    "passing threshold."
                )
            else:
                st.warning(
                    "The predicted grade is below the passing "
                    "threshold. Additional academic support "
                    "may be beneficial."
                )

        st.caption(
            f"Subject: {subject} | Prediction mode: {mode.title()} | "
            "Model output is an estimate, not a guaranteed result."
        )

        with st.expander("View submitted student information"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
