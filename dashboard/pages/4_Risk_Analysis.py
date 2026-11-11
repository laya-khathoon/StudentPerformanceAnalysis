from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from theme import apply_theme, render_hero
from src.models.prediction import predict_at_risk


st.set_page_config(
    page_title="Academic Risk Analysis",
    page_icon="⚠️",
    layout="wide"
)

apply_theme()

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "school"

math_df = pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")
por_df = pd.read_csv(DATA_DIR / "student-por.csv", sep=";")


render_hero(
    "Academic Risk Analysis",
    "Identify students who may need additional academic support using trained classification models.",
    "STUDENT SUPPORT & RISK MONITORING"
)

st.info(
    "Risk definition: A student is classified as at risk when "
    "the predicted probability of obtaining a final grade below "
    "10 out of 20 reaches the model's classification threshold."
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
        format_func=lambda x: (
            "Early Prediction" if x == "early"
            else "Later Prediction"
        )
    )

df = math_df if subject == "Mathematics" else por_df

if mode == "early":
    st.caption(
        "Early risk assessment excludes previous period grades G1 and G2."
    )
else:
    st.caption(
        "Later risk assessment includes previous period grades G1 and G2."
    )

st.divider()

st.subheader("Student Information")

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

with st.form("risk_assessment_form"):

    col1, col2 = st.columns(2)

    for index, column in enumerate(input_columns):

        series = df[column]

        with (col1 if index % 2 == 0 else col2):

            if pd.api.types.is_numeric_dtype(series):

                student_data[column] = st.number_input(
                    column.replace("_", " ").title(),
                    min_value=float(series.min()),
                    max_value=float(series.max()),
                    value=float(series.median()),
                    step=1.0,
                    key=f"risk_{column}"
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
                    key=f"risk_{column}"
                )

    submitted = st.form_submit_button(
        "Assess Academic Risk",
        width='stretch'
    )


if submitted:

    try:

        # Keep the submitted values for the review expander below.
        input_df = pd.DataFrame([student_data])

        try:
            prediction, probability = predict_at_risk(
                student_data, subject, mode
            )
        except FileNotFoundError:
            st.error("The classification model or preprocessor was not found.")
            st.stop()

        st.divider()
        st.subheader("Risk Assessment Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "At-Risk Probability",
                f"{probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Predicted Classification",
                "At Risk" if prediction == 1 else "Not At Risk"
            )

        st.progress(max(0.0, min(1.0, probability)))

        if prediction == 1:
            st.warning(
                "The model classifies this student as potentially "
                "at risk of scoring below 10. Consider reviewing "
                "their academic progress and support requirements."
            )
        else:
            st.success(
                "The model does not classify this student as "
                "at risk under the current classification threshold."
            )

        st.caption(
            f"Subject: {subject} | Prediction stage: {mode.title()} | "
            "Classification threshold: 0.50"
        )

        with st.expander("View submitted student information"):
            st.dataframe(input_df, width='stretch')

    except Exception as error:
        st.error(f"Risk assessment failed: {error}")


# Dataset-level risk overview
st.divider()
st.subheader("Dataset Risk Overview")

risk_df = df.copy()
risk_df["Risk Category"] = risk_df["G3"].apply(
    lambda grade: "At Risk" if grade < 10 else "Not At Risk"
)

risk_counts = (
    risk_df["Risk Category"]
    .value_counts()
    .reindex(["At Risk", "Not At Risk"], fill_value=0)
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(risk_df))

with col2:
    st.metric("At-Risk Records", int(risk_counts["At Risk"]))

with col3:
    st.metric(
        "At-Risk Percentage",
        f"{risk_counts['At Risk'] / len(risk_df) * 100:.1f}%"
    )

st.bar_chart(risk_counts)

st.caption(
    "Dataset risk categories are based on actual G3 grades. "
    "Individual risk assessments above are model predictions."
)
