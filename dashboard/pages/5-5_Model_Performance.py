from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from theme import apply_theme, render_hero


st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

apply_theme()

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

regression_path = REPORT_DIR / "regression_results.csv"
classification_path = REPORT_DIR / "classification_results.csv"


render_hero(
    "Model Performance & Explainability",
    "Evaluate machine learning models, compare prediction accuracy and understand the factors influencing academic risk.",
    "MACHINE LEARNING INSIGHTS"
)


# Load evaluation results
if not regression_path.exists() or not classification_path.exists():
    st.error("Evaluation result files are missing. Run model training first.")
    st.stop()

regression_df = pd.read_csv(regression_path)
classification_df = pd.read_csv(classification_path)


# Subject and mode filters
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


st.divider()


# Regression results
st.subheader("Regression Model Evaluation")

st.caption(
    "Regression models predict the final grade G3 on a 0–20 scale."
)

reg_subset = regression_df[
    (regression_df["Subject"] == subject) &
    (regression_df["Prediction Mode"] == mode)
].copy()

if not reg_subset.empty:

    reg_subset = reg_subset.sort_values("RMSE")

    best_reg = reg_subset.iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Best Regression Model", best_reg["Model"])

    with c2:
        st.metric("Best RMSE", f"{best_reg['RMSE']:.3f}")

    with c3:
        st.metric("Best R² Score", f"{best_reg['R2 Score']:.3f}")

    st.dataframe(
        reg_subset[
            ["Model", "MAE", "MSE", "RMSE", "R2 Score"]
        ].sort_values("RMSE"),
        width='stretch',
        hide_index=True
    )

else:
    st.warning("No regression results found for this selection.")


st.divider()


# Classification results
st.subheader("Academic Risk Classification")

st.caption(
    "Classification models identify students predicted to score below 10."
)

cls_subset = classification_df[
    (classification_df["Subject"] == subject) &
    (classification_df["Prediction Mode"] == mode)
].copy()

if not cls_subset.empty:

    cls_subset = cls_subset.sort_values(
        "F1 Score",
        ascending=False
    )

    best_cls = cls_subset.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Best Classifier", best_cls["Model"])

    with c2:
        st.metric("Accuracy", f"{best_cls['Accuracy']:.3f}")

    with c3:
        st.metric("F1 Score", f"{best_cls['F1 Score']:.3f}")

    with c4:
        st.metric("ROC-AUC", f"{best_cls['ROC-AUC']:.3f}")

    st.dataframe(
        cls_subset[
            [
                "Model", "Accuracy", "Precision",
                "Recall", "F1 Score", "ROC-AUC"
            ]
        ],
        width='stretch',
        hide_index=True
    )

    st.markdown("#### Confusion Matrix Details")

    c1, c2, c3, c4 = st.columns(4)

    for column, metric in zip(
        [c1, c2, c3, c4],
        ["TN", "FP", "FN", "TP"]
    ):
        with column:
            st.metric(metric, int(best_cls[metric]))

else:
    st.warning("No classification results found for this selection.")


st.divider()


# Evaluation figures
st.subheader("Model Evaluation Visualizations")

prefix = f"{subject.lower()}_{mode}"

figure_groups = [
    (
        "Confusion Matrix",
        f"{prefix}_confusion_matrix.png"
    ),
    (
        "ROC Curve",
        f"{prefix}_roc_curve.png"
    ),
    (
        "Regression RMSE Comparison",
        f"{prefix}_rmse_comparison.png"
    ),
    (
        "Regression R² Comparison",
        f"{prefix}_r2_comparison.png"
    )
]

for index in range(0, len(figure_groups), 2):

    col1, col2 = st.columns(2)

    for column, (title, filename) in zip(
        [col1, col2],
        figure_groups[index:index + 2]
    ):

        with column:

            image_path = FIGURE_DIR / filename

            st.markdown(f"**{title}**")

            if image_path.exists():
                st.image(
                    str(image_path),
                    width='stretch'
                )
            else:
                st.warning(f"Missing figure: {filename}")


st.divider()


# SHAP explainability
st.subheader("SHAP Model Explainability")

st.write(
    "SHAP explains how individual input features influence "
    "the model's prediction of the at-risk class."
)

shap_groups = [
    (
        "SHAP Summary Plot",
        f"{prefix}_shap_summary.png"
    ),
    (
        "SHAP Feature Importance",
        f"{prefix}_shap_importance.png"
    )
]

col1, col2 = st.columns(2)

for column, (title, filename) in zip(
    [col1, col2],
    shap_groups
):

    with column:

        st.markdown(f"**{title}**")

        image_path = FIGURE_DIR / filename

        if image_path.exists():
            st.image(
                str(image_path),
                width='stretch'
            )
        else:
            st.warning(f"Missing SHAP figure: {filename}")


st.divider()

st.caption(
    "Student Performance Analysis | Model evaluation based on "
    "the held-out test datasets."
)
