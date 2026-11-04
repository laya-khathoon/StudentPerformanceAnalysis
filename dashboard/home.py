from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "dashboard"

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from theme import apply_theme, render_hero, render_glass_card

apply_theme()

DATA_DIR = ROOT / "data" / "raw" / "school"

math_df = pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")
por_df = pd.read_csv(DATA_DIR / "student-por.csv", sep=";")

render_hero(
    "Every Student Has a Story.",
    "An intelligent academic analysis platform to understand student performance, explore learning patterns and identify students who may need academic support.",
    "SCHOOL PERFORMANCE INTELLIGENCE"
)

# Main statistics
st.subheader("Your Academic Workspace")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", len(math_df) + len(por_df))
c2.metric("Mathematics Students", len(math_df))
c3.metric("Portuguese Students", len(por_df))
c4.metric("Subjects Analysed", "02")

st.divider()

# Subject performance
st.subheader("Subject Performance")

math_avg = math_df["G3"].mean()
por_avg = por_df["G3"].mean()

left, right = st.columns(2)

with left:
    render_glass_card(
        "Mathematics",
        f"{len(math_df)} student records | Average final grade: {math_avg:.2f}/20",
        "SUBJECT 01"
    )

with right:
    render_glass_card(
        "Portuguese",
        f"{len(por_df)} student records | Average final grade: {por_avg:.2f}/20",
        "SUBJECT 02"
    )

comparison = pd.DataFrame({
    "Subject": ["Mathematics", "Portuguese"],
    "Average Final Grade": [math_avg, por_avg]
})

fig = px.bar(
    comparison,
    x="Subject",
    y="Average Final Grade",
    color="Subject",
    text_auto=".2f",
    title="Average Final Grade Comparison",
    color_discrete_map={
        "Mathematics": "#7FA88A",
        "Portuguese": "#8A8F96"
    }
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E6E8EB", family="Inter"),
    title_font=dict(color="#F1F3F5", size=18),
    showlegend=False,
    yaxis=dict(range=[0, 20], title="Average Grade (/20)"),
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Grade distribution
st.subheader("Understanding Academic Outcomes")

grade_distribution = pd.concat([
    math_df.assign(Subject="Mathematics"),
    por_df.assign(Subject="Portuguese")
], ignore_index=True)

fig = px.histogram(
    grade_distribution,
    x="G3",
    color="Subject",
    barmode="overlay",
    opacity=0.75,
    nbins=21,
    title="Final Grade Distribution Across Subjects",
    labels={
        "G3": "Final Grade",
        "count": "Number of Students"
    },
    color_discrete_map={
        "Mathematics": "#7FA88A",
        "Portuguese": "#8A8F96"
    }
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E6E8EB", family="Inter"),
    title_font=dict(color="#F1F3F5", size=18),
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Navigation section
st.subheader("Explore the Platform")

col1, col2 = st.columns(2)

with col1:
    render_glass_card(
        "Student Analysis",
        "Explore student demographics, study habits, attendance, academic performance and learning patterns.",
        "01 / DATA EXPLORATION"
    )

    st.write("")

    render_glass_card(
        "Performance Prediction",
        "Estimate final student grades using trained regression models and early or later prediction modes.",
        "02 / MACHINE LEARNING"
    )

with col2:
    render_glass_card(
        "Risk Analysis",
        "Identify students who may be at risk of scoring below the passing threshold.",
        "03 / ACADEMIC SUPPORT"
    )

    st.write("")

    render_glass_card(
        "Model Performance",
        "Compare machine learning models through evaluation metrics, ROC curves, confusion matrices and SHAP explanations.",
        "04 / MODEL EVALUATION"
    )

st.divider()

st.caption(
    "Data source: UCI Student Performance Dataset | "
    "Mathematics and Portuguese datasets are analysed separately."
)
