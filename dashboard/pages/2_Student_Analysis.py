from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT / "dashboard"

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from theme import apply_theme, render_hero

apply_theme()

DATA_DIR = ROOT / "data" / "raw" / "school"

try:
    math_df = pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")
    por_df = pd.read_csv(DATA_DIR / "student-por.csv", sep=";")

    render_hero(
        "Student Analysis",
        "Explore student demographics, learning habits, attendance patterns and academic outcomes.",
        "STUDENT INSIGHTS"
    )

    subject = st.selectbox(
        "Select Academic Subject",
        ["Mathematics", "Portuguese"]
    )

    df = math_df.copy() if subject == "Mathematics" else por_df.copy()

    st.sidebar.title("Analysis Filters")
    st.sidebar.caption("Refine the student records displayed below.")

    genders = sorted(df["sex"].dropna().unique().tolist())

    selected_genders = st.sidebar.multiselect(
        "Gender",
        genders,
        default=genders
    )

    min_age = int(df["age"].min())
    max_age = int(df["age"].max())

    selected_age = st.sidebar.slider(
        "Age Range",
        min_age,
        max_age,
        (min_age, max_age)
    )

    study_times = sorted(df["studytime"].dropna().unique().tolist())

    selected_study = st.sidebar.multiselect(
        "Study Time",
        study_times,
        default=study_times
    )

    filtered = df[
        df["sex"].isin(selected_genders)
        & df["age"].between(selected_age[0], selected_age[1])
        & df["studytime"].isin(selected_study)
    ].copy()

    st.subheader("Academic Snapshot")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Students", len(filtered))

    c2.metric(
        "Average Final Grade",
        f"{filtered['G3'].mean():.2f} / 20" if not filtered.empty else "N/A"
    )

    c3.metric(
        "Average Study Time",
        f"{filtered['studytime'].mean():.2f}" if not filtered.empty else "N/A"
    )

    c4.metric(
        "Average Absences",
        f"{filtered['absences'].mean():.2f}" if not filtered.empty else "N/A"
    )

    st.caption(f"Currently analyzing {subject} records.")

    if filtered.empty:
        st.warning("No students match the selected filters.")
        st.stop()

    def style_chart(fig):
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E6E8EB", family="Inter"),
            title_font=dict(color="#FFFFFF", size=17),
            margin=dict(l=20, r=20, t=60, b=20),
            legend_title_font=dict(color="#E6E8EB")
        )
        return fig

    st.divider()
    st.subheader("Academic Performance")

    left, right = st.columns(2)

    with left:
        grade_data = (
            filtered["G3"]
            .value_counts()
            .sort_index()
            .rename_axis("Final Grade")
            .reset_index(name="Students")
        )

        fig = px.bar(
            grade_data,
            x="Final Grade",
            y="Students",
            title="Final Grade Distribution",
            color="Students",
            color_continuous_scale=["#3A424C", "#7FA88A"]
        )

        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        fig = px.box(
            filtered,
            x="studytime",
            y="G3",
            title="Study Time vs Final Grade",
            labels={
                "studytime": "Weekly Study Time",
                "G3": "Final Grade"
            },
            color_discrete_sequence=["#7FA88A"]
        )

        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.divider()
    st.subheader("Student Characteristics")

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            filtered,
            x="age",
            color="sex",
            barmode="group",
            title="Student Age Distribution",
            color_discrete_sequence=["#7FA88A", "#8A8F96"]
        )

        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        failure_data = (
            filtered.groupby("failures", as_index=False)["G3"]
            .mean()
        )

        fig = px.bar(
            failure_data,
            x="failures",
            y="G3",
            title="Previous Failures vs Average Final Grade",
            labels={
                "failures": "Previous Failures",
                "G3": "Average Final Grade"
            },
            color_discrete_sequence=["#7FA88A"]
        )

        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.divider()
    st.subheader("Attendance Analysis")

    fig = px.scatter(
        filtered,
        x="absences",
        y="G3",
        color="studytime",
        hover_data=["age", "sex", "failures"],
        title="Absences vs Final Grade",
        labels={
            "absences": "Number of Absences",
            "G3": "Final Grade",
            "studytime": "Study Time"
        },
        color_continuous_scale=["#8A8F96", "#7FA88A"]
    )

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.divider()
    st.subheader("Student Records")

    st.caption(
        "The dataset does not contain personal student names or IDs. "
        "Records are displayed using their original row positions."
    )

    display_df = filtered.copy()
    display_df.insert(0, "Record Number", display_df.index + 1)

    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    st.caption("Source: UCI Student Performance Dataset")

except Exception as error:
    st.error("The Student Analysis page encountered an error.")
    st.exception(error)
