from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT / "dashboard"

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from theme import apply_theme, render_hero, render_glass_card

apply_theme()

DATA_DIR = ROOT / "data" / "raw" / "school"

math_df = pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")
por_df = pd.read_csv(DATA_DIR / "student-por.csv", sep=";")

render_hero(
    "Student Performance Overview",
    "A unified view of academic performance, student records and learning outcomes across two school subjects.",
    "ACADEMIC OVERVIEW"
)

st.subheader("Dataset at a Glance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mathematics Students", len(math_df))

with col2:
    st.metric("Portuguese Students", len(por_df))

with col3:
    st.metric("Total Subject Records", len(math_df) + len(por_df))

st.divider()

st.subheader("Academic Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Mathematics Average Grade", f"{math_df['G3'].mean():.2f} / 20")

with col2:
    st.metric("Portuguese Average Grade", f"{por_df['G3'].mean():.2f} / 20")

st.divider()

st.subheader("Final Grade Distribution")

subject = st.selectbox(
    "Choose a subject",
    ["Mathematics", "Portuguese"]
)

selected_df = math_df if subject == "Mathematics" else por_df

grade_data = (
    selected_df["G3"]
    .value_counts()
    .sort_index()
    .rename_axis("Final Grade")
    .reset_index(name="Number of Students")
)

fig = px.bar(
    grade_data,
    x="Final Grade",
    y="Number of Students",
    title=f"{subject} — Final Grade Distribution",
    template="plotly_dark",
    color="Number of Students",
    color_continuous_scale=["#3A424C", "#7FA88A"]
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E6E8EB", family="Inter"),
    title_font=dict(color="#FFFFFF", size=18),
    coloraxis_showscale=False,
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, width='stretch')

st.divider()

st.subheader("Dataset Preview")

st.caption(f"Showing the first 10 records from the {subject} dataset.")

st.dataframe(
    selected_df.head(10),
    width='stretch',
    hide_index=True
)

st.caption(
    "Source: UCI Student Performance Dataset. "
    "Mathematics and Portuguese records are analyzed separately."
)
