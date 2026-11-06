from pathlib import Path
import sys
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

PROJECT_ROOT = DASHBOARD_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fresh deployments (e.g. Streamlit Cloud) do not include the dataset;
# download it once at startup so every page can read it.
from src.models.ensure_artifacts import ensure_data

ensure_data()

st.set_page_config(
    page_title="Student Performance Explorer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

pages = {
    "Workspace": [
        st.Page("home.py", title="Home", icon="🏠", default=True),
        st.Page("pages/1_Overview.py", title="Overview", icon="📊"),
        st.Page("pages/2_Student_Analysis.py", title="Student Analysis", icon="📚"),
    ],
    "Prediction": [
        st.Page("pages/3_Performance_Prediction.py", title="Performance Prediction", icon="🎯"),
        st.Page("pages/4_Risk_Analysis.py", title="Risk Analysis", icon="⚠️"),
    ],
    "Evaluation": [
        st.Page("pages/5_Model_Performance.py", title="Model Performance", icon="📈"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
