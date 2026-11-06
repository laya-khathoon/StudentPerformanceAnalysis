from html import escape
import streamlit as st

THEME = {
    "page_bg": "#171C22",
    "background": "#1E252D",
    "surface": "#29323D",
    "surface_glow": "#3A424C",
    "foreground": "#F1F3F5",
    "text": "#E6E8EB",
    "muted": "#A3ADB8",
    "border": "rgba(255,255,255,0.12)",
    "divider": "#414B56",
    "primary": "#7FA88A",
    "primary_text": "#17221B",
    "accent": "#7FA88A",
}

def to_css():
    return """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700;800&display=swap');

    :root {
        --page-bg: #171C22;
        --background: #1E252D;
        --surface: #29323D;
        --surface-glow: #3A424C;
        --foreground: #F1F3F5;
        --text: #E6E8EB;
        --muted: #A3ADB8;
        --border: rgba(255,255,255,0.12);
        --divider: #414B56;
        --primary: #7FA88A;
        --primary-text: #17221B;
        --accent: #7FA88A;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: #171C22 !important;
        color: #E6E8EB !important;
        font-family: 'Inter', sans-serif !important;
    }

    .main .block-container {
        max-width: 1500px;
        padding: 2rem 3rem 4rem;
        background: #1E252D !important;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 28px;
        box-shadow: 0 25px 70px rgba(0,0,0,0.25);
        min-height: 94vh;
    }

    section[data-testid="stSidebar"] {
        background: #171C22 !important;
        border-right: 1px solid rgba(255,255,255,0.10);
    }

    section[data-testid="stSidebar"] * {
        color: #E6E8EB;
    }

    h1, h2, h3, h4 {
        color: #F1F3F5 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.035em;
    }

    p, li, label, span,
    [data-testid="stMarkdownContainer"] {
        color: #E6E8EB;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #A3ADB8 !important;
    }

    /* Alert boxes (st.info / warning / success / error) keep Streamlit's
       light pastel backgrounds, so force dark text and icons inside them;
       the app-wide light text color is unreadable on those backgrounds. */
    [data-testid="stAlert"],
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] li,
    [data-testid="stAlert"] a,
    [data-testid="stNotification"],
    [data-testid="stNotification"] p,
    [data-testid="stNotification"] span,
    [data-testid="stNotification"] div {
        color: #262730 !important;
    }

    [data-testid="stAlert"] svg,
    [data-testid="stNotification"] svg {
        fill: #262730 !important;
    }

    .colbo-hero {
        padding: 3.5rem;
        margin-bottom: 2.5rem;
        min-height: 290px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background:
            radial-gradient(circle at 80% 30%, rgba(127,168,138,0.20), transparent 35%),
            radial-gradient(circle at 20% 10%, #3A424C, transparent 60%),
            linear-gradient(135deg, #29323D, #1E252D);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 28px;
        box-shadow: 0 25px 65px rgba(0,0,0,0.30);
    }

    .colbo-eyebrow {
        color: #7FA88A;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.20em;
        margin-bottom: 1.2rem;
    }

    .colbo-hero-title {
        color: #F1F3F5;
        font-family: 'Poppins', sans-serif;
        font-size: clamp(2.5rem, 6vw, 5rem);
        font-weight: 700;
        letter-spacing: -0.05em;
        line-height: 1.05;
        margin-bottom: 1.2rem;
    }

    .colbo-hero-subtitle {
        color: #A3ADB8;
        font-size: 1rem;
        line-height: 1.8;
        max-width: 700px;
    }

    .colbo-card {
        min-height: 190px;
        padding: 2rem;
        background: radial-gradient(circle at 70% 0%, #3A424C, #29323D 75%);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 24px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.25);
    }

    .colbo-card-label {
        color: #7FA88A;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        margin-bottom: 1.2rem;
    }

    .colbo-card-title {
        color: #F1F3F5;
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .colbo-card-description {
        color: #A3ADB8;
        font-size: 0.9rem;
        line-height: 1.7;
    }

    [data-testid="stMetric"] {
        background: radial-gradient(circle at 80% 0%, #3A424C, #29323D 75%);
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.20);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: #A3ADB8 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #F1F3F5 !important;
        font-family: 'Poppins', sans-serif;
    }

    [data-testid="stPlotlyChart"] {
        background: #29323D;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 0.7rem;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: #29323D !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color: #E6E8EB !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background: #29323D !important;
        color: #E6E8EB !important;
    }

    [role="option"] {
        color: #E6E8EB !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: #7FA88A !important;
        color: #17221B !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 700;
        padding: 0.55rem 1.6rem;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #9BC0A5 !important;
        color: #17221B !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: #29323D !important;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        overflow: hidden;
    }

    [data-testid="stExpander"] {
        background: #29323D;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
    }

    hr {
        border-color: #414B56 !important;
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: #29323D;
        border-left: 3px solid #7FA88A;
    }

    [data-testid="stSidebarNav"] a:hover {
        background: #29323D;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    @media(max-width: 900px) {
        .main .block-container {
            padding: 1rem;
            margin: 0;
        }

        .colbo-hero {
            padding: 2rem;
        }
    }
    """

def apply_theme():
    st.markdown(f"<style>{to_css()}</style>", unsafe_allow_html=True)

def render_hero(title, subtitle="", eyebrow="STUDENT PERFORMANCE EXPLORER"):
    st.markdown(
        f"""
        <section class="colbo-hero">
            <div class="colbo-eyebrow">{escape(str(eyebrow))}</div>
            <div class="colbo-hero-title">{escape(str(title))}</div>
            <div class="colbo-hero-subtitle">{escape(str(subtitle))}</div>
        </section>
        """,
        unsafe_allow_html=True
    )

def render_glass_card(title, description="", label="ACADEMIC WORKSPACE"):
    st.markdown(
        f"""
        <section class="colbo-card">
            <div class="colbo-card-label">{escape(str(label))}</div>
            <div class="colbo-card-title">{escape(str(title))}</div>
            <div class="colbo-card-description">{escape(str(description))}</div>
        </section>
        """,
        unsafe_allow_html=True
    )
