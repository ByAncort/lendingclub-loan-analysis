import streamlit as st

st.set_page_config(
    page_title="LendingClub Risk Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding: 0 1rem 2rem 1rem; }

    h1, h2, h3 { color: #1a1a2e; }
    .stMetric {
        background-color: #f0f2f6;
        color: #1a1a2e;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    [data-testid="stMetricValue"] { color: #1a1a2e !important; }
    [data-testid="stMetricLabel"] { color: #4a4a4a !important; }
    .block-container { max-width: 100%; padding: 1.5rem 2rem; }
    hr { margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("# LendingClub")
st.sidebar.markdown("### Risk Intelligence Dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** LendingClub 2007-2018")
st.sidebar.markdown("**Objetivo:** Monitorear riesgo de default")

pages = [
    "01. Riesgo y Calidad de Cartera",
    "02. Rentabilidad / Pricing",
    "03. Perfil del Solicitante",
    "04. Aceptados vs Rechazados",
    "05. Dificultad Financiera",
    "06. Salud del Pipeline",
    "07. Predicción de Default",
    "08. Infraestructura Docker",
]

page = st.sidebar.radio("Navegación", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("*Proyecto Integrador - PCD*")

if page == pages[0]:
    from views.page01_riesgo import show
    show()
elif page == pages[1]:
    from views.page02_pricing import show
    show()
elif page == pages[2]:
    from views.page03_perfil import show
    show()
elif page == pages[3]:
    from views.page04_rechazados import show
    show()
elif page == pages[4]:
    from views.page05_dificultad import show
    show()
elif page == pages[5]:
    from views.page06_pipeline import show
    show()
elif page == pages[6]:
    from views.page07_predictor import show
    show()
elif page == pages[7]:
    from views.page08_docker import show
    show()
