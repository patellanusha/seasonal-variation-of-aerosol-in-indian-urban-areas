import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Environmental Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = PROJECT_ROOT / "output" / "eda_plots"

# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #07111f, #0b2239);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

/* Title */
.title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #b9d8f5;
    margin-bottom: 35px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.card-number {
    font-size: 38px;
    font-weight: 800;
    color: white;
}

.card-label {
    font-size: 17px;
    color: #b9d8f5;
}

/* Section headings */
.section {
    font-size: 30px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="title">📊 Environmental Data Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Seasonal analysis of Aerosol Optical Depth and Temperature'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Summary Cards
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-number">13</div>
        <div class="card-label">Cities</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-number">2012–2020</div>
        <div class="card-label">Study Period</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-number">1404</div>
        <div class="card-label">Monthly Records</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div class="card-number">4</div>
        <div class="card-label">Seasons</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def show_plot(filename, title):
    path = PLOTS_DIR / filename

    if path.exists():
        st.markdown(f"### {title}")
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Plot not found: {filename}")


# --------------------------------------------------
# Seasonal Analysis
# --------------------------------------------------

st.markdown(
    '<div class="section">🌦️ Seasonal Analysis</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    show_plot(
        "03_seasonal_avg_aod.png",
        "Average AOD by Season"
    )

with col2:
    show_plot(
        "04_seasonal_avg_temperature.png",
        "Average Temperature by Season"
    )


# --------------------------------------------------
# Monthly Trends
# --------------------------------------------------

st.markdown(
    '<div class="section">📈 Monthly Trends</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    show_plot(
        "01_monthly_aod_trend.png",
        "Monthly AOD Trend"
    )

with col2:
    show_plot(
        "02_monthly_temperature_trend.png",
        "Monthly Temperature Trend"
    )


# --------------------------------------------------
# City Comparison
# --------------------------------------------------

st.markdown(
    '<div class="section">🏙️ City-wise Comparison</div>',
    unsafe_allow_html=True
)

show_plot(
    "05_citywise_comparison.png",
    "AOD and Temperature Across Cities"
)


# --------------------------------------------------
# Correlation
# --------------------------------------------------

st.markdown(
    '<div class="section">🔗 AOD–Temperature Relationship</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    show_plot(
        "06_correlation_matrix.png",
        "Correlation Matrix"
    )

with col2:
    show_plot(
        "07_aod_vs_temperature_scatter.png",
        "AOD vs Temperature"
    )


# --------------------------------------------------
# Seasonal Correlation
# --------------------------------------------------

st.markdown(
    '<div class="section">🌡️ Seasonal Correlation</div>',
    unsafe_allow_html=True
)

show_plot(
    "11_seasonal_correlation_heatmap.png",
    "AOD–Temperature Correlation by City and Season"
)


# --------------------------------------------------
# Navigation
# --------------------------------------------------

st.write("")
st.write("")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    if st.button(
        "🤖 GO TO TEMPERATURE PREDICTION",
        use_container_width=True
    ):
        st.switch_page("pages/2_Predict_Temperature.py")