import streamlit as st

st.set_page_config(
    page_title="About Project",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

.stApp {
    background: linear-gradient(160deg, #060b17 0%, #0f1f38 35%, #16324a 65%, #1f4a56 100%);
    color: white;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
}

/* Main heading */
.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 500;
    color: white;
    margin-bottom: 10px;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.5px;
}

/* Subtitle */
.main-subtitle {
    text-align: center;
    font-size: 21px;
    color: #b9d8f5;
    margin-bottom: 40px;
    font-family: 'Inter', sans-serif;
    font-weight: 400;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

.card-title {
    font-size: 27px;
    font-weight: 700;
    margin-bottom: 12px;
    color: white;
    font-family: 'Space Grotesk', sans-serif;
}

.card-text {
    font-size: 18px;
    line-height: 1.7;
    color: #dcecff;
    font-family: 'Inter', sans-serif;
    font-weight: 400;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="main-title"> About the Project</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Seasonal Variation of Aerosol Effect on Temperature Analysis'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Project Overview
# -----------------------------

st.markdown("""
<div class="card">

<div class="card-title">
🔬 Project Overview
</div>

<div class="card-text">

This project analyzes the seasonal relationship between
<b>Aerosol Optical Depth (AOD)</b> and
<b>temperature</b> across selected urban regions of India.

Historical environmental data is analyzed across different
seasons to understand variations in aerosol levels and
temperature.

</div>

</div>
""", unsafe_allow_html=True)


# -----------------------------
# Objectives
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    <div class="card">

    <div class="card-title">
    🎯 Objectives
    </div>

    <div class="card-text">

    • Analyze seasonal AOD variation<br>
    • Analyze seasonal temperature variation<br>
    • Compare different Indian cities<br>
    • Study the relationship between AOD and temperature<br>
    • Develop a machine-learning based temperature prediction system

    </div>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="card">

    <div class="card-title">
    🛰️ Data Sources
    </div>

    <div class="card-text">

    • NASA Earthdata / Giovanni<br>
    • Aerosol Optical Depth (AOD)<br>
    • 2-meter air temperature<br>
    • Historical data from multiple years<br>
    • Data covering selected Indian cities

    </div>

    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# Technology
# -----------------------------

st.markdown("""
<div class="card">

<div class="card-title">
⚙️ Technologies Used
</div>

<div class="card-text">

<b>Programming:</b> Python<br>
<b>Data Processing:</b> Pandas, NumPy<br>
<b>Visualization:</b> Matplotlib, Seaborn<br>
<b>Machine Learning:</b> Scikit-learn, Random Forest<br>
<b>Web Application:</b> Streamlit<br>
<b>Data Source:</b> NASA Earthdata / Giovanni

</div>

</div>
""", unsafe_allow_html=True)


# -----------------------------
# Navigation
# -----------------------------

st.write("")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    if st.button(
        "📊 GO TO DASHBOARD",
        use_container_width=True
    ):
        st.switch_page("pages/2_Dashboard.py")
