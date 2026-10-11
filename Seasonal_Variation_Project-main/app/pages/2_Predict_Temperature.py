import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Temperature Prediction",
    page_icon="🌡️",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #07111f, #0b2239);
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */

.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

/* Subtitle */

.subtitle {
    text-align: center;
    font-size: 19px;
    color: #b9d8f5;
    margin-bottom: 40px;
}

/* Input card */

.input-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 30px;
    margin-top: 20px;
}

/* Result card */

.result-card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.20);
    border-radius: 20px;
    padding: 35px;
    text-align: center;
    margin-top: 30px;
}

.result-title {
    font-size: 22px;
    color: #b9d8f5;
}

.result-temperature {
    font-size: 55px;
    font-weight: 800;
    color: white;
    margin-top: 10px;
}

/* Info card */

.info-card {
    background: rgba(255,255,255,0.06);
    border-radius: 15px;
    padding: 20px;
    margin-top: 25px;
}

/* Warning */

.warning-card {
    background: rgba(255, 190, 80, 0.10);
    border: 1px solid rgba(255, 190, 80, 0.35);
    border-radius: 15px;
    padding: 18px;
    margin-top: 20px;
    color: #ffe3b0;
}

/* Tabs */

button[data-baseweb="tab"] {
    font-size: 17px;
    font-weight: 600;
}

/* Navigation buttons */

.nav-button {
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_model(filename):

    path = MODELS_DIR / filename

    if not path.exists():
        return None

    return joblib.load(path)


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_resource
def load_metadata():

    if not METADATA_PATH.exists():
        return None

    with open(METADATA_PATH, "r") as f:
        return json.load(f)


metadata = load_metadata()


# ============================================================
# CHECK METADATA
# ============================================================

if metadata is None:

    st.error("❌ models/model_metadata.json was not found.")

    st.info(
        f"Expected location:\n\n{METADATA_PATH}"
    )

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

rf_model = load_model(
    metadata["model_files"]["Random Forest"]
)

lr_model = load_model(
    metadata["model_files"]["Linear Regression"]
)


# ============================================================
# METADATA VALUES
# ============================================================

cities = metadata["cities"]

seasons = metadata["seasons"]

year_min = metadata["year_min"]

year_max = metadata["year_max"]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌡️ Temperature Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate temperature from Aerosol Optical Depth and environmental conditions'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TWO PREDICTION SECTIONS
# ============================================================

tab_historical, tab_future = st.tabs([
    "📅 Historical Estimate (2012–2020)",
    "🔮 Future Trend Forecast (Experimental)"
])


# ============================================================
# TAB 1 — HISTORICAL PREDICTION
# ============================================================

with tab_historical:

    if rf_model is None:

        st.error("❌ Random Forest model was not found.")

        st.info(
            f"Expected:\n\n"
            f"{MODELS_DIR / metadata['model_files']['Random Forest']}"
        )

        st.stop()


    st.markdown(
        '<div class="input-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        "## 🌍 Environmental Inputs"
    )

    st.write(
        "Estimate temperature for conditions within the study period "
        f"({year_min}–{year_max}) using the Random Forest model."
    )


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        city_h = st.selectbox(
            "🏙️ Select City",
            cities,
            key="city_hist"
        )

        season_h = st.selectbox(
            "🌦️ Select Season",
            seasons,
            key="season_hist"
        )

        aod_h = st.number_input(
            "🌫️ Aerosol Optical Depth (AOD)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.01,
            key="aod_hist"
        )


    with col2:

        year_h = st.number_input(
            "📅 Year",
            min_value=year_min,
            max_value=year_max,
            value=year_max,
            step=1,
            key="year_hist"
        )

        month_h = st.number_input(
            "📆 Month",
            min_value=1,
            max_value=12,
            value=6,
            step=1,
            key="month_hist"
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.write("")


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 ESTIMATE TEMPERATURE",
        use_container_width=True,
        key="predict_hist"
    ):

        input_data = pd.DataFrame({

            "AOD": [aod_h],

            "Year": [year_h],

            "Month": [month_h],

            "City": [city_h],

            "Season": [season_h]

        })


        try:

            pred = float(
                rf_model.predict(input_data)[0]
            )


            result_html = (

                '<div class="result-card">'

                '<div class="result-title">'
                '🌡️ Estimated Temperature'
                '</div>'

                f'<div class="result-temperature">'
                f'{pred:.2f} °C'
                f'</div>'

                '</div>'

            )


            st.markdown(
                result_html,
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="info-card">'
                '<h3>📋 Prediction Details</h3>'
                '</div>',
                unsafe_allow_html=True
            )


            d1, d2, d3, d4, d5 = st.columns(5)


            with d1:
                st.metric(
                    "City",
                    city_h
                )


            with d2:
                st.metric(
                    "Season",
                    season_h
                )


            with d3:
                st.metric(
                    "AOD",
                    f"{aod_h:.2f}"
                )


            with d4:
                st.metric(
                    "Year",
                    year_h
                )


            with d5:
                st.metric(
                    "Month",
                    month_h
                )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


    # --------------------------------------------------------
    # ABOUT HISTORICAL MODEL
    # --------------------------------------------------------

    with st.expander(
        "ℹ️ About this estimate"
    ):

        st.write("""
**Model:** Random Forest Regressor

**R²:** 0.947

**MAE:** approximately 1.00°C

This model estimates the temperature associated with a given:

- City
- Season
- Year
- Month
- AOD

The model was trained using data from 2012–2020.

Because Random Forest is a tree-based model, it should not be used to extrapolate far beyond the years on which it was trained.
""")


# ============================================================
# TAB 2 — FUTURE FORECAST
# ============================================================

with tab_future:


    st.markdown(
        '<div class="warning-card">'
        '⚠️ <b>Experimental.</b> '
        'The Random Forest model is not designed to extrapolate beyond '
        'the training period. This section uses Linear Regression to '
        'provide a rough directional estimate for future years.'
        '</div>',
        unsafe_allow_html=True
    )


    if lr_model is None:

        st.error(
            "❌ Linear Regression model was not found."
        )

        st.info(
            f"Expected:\n\n"
            f"{MODELS_DIR / metadata['model_files']['Linear Regression']}"
        )

        st.stop()


    st.markdown(
        '<div class="input-card">',
        unsafe_allow_html=True
    )


    st.markdown(
        "## 🔮 Future Temperature Scenario"
    )


    st.write(
        "Enter a future year and an assumed AOD value to obtain "
        "a rough temperature trend estimate."
    )


    col1, col2 = st.columns(2)


    with col1:

        city_f = st.selectbox(
            "🏙️ Select City",
            cities,
            key="city_fut"
        )


        season_f = st.selectbox(
            "🌦️ Select Season",
            seasons,
            key="season_fut"
        )


        aod_f = st.number_input(
            "🌫️ Assumed Aerosol Optical Depth (AOD)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.01,
            key="aod_fut"
        )


    with col2:

        year_f = st.number_input(
            "📅 Future Year",
            min_value=year_max + 1,
            max_value=year_max + 15,
            value=year_max + 5,
            step=1,
            key="year_fut"
        )


        month_f = st.number_input(
            "📆 Month",
            min_value=1,
            max_value=12,
            value=6,
            step=1,
            key="month_fut"
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.write("")


    # --------------------------------------------------------
    # FUTURE PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔮 FORECAST TEMPERATURE",
        use_container_width=True,
        key="predict_fut"
    ):


        input_data = pd.DataFrame({

            "AOD": [aod_f],

            "Year": [year_f],

            "Month": [month_f],

            "City": [city_f],

            "Season": [season_f]

        })


        try:

            pred = float(
                lr_model.predict(input_data)[0]
            )


            result_html = (

                '<div class="result-card">'

                '<div class="result-title">'
                '🌡️ Forecasted Temperature'
                '</div>'

                f'<div class="result-temperature">'
                f'{pred:.2f} °C'
                f'</div>'

                '</div>'

            )


            st.markdown(
                result_html,
                unsafe_allow_html=True
            )


            d1, d2, d3, d4, d5 = st.columns(5)


            with d1:
                st.metric(
                    "City",
                    city_f
                )


            with d2:
                st.metric(
                    "Season",
                    season_f
                )


            with d3:
                st.metric(
                    "AOD",
                    f"{aod_f:.2f}"
                )


            with d4:
                st.metric(
                    "Future Year",
                    year_f
                )


            with d5:
                st.metric(
                    "Month",
                    month_f
                )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


    # --------------------------------------------------------
    # ABOUT FUTURE FORECAST
    # --------------------------------------------------------

    with st.expander(
        "ℹ️ Why is this experimental?"
    ):

        st.write("""
**Model:** Linear Regression

**R²:** approximately 0.544

Linear Regression can produce values for years beyond 2020 because it models a mathematical relationship with Year.

However, its accuracy on the test data was substantially lower than Random Forest.

Therefore, this should be treated as a rough scenario estimate rather than a validated climate forecast.

A proper future forecasting system would require a dedicated time-series forecasting approach and future environmental inputs.
""")


# ============================================================
# NAVIGATION
# ============================================================

st.write("")
st.write("")


st.markdown(
    "### 📌 Continue",
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns([1, 1, 1])


# ============================================================
# HOME
# ============================================================

with col1:

    if st.button(
        "🏠 HOME",
        use_container_width=True
    ):

        st.switch_page(
            "app.py"
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

with col2:

    if st.button(
        "📚 ABOUT PROJECT",
        use_container_width=True
    ):

        st.switch_page(
            "pages/3_About_Project.py"
        )


# ============================================================
# ABOUT DEVELOPER
# ============================================================

with col3:

    if st.button(
        "👨‍💻 ABOUT DEVELOPER",
        use_container_width=True
    ):

        st.switch_page(
            "pages/4_About_Developer.py"
        )