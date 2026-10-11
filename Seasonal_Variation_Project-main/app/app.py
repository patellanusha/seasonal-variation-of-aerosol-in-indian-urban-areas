import streamlit as st

st.set_page_config(
    page_title="Urban Thermal Risk Monitoring System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Background */
    .stApp {
        background-image:
            linear-gradient(
                rgba(0,0,0,0.35),
                rgba(0,0,0,0.35)
            ),
            url("https://images.unsplash.com/photo-1614730321146-b6fa6a46bcb4");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Remove default padding */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }

    /* Title */
    .home-title {
        text-align: center;
        color: white;
        font-size: 55px;
        font-weight: 800;
        margin-top: 170px;
        text-shadow: 3px 3px 10px black;
    }

    /* Subtitle */
    .home-subtitle {
        text-align: center;
        color: white;
        font-size: 26px;
        font-weight: 500;
        margin-top: 20px;
        text-shadow: 2px 2px 8px black;
    }

    /* Start button position */
    div.stButton {
        display: flex;
        justify-content: center;
        margin-top: 180px;
    }

    /* Start button */
    div.stButton > button {
        width: 300px;
        height: 70px;

        font-size: 24px;
        font-weight: 700;

        border-radius: 14px;

        background-color: rgba(15, 20, 35, 0.95);

        color: white;

        border: 2px solid rgba(255,255,255,0.7);

        box-shadow: 0px 5px 20px rgba(0,0,0,0.5);

        transition: 0.3s;
    }

    /* Hover */
    div.stButton > button:hover {
        background-color: rgba(50,60,80,0.98);
        color: white;
        border: 2px solid white;
        transform: scale(1.05);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="home-title">Urban Thermal Risk Monitoring System</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# SUBTITLE
# --------------------------------------------------

st.markdown(
    '<div class="home-subtitle">'
    'Seasonal Variation of Aerosol Effect on Temperature Analysis'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# START BUTTON
# --------------------------------------------------

if st.button("START"):
    st.switch_page("pages/1_Dashboard.py")