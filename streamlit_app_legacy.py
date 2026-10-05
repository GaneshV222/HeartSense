"""
HeartSense – Main Streamlit Application

Cardiovascular Disease Prediction & Dynamic Risk Assessment System

Run with:
    python -m streamlit run app.py
"""

import streamlit as st

# ── Page config (must be first Streamlit command) ──────────────────────────────
st.set_page_config(
    page_title="HeartSense – CVD Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## HeartSense")
st.sidebar.markdown("**CVD Risk Assessment**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    options=[
        "Home",
        "My Assessment",
        "My History",
        "Analytics",
        "About",
    ],
    index=0,
)

st.sidebar.markdown("---")

# ── Route to the selected page ────────────────────────────────────────────────
if page == "Home":
    from src.ui.assessment import render_assessment
    render_assessment()

elif page == "My Assessment":
    from src.ui.assessment import render_assessment
    render_assessment()

elif page == "My History":
    from src.ui.history import render_history
    render_history()

elif page == "Analytics":
    from src.ui.analytics import render_analytics
    render_analytics()

elif page == "About":
    from src.ui.about import render_about
    render_about()
