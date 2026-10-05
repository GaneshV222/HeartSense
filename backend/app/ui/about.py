"""
HeartSense – About Page
"""

import streamlit as st

def render_about():
    st.title("About HeartSense")
    st.markdown("### Cardiovascular Disease Prediction System")
    
    st.markdown("---")
    
    st.markdown("#### Purpose")
    st.markdown("To provide a machine-learning-based cardiovascular risk assessment using patient clinical information.")
    
    st.markdown("#### How it Works")
    st.markdown("- **Advanced Analysis:** The system uses machine learning to analyze patient clinical information.")
    st.markdown("- **Historical Tracking:** Patient visits are logged, and historical data can be compared to identify health trends.")
    
    st.info("⚠️ **Important Disclaimer:** The system is intended for research and decision-support purposes only. It is **NOT** a medical diagnosis. Always consult a qualified healthcare professional.", icon="⚠️")
