"""
HeartSense – Home Page
"""

import streamlit as st

def render_home():
    st.title("Cardiovascular Health Assessment")
    st.markdown("Enter your health information or upload your medical report to assess your current cardiovascular risk.")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Please select **My Assessment** from the sidebar to begin your health evaluation.", icon="ℹ️")
        
        st.markdown("""
        ### What to expect:
        1. **Simple Input:** You can either fill out a straightforward form or upload an existing hospital report.
        2. **Fast Results:** Get an immediate AI-driven assessment of your cardiovascular health.
        3. **Track Over Time:** Return to see how your health changes compared to your previous visits.
        """)
