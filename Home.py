import streamlit as st

from utils import switch_page

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    initial_sidebar_state="collapsed",
)

st.write("# Welcome! 👋")

st.markdown(
    """
    This app allows you to explore various mathematical/statistical concepts and visualizations, along with machine learning models and algorithms.
    ### **📚 Topics**
"""
)

st.write("#### Mathematics")
if st.button("Norm Visualizer"):
    switch_page("Norm Visualizer")
st.write("#### Statistics")
st.write("#### Machine Learning")
