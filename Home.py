import streamlit as st

from utils import switch_page

st.set_page_config(
    page_title="Mathematics & Machine Learning Explorer",
    page_icon="👋",
    initial_sidebar_state="collapsed",
)

st.write("# Welcome! 👋")

st.markdown(
    """
    This app allows you to explore various mathematical/statistical concepts and visualizations, along with machine learning models and algorithms.
    ## **📚 Topics**
"""
)

st.write("##### Mathematics")
if st.button("Norm Visualizer"):
    switch_page("Norm Visualizer")
st.write("##### Statistics")
st.write("##### Machine Learning")
if st.button("Neural Network Separability"):
    switch_page("Neural Network Separability")
