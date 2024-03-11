import streamlit as st

from utils import switch_page

st.set_page_config(
    page_title="Mathematics & Machine Learning Explorer",
    page_icon="📚",
    initial_sidebar_state="collapsed",
)

st.write("# Welcome to the Mathematics & Machine Learning Explorer!")

st.markdown(
    """
    This app allows you to explore various mathematical/statistical concepts and visualizations, along with machine learning models and algorithms.
    ## **📚 Topics**
"""
)
st.markdown(
    """ <style>
                button {
                    background-color: #f63366;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    transition-duration: 0.4s;
                    cursor: pointer;
                    border-radius: 12px;
                }
                button:hover {
                    background-color: white;
                    color: black;
                    border: 2px solid #f63366;
                }
            </style>""",
    unsafe_allow_html=True,
)

st.write("##### Mathematics")
if st.button("Norm Visualizer"):
    switch_page("Norm Visualizer")
if st.button("Parking Problem Solver (MDP)"):
    switch_page("Parking Problem Solver")
st.write("##### Statistics")
st.write("##### Machine Learning")
if st.button("Neural Network Separability"):
    switch_page("Neural Network Separability")
if st.button("Gradient Descent"):
    switch_page("Gradient Descent")
