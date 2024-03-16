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
                    transition-duration: 0.2s;
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


col1, col2, col3, col4 = st.columns(4)

with col1:
    math_menu = st.popover("Mathematics")
    math_menu.page_link(
        "pages/Norm_Visualizer.py", label="Norm Visualizer", use_container_width=True
    )
    math_menu.page_link(
        "pages/Parking_Problem_Solver.py",
        label="Parking Problem Solver (MDP)",
        use_container_width=True,
    )

with col2:
    stats_menu = st.popover("Statistics (Coming Soon)")

with col3:
    ml_menu = st.popover("Machine Learning")
    ml_menu.page_link(
        "pages/Neural_Network_Separability.py",
        label="Neural Network Separability",
        use_container_width=True,
    )
    ml_menu.page_link(
        "pages/Gradient_Descent.py", label="Gradient Descent", use_container_width=True
    )

with col4:
    algo_menu = st.popover("Algorithms (Coming Soon)")
