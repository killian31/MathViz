import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Norm Visualizer", page_icon="♾️")
# This is automatically generated, do not modify
if st.button("Show code"):
    st.code(
        """import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Norm Visualizer", page_icon="♾️")


# Define the norm Np function
def norm_Np(x, p):
    return np.power(np.sum(np.abs(x) ** p), 1 / p)


# Streamlit UI
st.title("Np norm visualization in R2")
st.markdown(
    "This app visualizes the set $E = {x \\in \\mathbb{R}^2 : N_p(x) <= 1}$ for a given value of $p$, where $N_p(x) = (|x_1|^p + |x_2|^p)^(1/p)$."
)

# User input for p (slider)
p = st.slider("Choose a value for p (between 0 and 10):", 0.1, 10.0, 2.0, 0.1)

# Generate a grid of points for visualization
n_points = 400
x = np.linspace(-2, 2, n_points)
y = np.linspace(-2, 2, n_points)
X, Y = np.meshgrid(x, y)

# Calculate the set E using vectorized computation
Z = np.power(np.abs(X) ** p + np.abs(Y) ** p, 1 / p)

# Create a binary mask for the set E
E_mask = Z <= 1

# Plotting the set E
fig, ax = plt.subplots()
ax.imshow(
    E_mask,
    extent=[-2, 2, -2, 2],
    origin="lower",
    cmap="Blues",
    alpha=1.0,
    aspect="auto",
)

# Add zero axes
ax.axhline(0, color="black", linewidth=1)
ax.axvline(0, color="black", linewidth=1)

# Add dots at (-1, 0), (0, -1), (0, 1), and (1, 0)
dots = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])
ax.scatter(dots[:, 0], dots[:, 1], color="red", s=30, zorder=5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"p={p}")
st.pyplot(fig)

st.button("Re-run")
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


# Define the norm Np function
def norm_Np(x, p):
    return np.power(np.sum(np.abs(x) ** p), 1 / p)


# Streamlit UI
st.title("Np norm visualization in R2")
st.markdown(
    "This app visualizes the set $E = {x \\in \\mathbb{R}^2 : N_p(x) <= 1}$ for a given value of $p$, where $N_p(x) = (|x_1|^p + |x_2|^p)^(1/p)$."
)

# User input for p (slider)
p = st.slider("Choose a value for p (between 0 and 10):", 0.1, 10.0, 2.0, 0.1)

# Generate a grid of points for visualization
n_points = 400
x = np.linspace(-2, 2, n_points)
y = np.linspace(-2, 2, n_points)
X, Y = np.meshgrid(x, y)

# Calculate the set E using vectorized computation
Z = np.power(np.abs(X) ** p + np.abs(Y) ** p, 1 / p)

# Create a binary mask for the set E
E_mask = Z <= 1

# Plotting the set E
fig, ax = plt.subplots()
ax.imshow(
    E_mask,
    extent=[-2, 2, -2, 2],
    origin="lower",
    cmap="Blues",
    alpha=1.0,
    aspect="auto",
)

# Add zero axes
ax.axhline(0, color="black", linewidth=1)
ax.axvline(0, color="black", linewidth=1)

# Add dots at (-1, 0), (0, -1), (0, 1), and (1, 0)
dots = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])
ax.scatter(dots[:, 0], dots[:, 1], color="red", s=30, zorder=5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"p={p}")
st.pyplot(fig)

st.button("Re-run")
