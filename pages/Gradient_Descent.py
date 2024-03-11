import math as math

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
import streamlit as st
from matplotlib.animation import FuncAnimation

st.set_page_config(page_title="Gradient Descent", page_icon="📉")
# This is automatically generated, do not modify
if st.button('Show code'):
    st.code('''import math as math

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
import streamlit as st
from matplotlib.animation import FuncAnimation

st.set_page_config(page_title="Gradient Descent", page_icon="📉")

st.title("Gradient Descent")

st.markdown(
    """
## Gradient algorithm for logistic regression


The [binary logistic regression model](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) is implemented in `scklearn` with the function [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression).
The goal of this practical is to train a logistic regression model from scratch using gradient descent with backtracking line search.

### The logistic regression model
We recall some elements seen in class regarding the logistic regression model.

#### Joint observation: 
$\\big\\{x_1,\\ldots,x_n\\}\\subset \\mathbb{R}^p$, $\\big\\{y_1,\\ldots,y_n\\}\\subset \\{0,1\\}$. Denote by $X \\in \\mathbb{R}^{n \\times p}$ be the design matrix whose $i$-th row is $x_i \\in \\mathbb{R}^p$.

#### Predictor:
For $\\beta \\in \\mathbb{R}^p$ and $x \\in \\mathbb{R}^p$, predict $y = 0$ if the score $\\frac{\\exp(\\left\\langle x, \\beta \\right\\rangle )}{1 + \\exp(\\left\\langle x, \\beta \\right\\rangle )} \\leq \\frac{1}{2}$ and $y = 1$ otherwise

#### Model training:
Find the maximum likelihood, solve the following optimization problem
$$
{\\arg\\min}_{\\beta\\in \\mathbb{R}^p} \\qquad \\mathrm{loss}(\\beta) := \\sum_{i=1}^n - y_i  \\left\\langle \\beta, x_i\\right\\rangle + \\log(1 + \\exp( \\left\\langle \\beta, x_i\\right\\rangle)).
$$

#### Gradient of the loss:
Let $P(\\beta) \\in \\mathbb{R}^n$ bet the vector whose $i$-th entry is $\\frac{\\exp(\\left\\langle x_i, \\beta \\right\\rangle )}{1 + \\exp(\\left\\langle x_i, \\beta \\right\\rangle )}$, the gradient of $F$ is then given by
$$
\\nabla F(\\beta) = X^T(P(\\beta) - y)
$$

### Gradient descent with backtracking
We recall the gradient descent algorithm with backtracking line search. Choose $\\beta_0 \\in \\mathbb{R}^p$ and $L_0 > 0$ and iterate
- for $k = 0, \\ldots$:
- $L = L_0$
- $\\beta_{k+1} = \\beta_k - \\frac{1}{L} \\nabla F(\\beta_k)$
- while $F(\\beta_{k+1}) > F(\\beta_k) - \\frac{1}{2L} \\|\\nabla F(\\beta_k)\\|^2$:
    - $L = 2 \\times L$
    - $\\beta_{k+1} = \\beta_k - \\frac{1}{L} \\nabla F(\\beta_k)$
"""
)


def generate_data(seed, n, p):

    npr.seed(seed)

    n = 1000
    n2 = int(n / 2)

    y = np.zeros((n, 1))
    y[n2:n] = 1

    X = npr.normal(0, 1, (n, p + 1))
    print(X.shape)
    X[n2:n, :] = X[n2:n, :] + 3
    X[n2:n, 0] = X[n2:n, 0] + 2
    X[0:n2, :] = X[0:n2, :] + 1
    X[:, p] = 1

    return X, y


def data_to_df(X, y):
    X_ = X[:, 0:2]
    df = pd.DataFrame(X_)
    df["y"] = y
    # replace y = 0 by orange hexa color and y = 1 by blue hexa color
    df["y"] = df["y"].apply(lambda x: "#FFA500" if x == 0 else "#0000FF")
    df.columns = ["x1", "x2", "y"]

    return df


def plotReg(X, y, beta, s=20, title=""):
    ## Plot input data and classification boundary (only in 2D)
    ## X: design, the first two columns are plotted, the third column is full of ones for the intercept
    ## y: labels in {0, 1}
    ## beta: estimated parameter, third entry is the intercept.
    ## size: optional
    ## title: optional

    # Section 1
    # Plot data points for each class (y=0 and y=1)
    ysel = y.reshape((X.shape[0],))
    fig, ax = plt.subplots()
    ax.scatter(X[ysel == 0, 0], X[ysel == 0, 1], s=s, label="y = 0")
    ax.scatter(X[ysel == 1, 0], X[ysel == 1, 1], s=s, label="y = 1")

    # Section 2
    # Plot the decision boundary based on the parameter beta
    a1 = np.min(X[:, 0])
    a2 = np.max(X[:, 0])
    b1 = -(beta[0] * a1 + beta[2]) / beta[1]
    b2 = -(beta[0] * a2 + beta[2]) / beta[1]
    ax.plot([a1, a2], [b1, b2], c="black")

    # Section 3
    # Adjust the plot settings and add title and legend
    ax.set_title(title)
    # legend
    ax.legend()
    return fig, ax


def plotInitialScatter(ax, X, y):
    ysel = y.reshape((X.shape[0],))
    ax.scatter(X[ysel == 0, 0], X[ysel == 0, 1], label="y = 0")
    ax.scatter(X[ysel == 1, 0], X[ysel == 1, 1], label="y = 1")
    ax.legend()


def updateDecisionBoundary(ax, beta, line, X):
    a1 = np.min(X[:, 0])
    a2 = np.max(X[:, 0])
    b1 = -(beta[0] * a1 + beta[2]) / beta[1]
    b2 = -(beta[0] * a2 + beta[2]) / beta[1]
    line.set_data([a1, a2], [b1, b2])


def logisticLoss(X, y, beta):
    n = X.shape[0]
    logits = X.dot(beta)
    loss = -np.sum(y * logits - np.log(1 + np.exp(logits)))
    return loss


def logisticGrad(X, y, beta):
    # Evaluation of logistic loss gradient, X design, y labels, beta parameters
    n = X.shape[0]
    logits = X.dot(beta)
    grad = X.T.dot(1 / (1 + np.exp(-logits)) - y)
    return grad


def gradientDescentLogistic(X, y, betaZero, L0, max_iters):
    beta = betaZero.copy()
    loss_history = []
    grad_norm_squared_history = []

    chart = st.line_chart(loss_history, use_container_width=True)

    for k in range(max_iters):
        L = L0
        gradient = logisticGrad(X, y, beta)
        loss_current = logisticLoss(X, y, beta)
        grad_norm_squared = np.linalg.norm(gradient) ** 2

        # Initial step
        beta_new = beta - (1 / L) * gradient
        loss_new = logisticLoss(X, y, beta_new)

        # While loop condition directly in the while statement
        while loss_new > loss_current - (1 / (2 * L)) * grad_norm_squared:
            L = 2 * L
            beta_new = beta - (1 / L) * gradient
            loss_new = logisticLoss(X, y, beta_new)

        beta = beta_new
        loss_history.append(loss_new)
        grad_norm_squared_history.append(grad_norm_squared)
        last_rows = np.array(loss_history)
        chart.line_chart(last_rows)

    return beta, loss_history, grad_norm_squared_history


def gradientDescentLogisticWithAnimation(X, y, betaZero, L0, max_iters):
    beta = betaZero.copy()
    fig, ax = plt.subplots()

    # Set the axis limits to freeze them
    ax.set_xlim(np.min(X[:, 0]), np.max(X[:, 0]))
    ax.set_ylim(np.min(X[:, 1]), np.max(X[:, 1]))

    # Plot the initial scatter points
    plotInitialScatter(ax, X, y)

    # Plot an initial decision boundary
    (line,) = ax.plot([], [], c="black")  # Initialize an empty line

    # This function will be called for each frame of the animation
    def update(frame):
        nonlocal beta
        L = L0
        gradient = logisticGrad(X, y, beta)
        loss_current = logisticLoss(X, y, beta)
        grad_norm_squared = np.linalg.norm(gradient) ** 2

        beta_new = beta - (1 / L) * gradient
        loss_new = logisticLoss(X, y, beta_new)

        while loss_new > loss_current - (1 / (2 * L)) * grad_norm_squared:
            L = 2 * L
            beta_new = beta - (1 / L) * gradient
            loss_new = logisticLoss(X, y, beta_new)

        beta = beta_new
        updateDecisionBoundary(ax, beta, line, X)

    animation = FuncAnimation(fig, update, frames=range(max_iters), repeat=False)
    animation.save("logistic_regression.gif", writer="pillow", fps=10)
    return beta


pdim = 2
X, y = generate_data(0, 1000, pdim)
df = data_to_df(X, y)

st.subheader("Data")
st.scatter_chart(df, x="x1", y="x2", color="y")

# plotReg(X, y, np.array([1, 1, -3]), title="Initial classification boundary")
betaZero = np.zeros((pdim + 1, 1))
# streamlit input for betaZero 0, 1, 2
betaZero[0] = st.number_input("Initial beta 0", value=1)
betaZero[1] = st.number_input("Initial beta 1", value=1)
betaZero[2] = st.number_input("Initial beta 2", value=-3)

if st.button("Show initial classification boundary"):
    fig, _ = plotReg(X, y, betaZero, title="Initial classification boundary")
    st.pyplot(fig)


st.subheader("Training")

# learning rate
initial_learning_rate = st.selectbox(
    "Initial learning rate", [0.1, 0.01, 0.001, 0.0001], index=2
)
initial_learning_rate = float(initial_learning_rate)
max_iters = st.slider("Maximum number of iterations", 1, 2000, 1)

if st.button("Train and show loss visualization"):
    beta, loss_history, grad_norm_squared_history = gradientDescentLogistic(
        X, y, betaZero, initial_learning_rate, max_iters
    )

if st.button("Train and show classification animation"):
    beta = gradientDescentLogisticWithAnimation(
        X, y, betaZero, initial_learning_rate, max_iters
    )
    st.image("logistic_regression.gif")
''')


st.title("Gradient Descent")

st.markdown(
    """
## Gradient algorithm for logistic regression


The [binary logistic regression model](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) is implemented in `scklearn` with the function [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression).
The goal of this practical is to train a logistic regression model from scratch using gradient descent with backtracking line search.

### The logistic regression model
We recall some elements seen in class regarding the logistic regression model.

#### Joint observation: 
$\\big\\{x_1,\\ldots,x_n\\}\\subset \\mathbb{R}^p$, $\\big\\{y_1,\\ldots,y_n\\}\\subset \\{0,1\\}$. Denote by $X \\in \\mathbb{R}^{n \\times p}$ be the design matrix whose $i$-th row is $x_i \\in \\mathbb{R}^p$.

#### Predictor:
For $\\beta \\in \\mathbb{R}^p$ and $x \\in \\mathbb{R}^p$, predict $y = 0$ if the score $\\frac{\\exp(\\left\\langle x, \\beta \\right\\rangle )}{1 + \\exp(\\left\\langle x, \\beta \\right\\rangle )} \\leq \\frac{1}{2}$ and $y = 1$ otherwise

#### Model training:
Find the maximum likelihood, solve the following optimization problem
$$
{\\arg\\min}_{\\beta\\in \\mathbb{R}^p} \\qquad \\mathrm{loss}(\\beta) := \\sum_{i=1}^n - y_i  \\left\\langle \\beta, x_i\\right\\rangle + \\log(1 + \\exp( \\left\\langle \\beta, x_i\\right\\rangle)).
$$

#### Gradient of the loss:
Let $P(\\beta) \\in \\mathbb{R}^n$ bet the vector whose $i$-th entry is $\\frac{\\exp(\\left\\langle x_i, \\beta \\right\\rangle )}{1 + \\exp(\\left\\langle x_i, \\beta \\right\\rangle )}$, the gradient of $F$ is then given by
$$
\\nabla F(\\beta) = X^T(P(\\beta) - y)
$$

### Gradient descent with backtracking
We recall the gradient descent algorithm with backtracking line search. Choose $\\beta_0 \\in \\mathbb{R}^p$ and $L_0 > 0$ and iterate
- for $k = 0, \\ldots$:
- $L = L_0$
- $\\beta_{k+1} = \\beta_k - \\frac{1}{L} \\nabla F(\\beta_k)$
- while $F(\\beta_{k+1}) > F(\\beta_k) - \\frac{1}{2L} \\|\\nabla F(\\beta_k)\\|^2$:
    - $L = 2 \\times L$
    - $\\beta_{k+1} = \\beta_k - \\frac{1}{L} \\nabla F(\\beta_k)$
"""
)


def generate_data(seed, n, p):

    npr.seed(seed)

    n = 1000
    n2 = int(n / 2)

    y = np.zeros((n, 1))
    y[n2:n] = 1

    X = npr.normal(0, 1, (n, p + 1))
    print(X.shape)
    X[n2:n, :] = X[n2:n, :] + 3
    X[n2:n, 0] = X[n2:n, 0] + 2
    X[0:n2, :] = X[0:n2, :] + 1
    X[:, p] = 1

    return X, y


def data_to_df(X, y):
    X_ = X[:, 0:2]
    df = pd.DataFrame(X_)
    df["y"] = y
    # replace y = 0 by orange hexa color and y = 1 by blue hexa color
    df["y"] = df["y"].apply(lambda x: "#FFA500" if x == 0 else "#0000FF")
    df.columns = ["x1", "x2", "y"]

    return df


def plotReg(X, y, beta, s=20, title=""):
    ## Plot input data and classification boundary (only in 2D)
    ## X: design, the first two columns are plotted, the third column is full of ones for the intercept
    ## y: labels in {0, 1}
    ## beta: estimated parameter, third entry is the intercept.
    ## size: optional
    ## title: optional

    # Section 1
    # Plot data points for each class (y=0 and y=1)
    ysel = y.reshape((X.shape[0],))
    fig, ax = plt.subplots()
    ax.scatter(X[ysel == 0, 0], X[ysel == 0, 1], s=s, label="y = 0")
    ax.scatter(X[ysel == 1, 0], X[ysel == 1, 1], s=s, label="y = 1")

    # Section 2
    # Plot the decision boundary based on the parameter beta
    a1 = np.min(X[:, 0])
    a2 = np.max(X[:, 0])
    b1 = -(beta[0] * a1 + beta[2]) / beta[1]
    b2 = -(beta[0] * a2 + beta[2]) / beta[1]
    ax.plot([a1, a2], [b1, b2], c="black")

    # Section 3
    # Adjust the plot settings and add title and legend
    ax.set_title(title)
    # legend
    ax.legend()
    return fig, ax


def plotInitialScatter(ax, X, y):
    ysel = y.reshape((X.shape[0],))
    ax.scatter(X[ysel == 0, 0], X[ysel == 0, 1], label="y = 0")
    ax.scatter(X[ysel == 1, 0], X[ysel == 1, 1], label="y = 1")
    ax.legend()


def updateDecisionBoundary(ax, beta, line, X):
    a1 = np.min(X[:, 0])
    a2 = np.max(X[:, 0])
    b1 = -(beta[0] * a1 + beta[2]) / beta[1]
    b2 = -(beta[0] * a2 + beta[2]) / beta[1]
    line.set_data([a1, a2], [b1, b2])


def logisticLoss(X, y, beta):
    n = X.shape[0]
    logits = X.dot(beta)
    loss = -np.sum(y * logits - np.log(1 + np.exp(logits)))
    return loss


def logisticGrad(X, y, beta):
    # Evaluation of logistic loss gradient, X design, y labels, beta parameters
    n = X.shape[0]
    logits = X.dot(beta)
    grad = X.T.dot(1 / (1 + np.exp(-logits)) - y)
    return grad


def gradientDescentLogistic(X, y, betaZero, L0, max_iters):
    beta = betaZero.copy()
    loss_history = []
    grad_norm_squared_history = []

    chart = st.line_chart(loss_history, use_container_width=True)

    for k in range(max_iters):
        L = L0
        gradient = logisticGrad(X, y, beta)
        loss_current = logisticLoss(X, y, beta)
        grad_norm_squared = np.linalg.norm(gradient) ** 2

        # Initial step
        beta_new = beta - (1 / L) * gradient
        loss_new = logisticLoss(X, y, beta_new)

        # While loop condition directly in the while statement
        while loss_new > loss_current - (1 / (2 * L)) * grad_norm_squared:
            L = 2 * L
            beta_new = beta - (1 / L) * gradient
            loss_new = logisticLoss(X, y, beta_new)

        beta = beta_new
        loss_history.append(loss_new)
        grad_norm_squared_history.append(grad_norm_squared)
        last_rows = np.array(loss_history)
        chart.line_chart(last_rows)

    return beta, loss_history, grad_norm_squared_history


def gradientDescentLogisticWithAnimation(X, y, betaZero, L0, max_iters):
    beta = betaZero.copy()
    fig, ax = plt.subplots()

    # Set the axis limits to freeze them
    ax.set_xlim(np.min(X[:, 0]), np.max(X[:, 0]))
    ax.set_ylim(np.min(X[:, 1]), np.max(X[:, 1]))

    # Plot the initial scatter points
    plotInitialScatter(ax, X, y)

    # Plot an initial decision boundary
    (line,) = ax.plot([], [], c="black")  # Initialize an empty line

    # This function will be called for each frame of the animation
    def update(frame):
        nonlocal beta
        L = L0
        gradient = logisticGrad(X, y, beta)
        loss_current = logisticLoss(X, y, beta)
        grad_norm_squared = np.linalg.norm(gradient) ** 2

        beta_new = beta - (1 / L) * gradient
        loss_new = logisticLoss(X, y, beta_new)

        while loss_new > loss_current - (1 / (2 * L)) * grad_norm_squared:
            L = 2 * L
            beta_new = beta - (1 / L) * gradient
            loss_new = logisticLoss(X, y, beta_new)

        beta = beta_new
        updateDecisionBoundary(ax, beta, line, X)

    animation = FuncAnimation(fig, update, frames=range(max_iters), repeat=False)
    animation.save("logistic_regression.gif", writer="pillow", fps=10)
    return beta


pdim = 2
X, y = generate_data(0, 1000, pdim)
df = data_to_df(X, y)

st.subheader("Data")
st.scatter_chart(df, x="x1", y="x2", color="y")

# plotReg(X, y, np.array([1, 1, -3]), title="Initial classification boundary")
betaZero = np.zeros((pdim + 1, 1))
# streamlit input for betaZero 0, 1, 2
betaZero[0] = st.number_input("Initial beta 0", value=1)
betaZero[1] = st.number_input("Initial beta 1", value=1)
betaZero[2] = st.number_input("Initial beta 2", value=-3)

if st.button("Show initial classification boundary"):
    fig, _ = plotReg(X, y, betaZero, title="Initial classification boundary")
    st.pyplot(fig)


st.subheader("Training")

# learning rate
initial_learning_rate = st.selectbox(
    "Initial learning rate", [0.1, 0.01, 0.001, 0.0001], index=2
)
initial_learning_rate = float(initial_learning_rate)
max_iters = st.slider("Maximum number of iterations", 1, 2000, 1)

if st.button("Train and show loss visualization"):
    beta, loss_history, grad_norm_squared_history = gradientDescentLogistic(
        X, y, betaZero, initial_learning_rate, max_iters
    )

if st.button("Train and show classification animation"):
    beta = gradientDescentLogisticWithAnimation(
        X, y, betaZero, initial_learning_rate, max_iters
    )
    st.image("logistic_regression.gif")
