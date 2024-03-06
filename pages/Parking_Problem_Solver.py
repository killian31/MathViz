import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Parking Problem Solver", page_icon="🚗")

problem_explanation = """

In the context of Markov Decision Processes, the parking problem can be described as follows:

### State Space, $E$
The state space $E$ consists of $N$ tuples, each representing a parking slot. For a given slot $i$ (where $i = 1, ..., N$), the state can be either $(i, F)$ if the slot is free or $(i, T)$ if the slot is taken.

### Action Space, $A$
The action space depends on the current state:
- If $x = (i, T)$, the only possible action is to *continue*.
- If $x = (i, F)$, two actions are possible: *continue* or *park*.

### Reward Function
The reward function is defined as:
- $r((i, F), \\text{stop}) = i$, indicating the reward for parking in slot $i$.
- $r((i, F), \\text{continue}) = -c$, representing the choice to continue despite a free slot.
- $r((i, T), \\text{continue}) = -c$, for continuing when encountering a taken slot.
We use a constant $c \\in (0.5, 1)$ to represent the cost of continuing to search for a parking spot.

### Transition Probabilities
The probability of moving to the next state, given the current state and action, is described by $\\mathbb{P}$:
- $\\mathbb{P}[(i+1, F) | (i, T), \\text{continue}] = \\mathbb{P}[(i+1, F) | (i, F), \\text{continue}] = \\rho(i)$, where $\\rho(i)$ is a decreasing (linear) function in $i$, indicating the probability of the next slot being free.

### Value Function
The value function $V$ for the states is defined as:
- $V((N, F)) = N$, the value of parking in the last free slot.
- $V((N, T)) = 0$, the value of encountering the last slot taken.
- For a free slot $i$, $V((i, F)) = \\max(i, V((i+1, F))\\rho(i) + V((i+1, T))(1 - \\rho(i)))$, the maximum value between parking now or expecting future rewards.
- For a taken slot $i$, $V((i, T)) = \\rho(i)V((i+1, F)) + (1 - \\rho(i))V((i+1, T))$.

The objective is to determine the optimal policy: when to park immediately and when to continue searching for a closer slot.

### Solution Approach

To solve this problem, we propose using the Value 
Iteration algorithm, to find the optimal policy $\\pi^*$ that maximizes the expected 
reward. The value function $V(s)$ represents the maximum expected reward that can be 
obtained from state $s$. The algorithm iterates over all states, updating the value 
function based on the expected rewards of taking each possible action and transitioning
to subsequent states until the value function converges.

The optimal policy $\\pi^*(s)$ at each state $s$ is then determined by choosing the 
action that maximizes the expected reward given the current value function.


### Analytical Solution
**Initialization**: Start with an arbitrary value function $V_0$ and an empty policy $\\pi$.

**Value Iteration**: For each state $s \\in \\mathcal{S}$ in the state space $E$, update the value function
based on the expected reward of taking each possible action $a \\in \\mathcal{A}(s)$ and
transitioning to the next state $s' \\in \\mathcal{S}$. This involves calculating the 
expected reward for each action and choosing the maximum.

$$
V_{k+1}(s) = \\max_{a \\in \\mathcal{A}(s)} \\{\\mathcal{R}(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} \\mathcal{P}(s' | s, a)V_k(s')\\}
$$ 
where $\\gamma$ is the discount factor. In this problem, we assume $\\gamma = 1$.

**Policy Extraction**: After the value function converges, the optimal policy 
$\\pi^*(s)$ at each state $s$ is determined by choosing the action that maximizes the expected reward given the current value function.

$$
\\pi^*(s) = \\arg\\max_{a \\in \\mathcal{A}(s)} \\{\\mathcal{R}(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} \\mathcal{P}(s' | s, a)V(s')\\}
$$
"""
st.title("Parking Problem Solver")

# Display the problem explanation
st.markdown(problem_explanation)


# Initialize the parking lot with a given number of slots and probability of being taken
def initialize_parking_lot(N, p):
    return np.random.choice(["F", "T"], size=N, p=[1 - p, p])


def rho(i, N, p_start=0.9, p_end=0.1):
    """Linearly decreasing probability of a slot being free."""
    return p_start - (i / N) * (p_start - p_end)


# Compute the optimal policy using value iteration
def value_iteration(parking_lot, cost_of_continuing, p_start=0.9, p_end=0.1):
    N = len(parking_lot)
    V = np.zeros(N + 1)  # End of lot has value 0
    policy = np.zeros(N, dtype=int)  # Initialize policy: 0 for continue, 1 for park

    for i in range(N - 1, -1, -1):
        if parking_lot[i] == "F":
            # Calculate the expected value of continuing
            if i < N - 1:  # If not at the last slot
                rho_i = rho(i, N, p_start, p_end)
                continue_value = rho_i * V[i + 1] + (1 - rho_i) * (
                    V[i + 1] - cost_of_continuing
                )
            else:
                continue_value = 0  # Can't continue from the last slot
            park_value = i + 1  # Reward for parking now
            V[i] = max(park_value, continue_value - cost_of_continuing)
            policy[i] = park_value >= (continue_value - cost_of_continuing)
        else:
            V[i] = V[i + 1] - cost_of_continuing  # Adjust for the cost of continuing
            policy[i] = 0

    return policy


# Function to plot the parking lot and policy
def plot_parking_lot(parking_lot, policy):
    fig, ax = plt.subplots(figsize=(10, 2))
    N = len(parking_lot)
    ax.bar(
        range(N),
        [1] * N,
        color=["green" if s == "F" else "red" for s in parking_lot],
        edgecolor="black",
    )
    ax.set_xticks(range(N))
    # the x labels should have only one "Park" -Value and "" after the first "Park" -Value, and "Continue" -Value before the first "Park" -Value
    x_labels = []
    first_park_index = 0
    for i in range(N):
        if policy[i]:
            x_labels.append(f"Park")
            first_park_index = i
            break
        else:
            x_labels.append(f"Continue")
    for i in range(first_park_index + 1, N):
        x_labels.append(f"")

    ax.set_xticklabels(
        x_labels,
        rotation=90,
    )
    ax.set_yticks([])
    ax.set_title("Parking Lot Status and Optimal Policy")
    st.pyplot(fig)


N = st.slider("Number of Parking Slots", 1, 100, 20)
O = st.slider("Proportion of Taken Slots", 0.0, 1.0, 0.1, step=0.01)
p_start = st.slider("Initial Probability of Free Slot", 0.0, 1.0, 0.9)
p_end = st.slider("Final Probability of Free Slot", 0.0, p_start, 0.1)
cost_of_continuing = st.slider("Cost of Continuing", 0.51, 1.0, 0.6, step=0.01)

# parking_lot = initialize_parking_lot(N, p_start)  # if user forgets

if st.button("Generate Random Parking Lot"):
    parking_lot = initialize_parking_lot(N, O)
    # save the parking_lot in the session state
    st.session_state.parking_lot = parking_lot
    # Display the parking lot
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.bar(
        range(N),
        [1] * N,
        color=["green" if s == "F" else "red" for s in parking_lot],
        edgecolor="black",
    )
    ax.set_xticks(range(N))
    ax.set_xticklabels(
        [f"{i+1}" for i in range(N)],
        rotation=90,
    )
    ax.set_yticks([])
    ax.set_title("Parking Lot Status")
    st.pyplot(fig)

if st.button("Solve Parking Problem"):
    if "parking_lot" not in st.session_state:
        st.error("Please generate a parking lot first.")
        st.stop()
    policy = value_iteration(
        st.session_state.parking_lot, cost_of_continuing, p_start, p_end
    )
    plot_parking_lot(st.session_state.parking_lot, policy)
