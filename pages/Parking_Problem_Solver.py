import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Parking Problem Solver", page_icon="🚗")
# This is automatically generated, do not modify
if st.button('Show code'):
    st.code('''import random

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

We use a constant $c \\in \\mathbb{R_+}$ to represent the cost of continuing to search for a parking spot.

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

### Analytical solution using backward induction

The parking problem, formulated within the framework of Markov Decision Processes, finds
a practical solution through backward induction. 

**Initialization**: We start with a value function $V$ initialized for all slots, 
extending to $N+1$ to account for a hypothetical state beyond the last parking slot. 
This state acts as a boundary condition with $V[N+1]=0$.


**Backward Induction Process**:
The iterative process is as follows:
1. For each slot $i$ from $N$ down to $1$, we evaluate the decision-making process under
the assumption that the decision-maker is equipped with the knowledge of all subsequent
outcomes. This reverse evaluation is essential for incorporating the future impacts of
present decisions.
2. For each free slot $(i, F)$, we compute the decision to either park or continue. The
computation involves a comparison between the immediate reward of parking, $i$, and the
expected utility of continuing, which is a function of the value function of the 
subsequent slot $V[i+1]$ and the cost of continuing $c$. The probability $\\rho(i)$ 
that influences the transition to subsequent states adjusts dynamically with $i$.
3. For occupied slots $(i, T)$, the utility is derived from the value function of the
subsequent slot minus the cost of continuing, as parking is not an option.

**Recursion and Policy Determination**: The core of this process is encapsulated in the
recursion formula:
$$
V_{k+1}(s) = \\max_{a \\in \\mathcal{A}(s)} \\{\\mathcal{R}(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} \\mathcal{P}(s' | s, a)V_k(s')\\}
$$
where $\\gamma$ is the discount factor. In this problem, we assume $\\gamma = 1$.
- Alongside the value function, we concurrently develop an optimal policy $\\pi^*$, 
signifying at each slot whether to park or continue. This decision maximizes
the expected utility based on the calculated value function, effectively guiding the 
decision-maker towards the optimal action at each slot.

**Convergence**: This iterative process progresses until we reach the first slot,
culminating in a comprehensive policy $\\pi^*$ that covers the entire parking lot. This
policy gives the optimal action (park or continue) for every slot, derives from
maximizing the expected utility from that point forward.
"""

st.title("Parking Problem Solver")

st.markdown(problem_explanation)


def initialize_parking_lot(N, init_free_prop=0.9, end_free_prop=0.1):
    slots = ["F"] * N
    for i in range(N):
        if random.random() > rho(i, N, init_free_prop, end_free_prop):
            slots[i] = "T"
    return slots


def rho(i, N, p_start=0.9, p_end=0.1):
    """Linearly decreasing probability of a slot being free."""
    return p_start - ((i - 1) / (N - 1)) * (p_start - p_end)


def backward_induction(N, c, p_start=0.9, p_end=0.1, reward_func="i"):
    V_free = np.zeros(N + 1)
    V_taken = np.zeros(N + 1)
    policy = [None] * N

    V_free[N] = N
    V_taken[N] = 0

    for i in range(N - 1, -1, -1):
        prob_free_next = rho(i + 1, N, p_start, p_end)

        V_continue = (
            -c + prob_free_next * V_free[i + 1] + (1 - prob_free_next) * V_taken[i + 1]
        )
        if reward_func == "i":
            V_park = i + 1
        elif reward_func == "2i":
            V_park = 2 * (i + 1)
        elif reward_func == "i^2":
            V_park = (i + 1) ** 2
        else:
            raise ValueError("Invalid reward function")

        V_free[i] = max(V_park, V_continue)
        policy[i] = 1 if V_park >= V_continue else 0

        V_taken[i] = (
            -c + prob_free_next * V_free[i + 1] + (1 - prob_free_next) * V_taken[i + 1]
        )

    return V_free[:-1], policy


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
p_start = st.slider("Initial Probability of Free Slot", 0.0, 1.0, 0.9)
p_end = st.slider("Final Probability of Free Slot", 0.0, p_start, 0.1)
reward_func = st.selectbox(
    "Reward Function",
    ["i", "2i", "i^2"],
    format_func=lambda x: "Reward = " + x,
)
cost_of_continuing = st.slider("Cost of Continuing", 0.0, 10.0, 0.0, step=0.01)

if st.button("Generate Random Parking Lot"):
    parking_lot = initialize_parking_lot(N, init_free_prop=p_start, end_free_prop=p_end)
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
    values, policy = backward_induction(
        N, cost_of_continuing, p_start, p_end, reward_func
    )
    plot_parking_lot(st.session_state.parking_lot, policy)
''')


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

We use a constant $c \\in \\mathbb{R_+}$ to represent the cost of continuing to search for a parking spot.

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

### Analytical solution using backward induction

The parking problem, formulated within the framework of Markov Decision Processes, finds
a practical solution through backward induction. 

**Initialization**: We start with a value function $V$ initialized for all slots, 
extending to $N+1$ to account for a hypothetical state beyond the last parking slot. 
This state acts as a boundary condition with $V[N+1]=0$.


**Backward Induction Process**:
The iterative process is as follows:
1. For each slot $i$ from $N$ down to $1$, we evaluate the decision-making process under
the assumption that the decision-maker is equipped with the knowledge of all subsequent
outcomes. This reverse evaluation is essential for incorporating the future impacts of
present decisions.
2. For each free slot $(i, F)$, we compute the decision to either park or continue. The
computation involves a comparison between the immediate reward of parking, $i$, and the
expected utility of continuing, which is a function of the value function of the 
subsequent slot $V[i+1]$ and the cost of continuing $c$. The probability $\\rho(i)$ 
that influences the transition to subsequent states adjusts dynamically with $i$.
3. For occupied slots $(i, T)$, the utility is derived from the value function of the
subsequent slot minus the cost of continuing, as parking is not an option.

**Recursion and Policy Determination**: The core of this process is encapsulated in the
recursion formula:
$$
V_{k+1}(s) = \\max_{a \\in \\mathcal{A}(s)} \\{\\mathcal{R}(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} \\mathcal{P}(s' | s, a)V_k(s')\\}
$$
where $\\gamma$ is the discount factor. In this problem, we assume $\\gamma = 1$.
- Alongside the value function, we concurrently develop an optimal policy $\\pi^*$, 
signifying at each slot whether to park or continue. This decision maximizes
the expected utility based on the calculated value function, effectively guiding the 
decision-maker towards the optimal action at each slot.

**Convergence**: This iterative process progresses until we reach the first slot,
culminating in a comprehensive policy $\\pi^*$ that covers the entire parking lot. This
policy gives the optimal action (park or continue) for every slot, derives from
maximizing the expected utility from that point forward.
"""

st.title("Parking Problem Solver")

st.markdown(problem_explanation)


def initialize_parking_lot(N, init_free_prop=0.9, end_free_prop=0.1):
    slots = ["F"] * N
    for i in range(N):
        if random.random() > rho(i, N, init_free_prop, end_free_prop):
            slots[i] = "T"
    return slots


def rho(i, N, p_start=0.9, p_end=0.1):
    """Linearly decreasing probability of a slot being free."""
    return p_start - ((i - 1) / (N - 1)) * (p_start - p_end)


def backward_induction(N, c, p_start=0.9, p_end=0.1, reward_func="i"):
    V_free = np.zeros(N + 1)
    V_taken = np.zeros(N + 1)
    policy = [None] * N

    V_free[N] = N
    V_taken[N] = 0

    for i in range(N - 1, -1, -1):
        prob_free_next = rho(i + 1, N, p_start, p_end)

        V_continue = (
            -c + prob_free_next * V_free[i + 1] + (1 - prob_free_next) * V_taken[i + 1]
        )
        if reward_func == "i":
            V_park = i + 1
        elif reward_func == "2i":
            V_park = 2 * (i + 1)
        elif reward_func == "i^2":
            V_park = (i + 1) ** 2
        else:
            raise ValueError("Invalid reward function")

        V_free[i] = max(V_park, V_continue)
        policy[i] = 1 if V_park >= V_continue else 0

        V_taken[i] = (
            -c + prob_free_next * V_free[i + 1] + (1 - prob_free_next) * V_taken[i + 1]
        )

    return V_free[:-1], policy


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
p_start = st.slider("Initial Probability of Free Slot", 0.0, 1.0, 0.9)
p_end = st.slider("Final Probability of Free Slot", 0.0, p_start, 0.1)
reward_func = st.selectbox(
    "Reward Function",
    ["i", "2i", "i^2"],
    format_func=lambda x: "Reward = " + x,
)
cost_of_continuing = st.slider("Cost of Continuing", 0.0, 10.0, 0.0, step=0.01)

if st.button("Generate Random Parking Lot"):
    parking_lot = initialize_parking_lot(N, init_free_prop=p_start, end_free_prop=p_end)
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
    values, policy = backward_induction(
        N, cost_of_continuing, p_start, p_end, reward_func
    )
    plot_parking_lot(st.session_state.parking_lot, policy)
