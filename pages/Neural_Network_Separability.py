import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

st.set_page_config(
    page_title="Neural Network Separability",
    page_icon="https://github.com/killian31/SimpleNNViz/blob/main/boundary_plot.png",
)
# This is automatically generated, do not modify
if st.button("Show code"):
    st.code(
        '''import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

st.set_page_config(
    page_title="Neural Network Separability",
    page_icon="https://github.com/killian31/SimpleNNViz/blob/main/boundary_plot.png",
)


def list_of_tuples_to_numpy(lst):
    return np.array(lst)


# Function to generate points inside a circle
def generate_points_inside_circle(num_points, circle_center, circle_radius):
    points = []
    while len(points) < num_points:
        x1 = np.random.uniform(0, 10)
        x2 = np.random.uniform(0, 10)
        if (x1 - circle_center[0]) ** 2 + (
            x2 - circle_center[1]
        ) ** 2 <= circle_radius**2:
            points.append((x1, x2))
    return points


# Function to generate points outside a circle
def generate_points_outside_circle(num_points, circle_center, circle_radius):
    points = []
    while len(points) < num_points:
        x1 = np.random.uniform(0, 10)
        x2 = np.random.uniform(0, 10)
        if (x1 - circle_center[0]) ** 2 + (
            x2 - circle_center[1]
        ) ** 2 > circle_radius**2:
            points.append((x1, x2))
    return points


# Function to plot decision boundary
def plot_decision_boundary(model, points_inside, points_outside):
    model.eval()
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)

    # Convert points to NumPy arrays
    points_inside = list_of_tuples_to_numpy(points_inside)
    points_outside = list_of_tuples_to_numpy(points_outside)

    # Plot points inside and outside the circle
    ax.scatter(
        points_inside[:, 0], points_inside[:, 1], color="blue", label="Inside Circle"
    )
    ax.scatter(
        points_outside[:, 0], points_outside[:, 1], color="red", label="Outside Circle"
    )

    # Plot decision boundary
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    input_data = np.c_[xx.ravel(), yy.ravel()]
    inputs = torch.tensor(input_data, dtype=torch.float32)
    outputs = model(inputs).detach().numpy().reshape(xx.shape)
    ax.contour(
        xx, yy, outputs, levels=[0.5], colors="black", linestyles="dashed", linewidths=2
    )

    ax.set_title("Decision Boundary of the Neural Network")
    ax.set_xlabel("")
    ax.set_ylabel("")

    st.pyplot(fig)


def plot_interactive_3d_latent_space(model, points, labels):
    model.eval()
    with torch.no_grad():
        inputs = points.clone().detach()
        hidden_activations = model.fc1(inputs)
        hidden_activations = model.relu1(hidden_activations)
        hidden_activations = model.fc2(hidden_activations)
        hidden_activations = model.relu2(hidden_activations)
        hidden_activations = model.fc3(hidden_activations)

    fig = go.Figure()

    # Scatter plot colored by labels (0 or 1)
    fig.add_trace(
        go.Scatter3d(
            x=hidden_activations[:, 0],
            y=hidden_activations[:, 1],
            z=hidden_activations[:, 2],
            mode="markers",
            marker=dict(size=5, color=labels, colorscale="Viridis", opacity=0.8),
        )
    )

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis_title="Hidden Unit 1",
            yaxis_title="Hidden Unit 2",
            zaxis_title="Hidden Unit 3",
        ),
        scene_camera=dict(
            eye=dict(x=1.87, y=0.88, z=-0.64),
            up=dict(x=0, y=0, z=1),
        ),
    )
    # title
    fig.update_layout(title="3D Latent Space of the Neural Network")
    # augment size
    fig.update_layout(width=500, height=500)

    st.plotly_chart(fig)


class CustomDataset(Dataset):
    def __init__(self, points_inside, points_outside):
        self.data = torch.tensor(points_inside + points_outside, dtype=torch.float32)
        self.labels = torch.tensor(
            [0] * len(points_inside) + [1] * len(points_outside), dtype=torch.long
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class SimpleNN3D(nn.Module):
    def __init__(self):
        super(SimpleNN3D, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # Input layer (2 inputs, 10 hidden units)
        self.relu1 = nn.ReLU()  # Activation function for the first hidden layer
        self.fc2 = nn.Linear(10, 10)  # Second hidden layer (10 hidden units)
        self.relu2 = nn.ReLU()  # Activation function for the second hidden layer
        self.fc3 = nn.Linear(10, 3)  # Third hidden layer (3 hidden units)
        self.relu3 = nn.ReLU()  # Activation function for the third hidden layer
        self.fc4 = nn.Linear(3, 1)  # Output layer (1 output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return torch.sigmoid(x)  # Applying sigmoid activation for binary classification


circle_center_x = 5  # @param {type:"number"}
circle_center_y = 5  # @param {type:"number"}
circle_radius = 3  # @param {type:"number"}
number_points_inside = 200  # @param {type:"number"}
number_points_outside = 200  # @param {type:"number"}

# Circle parameters
circle_center = (circle_center_x, circle_center_y)
np.random.seed(0)
# Generate points inside and outside the circle
points_inside = generate_points_inside_circle(
    number_points_inside, circle_center, circle_radius
)
points_outside = generate_points_outside_circle(
    number_points_outside, circle_center, circle_radius + 0.2
)

points_inside_plot = np.array(points_inside)
points_outside_plot = np.array(points_outside)

# create pandas dataframe with inside and outside label
points_inside_df = pd.DataFrame(points_inside, columns=["x1", "x2"])
points_inside_df["label"] = 0
points_outside_df = pd.DataFrame(points_outside, columns=["x1", "x2"])
points_outside_df["label"] = 1
df = pd.concat([points_inside_df, points_outside_df])

st.title("Neural Network Separability")
st.write(
    """In this example, we will train a neural network to separate points from inside and 
    outside a circle."""
)
st.write("## Data")
st.write("The data consists of points inside and outside a circle.")
st.scatter_chart(df, x="x1", y="x2", color="label")

st.write("## Neural Network")
st.write(
    """We will use a simple neural network with 1 hidden layer to separate the
        points."""
)
model_3d = SimpleNN3D()

epochs = st.slider("Number of epochs", 10, 300, 300, 10)
optimizer = st.radio("Optimizer", ["adam", "sgd"])
learning_rate = st.radio("Learning rate", [0.01, 0.001, 0.0001])
batch_size = st.number_input("Batch size", 1, 64, 16, 8)
if st.button("Train Network"):
    # Create a custom dataset
    custom_dataset = CustomDataset(points_inside, points_outside)

    # Create a data loader
    train_loader = torch.utils.data.DataLoader(
        custom_dataset, batch_size=batch_size, shuffle=True
    )

    # Initialize the loss function, and optimizer

    criterion = nn.BCELoss()
    if optimizer == "adam":
        optimizer = optim.Adam(model_3d.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model_3d.parameters(), lr=learning_rate)

    losses = []
    accuracies = []

    # Training loop
    pbar = st.sidebar.progress(0)
    st.write("#### Loss Progress")
    chart = st.line_chart(losses, use_container_width=True)
    for epoch in range(epochs):
        model_3d.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_3d(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()

        # Compute accuracy
        model_3d.eval()
        with torch.no_grad():
            all_inputs = custom_dataset.data.clone().detach()
            all_labels = custom_dataset.labels.clone().detach().view(-1, 1)
            predictions = model_3d(all_inputs)
            predictions_rounded = predictions.round()
            accuracy = torch.sum(predictions_rounded == all_labels).item() / len(
                all_labels
            )
            accuracies.append(accuracy)

        # Store loss and update progress bar
        losses.append(loss.item())
        last_rows = np.array(losses)
        chart.line_chart(last_rows)
        pbar.progress((epoch + 1) / epochs)
    pbar.empty()
    st.write(f"Training complete!\nAccuracy: {accuracy:.2f}")

    plot_decision_boundary(model_3d, points_inside, points_outside)
    plot_interactive_3d_latent_space(
        model_3d, custom_dataset.data, custom_dataset.labels
    )
'''
    )


def list_of_tuples_to_numpy(lst):
    return np.array(lst)


# Function to generate points inside a circle
def generate_points_inside_circle(num_points, circle_center, circle_radius):
    points = []
    while len(points) < num_points:
        x1 = np.random.uniform(0, 10)
        x2 = np.random.uniform(0, 10)
        if (x1 - circle_center[0]) ** 2 + (
            x2 - circle_center[1]
        ) ** 2 <= circle_radius**2:
            points.append((x1, x2))
    return points


# Function to generate points outside a circle
def generate_points_outside_circle(num_points, circle_center, circle_radius):
    points = []
    while len(points) < num_points:
        x1 = np.random.uniform(0, 10)
        x2 = np.random.uniform(0, 10)
        if (x1 - circle_center[0]) ** 2 + (
            x2 - circle_center[1]
        ) ** 2 > circle_radius**2:
            points.append((x1, x2))
    return points


# Function to plot decision boundary
def plot_decision_boundary(model, points_inside, points_outside):
    model.eval()
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)

    # Convert points to NumPy arrays
    points_inside = list_of_tuples_to_numpy(points_inside)
    points_outside = list_of_tuples_to_numpy(points_outside)

    # Plot points inside and outside the circle
    ax.scatter(
        points_inside[:, 0], points_inside[:, 1], color="blue", label="Inside Circle"
    )
    ax.scatter(
        points_outside[:, 0], points_outside[:, 1], color="red", label="Outside Circle"
    )

    # Plot decision boundary
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    input_data = np.c_[xx.ravel(), yy.ravel()]
    inputs = torch.tensor(input_data, dtype=torch.float32)
    outputs = model(inputs).detach().numpy().reshape(xx.shape)
    ax.contour(
        xx, yy, outputs, levels=[0.5], colors="black", linestyles="dashed", linewidths=2
    )

    ax.set_title("Decision Boundary of the Neural Network")
    ax.set_xlabel("")
    ax.set_ylabel("")

    st.pyplot(fig)


def plot_interactive_3d_latent_space(model, points, labels):
    model.eval()
    with torch.no_grad():
        inputs = points.clone().detach()
        hidden_activations = model.fc1(inputs)
        hidden_activations = model.relu1(hidden_activations)
        hidden_activations = model.fc2(hidden_activations)
        hidden_activations = model.relu2(hidden_activations)
        hidden_activations = model.fc3(hidden_activations)

    fig = go.Figure()

    # Scatter plot colored by labels (0 or 1)
    fig.add_trace(
        go.Scatter3d(
            x=hidden_activations[:, 0],
            y=hidden_activations[:, 1],
            z=hidden_activations[:, 2],
            mode="markers",
            marker=dict(size=5, color=labels, colorscale="Viridis", opacity=0.8),
        )
    )

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis_title="Hidden Unit 1",
            yaxis_title="Hidden Unit 2",
            zaxis_title="Hidden Unit 3",
        ),
        scene_camera=dict(
            eye=dict(x=1.87, y=0.88, z=-0.64),
            up=dict(x=0, y=0, z=1),
        ),
    )
    # title
    fig.update_layout(title="3D Latent Space of the Neural Network")
    # augment size
    fig.update_layout(width=500, height=500)

    st.plotly_chart(fig)


class CustomDataset(Dataset):
    def __init__(self, points_inside, points_outside):
        self.data = torch.tensor(points_inside + points_outside, dtype=torch.float32)
        self.labels = torch.tensor(
            [0] * len(points_inside) + [1] * len(points_outside), dtype=torch.long
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class SimpleNN3D(nn.Module):
    def __init__(self):
        super(SimpleNN3D, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # Input layer (2 inputs, 10 hidden units)
        self.relu1 = nn.ReLU()  # Activation function for the first hidden layer
        self.fc2 = nn.Linear(10, 10)  # Second hidden layer (10 hidden units)
        self.relu2 = nn.ReLU()  # Activation function for the second hidden layer
        self.fc3 = nn.Linear(10, 3)  # Third hidden layer (3 hidden units)
        self.relu3 = nn.ReLU()  # Activation function for the third hidden layer
        self.fc4 = nn.Linear(3, 1)  # Output layer (1 output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return torch.sigmoid(x)  # Applying sigmoid activation for binary classification


circle_center_x = 5  # @param {type:"number"}
circle_center_y = 5  # @param {type:"number"}
circle_radius = 3  # @param {type:"number"}
number_points_inside = 200  # @param {type:"number"}
number_points_outside = 200  # @param {type:"number"}

# Circle parameters
circle_center = (circle_center_x, circle_center_y)
np.random.seed(0)
# Generate points inside and outside the circle
points_inside = generate_points_inside_circle(
    number_points_inside, circle_center, circle_radius
)
points_outside = generate_points_outside_circle(
    number_points_outside, circle_center, circle_radius + 0.2
)

points_inside_plot = np.array(points_inside)
points_outside_plot = np.array(points_outside)

# create pandas dataframe with inside and outside label
points_inside_df = pd.DataFrame(points_inside, columns=["x1", "x2"])
points_inside_df["label"] = 0
points_outside_df = pd.DataFrame(points_outside, columns=["x1", "x2"])
points_outside_df["label"] = 1
df = pd.concat([points_inside_df, points_outside_df])

st.title("Neural Network Separability")
st.write(
    """In this example, we will train a neural network to separate points from inside and 
    outside a circle."""
)
st.write("## Data")
st.write("The data consists of points inside and outside a circle.")
st.scatter_chart(df, x="x1", y="x2", color="label")

st.write("## Neural Network")
st.write(
    """We will use a simple neural network with 1 hidden layer to separate the
        points."""
)
model_3d = SimpleNN3D()

epochs = st.slider("Number of epochs", 10, 300, 300, 10)
optimizer = st.radio("Optimizer", ["adam", "sgd"])
learning_rate = st.radio("Learning rate", [0.01, 0.001, 0.0001])
batch_size = st.number_input("Batch size", 1, 64, 16, 8)
if st.button("Train Network"):
    # Create a custom dataset
    custom_dataset = CustomDataset(points_inside, points_outside)

    # Create a data loader
    train_loader = torch.utils.data.DataLoader(
        custom_dataset, batch_size=batch_size, shuffle=True
    )

    # Initialize the loss function, and optimizer

    criterion = nn.BCELoss()
    if optimizer == "adam":
        optimizer = optim.Adam(model_3d.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model_3d.parameters(), lr=learning_rate)

    losses = []
    accuracies = []

    # Training loop
    pbar = st.sidebar.progress(0)
    st.write("#### Loss Progress")
    chart = st.line_chart(losses, use_container_width=True)
    for epoch in range(epochs):
        model_3d.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_3d(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()

        # Compute accuracy
        model_3d.eval()
        with torch.no_grad():
            all_inputs = custom_dataset.data.clone().detach()
            all_labels = custom_dataset.labels.clone().detach().view(-1, 1)
            predictions = model_3d(all_inputs)
            predictions_rounded = predictions.round()
            accuracy = torch.sum(predictions_rounded == all_labels).item() / len(
                all_labels
            )
            accuracies.append(accuracy)

        # Store loss and update progress bar
        losses.append(loss.item())
        last_rows = np.array(losses)
        chart.line_chart(last_rows)
        pbar.progress((epoch + 1) / epochs)
    pbar.empty()
    st.write(f"Training complete!\nAccuracy: {accuracy:.2f}")

    plot_decision_boundary(model_3d, points_inside, points_outside)
    plot_interactive_3d_latent_space(
        model_3d, custom_dataset.data, custom_dataset.labels
    )
