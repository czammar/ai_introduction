import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Page Config ---
st.set_page_config(page_title="Gradient Ascent Explorer", layout="wide")
st.title("Gradient Ascent Optimization Explorer")
st.write("Adjust the parameters in the sidebar to see how the optimizer finds the peak!")

# --- Sidebar Inputs ---
st.sidebar.header("Hyperparameters")
num_iter = st.sidebar.slider(
    "Number of Iterations",
    10,
    3000,
    10)
lr = st.sidebar.number_input(
    "Learning Rate (lr)",
    min_value=0.00001,
    max_value=3.0,
    value=0.1,
    step=0.1)

st.sidebar.header("Initial Starting Point")
init_x = st.sidebar.slider("Initial x", -2.5, 2.5, -2.0)
init_y = st.sidebar.slider("Initial y", -2.5, 2.5, 2.5)

# --- Define the Surface ---
x_range = np.linspace(-3, 3, 100)
y_range = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = -np.exp(-(X**2 + Y**2))

# --- Gradient Ascent Logic ---
x, y = init_x, init_y
trajectory = []

for step in range(num_iter):
    reward = -np.exp(-(x**2 + y**2))
    trajectory.append((x, y, reward))

    # Gradients
    dx = 2 * x * reward
    dy = 2 * y * reward

    # Update (gradient ascent)
    x = x + lr * dx
    y = y + lr * dy

trajectory = np.array(trajectory)

# --- Show Final Results with Metrics ---
st.subheader("Final State")

# Create three columns for the metrics
m1, m2, m3 = st.columns(3)

with m1:
    st.metric(label="Final X", value=f"{x:.4f}")

with m2:
    st.metric(label="Final Y", value=f"{y:.4f}")

with m3:
    # 'reward' here is the last calculated Z value
    st.metric(label="Final Reward (Z)", value=f"{reward:.4f}")

# --- Visualization ---
fig = plt.figure(figsize=(5,6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, alpha=0.4, cmap='viridis')

# Plot the path
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
        color='red', marker='o', markersize=2, label="Optimization Path")

ax.set_title('Path of Gradient Ascent')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Reward (z)')
ax.legend()

# Display in Streamlit
st.pyplot(fig, use_container_width=False)

