import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
np.random.seed(0)
n_points = 10          # Number of points
iterations = 100       # Number of iterations for simulation
learning_rate = 0.1    # Step size for moving points
perplexity = 2.0       # Desired perplexity

# Initialize points randomly in 2D
y = np.random.rand(n_points, 2) * 10  # Initial positions in 2D space

# Function to calculate high-dimensional similarity probabilities
def calculate_p_matrix(data, perplexity):
    """Calculates pairwise similarities P_ij based on Gaussian kernel (high-dimensional space)."""
    n = data.shape[0]
    p_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(data[i] - data[j])
            p_ij = np.exp(-dist**2 / (2 * perplexity**2))
            p_matrix[i, j] = p_ij
            p_matrix[j, i] = p_ij
    p_matrix /= p_matrix.sum()
    return p_matrix

# Create a mock high-dimensional space similarity matrix for this example
X_high_dim = np.random.randn(n_points, 5)  # Mock high-dimensional data
P = calculate_p_matrix(X_high_dim, perplexity)

# Function to calculate low-dimensional similarity probabilities Q_ij using Student's t-distribution
def calculate_q_matrix(y):
    """Calculates pairwise similarities Q_ij based on Student's t-distribution (low-dimensional space)."""
    n = y.shape[0]
    q_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = np.linalg.norm(y[i] - y[j])**2
            q_ij = 1 / (1 + dist_sq)
            q_matrix[i, j] = q_ij
            q_matrix[j, i] = q_ij
    q_matrix /= q_matrix.sum()
    return q_matrix

# Calculate the gradients based on P and Q matrices
def calculate_gradient(y, P, Q):
    """Calculates the gradient to update positions based on the t-SNE cost function."""
    grad = np.zeros_like(y)
    for i in range(len(y)):
        for j in range(len(y)):
            if i != j:
                # Compute attractive or repulsive force
                force = (P[i, j] - Q[i, j]) * (y[i] - y[j])
                grad[i] += 2 * force  # Multiply by 2 as per t-SNE gradient form
    return grad

# Set up the plot
fig, ax = plt.subplots()
scat = ax.scatter(y[:, 0], y[:, 1], color='blue')
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

def update(frame):
    global y
    # Compute current low-dimensional similarities Q
    Q = calculate_q_matrix(y)
    
    # Calculate the gradient and update y
    gradient = calculate_gradient(y, P, Q)
    y -= learning_rate * gradient  # Gradient descent step
    
    # Update scatter plot
    scat.set_offsets(y)
    return scat,

# Create the animation
ani = FuncAnimation(fig, update, frames=iterations, interval=100, blit=True)
plt.show()