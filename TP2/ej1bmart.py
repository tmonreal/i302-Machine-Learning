import numpy as np
import matplotlib.pyplot as plt
from utils import get_best_coef, model_predict, load_data

# Load the data
X, y = load_data('/home/linar/Desktop/ML/Clases/Clase 3/i302/TP2/datasets/toy_dataset.pkl')

# Shuffle the data
data = list(zip(X, y))
np.random.shuffle(data)
X_shuffled, y_shuffled = zip(*data)
X_shuffled, y_shuffled = np.array(X_shuffled), np.array(y_shuffled)

# Split the shuffled data into training and validation sets
split_ratio = 0.75
split_index = int(split_ratio * len(X_shuffled))
X_train, X_val = X_shuffled[:split_index], X_shuffled[split_index:]
y_train, y_val = y_shuffled[:split_index], y_shuffled[split_index:]

# Define a list of values of M to try
M_values = range(1, 10)

# Lists to store training and validation errors
mse_train_values = []
mse_val_values = []

# Train and evaluate models for each value of M
for M in M_values:
    # Get the coefficients for the model
    w = get_best_coef(X_train, y_train, M)
    
    # Make predictions
    y_train_pred = model_predict(w, X_train)
    y_val_pred = model_predict(w, X_val)
    
    # Calculate mean squared errors
    mse_train = np.mean((y_train - y_train_pred) ** 2)
    mse_val = np.mean((y_val - y_val_pred) ** 2)
    
    # Append to lists
    mse_train_values.append(mse_train)
    mse_val_values.append(mse_val)

# Plot training and validation errors
plt.plot(M_values, mse_train_values, label='Training Error')
plt.plot(M_values, mse_val_values, label='Validation Error')
plt.xlabel('Grado del polinomio(M)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Error de Train y Val vs. Grado del polinomio')
plt.legend()
plt.show()

# Find the optimal value of M
best_M = M_values[np.argmin(mse_val_values)]
print(f'The best value of M is: {best_M}')