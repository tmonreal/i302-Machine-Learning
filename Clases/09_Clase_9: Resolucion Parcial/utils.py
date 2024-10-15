import numpy as np
import matplotlib.pyplot as plt

# Definir la función de costo
def cost_function(X, y, w):
    return np.mean((y - X @ w) ** 2)

# Implementar el gradiente descendente
def gradient_descent(X, y, w_init, learning_rate, n_iterations):
    w = w_init
    costs = []
    w_history = []  # Historial de pesos para graficar

    for _ in range(n_iterations):
        # Calcular el gradiente
        gradient = -2 * (X.T @ (y - X @ w)) / len(y)
        w = w - learning_rate * gradient
        costs.append(cost_function(X, y, w))
        w_history.append(w.copy())  # Guardar el historial de pesos
   
    return w, costs, np.array(w_history)

# Función para graficar el gradiente descendente
def plot_gradient_descent(X, y, w_histories, title):
    w0_range = np.linspace(-10, 10, 100)
    w1_range = np.linspace(-10, 10, 100)
    W0, W1 = np.meshgrid(w0_range, w1_range)

    # Determinar la dimensión de W
    num_weights = X.shape[1]
   
    # Calcular la función de costo para la malla de pesos
    Z = np.zeros(W0.shape)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            if num_weights == 2:
                Z[i, j] = cost_function(X, y, np.array([W0[i, j], W1[i, j]]))
            elif num_weights == 3:
                Z[i, j] = cost_function(X, y, np.array([W0[i, j], W1[i, j], 0]))  # El tercer peso se inicializa a 0


    plt.figure(figsize=(10, 7))
   
    # Plotear el contorno primero
    contour = plt.contour(W0, W1, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Función de Costo')

    # Graficar todas las trayectorias de gradiente descendente
    for w_history in w_histories:
        plt.plot(w_history[:, 0], w_history[:, 1], marker='o', alpha=0.01, markersize= 3)

    # Resaltar el último punto de cada trayectoria (cruces en negro)
    for w_history in w_histories:
        plt.scatter(w_history[-1, 0], w_history[-1, 1], s=100, marker='x', label='Mínimo encontrado')

    plt.title(title)
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.legend()
    plt.grid(True)
    plt.show()