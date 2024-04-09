import numpy as np
import matplotlib.pyplot as plt
import pickle

# Cargar los datos del archivo toy_dataset.pkl
with open('/home/linar/Desktop/ML/Clases/Clase 3/i302/TP2/datasets/toy_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['taret']
y = data['input_feature']

# Dividir los datos en conjuntos de entrenamiento y validación
# Aquí podemos usar una división simple, pero es recomendable usar validación cruzada para resultados más robustos
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Definir una función para entrenar y evaluar el modelo para un valor dado de M
def train_and_evaluate(X_train, y_train, X_val, y_val, M):
    # Construir matriz de características polinomiales
    X_train_poly = np.column_stack([X_train ** i for i in range(M+1)])
    X_val_poly = np.column_stack([X_val ** i for i in range(M+1)])
    
    # Calcular los parámetros del modelo usando mínimos cuadrados
    w = np.linalg.inv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train
    
    # Calcular las predicciones en el conjunto de entrenamiento y validación
    y_train_pred = X_train_poly @ w
    y_val_pred = X_val_poly @ w
    
    # Calcular el error cuadrático medio (MSE)
    mse_train = np.mean((y_train - y_train_pred) ** 2)
    mse_val = np.mean((y_val - y_val_pred) ** 2)
    
    return mse_train, mse_val

# Definir una lista de valores de M para probar
M_values = range(1, 15)

# Lista para almacenar los errores de entrenamiento y validación
mse_train_values = []
mse_val_values = []

# Entrenar y evaluar el modelo para cada valor de M
for M in M_values:
    mse_train, mse_val = train_and_evaluate(X_train, y_train, X_val, y_val, M)
    mse_train_values.append(mse_train)
    mse_val_values.append(mse_val)

# Graficar las curvas de error de entrenamiento y validación
plt.plot(M_values, mse_train_values, label='Training Error')
plt.plot(M_values, mse_val_values, label='Validation Error')
plt.xlabel('Degree of Polynomial (M)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training and Validation Error vs. Polynomial Degree')
plt.legend()
plt.show()

# Encontrar el valor óptimo de M
best_M = M_values[np.argmin(mse_val_values)]
print(f'El mejor valor de M es: {best_M}')