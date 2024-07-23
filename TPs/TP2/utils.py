import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    X, y = data["input_feature"], data["taret"]  # Corrected the keys here
    return X, y


def get_best_coef(X, Y, M):
    """
    Find the best weigths for X and Y
    """
    A = np.zeros((M, M)) # matriz de coeficientes
    S = np.zeros(2*M)
    b = np.zeros(M) # vector de resultados
    for i in range(len(X)):
        aux = Y[i]
        for j in range(M):
            b[j] = b[j]+aux
            aux = aux*X[i]
        aux = 1.0
        for j in range(2*M):
            S[j] = S[j]+aux
            aux = aux*X[i]
    for i in range(M):
        for j in range(M):
            A[i,j] = S[i+j]
            
    w = np.linalg.solve(A, b)
    assert np.allclose(np.dot(A, w), b)
    return w

def model_predict(w, X):
    """
    Predictions using polynomial with w params
    """
    poly = np.polynomial.polynomial.Polynomial(w)   
    return poly(X)