import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_data(N=10, noise=0.1, X_interval=(0,1), gt_fn=None, seed=42):
    """
    Generates synthetic data based on a ground truth function with added noise.

    Parameters:
    N (int): Number of data points to generate. Default is 10.
    noise (float): Standard deviation of the Gaussian noise to add to the data. Default is 0.1.
    X_interval (tuple): Interval (min, max) from which to draw the input data points. Default is (0, 1).
    gt_fn (callable): Ground truth function that takes an array of inputs and returns the corresponding outputs.
                      This function must be provided by the user.
    seed (int): Seed for the random number generator to ensure reproducibility. Default is 42.

    Returns:
    tuple: A tuple containing:
        - X (ndarray): The generated input data points.
        - Y (ndarray): The noisy output data points.
        - Y_true (ndarray): The true output data points (without noise).

    Raises:
    ValueError: If gt_fn is not provided.
    """
    generator = np.random.RandomState(seed=42)
    if not gt_fn:
        raise ValueError("You must define a ground truth function")
    X = generator.uniform(*X_interval, size=N)
    Y_true = gt_fn(X)
    Y = Y_true + 0.1*generator.standard_normal(Y_true.shape)
    return X, Y, Y_true


def get_best_coef(X, Y, M):
    """
    Finds the best coefficients for fitting a polynomial of degree M-1 to the data.

    This function calculates the coefficients for a polynomial that best fits the given data points (X, Y)
    by solving a system of linear equations derived from the polynomial basis functions.

    Parameters:
    X (array-like): The input data points.
    Y (array-like): The output data points corresponding to X.
    M (int): The number of coefficients to find, which corresponds to the degree of the polynomial plus one.

    Returns:
    ndarray: An array of the best-fit polynomial coefficients of length M.

    Raises:
    LinAlgError: If the linear system cannot be solved (e.g., if the matrix is singular).
    
    Notes:
    - This method uses a least squares approach to find the best fit polynomial coefficients by constructing
      and solving the normal equations.
    - The function assumes that the lengths of X and Y are the same and that M is less than or equal to the number of data points.
    """
    A = np.zeros((M, M)) # matriz de coeficientes
    S = np.zeros(2*M)  # sumas de potencias de X
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