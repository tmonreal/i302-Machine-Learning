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

def get_best_coef(X, Y, M, basis_fn=None):
    if basis_fn is None:
        # Default to polynomial basis if none provided
        # def basis_fn(x, degree): return x**degree
        basis_fn = lambda x, degree: x**degree

    # Construct the design matrix using the basis function
    PHI = np.zeros((len(X), M))
    for i in range(M):
        PHI[:, i] = basis_fn(X, i)
    
    # Solve for the coefficients using the normal equation
    PHI_T_PHI = np.dot(PHI.T, PHI)
    PHI_T_Y = np.dot(PHI.T, Y)
    PHI_T_PHI_inv = np.linalg.inv(PHI_T_PHI)  
    w = np.dot(PHI_T_PHI_inv, PHI_T_Y)
    
    return w

def model_predict(w, X, basis_fn=None):
    if basis_fn is None:
        basis_fn = lambda x, degree: x**degree
    
    PHI = np.zeros((len(X), len(w)))
    for i in range(len(w)):
        PHI[:, i] = basis_fn(X, i)
    
    return np.dot(PHI, w)

def get_best_coef_reg(X, Y, M, basis_fn=None):
    if basis_fn is None:
        # Default to polynomial basis if none provided
        basis_fn = lambda x, degree: x**degree

    # Construct the design matrix using the basis function
    PHI = np.zeros((len(X), M))
    for i in range(M):
        PHI[:, i] = basis_fn(X, i)
    
    # Solve for the coefficients using the normal equation
    PHI_T_PHI = np.dot(PHI.T, PHI) + np.eye(PHI.shape[1]) * 1e-8
    PHI_T_Y = np.dot(PHI.T, Y)
    PHI_T_PHI_inv = np.linalg.inv(PHI_T_PHI)  
    w = np.dot(PHI_T_PHI_inv, PHI_T_Y)
    
    return w