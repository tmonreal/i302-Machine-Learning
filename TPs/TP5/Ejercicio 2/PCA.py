import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X):
        # Compute the mean of the data
        self.mean = np.mean(X, axis=0).values  # Convert mean to NumPy array
        # Center the data
        X_centered = X - self.mean
        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        # Compute the eigenvalues and eigenvectors of the covariance matrix
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(cov_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvectors = self.eigenvectors[:, idx]
        # Select the top n_components eigenvectors as principal components
        self.components = self.eigenvectors[:, :self.n_components]


    def transform(self, X):
        # Project the data onto the new feature space defined by the principal components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def reconstruct(self, X_transformed):
        # Reconstruct the original data from the reduced feature space
        return np.dot(X_transformed, self.components.T) + self.mean
    
    def inverse_transform(self, X_transformed):
        # Reconstruct the original data from the reduced feature space
        return np.dot(X_transformed, self.components.T) + self.mean
