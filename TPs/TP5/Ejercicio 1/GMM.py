import numpy as np
import matplotlib.pyplot as plt

class GMM:
    def __init__(self, n_components=1, max_iter=100, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.means = None
        self.covariances = None
        self.priors = None
        self.log_likelihoods = None

    def _initialize_parameters(self, X):
        np.random.seed(self.random_state)
        n_samples, _ = X.shape
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_indices]
        self.covariances = [np.cov(X.T)] * self.n_components
        self.priors = np.ones(self.n_components) / self.n_components

    def _compute_likelihoods(self, X):
        likelihoods = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            diff = X - self.means[k]
            exponent_term = np.exp(-0.5 * np.sum(np.dot(diff, np.linalg.inv(self.covariances[k])) * diff, axis=1))
            likelihoods[:, k] = exponent_term / np.sqrt(np.linalg.det(self.covariances[k]))
        return likelihoods

    def _compute_posteriors(self, X):
        likelihoods = self._compute_likelihoods(X)
        weighted_likelihoods = likelihoods * self.priors
        posteriors = weighted_likelihoods / np.sum(weighted_likelihoods, axis=1, keepdims=True)
        return posteriors

    def _update_parameters(self, X, posteriors):
        self.means = np.dot(posteriors.T, X) / np.sum(posteriors, axis=0)[:, np.newaxis]
        self.covariances = []
        self.priors = np.mean(posteriors, axis=0)
        for k in range(self.n_components):
            diff = X - self.means[k]
            covariance = np.dot((diff * posteriors[:, k][:, np.newaxis]).T, diff) / np.sum(posteriors[:, k])
            self.covariances.append(covariance)
        self.covariances = np.array(self.covariances)

    def _compute_log_likelihood(self, X):
        likelihoods = self._compute_likelihoods(X)
        return np.sum(np.log(np.sum(likelihoods * self.priors, axis=1)))

    def fit(self, X):
        self._initialize_parameters(X)
        prev_log_likelihood = -np.inf
        self.log_likelihoods = []
        for _ in range(self.max_iter):
            posteriors = self._compute_posteriors(X)
            self._update_parameters(X, posteriors)
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihoods.append(log_likelihood)
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

    def predict(self, X):
        posteriors = self._compute_posteriors(X)
        return np.argmax(posteriors, axis=1)
    
def plot_clusters_gmm(X, means, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='tab20')
    plt.scatter(means[:, 0], means[:, 1], s=100, c='red', marker='X')
    plt.title(title)
    plt.xlabel('Feature A')
    plt.ylabel('Feature B')
    plt.show()