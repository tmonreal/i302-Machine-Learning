import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        # Initialize centroids randomly from the data points
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X)

            # Update centroids
            new_centroids = self._calculate_centroids(X, labels)

            # Check for convergence (if centroids do not change)
            if np.all(new_centroids == self.centroids):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        # Calculate the distance from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        # Assign each point to the nearest centroid
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, X, labels):
        # Calculate new centroids as the mean of all points assigned to each centroid
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

    def predict(self, X):
        return self._assign_clusters(X)

# Function to calculate the Within-Cluster Sum of Squares (WCSS) for a given number of clusters
def calculate_wcss(X, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    centroids = kmeans.centroids
    labels = kmeans.predict(X)
    wcss = np.sum((X - centroids[labels]) ** 2)
    return wcss

# Function to plot the elbow method
def plot_elbow_method(X, max_clusters=10):
    wcss_values = [calculate_wcss(X, k) for k in range(1, max_clusters + 1)]
    plt.plot(range(1, max_clusters + 1), wcss_values, marker='o', linestyle='-')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares')  
    plt.show()