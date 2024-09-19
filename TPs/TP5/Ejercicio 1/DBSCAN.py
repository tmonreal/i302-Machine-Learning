import numpy as np
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, epsilon=1.0, min_samples=5):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.labels = None

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _find_neighbors(self, X, point_index):
        distances = np.sqrt(np.sum((X - X[point_index]) ** 2, axis=1))
        neighbors = np.where(distances <= self.epsilon)[0]
        return neighbors

    def _expand_cluster(self, X, point_index, neighbors, cluster_id):
        self.labels[point_index] = cluster_id
        i = 0
        while i < len(neighbors):
            current_point_index = neighbors[i]
            if self.labels[current_point_index] == -1:  # No asignado aún
                self.labels[current_point_index] = cluster_id
            elif self.labels[current_point_index] == 0:  # Punto frontera
                self.labels[current_point_index] = cluster_id
                current_point_neighbors = self._find_neighbors(X, current_point_index)
                if len(current_point_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, current_point_neighbors))
            i += 1

    def fit_predict(self, X):
        self.labels = np.zeros(X.shape[0], dtype=int)  # 0 para puntos no clasificados, -1 para ruido
        current_cluster_id = 0
        for i, point in enumerate(X):
            if self.labels[i] != 0:  # Ya clasificado
                continue
            neighbors = self._find_neighbors(X, i)
            if len(neighbors) < self.min_samples:  # Punto de ruido
                self.labels[i] = -1
                continue
            current_cluster_id += 1
            self._expand_cluster(X, i, neighbors, current_cluster_id)
        return self.labels
    
def plot_clusters_dbscan(X, labels, title):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))  # Colores basados en el esquema 'tab20'
    
    plt.figure(figsize=(8, 6))
    for cluster_label, color in zip(unique_labels, colors):
        cluster_points = X[labels == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], s=20, label=f'Cluster {cluster_label}')
    
    plt.title(title)
    plt.xlabel('Feature A')
    plt.ylabel('Feature B')
    plt.legend()
    plt.show()