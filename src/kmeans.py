import numpy as np
import random

# Our implementation of the K-means clustering algorithm.
def kmeans_implementation(X, k, max_iters=300, tol=0.0001, random_state=42) -> tuple[np.ndarray, np.ndarray]:
    """
    A simple implementation of the K-means clustering algorithm.
    Parameters:
    - X: Input data, scaled numpy array of shape (n_samples, n_features).
    - k: Number of clusters.
    - max_iters: Maximum number of iterations.
    - tol: Tolerance to declare convergence.
    - random_state: Seed for reproducibility.
    """

    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    data = np.asarray(X)
    n_samples, n_features = data.shape

    # Randomly choose initial centroids
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[indices, :]

    for iteration in range(max_iters):
        # Assign points to nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            points = data[labels == i]
            if len(points) > 0:
                new_centroids[i] = points.mean(axis=0)
            else:
                # Reinitialize empty cluster randomly
                new_centroids[i] = data[np.random.randint(0, n_samples)]

        # Check for convergence
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            print(f"Converged after {iteration+1} iterations.")
            break

        centroids = new_centroids

    return labels, centroids
class OurKmeans:
    def __init__(self, centroids: np.ndarray):
        self.centroids = centroids  # shape: (k, n_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each row in X to the nearest centroid."""
        X = np.asarray(X)
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)
