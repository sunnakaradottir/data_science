import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



def kmeans_implementation(df, k, max_iters=300, tol=0.0001, random_state=None):
    """
    Our implementation of the K-means clustering algorithm.
    """

    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    data = df.to_numpy()
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

    # Convert results back to pandas objects
    centroids_df = pd.DataFrame(centroids, columns=df.columns)
    labels_series = pd.Series(labels, name="Cluster")

    return labels_series, centroids



# Test:

def plot_clusters(df, labels):

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels)
    plt.title("K-Means Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster ID")
    plt.show()




def elbow_method(data, k_values=range(1, 11)) -> list[int]:
    """
    Apply the elbow method to determine the optimal number of clusters for KMeans.
    Parameters:
    - data: The input data for clustering.
    - k_values: A list of integers representing the number of clusters to try.

    Returns:
    - A list of inertia values corresponding to each k in k_values.
    """
    inertia_values = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42) # set random_state for reproducibility
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_) # inertia is the sum of squared distances to nearest cluster center

    return inertia_values

def plot_elbow_method(k_values: list[int], inertia_values: list[int]) -> None:
    """
    Plot the results of the elbow method.
    Parameters:
    - k_values: A list of integers representing the number of clusters.
    - inertia_values: A list of inertia values corresponding to each k in k_values.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_values, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()



df = pd.read_csv('./data/creditcard.csv').drop(columns=['Class'])
labels, centroids = kmeans_implementation(df, k=3, random_state=42)
#plot_clusters(df, labels)

k_values = list(range(1, 11))
plot_elbow_method(k_values, elbow_method(df, k_values))


#print(f"Cluster Sizes: {np.bincount(labels) / len(labels)}")




