import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pathlib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path
#from cluster import elbow_method, plot_elbow_method



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





#df = pd.read_csv('./data/creditcard.csv').drop(columns=['Class'])
#labels, centroids = kmeans_implementation(df, k=3, random_state=42)
#plot_clusters(df, labels)

#k_values = list(range(1, 11))
#plot_elbow_method(k_values, elbow_method(df, k_values))

df = pd.read_csv('./data/creditcard.csv').drop(columns=['Class'])
labels, centroids = kmeans_implementation(df, k=8, random_state=42)
#plot_clusters(df, labels)

pca = PCA(n_components=2)
reduced = pca.fit_transform(df)

plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels)
plt.title("K-Means Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster ID")
plt.show()

#print(f"Cluster Sizes: {np.bincount(labels) / len(labels)}")




