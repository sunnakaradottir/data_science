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




def centroid(cluster_df):
    """Compute the centroid (mean) of a cluster DataFrame."""
    return cluster_df.mean(axis=0)

def kmeans2(points_df, k=2, max_iters=100, tol=1e-4, random_state=None):
  
    if random_state is not None:
        np.random.seed(random_state)

    n = len(points_df)
    feature_names = points_df.columns

    # Step 0: Initialization — pick k random points as initial centroids
    idx = np.random.choice(n, size=k, replace=False)
    representatives = [points_df.iloc[i] for i in idx]

    # Initialize cluster containers
    clusters = [pd.DataFrame(columns=feature_names) for _ in range(k)]
    labels = np.zeros(n, dtype=int)

    for iteration in range(max_iters):
        # Reset clusters
        clusters = [pd.DataFrame(columns=feature_names) for _ in range(k)]

        # Step 1: Assign each point to nearest cluster
        for i, (_, p) in enumerate(points_df.iterrows()):
            # Compute distances to each cluster centroid
            distances = []
            for c in range(k):
                if len(clusters[c]) == 0:
                    center = representatives[c]
                else:
                    center = centroid(clusters[c])
                d = np.sqrt(((p - center) ** 2).sum())
                distances.append(d)

            # Assign to nearest cluster
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx] = pd.concat([clusters[cluster_idx], p.to_frame().T], ignore_index=True)
            labels[i] = cluster_idx

        # Step 2: Update representatives (centroids)
        new_representatives = []
        for c in range(k):
            if len(clusters[c]) > 0:
                new_representatives.append(centroid(clusters[c]))
            else:
                # Handle empty cluster by picking a random point
                new_representatives.append(points_df.sample(1, random_state=random_state).iloc[0])

        # Step 3: Check for convergence
        shift = sum(np.linalg.norm(new_representatives[i] - representatives[i]) for i in range(k))
        if shift < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

        representatives = new_representatives

    # Combine final centroids into a DataFrame
    centroids_df = pd.DataFrame(representatives, columns=feature_names)

    labels_series = pd.Series(labels, name='Cluster')

    return labels_series, centroids_df, clusters

df = pd.read_csv('./data/creditcard.csv').drop(columns=['Class'])
labels, centroids, clusters = kmeans2(df, k=3)
plot_clusters(df, labels)


#print(f"Cluster Sizes: {np.bincount(labels) / len(labels)}")




