import os
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Tuple, Union
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import MiniBatchKMeans
from collections import namedtuple
KMeansType = Union[KMeans, MiniBatchKMeans]
KMeansSearchResult = namedtuple("KMeansSearchResult", ["best_k", "best_metric", "metric_name", "scores"])
from src.data import load_base_data, load_scaler
from joblib import dump, load
from src.kmeans import kmeans_implementation, OurKmeans





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

def perform_kmeans_clustering(data, n_clusters: int) -> OurKmeans:
    """
    Perform KMeans clustering on the given data.
    Parameters:
    - data: The input data for clustering.
    - n_clusters: The number of clusters to form.

    Returns:
    - The fitted KMeans model.
    """
    _, centroids = kmeans_implementation(data, n_clusters)
    kmeans = OurKmeans(centroids=centroids)
    return kmeans

def fit_k_means_and_save(
    n_clusters: int,
    kmeans_out_path: str = '../data/kmeans.joblib',
):
    """
    Fit KMeans on the (scaled) base data and save the model to disk.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))

    # 1) Load base data (only normals)
    df = load_base_data()
    X = df.drop(columns=["Class"]).values

    # 2) Load scaler and scale
    scaler = load_scaler()
    X_scaled = scaler.transform(X)

    # 3) Fit KMeans once
    kmeans = perform_kmeans_clustering(X_scaled, n_clusters=n_clusters)

    # print size of each cluster as percentage and save as text file
    labels = kmeans.predict(X_scaled)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    print("Cluster distribution:")
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c} samples ({(c/total)*100:.2f}%)")
    cluster_dist_path = os.path.join(dir_path, '../data/cluster_distribution.txt')
    with open(cluster_dist_path, 'w') as f:
        f.write("Cluster distribution:\n")
        for u, c in zip(unique, counts):
            f.write(f"Cluster {u}: {c} samples ({(c/total)*100:.2f}%)\n")
    print(f"Cluster distribution saved to {cluster_dist_path}")

    # 4) Save model to disk
    kmeans_path = os.path.join(dir_path, kmeans_out_path)
    dump(kmeans, kmeans_path)
    print(f"KMeans model saved to {kmeans_path}")
    return kmeans  # optional, but handy


def load_kmeans_model(kmeans_path: str = '../data/kmeans.joblib') -> KMeans:
    """
    Load a saved KMeans model from disk.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(dir_path, kmeans_path)
    kmeans = load(full_path)
    return kmeans


def search_k(
    X: np.ndarray,
    k_range: range = range(3, 11),
    random_state: int = 42,
    n_init: Union[int, str] = "auto",
    batch_size: int = 4096,
    max_iter: int = 300,
) -> KMeansSearchResult:
    """
    Try a range of k and compute inertia & silhouette for each.
    Returns the k that maximizes silhouette (primary) with all scores for plotting.

    Notes:
    - Silhouette needs k>=2 and < n_samples; caller should ensure k_range is valid.
    - X is assumed already scaled.
    -  silhouette_score = (b - a) / max(a, b)
      where a = mean intra-cluster distance, b = mean nearest-cluster distance.
    """
    rows: List[Tuple[int, float, float]] = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter)

        labels = km.fit_predict(X)
        inertia = float(km.inertia_)
        # For very imbalanced or large datasets, silhouette on a sample keeps this fast.
        # Here: sample up to 20k points for silhouette.
        if X.shape[0] > 20000:
            idx = np.random.RandomState(0).choice(X.shape[0], size=20000, replace=False)
            sil = float(silhouette_score(X[idx], labels[idx]))
        else:
            sil = float(silhouette_score(X, labels)) # look for silhouette score near maximum

        db = davies_bouldin_score(X, labels) # look for Davies-Bouldin index low
        ch = calinski_harabasz_score(X, labels) # look for Calinski-Harabasz index stable or high
        rows.append((k, inertia, sil, db, ch))

    scores = pd.DataFrame(rows, columns=["k", "inertia", "silhouette", "davies_bouldin", "calinski_harabasz"])
    # Choose k by maximum silhouette; break ties by lower inertia.
    best_row = scores.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]
    return KMeansSearchResult(
        best_k=int(best_row.k),
        best_metric=float(best_row.silhouette),
        metric_name="silhouette",
        scores=scores,
    )

def cluster_enricher(df: pd.DataFrame, X_scaled: np.ndarray, kmeans: KMeans) -> pd.DataFrame:
    """
    Add cluster assignments and distance-to-centroid features to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame (rows must align with X_scaled).
    kmeans : sklearn.cluster.KMeans
        Fitted KMeans model.
    X_scaled : np.ndarray
        Scaled feature matrix used for clustering.
    Returns
    -------
    pd.DataFrame
        Copy of df with two new columns added.
    """
    # Predict cluster assignments
    labels = kmeans.predict(X_scaled)

    # Compute distances to respective centroids
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[labels], axis=1)

    # Add new columns to DataFrame
    df_out = df.copy()
    df_out['cluster_id'] = labels
    df_out['dist_to_centroid'] = distances
    return df_out

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def get_avg_per_cluster(df: pd.DataFrame) -> None:
    """
    Prints the average of 'Amount', 'Time', 'V1', 'V2', and 'V12' per cluster.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the clustered data with a 'cluster_id' column.
    """
    cluster_means = df.groupby('cluster_id')[['Amount', 'Time', 'V1', 'V2', 'V12']].mean()

    print(cluster_means)
    
def visualize_clusters_pca(X_scaled: np.ndarray, labels: np.ndarray, n_components: int = 2) -> None:
    """
    Perform PCA for visualization and plot clusters in 2D.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix (e.g., StandardScaler.transform output).
    labels : np.ndarray
        Cluster labels for each sample.
    n_components : int, optional
        Number of PCA components for projection (default is 2).
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=2)
    plt.title("PCA projection colored by cluster")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.show()


def plot_search_k_result(result) -> None:
    df = result.scores

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # 1) Silhouette
    axes[0].plot(df["k"], df["silhouette"], marker="o")
    axes[0].set_title("Silhouette (↑ better)")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("score")

    # 2) Davies-Bouldin
    axes[1].plot(df["k"], df["davies_bouldin"], marker="o")
    axes[1].set_title("Davies-Bouldin (↓ better)")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("score")

    # 3) Calinski-Harabasz
    axes[2].plot(df["k"], df["calinski_harabasz"], marker="o")
    axes[2].set_title("Calinski-Harabasz (↑ better)")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("score")

    plt.suptitle(f"Best k by {result.metric_name}: {result.best_k} (score={result.best_metric:.3f})", y=1.05)
    plt.tight_layout()
    plt.show()

