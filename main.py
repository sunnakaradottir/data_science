from src.data import load_credit_card_data, load_credit_card_data_normal, load_credit_card_data_fraud, scale_data
from src.cluster import elbow_method, plot_elbow_method, perform_kmeans_clustering, search_k, visualize_clusters_pca, plot_search_k_result, get_avg_per_cluster, cluster_enricher
from src.train import run
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def prep() -> tuple[np.ndarray, np.ndarray]:
    df = load_credit_card_data_normal()
    X = df.drop(columns=['Class'])
    X = scale_data(X)
    km = perform_kmeans_clustering(X, n_clusters=3)
    labels = km.labels_
    return X, labels

def main_single_cluster():
    X, labels = prep()
    run(X, labels, cluster_id=2, sweep=True)


def main():
   main_single_cluster()


if __name__ == "__main__":
    main()