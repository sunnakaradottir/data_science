from src.data import load_credit_card_data, load_credit_card_data_normal, load_credit_card_data_fraud, load_base_data, fit_and_save_scaler, load_scaler, prepare_tune_split
from src.cluster import elbow_method, plot_elbow_method, perform_kmeans_clustering, search_k, visualize_clusters_pca, plot_search_k_result, get_avg_per_cluster, cluster_enricher
from src.train import run
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def prep() -> tuple[np.ndarray, np.ndarray]:
    # 1) Get normal only transactions
    df = load_base_data()
    # 2) Drop the class column (label)
    X = df.drop(columns=['Class'])
    # 3) load the pre-fitted scaler
    scaler = load_scaler()
    # 4) Scale!
    X_scaled = scaler.transform(X)
    # 5) Perform KMeans clustering to get cluster labels
    km = perform_kmeans_clustering(X_scaled, n_clusters=3)
    labels = km.labels_
    # 6) Return scaled data and labels
    return X_scaled, labels

def main_setup():
    """
    Run once (or whenever we need to tbh) to:
    - create base and tune splits
    - fit scaler on all normal transactions (minus ones in tune set ca. 2500 samples)
    """
    prepare_tune_split(times=5)
    fit_and_save_scaler(path_to_data='../data/base_data.csv')
    # fit clustering and save kmeans model
    


def main_single_cluster():
    X, labels = prep()
    run(X, labels, cluster_id=2, sweep=True)


def main():
    # Uncomment to run setup (create splits and fit scaler)
    # main_setup()

   main_single_cluster()


if __name__ == "__main__":
    main()