from src.data import load_credit_card_data, load_credit_card_data_normal, load_credit_card_data_fraud, scale_data
from src.cluster import elbow_method, plot_elbow_method, perform_kmeans_clustering, search_k, visualize_clusters_pca, plot_search_k_result, get_avg_per_cluster, cluster_enricher
from src.train import run
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np



def main():
    # Load the credit card transaction data
    df = load_credit_card_data_normal()

    # Select features for clustering (excluding the 'Class' column)
    X = df.drop(columns=['Class'])

    # Scale the data
    # k means is distance based, so large differences in scale will affect the results
    X = scale_data(X)

    # sample a subset for faster experimentation e.g. 10%
    np.random.seed(42)
    sample_indices = np.random.choice(X.shape[0], size=int(0.1 * X.shape[0]), replace=False)
    X = X[sample_indices]

    # best k to compare with elbow method
    # search_result = search_k(X, k_range=range(2, 11)) # NOTE: this will tell us the max of k_range is usually best (heell naw)
    
    # print("Searching for the best k using search_k function...")
    # plot_search_k_result(search_result) #NOTE: plotting silhouette vs davies bouldin vs calinski harabasz will give us k=4 or k=8 (perchance)
    
    ks = [3,4,8]
    kmeans_list = [perform_kmeans_clustering(X, n_clusters=k) for k in ks] 
    for k, kmeans in zip(ks, kmeans_list):
        print("k=" + str(k))
        print(f"Cluster Sizes: {np.bincount(kmeans.labels_) / len(kmeans.labels_)}")    

    # Choose k=3 from kmeans_list
    kmeans = kmeans_list[0]
    labels = kmeans.labels_

    run(X, labels)

if __name__ == "__main__":
    main()