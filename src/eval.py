import pandas as pd
from src.data import load_scaler
from src.cluster import load_kmeans_model
import torch
import numpy as np


def evaluate_thresholds(model, scaler, tune_data_path='/data/tune_data.csv', cluster_id: int = 0) -> None:
    """
    Evaluate different classification thresholds on the tune dataset to find the optimal threshold.
    """

    df = pd.read_csv(tune_data_path)

    # Separate features and labels
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values 
    
    scaler = load_scaler()
    X_scaled = scaler.transform(X)   # class labels NOT touched

    #TODO: GET ONLY DATA CORRESPONDING TO THIS CLUSTER
    # Data was scaled before training model
    kmeans_model = load_kmeans_model()
    cluster_label = kmeans_model.predict(X_scaled)

    mask = cluster_label == cluster_id
    X_cluster = X_scaled[mask]
    y = y[mask]

    # Convert to torch tensor
    X_t = torch.tensor(X_cluster, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        outputs = model(X_t)
        X_hat = outputs["x_hat"]
        errors = torch.mean((X_hat - X_t)**2, dim=1).cpu().numpy()


    thresholds = np.linspace(errors.min(), errors.max(), 50)

    results = []

    for t in thresholds:
        preds = (errors > t).astype(int)    # 1 = fraud predicted
        
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        tn = ((preds == 0) & (y == 0)).sum()

        recall = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)

        results.append((t, recall, fpr))

    # Store results as a readible file to analyze later
    results_df = pd.DataFrame(results, columns=["Threshold", "Recall", "FPR"])
    results_df.to_csv("../results/threshold_tuning_results.csv", index=False)



    # fall 1: 
    #   determine'ar threshold per cluster
    #   gefur út stats fyrir það
    #   (fjórar mism skrár, með stats um hvern cluster)

    # fall 2:
    #   lpadar modelum
    #   notar threshold
    #   tekur punkt
    #   assignar á autencoder


def final_run(thresholds=[0.5, 0.5, 0.5, 0.5], tune_data_path='../data/tune_data.csv'):

    df = pd.read_csv(tune_data_path)

    # Separate features and labels
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values 
    
    scaler = load_scaler()
    X_scaled = scaler.transform(X)   

    kmeans_model = load_kmeans_model()
    X_scaled["cluster_label"] = kmeans_model.predict(X_scaled)


    # Split X_scaled into clusters
    clusters = {}
    for cid in range(len(thresholds)):
        clusters[cid] = X_scaled[X_scaled["cluster_label"] == cid]


    # Tekur punkt og hendir í rétt módel


    # assignar módel
 