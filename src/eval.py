import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_reconstruction_errors(model, X):
    """Pass X through AutoEncoder, return MSE per sample."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(X_tensor)['x_hat']
        errors = F.mse_loss(reconstructed, X_tensor, reduction='none').mean(dim=1)
    return errors.cpu().numpy()


def find_optimal_threshold(errors, y_true, n_thresholds=200):
    """Try many thresholds, return the one with best F1. Returns (threshold, f1, results_df)."""
    thresholds = np.linspace(errors.min(), errors.max(), n_thresholds)
    best_f1, best_threshold = 0, 0
    all_results = []

    for threshold in thresholds:
        preds = (errors > threshold).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        all_results.append({'threshold': threshold, 'precision': precision, 'recall': recall, 'f1': f1})
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    return best_threshold, best_f1, pd.DataFrame(all_results)


def calculate_metrics(y_true, y_pred):
    """Return dict with accuracy, precision, recall, f1."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
