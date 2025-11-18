import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

def prepare_tune_split(
    source_path: str = '../data/creditcard.csv',
    tune_out_path: str = '../data/tune_data.csv',
    base_out_path: str = '../data/base_data.csv',
    seed: int = 42,
    times: int = 5,   # how many normals per fraud in tune set
) -> dict:
    """
    Create:
    - tune_data.csv: all fraud + `times` * n_fraud normals (or fewer if not enough)
    - base_data.csv: the rest of the data (here: all remaining normal transactions)

    base_data.csv will be used for clustering and AE training.
    tune_data.csv will be used ONLY for threshold tuning.
    """

    # Resolve paths relative to this file
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, source_path)
    tune_path = os.path.join(dir_path, tune_out_path)
    base_path = os.path.join(dir_path, base_out_path)

    # Load full dataset
    df = pd.read_csv(data_path)
    if 'Class' not in df.columns:
        raise ValueError("Expected a 'Class' column in the dataset")

    # Partition by class
    df_normal = df[df['Class'] == 0].copy()
    df_fraud  = df[df['Class'] == 1].copy()

    n_fraud = len(df_fraud)
    if n_fraud == 0:
        raise ValueError("No fraudulent samples (Class==1) found in dataset")

    # --- Build tune set: all fraud + times * fraud normals (sampled) ---
    n_norm_for_tune = min(len(df_normal), n_fraud * times)
    df_normal_tune = df_normal.sample(n=n_norm_for_tune, random_state=seed, replace=False)

    # Indices in the ORIGINAL df that will be in tune_df
    used_idx = df_fraud.index.union(df_normal_tune.index)

    # Build tune_df directly from df using those indices, then shuffle
    tune_df = df.loc[used_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # --- Base set: everything not in tune_df ---
    base_df = df.drop(index=used_idx).reset_index(drop=True)

    # Sanity: base_df should contain only normal (Class == 0)
    assert (base_df['Class'] == 0).all(), "base_df should contain only normal samples"

    # Save CSVs
    tune_df.to_csv(tune_path, index=False)
    base_df.to_csv(base_path, index=False)

    summary = {
        "paths": {
            "tune": tune_path,
            "base": base_path,
        },
        "counts": {
            "total": int(len(df)),
            "normal_total": int(len(df_normal)),
            "fraud_total": int(n_fraud),
            "tune_normal": int((tune_df['Class'] == 0).sum()),
            "tune_fraud": int((tune_df['Class'] == 1).sum()),
            "base_normal": int((base_df['Class'] == 0).sum()),
        },
    }

    print("Splits created and saved to:")
    print(f"  tune -> {tune_path}  (normal={summary['counts']['tune_normal']}, fraud={summary['counts']['tune_fraud']})")
    print(f"  base -> {base_path}  (normal={summary['counts']['base_normal']})")

    return summary



# Raw data loaders
def load_base_data(path: str = '../data/base_data.csv') -> pd.DataFrame:
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, path)
    return pd.read_csv(data_path)

def load_credit_card_data(path='../data/creditcard.csv') -> pd.DataFrame:
    """
    Load the credit card transaction data from a CSV file.
    Converts the 'Class' column to int8 and columns V1 to V28 to float16 to optimize memory usage.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir_path, path)
    df = pd.read_csv(path)
    df['Class'] = df['Class'].astype('int8')
    for col in df.columns[1:29]:
        df[col] = df[col].astype('float16')
    return df

def load_credit_card_data_normal(path='../data/creditcard.csv') -> pd.DataFrame:
    """
    Load the credit card transaction data (without the fraudulent transactions) from a CSV file.
    Converts the 'Class' column to int8 and columns V1 to V28 to float16 to optimize memory usage.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir_path, path)
    df = pd.read_csv(path)
    df = df[df['Class'] == 0]  # Keep only normal transactions
    df['Class'] = df['Class'].astype('int8')
    for col in df.columns[1:29]:
        df[col] = df[col].astype('float16')
    return df

def load_credit_card_data_fraud(path='../data/creditcard.csv') -> pd.DataFrame:
    """
    Load the credit card transaction data (only the fraudulent transactions) from a CSV file.
    Converts the 'Class' column to int8 and columns V1 to V28 to float16 to optimize memory usage.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir_path, path)
    df = pd.read_csv(path)
    df = df[df['Class'] == 1]  # Keep only fraudulent transactions
    df['Class'] = df['Class'].astype('int8')
    for col in df.columns[1:29]:
        df[col] = df[col].astype('float16')
    return df

# Scaler stuff

def fit_and_save_scaler(
    path_to_data: str = '../data/base_data.csv',
    path_to_save: str = '../data/scaler.joblib'
):
    """
    Fit a StandardScaler on NORMAL ONLY data from base_data.csv and save it.
    """

    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, path_to_data)
    save_path = os.path.join(dir_path, path_to_save)

    df = pd.read_csv(data_path)
    if 'Class' not in df.columns:
        raise ValueError("Expected a 'Class' column in the dataset")

    # base_data.csv should already be only normals, but we can be safe:
    df_norm = df[df['Class'] == 0]
    X = df_norm.drop(columns=['Class'])

    scaler = StandardScaler().fit(X)
    dump(scaler, save_path)
    print(f"Scaler fitted on base normal data and saved to {save_path}")

    return scaler

def load_scaler(path: str = '../data/scaler.joblib') -> StandardScaler:
    dir_path = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(dir_path, path))