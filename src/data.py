import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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

def scale_data(df: pd.DataFrame) -> np.ndarray:
    """
    Scale the features of the credit card transaction data using StandardScaler.
    It expects the input DataFrame to not include the 'Class' column.
    """
    from sklearn.preprocessing import StandardScaler


    scaler = StandardScaler().fit(df)
    scaled_features = scaler.transform(df)

    return scaled_features

