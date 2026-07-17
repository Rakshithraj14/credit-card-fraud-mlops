# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/creditcard.csv"
RANDOM_STATE = 42

def load_data(path: str = DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y

def split_data(X, y, test_size: float = 0.2):
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )