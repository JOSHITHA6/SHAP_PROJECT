# utils/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(df, target):
    df = df.copy()

    # ---------------- HANDLE MISSING VALUES ---------------- #
    df = df.dropna()

    # ---------------- SEPARATE FEATURES & TARGET ---------------- #
    X = df.drop(columns=[target])
    y = df[target]

    # ---------------- ENCODE CATEGORICAL FEATURES ---------------- #
    X = pd.get_dummies(X)

    # ---------------- ENCODE TARGET (if classification) ---------------- #
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # ---------------- SCALING ---------------- #
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler