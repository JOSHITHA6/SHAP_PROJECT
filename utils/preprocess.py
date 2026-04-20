import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target):

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    # Convert back to DataFrame (VERY IMPORTANT for SHAP)
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns
    )

    return X_train, X_test, y_train, y_test