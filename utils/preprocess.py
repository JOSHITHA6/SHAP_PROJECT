import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=['int64','float64']).columns
    cat_cols = X.select_dtypes(include=['object','category']).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    raw_names = preprocessor.get_feature_names_out()

    # CLEAN NAMES
    feature_names = []
    for name in raw_names:
        if "__" in name:
            name = name.split("__")[1]

        if "_" in name and name.split("_")[-1].isalpha():
            parts = name.split("_")
            feature_names.append(f"{parts[0]} = {parts[-1]}")
        else:
            feature_names.append(name)

    X_test_original = X.iloc[:len(X_test)]

    return (
        X_train, X_test,
        y_train, y_test,
        X_test_original,
        preprocessor,
        feature_names,
        X.columns,
        df
    )