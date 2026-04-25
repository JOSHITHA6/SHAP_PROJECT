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

    feature_names = preprocessor.get_feature_names_out()

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