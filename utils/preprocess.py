import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    original_columns = X.columns.tolist()
    
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    transformers = [('num', numeric_transformer, numeric_cols)]
    
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if y.nunique() <= 10 else None
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_test_original = X_test.copy()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = numeric_cols.copy()
    
    if categorical_cols:
        try:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features = ohe.get_feature_names_out(categorical_cols)
            feature_names.extend(list(cat_features))
        except:
            for cat_col in categorical_cols:
                unique_vals = X[cat_col].dropna().unique()[:10]
                for val in unique_vals:
                    feature_names.append(f"{cat_col}_{val}")
    
    if len(feature_names) != X_train_processed.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
    
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    
    return (X_train_processed, X_test_processed, y_train, y_test, X_test_original,
            preprocessor, feature_names, original_columns, df)