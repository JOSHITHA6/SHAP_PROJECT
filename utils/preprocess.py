import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df, target_column):
    """
    Preprocess data for machine learning
    """
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Store original column names
    original_columns = X.columns.tolist()
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # If no categorical columns, create empty list
    if not categorical_cols:
        categorical_cols = []
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Only create categorical transformer if there are categorical columns
    transformers = [('num', numeric_transformer, numeric_cols)]
    
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # Split data (80/20)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if y.nunique() <= 10 else None
        )
    except:
        # If stratification fails due to too many categories
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Store original test data for display
    X_test_original = X_test.copy()
    
    # Fit preprocessor and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Generate feature names - SIMPLE AND RELIABLE APPROACH
    feature_names = []
    
    # Add numeric feature names
    feature_names.extend(numeric_cols)
    
    # Add categorical feature names if they exist
    if categorical_cols:
        try:
            # Get the onehot encoder
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            # Get feature names
            cat_feature_names = ohe.get_feature_names_out(categorical_cols)
            feature_names.extend(list(cat_feature_names))
        except:
            # If that fails, create generic names
            for cat_col in categorical_cols:
                # Get unique values from the original data (limit to 10)
                unique_vals = X[cat_col].dropna().unique()[:10]
                for val in unique_vals:
                    feature_names.append(f"{cat_col}_{val}")
            
            # If still no features, create generic names
            if len(feature_names) == len(numeric_cols):
                n_cat_features = X_train_processed.shape[1] - len(numeric_cols)
                for i in range(n_cat_features):
                    feature_names.append(f"cat_feature_{i}")
    
    # Final safety check - ensure feature_names length matches
    if len(feature_names) != X_train_processed.shape[1]:
        # Regenerate with generic names
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
    
    # Convert to DataFrame for better handling
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    
    return (
        X_train_processed, X_test_processed,
        y_train, y_test,
        X_test_original,
        preprocessor,
        feature_names,
        original_columns,
        df
    )