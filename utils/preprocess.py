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
    
    Parameters:
    - df: Input dataframe
    - target_column: Name of target column
    
    Returns:
    - X_train, X_test, y_train, y_test: Split data
    - X_test_original: Original test features (for display)
    - preprocessor: Fitted preprocessor
    - feature_names: Names of features after preprocessing
    - original_columns: Original column names
    - df_full: Full dataframe
    """
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Store original column names
    original_columns = X.columns.tolist()
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y if y.nunique() <= 10 else None
    )
    
    # Store original test data for display
    X_test_original = X_test.copy()
    
    # Fit preprocessor and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing - FIXED VERSION
    feature_names = []
    
    # Add numeric feature names
    feature_names.extend(numeric_cols)
    
    # Add categorical feature names correctly
    for cat_col in categorical_cols:
        try:
            # FIX: Get the onehot encoder and call get_feature_names_out correctly
            onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            
            # For sklearn versions 1.0+
            if hasattr(onehot_encoder, 'get_feature_names_out'):
                # Get feature names for this column
                cats = onehot_encoder.get_feature_names_out([cat_col])
                feature_names.extend(cats)
            else:
                # Fallback for older sklearn versions
                cats = [f"{cat_col}_{cat}" for cat in onehot_encoder.categories_[0]]
                feature_names.extend(cats)
        except Exception as e:
            # Fallback: generate names manually
            if hasattr(onehot_encoder, 'categories_'):
                categories = onehot_encoder.categories_[0]
                cats = [f"{cat_col}_{cat}" for cat in categories]
                feature_names.extend(cats)
            else:
                # If all else fails, use indexed names
                n_categories = len(preprocessor.transformers_[1][2])  # Get categorical columns count
                feature_names.extend([f"{cat_col}_cat_{i}" for i in range(n_categories)])
    
    # Ensure feature_names length matches processed data
    if len(feature_names) != X_train_processed.shape[1]:
        print(f"Warning: feature_names length ({len(feature_names)}) doesn't match X_train_processed shape ({X_train_processed.shape[1]})")
        print("Regenerating feature names...")
        
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