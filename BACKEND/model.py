from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def train_model(X_train, y_train, task, model_type):
    """
    Train a model with hyperparameter tuning
    
    Parameters:
    - X_train: Training features
    - y_train: Training target
    - task: 'Classification' or 'Regression'
    - model_type: Type of model to train
    
    Returns:
    - Trained model
    """
    
    if task == "Classification":
        
        if model_type == "Random Forest":
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
            cv_folds = 3
            
        else:  # Logistic Regression
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            param_grid = {
                "C": [0.1, 1, 10],
                "solver": ['lbfgs', 'liblinear']
            }
            cv_folds = 3

    else:  # Regression
        
        if model_type == "Random Forest":
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
            cv_folds = 3
            
        else:  # Linear Regression
            model = LinearRegression()
            param_grid = {}  # No hyperparameters to tune
            cv_folds = None
    
    # Perform GridSearch if we have parameters to tune
    if param_grid:
        # Use fewer folds for small datasets
        if len(X_train) < 100:
            cv_folds = 2
        
        grid = GridSearchCV(
            model, 
            param_grid, 
            cv=cv_folds, 
            n_jobs=-1,
            verbose=0,
            scoring='accuracy' if task == 'Classification' else 'r2'
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_
    
    else:
        model.fit(X_train, y_train)
        return model