from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
warnings.filterwarnings('ignore')


def train_model(X_train, y_train, task, model_type):
    """
    Train and return a fitted model.

    Parameters
    ----------
    X_train    : Training features
    y_train    : Training target
    task       : 'Classification' or 'Regression'
    model_type : 'Random Forest' or 'Logistic Regression' (classification)
                 'Random Forest' or 'Linear Regression'  (regression)
    """

    if task == "Classification":
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:  # Logistic Regression
            model = LogisticRegression(max_iter=1000, random_state=42)

    else:  # Regression
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:  # Linear Regression
            model = LinearRegression()

    model.fit(X_train, y_train)
    return model