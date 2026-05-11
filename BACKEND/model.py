from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR


def train_model(X_train, y_train, task: str, model_type: str):
    """
    Train and return a fitted model.

    Parameters
    ----------
    X_train    : array-like of shape (n_samples, n_features)
    y_train    : array-like of shape (n_samples,)
    task       : "Classification" | "Regression"
    model_type : one of the strings shown in the UI drop-down
    """
    if task == "Classification":
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "SVM":
            # probability=True lets us use LinearExplainer / KernelExplainer
            model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown classification model: {model_type}")

    elif task == "Regression":
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "SVR":
            model = SVR()
        else:
            raise ValueError(f"Unknown regression model: {model_type}")

    else:
        raise ValueError(f"Unknown task: {task}")

    model.fit(X_train, y_train)
    return model