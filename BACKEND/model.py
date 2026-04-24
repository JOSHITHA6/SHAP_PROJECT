from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

def train_model(X_train, y_train, task, model_type):

    # ================= CLASSIFICATION =================
    if task == "Classification":

        if model_type == "Random Forest":
            model = RandomForestClassifier()

        elif model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        elif model_type == "Linear Regression":
            raise ValueError("Linear Regression cannot be used for Classification")

    # ================= REGRESSION =================
    else:

        if model_type == "Random Forest":
            model = RandomForestRegressor()

        elif model_type == "Linear Regression":
            model = LinearRegression()

        elif model_type == "Logistic Regression":
            raise ValueError("Logistic Regression cannot be used for Regression")

    model.fit(X_train, y_train)
    return model