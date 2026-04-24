from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

def train_model(X_train, y_train, task, model_type):

    if task == "Classification":

        if model_type == "Random Forest":
            model = RandomForestClassifier(class_weight='balanced')

        elif model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

    else:

        if model_type == "Random Forest":
            model = RandomForestRegressor()

        elif model_type == "Linear Regression":
            model = LinearRegression()

    model.fit(X_train, y_train)
    return model