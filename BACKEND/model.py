from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train, task, model_type):

    if task == "Classification":

        if model_type == "Random Forest":
            model = RandomForestClassifier(random_state=42)

            param_grid = {
                "n_estimators": [100, 150],
                "max_depth": [None, 10, 20]
            }

        else:
            model = LogisticRegression(max_iter=1000)

            param_grid = {
                "C": [0.1, 1, 10]
            }

    else:  # Regression

        if model_type == "Random Forest":
            model = RandomForestRegressor(random_state=42)

            param_grid = {
                "n_estimators": [100, 150],
                "max_depth": [None, 10]
            }

        else:
            model = LinearRegression()
            param_grid = {}

    # GridSearch (safe + fast)
    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_

    else:
        model.fit(X_train, y_train)
        return model