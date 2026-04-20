from sklearn.model_selection import train_test_split

# Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor


def train_model(df, target, task, model_type):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model mappings
    classification_models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier()
    }

    regression_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
        "XGBoost": XGBRegressor()
    }

    # Select model
    if task == "Classification":
        model = classification_models.get(model_type)
    else:
        model = regression_models.get(model_type)

    if model is None:
        raise ValueError("Invalid model selected")

    model.fit(X_train, y_train)

    return model, X_test