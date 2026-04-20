import shap
import matplotlib.pyplot as plt


def explain_model(model, X, model_type):
    # Choose correct SHAP explainer
    if model_type in ["Random Forest", "XGBoost"]:
        explainer = shap.TreeExplainer(model)

    elif model_type in ["Linear Regression", "Logistic Regression"]:
        explainer = shap.LinearExplainer(model, X)

    else:
        explainer = shap.KernelExplainer(model.predict, X)

    shap_values = explainer(X)

    # Plot (global explanation)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)

    return fig