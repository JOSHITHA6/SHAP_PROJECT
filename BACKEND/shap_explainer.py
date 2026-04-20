import shap
import matplotlib.pyplot as plt

def explain_model(model, X, model_type):

    # 🔥 TAKE SMALL SAMPLE (IMPORTANT)
    X_sample = X[:50]

    if model_type in ["Random Forest", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

    elif model_type in ["Linear Regression", "Logistic Regression"]:
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

    else:
        explainer = shap.KernelExplainer(model.predict, X_sample)
        shap_values = explainer.shap_values(X_sample)

    # Plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, show=False)

    return fig