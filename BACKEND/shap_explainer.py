import shap
import matplotlib.pyplot as plt

def generate_shap_plots(model, X_sample, X_single=None):

    # Use TreeExplainer explicitly (more stable)
    explainer = shap.TreeExplainer(model)

    # GLOBAL
    shap_values = explainer(X_sample, check_additivity=False)

    fig_global = plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)

    fig_local = None

    # LOCAL
    if X_single is not None:
        shap_single = explainer(X_single, check_additivity=False)

        fig_local = plt.figure()
        shap.plots.waterfall(shap_single[0], show=False)

    return fig_global, fig_local