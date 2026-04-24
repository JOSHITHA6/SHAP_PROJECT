import shap
import matplotlib.pyplot as plt

def generate_shap_plots(model, X_sample, X_single=None):

    explainer = shap.Explainer(model, X_sample)

    # GLOBAL
    shap_values = explainer(X_sample)
    fig_global = plt.figure()
    shap.plots.beeswarm(shap_values, show=False)

    fig_local = None

    # LOCAL (only if row selected)
    if X_single is not None:
        shap_single = explainer(X_single)
        fig_local = plt.figure()
        shap.plots.waterfall(shap_single[0], show=False)

    return fig_global, fig_local