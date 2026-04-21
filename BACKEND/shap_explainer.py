import shap
import matplotlib.pyplot as plt

def generate_shap_plots(model, X_sample):

    # Universal SHAP (works for all models)
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample, check_additivity=False)

    # -------- GLOBAL (Beeswarm / Violin-like) --------
    fig_global, ax1 = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)

    # -------- LOCAL (Waterfall) --------
    fig_local, ax2 = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)

    return fig_global, fig_local