import shap
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

# 🔥 CRITICAL FIX (prevents $f(x)$ crash)
matplotlib.rcParams['text.usetex'] = False


def generate_shap_plots(model, X_sample, X_single=None, feature_names=None, task="Regression"):

    explainer = shap.Explainer(model)

    # -------- GLOBAL EXPLANATION --------
    shap_values = explainer(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="violin", show=False)

    fig_global = plt.gcf()
    plt.close()

    # -------- LOCAL EXPLANATION --------
    fig_local = None

    if X_single is not None:
        shap_single = explainer(X_single)

        plt.figure()
        shap.plots.waterfall(shap_single[0], show=False)

        fig_local = plt.gcf()
        plt.close()

    return fig_global, fig_local, shap_values.values