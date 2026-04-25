import shap
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.rcParams['text.usetex'] = False


def generate_shap_plots(model, X_sample, X_single=None, feature_names=None, task="Regression"):

    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)

    explainer = shap.Explainer(model)

    shap_values = explainer(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="violin", show=False)

    fig_global = plt.gcf()
    plt.close()

    fig_local = None

    if X_single is not None:
        if not isinstance(X_single, pd.DataFrame):
            X_single = pd.DataFrame(X_single, columns=feature_names)

        shap_single = explainer(X_single)

        plt.figure()
        shap.plots.waterfall(shap_single[0], show=False)

        fig_local = plt.gcf()
        plt.close()

    return fig_global, fig_local, shap_values.values