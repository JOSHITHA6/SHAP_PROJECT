import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_shap_plots(model, X_sample, X_single=None, feature_names=None, task="Regression"):

    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)

    # Choose explainer
    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample)

    shap_values = explainer(X_sample)

    # 🔥 HANDLE CLASSIFICATION OUTPUT
    if isinstance(shap_values.values, np.ndarray) and len(shap_values.values.shape) == 3:
        shap_vals = shap_values.values[:, :, 1]  # class 1
    else:
        shap_vals = shap_values.values

    plt.close('all')

    # ===== GLOBAL =====
    fig_global = plt.figure()

    shap.summary_plot(
        shap_vals,
        X_sample,
        plot_type="violin",
        show=False
    )

    # ===== LOCAL =====
    fig_local = None

    if X_single is not None:

        if not isinstance(X_single, pd.DataFrame):
            X_single = pd.DataFrame(X_single, columns=feature_names)

        shap_single = explainer(X_single)

        if isinstance(shap_single.values, np.ndarray) and len(shap_single.values.shape) == 3:
            values = shap_single.values[0, :, 1]
            base = shap_single.base_values[0, 1]
        else:
            values = shap_single.values[0]
            base = shap_single.base_values[0]

        fig_local = plt.figure()

        shap.waterfall_plot(
            shap.Explanation(
                values=values,
                base_values=base,
                feature_names=feature_names
            ),
            show=False
        )

    return fig_global, fig_local, shap_vals