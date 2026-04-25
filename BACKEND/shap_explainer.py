import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_shap_plots(model, X_sample, X_single=None, feature_names=None, task="Regression"):

    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)

    # -------- EXPLAINER --------
    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample)

    shap_values = explainer(X_sample)

    # -------- HANDLE CLASSIFICATION --------
    if isinstance(shap_values.values, np.ndarray) and len(shap_values.values.shape) == 3:
        shap_vals = shap_values.values[:, :, 1]
    else:
        shap_vals = shap_values.values

    # -------- GLOBAL --------
    plt.close('all')
    fig_global = plt.figure()

    shap.summary_plot(shap_vals, X_sample, plot_type="violin", show=False)

    plt.xlabel("Impact on Prediction", fontsize=12)
    plt.ylabel("Features", fontsize=12)

    # -------- LOCAL --------
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

        # 🔥 take top features (no grey confusion)
        idx = np.argsort(np.abs(values))[::-1][:8]

        values = values[idx]
        names = np.array(feature_names)[idx]

        fig_local = plt.figure()

        shap.waterfall_plot(
            shap.Explanation(
                values=values,
                base_values=base,
                feature_names=names
            ),
            show=False
        )

        # 🔥 FORCE LABELS INSIDE GRAPH (THIS IS THE FIX)
        fig = plt.gcf()

        fig.text(0.5, 0.02, "Impact on Prediction", ha='center', fontsize=12)
        fig.text(0.02, 0.5, "Features", va='center', rotation='vertical', fontsize=12)

    return fig_global, fig_local, shap_vals