import shap
import matplotlib.pyplot as plt
import numpy as np

def generate_shap_plots(model, X_sample, X_single=None):

    # ================= AUTO EXPLAINER =================
    if hasattr(model, "estimators_"):
        # Tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)

    else:
        # Linear / Logistic models
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer(X_sample)

    # ================= GLOBAL PLOT =================
    fig_global = plt.figure()

    shap.summary_plot(
        shap_values.values,
        X_sample,
        show=False
    )

    # ================= LOCAL PLOT =================
    fig_local = None

    if X_single is not None:

        # Recompute SHAP for single row
        if hasattr(model, "estimators_"):
            shap_single = shap.TreeExplainer(model)(X_single)
        else:
            shap_single = shap.LinearExplainer(model, X_sample)(X_single)

        fig_local = plt.figure()

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_single.values[0],
                base_values=shap_single.base_values[0],
                data=X_single.iloc[0]
            ),
            show=False
        )

    return fig_global, fig_local