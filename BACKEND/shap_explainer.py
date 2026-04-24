import shap
import matplotlib.pyplot as plt

def generate_shap_plots(model, X_sample, X_single=None):

    # Choose explainer
    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample)

    shap_values = explainer(X_sample)

    # ===== GLOBAL =====
    plt.close('all')
    fig_global = plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="violin",
        show=False
    )

    plt.xlabel("SHAP value (impact on prediction)")
    plt.ylabel("Features")

    # ===== LOCAL =====
    fig_local = None

    if X_single is not None:
        shap_single = explainer(X_single)

        fig_local = plt.figure()

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_single.values[0],
                base_values=shap_single.base_values[0],
                data=X_single[0]
            ),
            show=False
        )

    return fig_global, fig_local