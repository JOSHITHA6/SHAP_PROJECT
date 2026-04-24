import shap
import matplotlib.pyplot as plt
import pandas as pd

def generate_shap_plots(
    model,
    X_sample,
    X_single=None,
    feature_names=None,
    original_row=None
):

    # Convert to DataFrame for SHAP
    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)

    # Choose explainer
    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample)

    shap_values = explainer(X_sample)

    plt.close('all')

    # ===== GLOBAL =====
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

        if not isinstance(X_single, pd.DataFrame):
            X_single = pd.DataFrame(X_single, columns=feature_names)

        shap_single = explainer(X_single)

        fig_local = plt.figure()

        # 🔥 USE ORIGINAL VALUES HERE
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_single.values[0],
                base_values=shap_single.base_values[0],
                data=original_row.iloc[0],   # ✅ ONLY ORIGINAL VALUES
                feature_names=feature_names
            ),
            show=False
        )

    return fig_global, fig_local