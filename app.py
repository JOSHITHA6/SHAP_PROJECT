import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots
from utils.preprocess import preprocess_data

from sklearn.metrics import accuracy_score, r2_score

# ====================================================
# CONFIG
# ====================================================
st.set_page_config(layout="wide")

# ====================================================
# CSS
# ====================================================
st.markdown("""
<style>
.stApp { background-color: #f8fafc; }
.section-box {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
.divider {
    border-left: 2px solid #e2e8f0;
    height: 100%;
}
</style>
""", unsafe_allow_html=True)

# ====================================================
# TITLE
# ====================================================
st.title("📊 SHAP Explainability Tool")

# ====================================================
# LAYOUT
# ====================================================
col1, col_gap, col2 = st.columns([1, 0.05, 1])

# ====================================================
# LEFT SIDE
# ====================================================
with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        st.session_state["df"] = pd.read_csv(file)

    df = st.session_state.get("df", None)

    if df is not None:
        st.dataframe(df, height=350)
        target = st.selectbox("Target Column", df.columns)
    else:
        target = None

    task = st.radio("Task", ["Classification", "Regression"])

    model_type = st.selectbox(
        "Model",
        ["Random Forest", "Linear/Logistic Regression"]
    )

    if st.button("Run Model"):
        st.session_state["run"] = True

    run = st.session_state.get("run", False)

    st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# RIGHT SIDE
# ====================================================
with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    if run and df is not None and target is not None:

        if "model" not in st.session_state:

            X_train, X_test, y_train, y_test = preprocess_data(df, target)
            model = train_model(X_train, y_train, task, model_type)

            st.session_state["model"] = model
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test

        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        y_pred = model.predict(X_test)

        # ====================================================
        # METRICS
        # ====================================================
        if task == "Classification":
            st.success(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
        else:
            st.success(f"R² Score: {r2_score(y_test, y_pred):.3f}")

        # ====================================================
        # SHAP
        # ====================================================
        st.markdown("## 🔍 SHAP Explanation")

        X_sample = X_test.sample(min(50, len(X_test)), random_state=42)

        tab1, tab2 = st.tabs(["Global", "Local"])

        # ====================================================
        # GLOBAL
        # ====================================================
        with tab1:

            plot_type = st.selectbox(
                "Select Graph Type",
                ["Bar", "Beeswarm", "Violin"]
            )

            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample, check_additivity=False)

            shap_array = shap_values.values
            features = X_sample.columns

            # ---------- PLOTS ----------
            if plot_type == "Bar":
                abs_vals = np.abs(shap_array).mean(axis=0)
                idx = np.argsort(abs_vals)

                fig = plt.figure()
                plt.barh(features[idx], abs_vals[idx])
                plt.title("Feature Importance")
                st.pyplot(fig)

            elif plot_type == "Beeswarm":
                fig = plt.figure()
                shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(fig)

            elif plot_type == "Violin":
                fig = plt.figure()
                shap.summary_plot(shap_values, X_sample, plot_type="violin", show=False)
                st.pyplot(fig)

                st.info("""
Violin plot shows distribution of impact.
Left → decreases prediction
Right → increases prediction
Width → number of data points
""")

            # ====================================================
            # CORRECT TREND LOGIC (WEIGHTED)
            # ====================================================
            abs_vals = np.abs(shap_array).mean(axis=0)
            percent = abs_vals / abs_vals.sum() * 100

            st.markdown("### 📌 Feature Insights")

            for i in np.argsort(abs_vals)[::-1][:5]:

                feat = features[i]

                pos_strength = np.sum(shap_array[:, i][shap_array[:, i] > 0])
                neg_strength = -np.sum(shap_array[:, i][shap_array[:, i] < 0])

                total = pos_strength + neg_strength

                if total == 0:
                    trend = "no impact"
                elif pos_strength / total > 0.6:
                    trend = "mostly increases"
                elif neg_strength / total > 0.6:
                    trend = "mostly decreases"
                else:
                    trend = "mixed effect"

                st.write(f"• {feat}: {percent[i]:.1f}% → {trend}")

        # ====================================================
        # LOCAL
        # ====================================================
        with tab2:

            total_rows = len(X_test)

            row = st.number_input(
                "Select Row",
                1,
                total_rows,
                1,
                key="row"
            )

            X_single = X_test.iloc[[row-1]]

            st.dataframe(X_single)

            _, fig_local = generate_shap_plots(model, X_sample, X_single)
            st.pyplot(fig_local)

    else:
        st.info("Upload dataset and run model")

    st.markdown('</div>', unsafe_allow_html=True)