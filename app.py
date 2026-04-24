import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots
from utils.preprocess import preprocess_data

from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(layout="wide")
st.title("📊 SHAP Explainability Tool")

col1, col2 = st.columns(2)

# ================= LEFT =================
with col1:

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        st.session_state["df"] = pd.read_csv(file)

    df = st.session_state.get("df", None)

    if df is not None:
        st.dataframe(df.head(20))
        target = st.selectbox("Target Column", df.columns)
    else:
        target = None

    task = st.radio("Task", ["Classification", "Regression"])

    model_type = st.selectbox(
        "Model",
        ["Random Forest", "Linear Regression", "Logistic Regression"]
    )

    # ================= STRICT VALIDATION =================
    invalid_combo = False

    if task == "Classification" and model_type == "Linear Regression":
        st.error("❌ Linear Regression is NOT suitable for Classification")
        invalid_combo = True

    if task == "Regression" and model_type == "Logistic Regression":
        st.error("❌ Logistic Regression is NOT suitable for Regression")
        invalid_combo = True

    if st.button("Run Model"):
        if not invalid_combo:
            st.session_state["run"] = True

    run = st.session_state.get("run", False)

# ================= RIGHT =================
with col2:

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

        # ================= PREDICTIONS =================
        if task == "Classification":
            if hasattr(model, "predict_proba"):
                y_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test).astype(int)
        else:
            y_pred = model.predict(X_test)

        # ================= METRICS =================
        if task == "Classification":
            st.success(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
        else:
            st.success(f"R² Score: {r2_score(y_test, y_pred):.3f}")

        st.markdown("## 🔍 SHAP Explanation")

        X_sample = X_test.sample(min(50, len(X_test)), random_state=42)

        tab1, tab2 = st.tabs(["Global", "Local"])

        # ================= GLOBAL =================
        with tab1:

            plot_type = st.selectbox(
                "Graph Type",
                ["Violin"]
                '''["Bar", "Beeswarm", "Violin"]'''
            )

            # ================= EXPLAINER =================
            try:
                if hasattr(model, "estimators_"):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.LinearExplainer(model, X_sample)

                shap_values = explainer(X_sample)

            except:
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)

            shap_array = shap_values.values
            features = X_sample.columns

            # ✅ FIX: convert to numpy
            feature_names = np.array(features)

            # ================= PLOTS =================
            if plot_type == "Bar":
                vals = np.abs(shap_array).mean(axis=0)
                idx = np.argsort(vals)

                fig = plt.figure()
                plt.barh(feature_names[idx], vals[idx])
                plt.title("Feature Importance")
                st.pyplot(fig)

            '''elif plot_type == "Beeswarm":
                fig = plt.figure()
                shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(fig)'''

            '''else:
                fig = plt.figure()
                shap.summary_plot(shap_values, X_sample, plot_type="violin", show=False)
                st.pyplot(fig)'''

            # ================= INSIGHTS =================
            vals = np.abs(shap_array).mean(axis=0)
            percent = vals / vals.sum() * 100

            increase, decrease, mixed = [], [], []

            for i in np.argsort(vals)[::-1][:6]:

                feat = feature_names[i]

                pos = np.sum(shap_array[:, i][shap_array[:, i] > 0])
                neg = -np.sum(shap_array[:, i][shap_array[:, i] < 0])
                total = pos + neg

                if total == 0:
                    continue

                if pos / total > 0.6:
                    increase.append((feat, percent[i]))
                elif neg / total > 0.6:
                    decrease.append((feat, percent[i]))
                else:
                    mixed.append((feat, percent[i]))

            st.markdown("### 📌 Feature Impact")

            if increase:
                st.markdown("### 🔺 Increases Prediction")
                for f, p in increase:
                    st.write(f"{f} → {p:.1f}%")

            if decrease:
                st.markdown("### 🔻 Decreases Prediction")
                for f, p in decrease:
                    st.write(f"{f} → {p:.1f}%")

            if mixed:
                st.markdown("### ⚖️ Mixed Effect")
                for f, p in mixed:
                    st.write(f"{f} → {p:.1f}%")

        # ================= LOCAL =================
        with tab2:

            total = len(X_test)
            row = st.number_input("Select Row", 1, total, 1)

            st.markdown("### 🧾 Original Input")
            st.dataframe(df.iloc[[row-1]])

            st.markdown("### ⚙️ Model Input")
            X_single = X_test.iloc[[row-1]]
            st.dataframe(X_single)

            _, fig_local = generate_shap_plots(model, X_sample, X_single)
            st.pyplot(fig_local)

    else:
        st.info("Upload dataset and run model")