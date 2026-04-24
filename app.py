import streamlit as st
import pandas as pd
import shap

from utils.preprocess import preprocess_data
from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

st.set_page_config(layout="wide")
st.title("SHAP- AI MODEL EXPLAINABILITY TOOL")

col1, col2 = st.columns(2)

# ================= LEFT =================
with col1:

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        target = st.selectbox("Target Column", df.columns)

        # Auto detect task
        if df[target].nunique() <= 10:
            task = "Classification"
        else:
            task = "Regression"

        st.write(f"Detected Task: {task}")

        if task == "Classification":
            model_type = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
        else:
            model_type = st.selectbox("Model", ["Random Forest", "Linear Regression"])

        if st.button("Run Model"):

            (
                X_train, X_test,
                y_train, y_test,
                X_test_original,
                preprocessor,
                feature_cols
            ) = preprocess_data(df, target)

            model = train_model(X_train, y_train, task, model_type)

            st.session_state.update({
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "X_test_original": X_test_original,
                "preprocessor": preprocessor,
                "feature_cols": feature_cols,
                "task": task
            })

# ================= RIGHT =================
with col2:

    if "model" in st.session_state:

        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        task = st.session_state["task"]

        # ===== PREDICTIONS =====
        if task == "Classification":
            y_pred = model.predict(X_test)

            st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
            st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
            st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

        else:
            y_pred = model.predict(X_test)

            st.success(f"R²: {r2_score(y_test, y_pred):.2f}")
            st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
            st.write(f"RMSE: {mean_squared_error(y_test, y_pred)**0.5:.2f}")

        st.markdown("## 🔍 Global SHAP")

        fig_global, _ = generate_shap_plots(model, X_test)
        st.pyplot(fig_global)

        # ===== LOCAL =====
        st.markdown("## 🔍 Local Explanation")

        option = st.radio("Choose Option", ["Select Row", "Enter New Data"])

        if option == "Select Row":
            row = st.number_input("Row", 1, len(X_test), 1)
            X_single = X_test[row-1:row]

            _, fig_local = generate_shap_plots(model, X_test, X_single)
            st.pyplot(fig_local)

        else:
            st.markdown("### Enter Feature Values")

            input_data = {}

            for col in st.session_state["feature_cols"]:
                if col != st.session_state["feature_cols"][-1]:
                    input_data[col] = st.number_input(col)

            if st.button("Predict"):

                new_df = pd.DataFrame([input_data])
                new_processed = st.session_state["preprocessor"].transform(new_df)

                pred = model.predict(new_processed)

                st.success(f"Prediction: {pred[0]}")

                _, fig_local = generate_shap_plots(model, X_test, new_processed)
                st.pyplot(fig_local)