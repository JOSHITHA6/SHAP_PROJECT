import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from utils.preprocess import preprocess_data
from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

# ================= CONFIG =================
st.set_page_config(layout="wide")
st.title("SHAP- AI MODEL EXPLAINABILITY TOOL")

# ================= PAGE STATE =================
if "page" not in st.session_state:
    st.session_state["page"] = "input"

# =========================================================
# ======================= PAGE 1 ===========================
# ======================= INPUT ============================
# =========================================================
if st.session_state["page"] == "input":

    st.subheader("📥 Input Panel")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state["df"] = df
    else:
        df = None

    if df is not None:

        target = st.selectbox("Target Column", df.columns)

        # Auto detect task
        if df[target].nunique() <= 10:
            task = "Classification"
        else:
            task = "Regression"

        st.write(f"Detected Task: {task}")

        # Model selection
        if task == "Classification":
            model_type = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
        else:
            model_type = st.selectbox("Model", ["Random Forest", "Linear Regression"])

        # ================= VALIDATION =================
        ready = target is not None and model_type is not None

        run_btn = st.button("🚀 Run Model", disabled=not ready)

        if run_btn:

            (
                X_train, X_test,
                y_train, y_test,
                X_test_original,
                preprocessor,
                feature_cols
            ) = preprocess_data(df, target)

            model = train_model(X_train, y_train, task, model_type)

            # Save everything
            st.session_state.update({
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "X_test_original": X_test_original.reset_index(drop=True),
                "preprocessor": preprocessor,
                "feature_cols": feature_cols,
                "task": task
            })

            # 🔥 SWITCH PAGE
            st.session_state["page"] = "output"
            st.rerun()

# =========================================================
# ======================= PAGE 2 ===========================
# ======================= OUTPUT ===========================
# =========================================================
elif st.session_state["page"] == "output":

    st.subheader("📊 Output Panel")

    # Back button
    if st.button("⬅️ Back to Input"):
        st.session_state["page"] = "input"
        st.rerun()

    if "model" in st.session_state:

        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        X_test_original = st.session_state["X_test_original"]
        task = st.session_state["task"]

        # ================= TEST DATA =================
        st.markdown("### 📄 Test Dataset (20%)")
        st.dataframe(X_test_original)

        # ================= METRICS =================
        st.markdown("### 📈 Model Performance")

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
            st.write(f"RMSE: {(mean_squared_error(y_test, y_pred))**0.5:.2f}")

        # ================= TABS =================
        tab1, tab2 = st.tabs(["🌍 Global Explainability", "🔍 Local Explainability"])

        # ================= GLOBAL =================
        with tab1:
            st.markdown("### SHAP Global Explanation")

            fig_global, _ = generate_shap_plots(model, X_test)
            st.pyplot(fig_global)

        # ================= LOCAL =================
        with tab2:

            option = st.radio("Choose Option", ["Select Row", "Enter New Data"])

            # -------- OPTION A --------
            if option == "Select Row":
                row = st.number_input("Row Number", 1, len(X_test), 1)

                X_single = X_test[row-1:row]

                _, fig_local = generate_shap_plots(model, X_test, X_single)
                st.pyplot(fig_local)

            # -------- OPTION B --------
            else:
                st.markdown("### ✏️ Enter New Data")

                input_data = {}

                for col in st.session_state["feature_cols"]:
                    input_data[col] = st.number_input(f"{col}", key=col)

                # Enable only if all filled
                filled = all(v is not None for v in input_data.values())

                predict_btn = st.button("Predict", disabled=not filled)

                if predict_btn:

                    new_df = pd.DataFrame([input_data])
                    new_processed = st.session_state["preprocessor"].transform(new_df)

                    pred = model.predict(new_processed)

                    st.success(f"Prediction: {pred[0]}")

                    _, fig_local = generate_shap_plots(model, X_test, new_processed)
                    st.pyplot(fig_local)

    else:
        st.warning("No model found. Go back and run again.")