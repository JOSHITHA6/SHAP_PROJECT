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
# ======================= INPUT PAGE =======================
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

        if task == "Classification":
            model_type = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
        else:
            model_type = st.selectbox("Model", ["Random Forest", "Linear Regression"])

        # Enable Run only when ready
        ready = target is not None and model_type is not None

        if st.button("🚀 Run Model", disabled=not ready):

            (
                X_train, X_test,
                y_train, y_test,
                X_test_original,
                preprocessor,
                feature_cols
            ) = preprocess_data(df, target)

            model = train_model(X_train, y_train, task, model_type)

            # Combine test + target for display
            test_display = X_test_original.copy()
            test_display[target] = y_test.values
            test_display = test_display.reset_index(drop=True)

            st.session_state.update({
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "test_display": test_display,
                "preprocessor": preprocessor,
                "feature_cols": feature_cols,
                "target": target,
                "task": task
            })

            st.session_state["page"] = "output"
            st.rerun()

# =========================================================
# ======================= OUTPUT PAGE ======================
# =========================================================
elif st.session_state["page"] == "output":

    st.subheader("📊 Output Panel")

    if st.button("⬅️ Back"):
        st.session_state["page"] = "input"
        st.rerun()

    model = st.session_state["model"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    test_display = st.session_state["test_display"]
    feature_cols = st.session_state["feature_cols"]
    preprocessor = st.session_state["preprocessor"]
    task = st.session_state["task"]
    target = st.session_state["target"]

    # ================= TEST DATA =================
    st.markdown("### 📄 Test Dataset (20%)")
    st.dataframe(test_display)

    # ================= METRICS =================
    st.markdown("### 📈 Model Performance")

    y_pred = model.predict(X_test)

    if task == "Classification":
        st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.2f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.2f}")
    else:
        st.success(f"R²: {r2_score(y_test, y_pred):.2f}")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"RMSE: {(mean_squared_error(y_test, y_pred))**0.5:.2f}")

    # ================= TABS =================
    tab1, tab2 = st.tabs(["🌍 Global Explainability", "🔍 Local Explainability"])

    # ================= GLOBAL =================
    with tab1:

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

            # 🔥 FORM (NO FLICKER)
            with st.form("manual_input_form"):

                input_data = {}

                for i, col in enumerate(feature_cols):
                    input_data[col] = st.number_input(
                        f"Feature {i+1}",
                        key=f"input_{col}"
                    )

                submitted = st.form_submit_button("Predict")

                if submitted:

                    new_df = pd.DataFrame([input_data])

                    # Ensure correct column order
                    new_df = new_df[feature_cols]

                    new_processed = preprocessor.transform(new_df)

                    pred = model.predict(new_processed)

                    st.success(f"Prediction: {pred[0]}")

                    _, fig_local = generate_shap_plots(model, X_test, new_processed)
                    st.pyplot(fig_local)