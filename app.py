import streamlit as st
import pandas as pd

from utils.preprocess import preprocess_data
from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

# ================= CONFIG =================
st.set_page_config(layout="wide")
st.title("Explainable AI Dashboard (SHAP-Based Insights)")

# ================= SESSION STATE =================
if "page" not in st.session_state:
    st.session_state.page = "input"

# =========================================================
# ======================= INPUT PAGE =======================
# =========================================================
if st.session_state.page == "input":

    st.subheader("📥 Input Panel")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
    else:
        df = None

    if df is not None:

        target = st.selectbox("Target Column", df.columns)

        task = "Classification" if df[target].nunique() <= 10 else "Regression"
        st.write(f"Detected Task: {task}")

        model_type = st.selectbox(
            "Model",
            ["Random Forest", "Logistic Regression"]
            if task == "Classification"
            else ["Random Forest", "Linear Regression"]
        )

        if st.button("🚀 Run Model"):

            (
                X_train, X_test,
                y_train, y_test,
                X_test_original,
                preprocessor,
                feature_names
            ) = preprocess_data(df, target)

            model = train_model(X_train, y_train, task, model_type)

            test_display = X_test_original.copy()
            test_display[target] = y_test.values
            test_display = test_display.reset_index(drop=True)

            st.session_state.update({
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "test_display": test_display,
                "preprocessor": preprocessor,
                "feature_names": feature_names,
                "task": task
            })

            st.session_state.page = "output"
            st.rerun()

# =========================================================
# ======================= OUTPUT PAGE ======================
# =========================================================
elif st.session_state.page == "output":

    if st.button("⬅️ Back"):
        st.session_state.page = "input"
        st.rerun()

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    test_display = st.session_state.test_display
    feature_names = st.session_state.feature_names
    task = st.session_state.task

    # ================= SPACED LAYOUT =================
    left, spacer, right = st.columns([1.1, 0.1, 1.4])

    # ================= LEFT SIDE =================
    with left:

        st.markdown("### 📄 Test Dataset (20%)")
        st.dataframe(test_display, height=350)

        st.markdown("---")

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

    # ================= RIGHT SIDE =================
    with right:

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🌍 Global Explainability", "🔍 Local Explainability"])

        # ================= GLOBAL =================
        with tab1:

            with st.spinner("Generating SHAP..."):
                fig_global, _ = generate_shap_plots(
                    model,
                    X_test[:100],
                    feature_names=feature_names
                )

            st.pyplot(fig_global)

        # ================= LOCAL =================
        with tab2:

            option = st.radio(
                "Choose Option",
                ["Select Row", "Enter New Data"]
            )

            # -------- OPTION 1 --------
            if option == "Select Row":

                row = st.number_input("Row Number", 1, len(X_test), 1)

                X_single = X_test[row-1:row]

                _, fig_local = generate_shap_plots(
                    model,
                    X_test[:100],
                    X_single,
                    feature_names=feature_names
                )

                st.pyplot(fig_local)

            # -------- OPTION 2 --------
            else:

                st.info("Fill all fields and click Predict")

                with st.form("manual_input_form"):

                    input_data = {}
                    cols = st.columns(2)

                    for i, col in enumerate(feature_names):
                        with cols[i % 2]:
                            input_data[col] = st.text_input(col)

                    submit = st.form_submit_button("Predict")

                    if submit:

                        if any(v.strip() == "" for v in input_data.values()):
                            st.error("⚠️ Please fill all fields")
                        else:
                            try:
                                for k in input_data:
                                    input_data[k] = float(input_data[k])

                                new_df = pd.DataFrame([input_data])
                                new_processed = new_df.values

                                pred = model.predict(new_processed)

                                st.success(f"Prediction: {pred[0]}")

                                _, fig_local = generate_shap_plots(
                                    model,
                                    X_test[:100],
                                    new_processed,
                                    feature_names=feature_names
                                )

                                st.pyplot(fig_local)

                            except:
                                st.error("⚠️ Enter valid numeric values")