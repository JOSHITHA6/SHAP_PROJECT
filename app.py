import streamlit as st
import pandas as pd
import numpy as np

from utils.preprocess import preprocess_data
from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots

from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(layout="wide")
st.title("SHAP - AI Model Explainability Tool")

if "page" not in st.session_state:
    st.session_state.page = "input"

# ================= INPUT =================
if st.session_state.page == "input":

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        target = st.selectbox("Target Column", df.columns)

        task = "Classification" if df[target].nunique() <= 10 else "Regression"
        st.write(f"Detected Task: {task}")

        model_type = st.selectbox(
            "Model",
            ["Random Forest", "Logistic Regression"]
            if task == "Classification"
            else ["Random Forest", "Linear Regression"]
        )

        if st.button("Run Model"):

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

            st.session_state.update({
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "test_display": test_display.reset_index(drop=True),
                "feature_names": feature_names,
                "task": task,
                "preprocessor": preprocessor
            })

            st.session_state.page = "output"
            st.rerun()

# ================= OUTPUT =================
else:

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    test_display = st.session_state.test_display
    feature_names = st.session_state.feature_names
    task = st.session_state.task
    preprocessor = st.session_state.preprocessor

    left, _, right = st.columns([1.1, 0.1, 1.4])

    # LEFT
    with left:
        st.subheader("Test Data (20%)")
        st.dataframe(test_display, height=300)

        y_pred = model.predict(X_test)

        if task == "Classification":
            st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        else:
            st.success(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # RIGHT
    with right:

        tab1, tab2 = st.tabs(["Global", "Local"])

        # ===== GLOBAL =====
        with tab1:

            fig, _, shap_vals = generate_shap_plots(
                model, X_test[:100], feature_names=feature_names, task=task
            )
            st.pyplot(fig)

            # 🔥 EXPLANATION
            vals = np.abs(shap_vals).mean(axis=0)
            perc = vals / vals.sum() * 100

            st.markdown("### 📌 Key Insights")

            for i in np.argsort(vals)[::-1][:5]:
                direction = "↑ increases" if np.mean(shap_vals[:, i]) > 0 else "↓ decreases"
                st.write(f"{feature_names[i]} → {perc[i]:.1f}% → {direction} prediction")

        # ===== LOCAL =====
        with tab2:

            option = st.radio("Choose Option", ["Select Row", "Enter New Data"])

            if option == "Select Row":

                row = st.number_input("Row", 1, len(X_test), 1)
                X_single = X_test[row-1:row]

                _, fig_local, _ = generate_shap_plots(
                    model, X_test[:100], X_single, feature_names, task
                )

                st.pyplot(fig_local)

            else:

                st.info("Enter all values")

                # 🔥 ALL INPUTS AT ONCE
                with st.form("input_form"):

                    inputs = {}
                    cols = st.columns(2)

                    for i, col in enumerate(feature_names):
                        with cols[i % 2]:
                            inputs[col] = st.text_input(col)

                    submit = st.form_submit_button("Predict")

                if submit:

                    try:
                        clean = {k: float(v.strip()) for k, v in inputs.items()}

                        new_df = pd.DataFrame([clean])
                        new_processed = preprocessor.transform(new_df)

                        pred = model.predict(new_processed)
                        st.success(f"Prediction: {pred[0]}")

                        _, fig_local, shap_vals = generate_shap_plots(
                            model, X_test[:100], new_processed, feature_names, task
                        )

                        st.pyplot(fig_local)

                    except:
                        st.error("⚠️ Enter valid numeric values")