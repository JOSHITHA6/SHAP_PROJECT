import streamlit as st
import pandas as pd

from utils.preprocess import preprocess_data
from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

st.set_page_config(layout="wide")
st.title("Explainable AI Dashboard (SHAP-Based Insights)")

# ================= PAGE STATE =================
if "page" not in st.session_state:
    st.session_state["page"] = "input"

# ================= INPUT PAGE =================
if st.session_state["page"] == "input":

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state["df"] = df
    else:
        df = None

    if df is not None:

        target = st.selectbox("Target Column", df.columns)

        task = "Classification" if df[target].nunique() <= 10 else "Regression"

        st.write(f"Detected Task: {task}")

        model_type = st.selectbox(
            "Model",
            ["Random Forest", "Logistic Regression"] if task == "Classification"
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
                "test_display": test_display,
                "preprocessor": preprocessor,
                "feature_names": feature_names,
                "task": task
            })

            st.session_state["page"] = "output"
            st.rerun()

# ================= OUTPUT PAGE =================
else:

    model = st.session_state["model"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    test_display = st.session_state["test_display"]
    feature_names = st.session_state["feature_names"]
    preprocessor = st.session_state["preprocessor"]
    task = st.session_state["task"]

    col1, col2 = st.columns(2)

    # LEFT
    with col1:
        st.dataframe(test_display)

        st.divider()

        y_pred = model.predict(X_test)

        if task == "Classification":
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
        else:
            st.write("R²:", r2_score(y_test, y_pred))

    # RIGHT
    with col2:

        tab1, tab2 = st.tabs(["Global", "Local"])

        with tab1:
            fig, _ = generate_shap_plots(model, X_test, feature_names=feature_names)
            st.pyplot(fig)

        with tab2:

            with st.form("input_form"):

                input_data = {}

                for i, col in enumerate(feature_names):
                    input_data[col] = st.text_input(col)

                submit = st.form_submit_button("Predict")

                if submit:

                    input_data = {k: float(v) for k, v in input_data.items()}

                    new_df = pd.DataFrame([input_data])

                    new_processed = new_df.values  # already aligned

                    pred = model.predict(new_processed)

                    st.success(f"Prediction: {pred[0]}")

                    _, fig_local = generate_shap_plots(
                        model,
                        X_test,
                        new_processed,
                        feature_names=feature_names
                    )

                    st.pyplot(fig_local)