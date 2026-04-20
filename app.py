import streamlit as st
import pandas as pd

from backend.model import train_model
from backend.shap_explainer import explain_model

st.title("SHAP Explainability Tool")

# Upload dataset
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Select target column
    target = st.selectbox("Select Target Column", df.columns)

    # Select task
    task = st.radio("Select Task", ["Classification", "Regression"])

    # Dynamic model selection
    if task == "Classification":
        model_type = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]
        )
    else:
        model_type = st.selectbox(
            "Select Model",
            ["Linear Regression", "Random Forest", "SVR", "XGBoost"]
        )

    if st.button("Run Model"):
        model, X_test = train_model(df, target, task, model_type)

        st.success("Model trained successfully!")

        fig = explain_model(model, X_test, model_type)

        st.pyplot(fig)