import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd

from BACKEND.model import train_model
from BACKEND.shap_explainer import explain_model

st.title("SHAP Explainability Tool")

# Upload dataset
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.write("Dataset Preview:")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    task = st.radio("Select Task", ["Classification", "Regression"])

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
        st.write("Running model...")