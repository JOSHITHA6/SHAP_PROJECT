
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd

st.markdown("""
<style>

/* Center main container */
.block-container {
    max-width: 700px;
    margin: auto;
}

/* Card box */
.custom-box {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

/* Upload box styling */
.stFileUploader {
    background-color: #f1f3f6;
    padding: 15px;
    border-radius: 10px;
}

/* Info message full width */
.full-width {
    width: 100%;
}

/* Title center */
h1 {
    text-align: center;
}

</style>
""", unsafe_allow_html=True)
from BACKEND.model import train_model
from BACKEND.shap_explainer import explain_model

st.title("SHAP Explainability Tool")

# Upload dataset
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.success("Dataset Preview:")
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
        with st.spinner("Training model and generating SHAP explanations..."):
            model = train_model(df, target, model_type)
            fig = explain_model(model, df.drop(columns=[target]), model_type)
        st.pyplot(fig)
       
else:
    st.info("Please upload a dataset to get started.")