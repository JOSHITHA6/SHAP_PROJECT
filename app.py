import streamlit as st
import pandas as pd

from backend.model import train_model
from backend.shap_explainer import get_shap_values

st.set_page_config(layout="wide")

# Title
st.title("SHAP – Explains ML Models")
st.caption("Explain machine learning model predictions using SHapley values")

# Layout
col1, col2 = st.columns(2)

# ---------------- LEFT SIDE ---------------- #
with col1:
    st.subheader("CONFIGURE SHAP")

    # Upload Dataset
    uploaded_file = st.file_uploader("Upload Dataset")

    # Task selection
    task = st.radio("Select Your Task", ["Classification", "Regression"])

    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "Linear Regression"]
    )

    run_button = st.button("Run")

# ---------------- RIGHT SIDE ---------------- #
with col2:
    st.subheader("OUTPUT SCREEN")

    st.markdown("### OUTPUT PREDICTION")
    prediction_placeholder = st.empty()

    tab1, tab2 = st.tabs(["Local Explainability", "Global Explainability"])

    with tab1:
        local_placeholder = st.empty()

    with tab2:
        global_placeholder = st.empty()


# ---------------- LOGIC ---------------- #
if run_button and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    model, X_test = train_model(df, task, model_type)

    prediction_placeholder.success("Model trained successfully!")

    shap_values, plot = get_shap_values(model, X_test, task)

    with tab1:
        st.pyplot(plot)

    with tab2:
        st.write("Global explanation coming soon...")