import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
.block-container {
    max-width: 800px;
    margin: auto;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
.stButton>button {
    width: 100%;
    height: 45px;
    border-radius: 8px;
    background-color: #4a7cff;
    color: white;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1 style='text-align:center;'>SHAP Explainability Tool</h1>", unsafe_allow_html=True)

# ---------- CARD ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

# ---------- FORM ----------
with st.form("shap_form"):

    st.subheader("Upload CSV")
    file = st.file_uploader("", type=["csv"])

    st.subheader("Dataset Preview")
    df = None
    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df.head())
    else:
        st.info("Please upload a dataset to get started.")

    st.subheader("Select Target Column")
    if df is not None:
        target = st.selectbox("", df.columns)
    else:
        target = st.selectbox("", ["Upload dataset first"], disabled=True)

    st.subheader("Select Task")
    task = st.radio("", ["Classification", "Regression"])

    st.subheader("Select Model")
    if task == "Classification":
        model = st.selectbox("", ["Random Forest", "Logistic Regression", "SVM", "XGBoost"])
    else:
        model = st.selectbox("", ["Linear Regression", "Random Forest", "SVR", "XGBoost"])

    run = st.form_submit_button("Run Model")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- RUN LOGIC ----------
if run:
    if file is None:
        st.warning("Please upload a dataset first")
    else:
        st.success("Running model...")