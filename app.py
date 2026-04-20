import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>

/* Page width */
.block-container {
    max-width: 1100px;
    margin: auto;
}

/* Neon card style */
.card {
    background: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0px 0px 15px rgba(0, 123, 255, 0.3);
}

/* Section titles */
.section-title {
    font-weight: 600;
    margin-bottom: 10px;
    color: #2c3e50;
}

/* Blue theme inputs */
.stSelectbox, .stRadio {
    border-radius: 8px;
}

/* Run button (green) */
.stButton>button {
    width: 100%;
    height: 45px;
    border-radius: 8px;
    background-color: #28a745;
    color: white;
    font-size: 16px;
    font-weight: bold;
}

/* Output box */
.output-box {
    background-color: #f4f8ff;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #d0e3ff;
}

</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1 style='text-align:center;'>SHAP Explainability Tool</h1>", unsafe_allow_html=True)

# ---------- LAYOUT ----------
col1, col2 = st.columns(2)

# ---------- LEFT: INPUT ----------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">CONFIGURE SHAP</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload Dataset", type=["csv"])

    df = None
    if file is not None:
        df = pd.read_csv(file)
        st.success("Dataset Loaded")
        st.dataframe(df.head())

    st.markdown("Select Target Column")
    if df is not None:
        target = st.selectbox("", df.columns)
    else:
        target = st.selectbox("", ["Upload dataset first"], disabled=True)

    st.markdown("Select Task")
    task = st.radio("", ["Classification", "Regression"])

    st.markdown("Select Model")
    if task == "Classification":
        model = st.selectbox("", ["Random Forest", "Logistic Regression", "SVM", "XGBoost"])
    else:
        model = st.selectbox("", ["Linear Regression", "Random Forest", "SVR", "XGBoost"])

    run = st.button("Run Model")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------- RIGHT: OUTPUT ----------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">OUTPUT SCREEN</div>', unsafe_allow_html=True)

    if run and file is not None:
        st.success("Model executed successfully!")

        st.markdown('<div class="output-box">', unsafe_allow_html=True)
        st.write("Prediction will appear here")
        st.markdown('</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Local Explainability", "Global Explainability"])

        with tab1:
            st.write("Local SHAP explanation will appear here")

        with tab2:
            st.write("Global SHAP explanation will appear here")

    else:
        st.info("Run the model to see results")

    st.markdown('</div>', unsafe_allow_html=True)