import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>

/* Main container width */
.block-container {
    max-width: 1100px;
    margin: auto;
}

/* OUTER NEON BOX */
.outer-box {
    background: white;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 0 20px rgba(0, 123, 255, 0.4);
}

/* Divider line */
.divider {
    border-left: 2px solid #e0e6f0;
    height: 100%;
}

/* Section titles */
.section-title {
    font-weight: 600;
    margin-bottom: 10px;
    color: #2c3e50;
}

/* Run button */
.stButton>button {
    background-color: #28a745;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 16px;
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

# ---------- OUTER BOX ----------
st.markdown('<div class="outer-box">', unsafe_allow_html=True)

col1, col_mid, col2 = st.columns([1, 0.05, 1])

# ---------- LEFT (INPUT) ----------
with col1:
    st.markdown('<div class="section-title">CONFIGURE SHAP</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload Dataset", type=["csv"])

    df = None
    if file is not None:
        df = pd.read_csv(file)

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


# ---------- MIDDLE DIVIDER ----------
with col_mid:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ---------- RIGHT (OUTPUT) ----------
with col2:
    st.markdown('<div class="section-title">OUTPUT SCREEN</div>', unsafe_allow_html=True)

    if run and file is not None:
        st.success("Model executed successfully!")

        st.markdown('<div class="output-box">', unsafe_allow_html=True)
        st.write("Prediction will appear here")
        st.markdown('</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Local Explainability", "Global Explainability"])

        with tab1:
            st.write("Local SHAP explanation here")

        with tab2:
            st.write("Global SHAP explanation here")

    else:
        st.info("Run the model to see results")

# ---------- CLOSE BOX ----------
st.markdown('</div>', unsafe_allow_html=True)