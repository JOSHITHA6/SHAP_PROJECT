import streamlit as st
import pandas as pd

from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots
from utils.preprocess import preprocess_data

from sklearn.metrics import accuracy_score, r2_score

# ====================================================
# CONFIG
# ====================================================
st.set_page_config(layout="wide")

# ====================================================
# 🎨 CSS (ALIGNMENT FIXED)
# ====================================================
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f8fafc;
}

/* Section Box */
.section-box {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    height: 100%;
}

/* Divider */
.divider {
    border-left: 2px solid #e2e8f0;
    height: 100%;
    margin: auto;
}

/* FORCE TOP ALIGNMENT */
[data-testid="column"] > div {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

</style>
""", unsafe_allow_html=True)

# ====================================================
# TITLE
# ====================================================
st.title("📊 SHAP Explainability Tool")

# ====================================================
# LAYOUT (WITH GAP)
# ====================================================
col1, col_gap, col2 = st.columns([1, 0.08, 1])

# ====================================================
# LEFT SIDE
# ====================================================
with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    st.markdown("### ⚙️ Configure SHAP")
    st.markdown("<br>", unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV", type=["csv"])

    df = None
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        target = st.selectbox("Select Target Column", df.columns)

    task = st.radio("Task", ["Classification", "Regression"])

    model_type = st.selectbox(
        "Model",
        ["Random Forest", "Linear/Logistic Regression"]
    )

    run = st.button("🚀 Run Model")

    st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# DIVIDER
# ====================================================
with col_gap:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ====================================================
# RIGHT SIDE
# ====================================================
with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    st.markdown("### 📊 Output")
    st.markdown("<br>", unsafe_allow_html=True)

    if run and df is not None:

        with st.spinner("Running model + SHAP... ⏳"):

            # -------- PREPROCESS --------
            X_train, X_test, y_train, y_test = preprocess_data(df, target)

            # -------- MODEL --------
            model = train_model(X_train, y_train, task, model_type)

            y_pred = model.predict(X_test)

        # -------- METRICS --------
        st.markdown('<div style="margin-top:0px;">', unsafe_allow_html=True)

        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {round(acc*100,2)}%")
        else:
            score = r2_score(y_test, y_pred)
            st.success(f"R² Score: {round(score,3)}")

        st.markdown('</div>', unsafe_allow_html=True)

        # ====================================================
        # SHAP
        # ====================================================
        st.markdown("## 🔍 SHAP Explanation")

        X_sample = X_test.iloc[:30]

        fig_global, fig_local = generate_shap_plots(model, X_sample)

        # -------- TABS --------
        tab1, tab2 = st.tabs(["🌍 Global Explanation", "🔍 Local Explanation"])

        with tab1:
            st.pyplot(fig_global)

        with tab2:
            st.pyplot(fig_local)

    else:
        st.info("Upload dataset and click Run")

    st.markdown('</div>', unsafe_allow_html=True)