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
# CSS
# ====================================================
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}

.section-box {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    height: 100%;
}

.divider {
    border-left: 2px solid #e2e8f0;
    height: 100%;
    margin: auto;
}

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
# LAYOUT
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

        # ✅ FULL DATASET (SCROLLABLE)
        st.dataframe(df, use_container_width=True, height=400)

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
        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {round(acc*100,2)}%")
        else:
            score = r2_score(y_test, y_pred)
            st.success(f"R² Score: {round(score,3)}")

        # ====================================================
        # SHAP
        # ====================================================
        st.markdown("## 🔍 SHAP Explanation")

        # GLOBAL SAMPLE
        X_sample = X_test.iloc[:30]

        tab1, tab2 = st.tabs(["🌍 Global Explanation", "🔍 Local Explanation"])

        # ====================================================
        # 🌍 GLOBAL
        # ====================================================
        with tab1:
            fig_global, _ = generate_shap_plots(model, X_sample, None)
            st.pyplot(fig_global)

        # ====================================================
        # 🔍 LOCAL (USER FRIENDLY INDEXING)
        # ====================================================
        with tab2:

            total_rows = len(X_test)

            st.markdown(f"Select row between **1 and {total_rows}**")

            row_number = st.number_input(
                "Row Number",
                min_value=1,
                max_value=total_rows,
                value=1,
                step=1
            )

            # 🔥 Convert to 0-based index
            row_index = row_number - 1

            X_single = X_test.iloc[[row_index]]

            # Optional preview
            st.markdown("### Selected Row Preview")
            st.dataframe(X_single)

            _, fig_local = generate_shap_plots(model, X_sample, X_single)

            st.pyplot(fig_local)

    else:
        st.info("Upload dataset and click Run")

    st.markdown('</div>', unsafe_allow_html=True)