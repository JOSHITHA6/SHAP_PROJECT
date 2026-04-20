import streamlit as st
import pandas as pd

from backend.model import train_model
from backend.shap_explainer import generate_shap_plots
from utils.preprocess import preprocess_data

from sklearn.metrics import accuracy_score, r2_score

# ====================================================
# CONFIG
# ====================================================
st.set_page_config(layout="wide")
st.title("📊 SHAP Explainability Tool")

col1, col2 = st.columns([1, 1])

# ====================================================
# LEFT SIDE (INPUT)
# ====================================================
with col1:
    st.subheader("⚙️ Configure SHAP")

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


# ====================================================
# RIGHT SIDE (OUTPUT)
# ====================================================
with col2:
    st.subheader("📊 Output Screen")

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
            # 🔥 SHAP (FAST + SAFE)
            # ====================================================
            st.markdown("## 🔍 SHAP Explanation")

            # ⚡ Sample for speed
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