import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import shap
import matplotlib.pyplot as plt

# ====================================================
# PAGE CONFIG
# ====================================================
st.set_page_config(layout="wide")
st.title("📊 SHAP Explainability Tool")

# ====================================================
# SESSION STATE
# ====================================================
if "trained" not in st.session_state:
    st.session_state.trained = False

# ====================================================
# LAYOUT
# ====================================================
col1, col2 = st.columns([1,1])

# ====================================================
# LEFT SIDE (INPUT)
# ====================================================
with col1:
    st.subheader("⚙️ Configure SHAP")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        target = st.selectbox("Select Target Column", df.columns)

    task = st.radio("Task", ["Classification", "Regression"])

    model_type = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

    run = st.button("🚀 Run Model")


# ====================================================
# RIGHT SIDE (OUTPUT)
# ====================================================
with col2:
    st.subheader("📊 Output Screen")

    if run and uploaded_file:

        # -------- PREPROCESS --------
        df = pd.get_dummies(df, drop_first=True)

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # -------- MODEL --------
        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=2000)
        else:
            model = RandomForestClassifier()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success("✅ Model Trained Successfully")
        st.write(f"Accuracy: {round(acc*100,2)}%")

        # ====================================================
        # SHAP (FAST VERSION 🔥)
        # ====================================================
        st.markdown("### 🔍 SHAP Explanation")

        # ⚡ Take small sample to avoid lag
        X_sample = X_test[:50]

        try:
            if model_type == "Random Forest":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

            else:
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)

            # -------- GLOBAL EXPLANATION --------
            st.markdown("#### 🌍 Global Explanation")

            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig)

            # -------- LOCAL EXPLANATION --------
            st.markdown("#### 🔍 Local Explanation")

            fig2, ax2 = plt.subplots()
            shap.plots.waterfall(shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_sample[0]
            ), show=False)

            st.pyplot(fig2)

        except Exception as e:
            st.error("SHAP failed (try smaller dataset)")
            st.write(e)

    else:
        st.info("Run the model to see results")