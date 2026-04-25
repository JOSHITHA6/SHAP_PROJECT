import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.preprocess import preprocess_data
from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots

from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(layout="wide")
st.title("SHAP - AI Model Explainability Tool")

# ================= SESSION =================
if "page" not in st.session_state:
    st.session_state.page = "input"

# ================= INPUT PAGE =================
if st.session_state.page == "input":

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        target = st.selectbox("Target Column", df.columns)

        X = df.drop(columns=[target])

        num_cols = X.select_dtypes(include=['int64','float64']).columns
        cat_cols = X.select_dtypes(include=['object','category']).columns

        task = "Classification" if df[target].nunique() <= 10 else "Regression"
        st.write(f"Detected Task: {task}")

        model_type = st.selectbox(
            "Model",
            ["Random Forest", "Logistic Regression"]
            if task == "Classification"
            else ["Random Forest", "Linear Regression"]
        )

        st.markdown("## 🔢 Enter Input Data")

        # 🔥 Dynamic input form
        with st.form("input_form"):

            input_data = {}
            cols = st.columns(2)

            for i, col in enumerate(X.columns):

                with cols[i % 2]:

                    if col in cat_cols:
                        input_data[col] = st.selectbox(col, df[col].dropna().unique())
                    else:
                        input_data[col] = st.text_input(col)

            submit = st.form_submit_button("Run Model")

        if submit:
            try:
                clean = {}

                for col in input_data:
                    if col in num_cols:
                        clean[col] = float(str(input_data[col]).strip())
                    else:
                        clean[col] = input_data[col]

                new_input_df = pd.DataFrame([clean])

                (
                    X_train, X_test,
                    y_train, y_test,
                    X_test_original,
                    preprocessor,
                    feature_names
                ) = preprocess_data(df, target)

                model = train_model(X_train, y_train, task, model_type)

                new_processed = preprocessor.transform(new_input_df)

                pred = model.predict(new_processed)

                st.session_state.update({
                    "model": model,
                    "X_test": X_test,
                    "y_test": y_test,
                    "test_display": X_test_original.reset_index(drop=True),
                    "feature_names": feature_names,
                    "task": task,
                    "new_processed": new_processed,
                    "prediction": pred[0]
                })

                st.session_state.page = "output"
                st.rerun()

            except:
                st.error("⚠️ Please enter valid numeric values")

# ================= OUTPUT PAGE =================
else:

    if st.button("⬅️ Back"):
        st.session_state.page = "input"
        st.rerun()

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    test_display = st.session_state.test_display
    feature_names = st.session_state.feature_names
    task = st.session_state.task
    new_processed = st.session_state.new_processed
    prediction = st.session_state.prediction

    left, _, right = st.columns([1.1, 0.1, 1.4])

    # LEFT PANEL
    with left:
        st.subheader("📄 Test Data (20%)")
        st.dataframe(test_display, height=300)

        y_pred = model.predict(X_test)

        if task == "Classification":
            st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        else:
            st.success(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # RIGHT PANEL
    with right:

        tab1, tab2 = st.tabs(["🌍 Global", "🔍 Local"])

        # GLOBAL
        with tab1:

            fig, _, shap_vals = generate_shap_plots(
                model, X_test[:100], feature_names=feature_names, task=task
            )
            st.pyplot(fig)

            st.markdown("### 📌 Key Insights")

            vals = np.abs(shap_vals).mean(axis=0)
            perc = vals / vals.sum() * 100

            for i in np.argsort(vals)[::-1][:5]:
                direction = "⬆️ increases" if np.mean(shap_vals[:, i]) > 0 else "⬇️ decreases"

                st.markdown(f"""
                <div style="padding:10px;border-radius:8px;margin-bottom:8px;background:#f8f9fa;">
                <b>{feature_names[i]}</b><br>
                Contribution: {perc[i]:.1f}%<br>
                Effect: {direction}
                </div>
                """, unsafe_allow_html=True)

        # LOCAL
        with tab2:

            st.success(f"Prediction: {prediction}")

            _, fig_local, shap_vals = generate_shap_plots(
                model, X_test[:100], new_processed, feature_names, task
            )

            st.pyplot(fig_local)

            st.markdown("### 📌 Explanation")

            values = shap_vals.mean(axis=0)

            explanations = sorted(
                zip(feature_names, values),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            for feat, val in explanations[:5]:
                direction = "⬆️ increases" if val > 0 else "⬇️ decreases"

                st.markdown(f"""
                <div style="padding:10px;border-radius:8px;margin-bottom:8px;background:#f1f1f1;">
                <b>{feat}</b><br>
                Impact: {val:.2f}<br>
                Effect: {direction}
                </div>
                """, unsafe_allow_html=True)