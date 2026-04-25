import streamlit as st
import pandas as pd
import numpy as np

from utils.preprocess import preprocess_data
from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots

from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(layout="wide")

root = st.container()

# ---------------- STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "input"

# ================= INPUT PAGE =================
if st.session_state.page == "input":

    with root:

        st.title("SHAP - AI Model Explainability Tool")

        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            df = pd.read_csv(file)

            target = st.selectbox("Target Column", df.columns)

            task = "Classification" if df[target].nunique() <= 10 else "Regression"
            st.write(f"Detected Task: {task}")

            model_type = st.selectbox(
                "Model",
                ["Random Forest", "Logistic Regression"]
                if task == "Classification"
                else ["Random Forest", "Linear Regression"]
            )

            if st.button("🚀 Run Model"):

                (
                    X_train, X_test,
                    y_train, y_test,
                    X_test_original,
                    preprocessor,
                    feature_names,
                    original_columns,
                    df_full
                ) = preprocess_data(df, target)

                # include target column
                test_display = X_test_original.copy()
                test_display[target] = y_test.values

                model = train_model(X_train, y_train, task, model_type)

                st.session_state.update({
                    "model": model,
                    "X_test": X_test,
                    "y_test": y_test,
                    "test_display": test_display.reset_index(drop=True),
                    "feature_names": feature_names,
                    "task": task,
                    "preprocessor": preprocessor,
                    "original_columns": original_columns,
                    "df_full": df_full,
                    "target": target
                })

                st.session_state.page = "output"
                st.rerun()

# ================= OUTPUT PAGE =================
elif st.session_state.page == "output":

    with root:

        st.title("📊 Model Output Dashboard")

        if st.button("⬅️ Back"):
            st.session_state.page = "input"
            st.rerun()

        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        test_display = st.session_state.test_display
        feature_names = st.session_state.feature_names
        task = st.session_state.task

        left, _, right = st.columns([1.1, 0.1, 1.4])

        # -------- LEFT --------
        with left:
            st.subheader("📄 Test Dataset (20%)")
            st.dataframe(test_display, height=350)

            st.markdown("---")
            st.subheader("📈 Model Performance")

            y_pred = model.predict(X_test)

            if task == "Classification":
                st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            else:
                st.success(f"R² Score: {r2_score(y_test, y_pred):.2f}")

        # -------- RIGHT --------
        with right:

            tab1, tab2 = st.tabs(["🌍 Global Explainability", "🔍 Local Explainability"])

            # ---------- GLOBAL ----------
            with tab1:

                fig, _, shap_vals = generate_shap_plots(
                    model, X_test[:100], feature_names=feature_names, task=task
                )

                st.pyplot(fig)

                st.markdown("### 📌 Global Insights (Simple Explanation)")

                vals = np.abs(shap_vals).mean(axis=0)
                perc = (vals / vals.sum()) * 100

                top_idx = np.argsort(vals)[::-1][:5]

                for i in top_idx:
                    direction = "increases" if np.mean(shap_vals[:, i]) > 0 else "decreases"

                    st.markdown(f"""
                    <div style="
                        padding:12px;
                        margin-bottom:10px;
                        background:#f8f9fa;
                        border-radius:10px;
                        border-left:6px solid {'green' if direction=='increases' else 'red'};
                        font-size:15px;">
                    <b>{feature_names[i]}</b><br>
                    👉 Contributes <b>{perc[i]:.1f}%</b><br>
                    👉 Usually <b>{direction}</b> the prediction
                    </div>
                    """, unsafe_allow_html=True)

            # ---------- LOCAL ----------
            with tab2:

                st.markdown("### Select Row for Explanation")

                row = st.number_input("Row Number", 1, len(X_test), 1)

                # show selected row
                st.markdown("### 📄 Selected Row Data")
                st.dataframe(test_display.iloc[[row-1]])

                X_single = X_test[row-1:row]

                _, fig_local, shap_vals = generate_shap_plots(
                    model, X_test[:100], X_single, feature_names, task
                )

                st.pyplot(fig_local)

                # ✅ NEW ADDITION (AS YOU ASKED)
                st.markdown("**X-axis:** Impact on Prediction")
                st.markdown("**Y-axis:** Features")

                st.markdown("### 📌 Why this prediction?")

                shap_row = shap_vals[row-1]

                vals = np.abs(shap_row)
                perc = (vals / vals.sum()) * 100

                pairs = sorted(
                    zip(feature_names, shap_row, perc),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                for feat, val, p in pairs[:5]:
                    direction = "increases" if val > 0 else "decreases"

                    st.markdown(f"""
                    <div style="
                        padding:12px;
                        margin-bottom:10px;
                        background:#f8f9fa;
                        border-radius:10px;
                        border-left:6px solid {'green' if val>0 else 'red'};
                        font-size:15px;">
                    <b>{feat}</b><br>
                    👉 Contribution: <b>{p:.1f}%</b><br>
                    👉 This feature <b>{direction}</b> the prediction
                    </div>
                    """, unsafe_allow_html=True)