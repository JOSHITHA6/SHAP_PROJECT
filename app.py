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
        target = st.session_state.target

        left, _, right = st.columns([1.1, 0.1, 1.4])

        # -------- LEFT --------
        with left:
            st.subheader("📄 Test Dataset (20%)")
            st.dataframe(test_display, height=350)

            st.markdown("---")
            st.subheader("📈 Model Performance")

            y_pred = model.predict(X_test)

            if task == "Classification":
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Accuracy: {acc:.2f}")
            else:
                r2 = r2_score(y_test, y_pred)
                st.success(f"R² Score: {r2:.2f}")

        # -------- RIGHT --------
        with right:

            tab1, tab2 = st.tabs(["🌍 Global Explainability", "🔍 Local Explainability"])

            # ---------- GLOBAL ----------
            with tab1:

                fig, _, shap_vals = generate_shap_plots(
                    model, X_test[:100], feature_names=feature_names, task=task
                )

                st.pyplot(fig)

                st.markdown("### 🎯 Overall Model Behavior")

                if task == "Classification":
                    majority = (model.predict(X_test) == 1).mean()
                    label = "YES" if majority >= 0.5 else "NO"
                    st.success(f"👉 Model tends to predict: {label}")
                else:
                    avg = model.predict(X_test).mean()
                    st.success(f"👉 Average prediction: {round(avg,3)}")

                st.markdown("### 📌 Why this behavior?")

                vals = np.abs(shap_vals).mean(axis=0)
                perc = (vals / vals.sum()) * 100

                top_idx = np.argsort(vals)[::-1][:5]

                for i in top_idx:
                    direction = "increases" if np.mean(shap_vals[:, i]) > 0 else "decreases"

                    st.markdown(f"""
                    <div style="padding:10px;background:#f8f9fa;margin-bottom:8px;border-radius:8px;">
                    <b>{feature_names[i]}</b><br>
                    {perc[i]:.1f}% contribution → {direction}
                    </div>
                    """, unsafe_allow_html=True)

            # ---------- LOCAL ----------
            with tab2:

                row = st.number_input("Select Row", 1, len(X_test), 1)

                st.markdown("### 📄 Selected Row Data")
                st.dataframe(test_display.iloc[[row-1]])

                X_single = X_test[row-1:row]

                pred = model.predict(X_single)[0]
                actual = y_test.iloc[row-1]

                st.markdown("### 🎯 Prediction")

                if task == "Classification":

                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(X_single)[0]
                        confidence = np.max(prob)
                    else:
                        confidence = 0.7

                    pred_label = "YES" if pred == 1 else "NO"
                    actual_label = "YES" if actual == 1 else "NO"

                    if pred == actual:
                        st.success(f"✔ {pred_label} (Confidence: {confidence:.2f})")
                    else:
                        st.warning(f"⚠️ {pred_label} (Confidence: {confidence:.2f})")
                        st.info(f"Actual: {actual_label}")

                else:
                    st.success(f"{round(pred,3)} (Actual: {round(actual,3)})")

                # SHAP LOCAL
                _, fig_local, shap_vals = generate_shap_plots(
                    model, X_test[:100], X_single, feature_names, task
                )

                st.pyplot(fig_local)

                # ✅ ADDED AS YOU ASKED
                st.markdown("**X-axis = Impact on Prediction**")
                st.markdown("**Y-axis = Features**")

                st.markdown("### 📌 Why this prediction?")

                shap_row = shap_vals[row-1]

                vals = np.abs(shap_row)
                perc = (vals / vals.sum()) * 100

                for feat, val, p in sorted(zip(feature_names, shap_row, perc), key=lambda x: abs(x[1]), reverse=True)[:5]:
                    st.markdown(f"{feat}: {p:.1f}% → {'increase' if val>0 else 'decrease'}")