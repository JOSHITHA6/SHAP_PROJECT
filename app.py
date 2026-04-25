import streamlit as st
import pandas as pd
import numpy as np

from utils.preprocess import preprocess_data
from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots

from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(layout="wide")

# ---------------- STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "input"

# ---------------- INPUT PAGE ----------------
if st.session_state.page == "input":

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

            model = train_model(X_train, y_train, task, model_type)

            st.session_state.update({
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "test_display": X_test_original.reset_index(drop=True),
                "feature_names": feature_names,
                "task": task,
                "preprocessor": preprocessor,
                "original_columns": original_columns,
                "df_full": df_full
            })

            st.session_state.page = "output"
            st.rerun()

# ---------------- OUTPUT PAGE ----------------
else:

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
    preprocessor = st.session_state.preprocessor
    original_columns = st.session_state.original_columns
    df_full = st.session_state.df_full

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

            vals = np.abs(shap_vals).mean(axis=0)
            perc = vals / vals.sum() * 100

            st.markdown("### 📌 Key Feature Insights")

            for i in np.argsort(vals)[::-1][:5]:
                direction = "⬆️ increases" if np.mean(shap_vals[:, i]) > 0 else "⬇️ decreases"

                st.markdown(f"""
                <div style="padding:10px;margin-bottom:8px;background:#f8f9fa;border-radius:8px;">
                <b>{feature_names[i]}</b><br>
                Contribution: {perc[i]:.1f}%<br>
                Effect: {direction}
                </div>
                """, unsafe_allow_html=True)

        # ---------- LOCAL ----------
        with tab2:

            option = st.radio("Choose Option", ["Select Row", "Enter New Data"])

            # SELECT ROW
            if option == "Select Row":

                row = st.number_input("Row Number", 1, len(X_test), 1)

                X_single = X_test[row-1:row]

                _, fig_local, _ = generate_shap_plots(
                    model, X_test[:100], X_single, feature_names, task
                )

                st.pyplot(fig_local)

            # ENTER NEW DATA
            else:

                st.markdown("### Enter Values")

                with st.form("new_input"):

                    input_data = {}
                    cols = st.columns(2)

                    for i, col in enumerate(original_columns):

                        with cols[i % 2]:

                            dtype = df_full[col].dtype

                            if dtype == "object":
                                input_data[col] = st.selectbox(col, df_full[col].dropna().unique())

                            elif set(df_full[col].dropna().unique()).issubset({0,1}):
                                input_data[col] = st.selectbox(col, [0,1])
                                st.caption("0 → Not Present, 1 → Present")

                            else:
                                input_data[col] = st.text_input(col)

                    submit = st.form_submit_button("Predict")

                if submit:
                    try:
                        clean = {}

                        for col in input_data:
                            if df_full[col].dtype != "object":
                                clean[col] = float(str(input_data[col]).strip())
                            else:
                                clean[col] = input_data[col]

                        new_df = pd.DataFrame([clean])
                        new_processed = preprocessor.transform(new_df)

                        pred = model.predict(new_processed)

                        st.success(f"Prediction: {pred[0]}")

                        _, fig_local, _ = generate_shap_plots(
                            model, X_test[:100], new_processed, feature_names, task
                        )

                        st.pyplot(fig_local)

                    except:
                        st.error("⚠️ Enter valid numeric values")