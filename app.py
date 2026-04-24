import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

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

    if file is not None:
        st.session_state["df"] = pd.read_csv(file)

    df = st.session_state.get("df", None)

    if df is not None:
        st.dataframe(df, use_container_width=True, height=400)
        target = st.selectbox("Select Target Column", df.columns)
    else:
        target = None

    task = st.radio("Task", ["Classification", "Regression"])

    model_type = st.selectbox(
        "Model",
        ["Random Forest", "Linear/Logistic Regression"]
    )

    if st.button("🚀 Run Model"):
        st.session_state["run"] = True

    run = st.session_state.get("run", False)

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

    if run and df is not None and target is not None:

        if "model" not in st.session_state:

            with st.spinner("Running model + SHAP... ⏳"):

                X_train, X_test, y_train, y_test = preprocess_data(df, target)
                model = train_model(X_train, y_train, task, model_type)

                st.session_state["model"] = model
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test

        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        y_pred = model.predict(X_test)

        # ====================================================
        # METRICS
        # ====================================================
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

        X_sample = X_test.sample(min(30, len(X_test)), random_state=42)

        tab1, tab2 = st.tabs(["🌍 Global Explanation", "🔍 Local Explanation"])

        # ====================================================
        # 🌍 GLOBAL (BAR PLOT + INSIGHTS)
        # ====================================================
        with tab1:

            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample, check_additivity=False)

            # -------- Mean SHAP ----------
            mean_shap = shap_values.values.mean(axis=0)
            abs_shap = np.abs(shap_values.values).mean(axis=0)

            feature_names = X_sample.columns

            # -------- BAR PLOT ----------
            fig_bar = plt.figure()
            sorted_idx = np.argsort(abs_shap)

            plt.barh(feature_names[sorted_idx], abs_shap[sorted_idx])
            plt.xlabel("Mean |SHAP Value| (Impact)")
            plt.title("Feature Importance (Global)")
            st.pyplot(fig_bar)

            # -------- PERCENT ----------
            total = abs_shap.sum()
            percent_contribution = (abs_shap / total) * 100

            # -------- TOP K ----------
            TOP_K = 5
            sorted_idx_desc = np.argsort(abs_shap)[::-1][:TOP_K]

            st.markdown("### 📌 Key Feature Insights")

            for i in sorted_idx_desc:
                feature = feature_names[i]
                direction = mean_shap[i]
                percent = percent_contribution[i]

                if direction > 0:
                    st.write(f"🔺 {feature}: **{percent:.1f}%** contribution → increases prediction")
                else:
                    st.write(f"🔻 {feature}: **{percent:.1f}%** contribution → decreases prediction")

            # -------- GUIDE ----------
            st.markdown("""
### 📈 Interpretation Guide

- Bars show how important each feature is  
- Percentage shows contribution strength  
- 🔺 means increases prediction  
- 🔻 means decreases prediction  
""")

        # ====================================================
        # 🔍 LOCAL
        # ====================================================
        with tab2:

            total_rows = len(X_test)

            st.markdown(f"""
📊 Total rows in dataset: **{len(df)}**  
🧪 Rows used for SHAP (test set): **{total_rows}**

Select row from **test dataset** (1 to {total_rows})
""")

            row_number = st.number_input(
                "Row Number",
                min_value=1,
                max_value=total_rows,
                value=st.session_state.get("row_number", 1),
                step=1,
                key="row_number"
            )

            row_index = row_number - 1
            X_single = X_test.iloc[[row_index]]

            st.markdown("### Selected Row Preview")
            st.dataframe(X_single)

            _, fig_local = generate_shap_plots(model, X_sample, X_single)
            st.pyplot(fig_local)

    else:
        st.info("Upload dataset and click Run")

    st.markdown('</div>', unsafe_allow_html=True)