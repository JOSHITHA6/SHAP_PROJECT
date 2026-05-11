import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import shap
from utils.preprocess import preprocess_data
from BACKEND.model import train_model
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression

st.set_page_config(layout="wide", page_title="SHAP AI Explainability Tool", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}
.explanation-box {
    padding: 15px;
    margin-bottom: 12px;
    background: #ffffff;
    border-radius: 8px;
    border-left: 4px solid;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.prediction-card-light {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e0e0e0;
    margin: 20px 0;
}
.prediction-card-light h1 {
    margin: 0;
    font-size: 32px;
    font-weight: 600;
    color: #333;
}
.metric-card-light {
    background: #ffffff;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e0e0e0;
    margin: 20px 0;
}
.metric-card-light h1 {
    margin: 10px 0 0 0;
    font-size: 36px;
    font-weight: 600;
    color: #2c3e50;
}
.stButton > button {
    background: #667eea;
    color: white;
    border: none;
    padding: 10px 30px;
    border-radius: 8px;
    width: 100%;
}
.stRadio > div {
    display: flex;
    gap: 20px;
    justify-content: center;
}
.axis-label {
    background: #f8f9fa;
    padding: 8px 15px;
    border-radius: 6px;
    margin: 8px 0;
    border-left: 3px solid #667eea;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "input"
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "shap_values_global" not in st.session_state:
    st.session_state.shap_values_global = None
if "shap_explainer" not in st.session_state:
    st.session_state.shap_explainer = None


# ─────────────────────────────────────────────
# SHAP HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_shap_explainer(model, X_background):
    """Return the appropriate SHAP explainer for the model type."""
    model_name = type(model).__name__
    tree_models = ("RandomForestClassifier", "RandomForestRegressor",
                   "GradientBoostingClassifier", "GradientBoostingRegressor",
                   "XGBClassifier", "XGBRegressor",
                   "DecisionTreeClassifier", "DecisionTreeRegressor",
                   "ExtraTreesClassifier", "ExtraTreesRegressor")
    linear_models = ("LinearRegression", "LogisticRegression", "Ridge", "Lasso",
                     "ElasticNet", "SGDClassifier", "SGDRegressor")

    if model_name in tree_models:
        return shap.TreeExplainer(model)
    elif model_name in linear_models:
        return shap.LinearExplainer(model, X_background)
    else:
        # KernelExplainer works for any model but is slow; use a small background
        bg = shap.sample(X_background, min(50, len(X_background)))
        return shap.KernelExplainer(model.predict, bg)


def compute_shap_values(explainer, X, task, model):
    """
    Compute SHAP values and always return a clean 2-D array (n_samples x n_features).

    Handles three formats that different shap versions return:
      - list of arrays  -> old TreeExplainer style
      - 3-D array (n_samples, n_features, n_classes) -> newer shap style
      - 2-D array (n_samples, n_features) -> regression / linear models
    """
    raw = explainer.shap_values(X)
    arr = np.array(raw)

    # Case 1: list of 2-D arrays, one per class -> (n_classes, n_samples, n_features)
    if isinstance(raw, list):
        arr = np.stack(raw, axis=0)
        if arr.shape[0] == 2:
            return arr[1]                      # binary -> class-1
        else:
            best = int(np.argmax(np.abs(arr).mean(axis=(1, 2))))
            return arr[best]

    # Case 2: 3-D array (n_samples, n_features, n_classes) -> newer shap
    if arr.ndim == 3:
        if arr.shape[2] == 2:
            return arr[:, :, 1]               # binary -> class-1
        else:
            best = int(np.argmax(np.abs(arr).mean(axis=(0, 1))))
            return arr[:, :, best]

    # Case 3: already 2-D (n_samples, n_features) -> regression
    return arr


def global_shap_summary(shap_vals, feature_names, top_n=10):
    """
    Mean |SHAP| per feature -> (indices sorted desc, mean_abs_shap, mean_signed_shap).
    """
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim != 2:
        shap_vals = shap_vals.reshape(shap_vals.shape[0], -1)
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    mean_sign = shap_vals.mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:top_n].tolist()
    return top_idx, mean_abs, mean_sign


def local_shap_row(shap_vals, row_idx):
    """Return the SHAP vector for a single row (1-D, length = n_features)."""
    return shap_vals[row_idx]


# ─────────────────────────────────────────────
# PAGE 1 – INPUT
# ─────────────────────────────────────────────

if st.session_state.page == "input":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("🔮 SHAP – AI Model Explainability Tool")
    st.markdown("*Upload any dataset, train a model, and understand WHY it makes predictions*")
    st.markdown("---")

    file = st.file_uploader("📁 Upload CSV File", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.success(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        st.session_state.original_df = df.copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            target = st.selectbox("🎯 Target Column", df.columns)
        with col2:
            unique_count = df[target].nunique()
            task = "Classification" if unique_count <= 10 else "Regression"
            st.info(f"📋 Task: {task} ({unique_count} unique values)")
        with col3:
            if task == "Classification":
                model_type = st.selectbox("🤖 Model", ["Random Forest", "Logistic Regression"])
            else:
                model_type = st.selectbox("🤖 Model", ["Random Forest", "Linear Regression"])

        if st.button("🚀 Run Model", type="primary"):
            with st.spinner("Processing…"):
                start = time.time()

                (X_train, X_test, y_train, y_test, X_test_original,
                 preprocessor, feature_names, original_columns, _) = preprocess_data(df, target)

                test_display = X_test_original.copy()
                test_display[target] = y_test.values

                model = train_model(X_train, y_train, task, model_type)

                # ── Compute SHAP values for the whole test set (global)
                with st.spinner("Computing SHAP values…"):
                    explainer   = get_shap_explainer(model, X_train)
                    shap_values = compute_shap_values(explainer, X_test, task, model)

                st.session_state.update({
                    "model":            model,
                    "X_train":          X_train,
                    "X_test":           X_test,
                    "y_test":           y_test,
                    "test_display":     test_display.reset_index(drop=True),
                    "feature_names":    feature_names,
                    "task":             task,
                    "target_name":      target,
                    "preprocessor":     preprocessor,
                    "original_columns": original_columns,
                    "y_pred":           model.predict(X_test),
                    "shap_values":      shap_values,   # shape (n_test, n_features)
                    "shap_explainer":   explainer,
                })
                st.session_state.page = "output"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2 – OUTPUT
# ─────────────────────────────────────────────

elif st.session_state.page == "output":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("📊 Model Results")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⬅️ Back to Input"):
            st.session_state.page = "input"
            st.rerun()

    st.markdown("---")
    st.subheader("📄 Test Dataset")
    st.dataframe(st.session_state.test_display, height=300, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Model Performance")
    if st.session_state.task == "Classification":
        acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
        st.markdown(f'<div class="metric-card-light"><h3>🎯 Accuracy</h3><h1>{acc:.2%}</h1></div>',
                    unsafe_allow_html=True)
    else:
        r2 = r2_score(st.session_state.y_test, st.session_state.y_pred)
        st.markdown(f'<div class="metric-card-light"><h3>📊 R² Score</h3><h1>{r2:.3f}</h1></div>',
                    unsafe_allow_html=True)

    if st.button("🔍 View Explanations", type="primary"):
        st.session_state.page = "explanation"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 3 – EXPLANATION
# ─────────────────────────────────────────────

elif st.session_state.page == "explanation":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    model           = st.session_state.model
    X_train         = st.session_state.X_train
    X_test          = st.session_state.X_test
    y_test          = st.session_state.y_test
    test_display    = st.session_state.test_display
    feature_names   = st.session_state.feature_names
    task            = st.session_state.task
    preprocessor    = st.session_state.preprocessor
    original_columns= st.session_state.original_columns
    y_pred          = st.session_state.y_pred
    shap_values     = st.session_state.shap_values       # (n_test, n_features)
    explainer       = st.session_state.shap_explainer

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⬅️ Back to Results"):
            st.session_state.page = "output"
            st.rerun()

    st.title("📖 Model Explainability")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        exp_type = st.radio("Select", ["🌍 Global Explanation", "🔍 Local Explanation"], horizontal=True)

    st.markdown("---")


    # ══════════════════════════════════════════
    # GLOBAL EXPLANATION  (mean signed SHAP)
    # ══════════════════════════════════════════
    if exp_type == "🌍 Global Explanation":
        st.subheader("🌍 How Features Impact Predictions (across ALL samples)")

        top_idx, mean_abs, mean_sign = global_shap_summary(shap_values, feature_names, top_n=10)

        top_features  = [feature_names[i] for i in top_idx]
        top_mean_sign = [float(mean_sign[i]) for i in top_idx]   # signed: +ve or -ve
        top_mean_abs  = [float(mean_abs[i])  for i in top_idx]   # for % calculation

        # Color: green = increases prediction, red = decreases
        colors = ['#28a745' if s >= 0 else '#dc3545' for s in top_mean_sign]

        # ── Chart: signed SHAP bars (left = decreases, right = increases)
        fig, ax = plt.subplots(figsize=(13, 8))
        bars = ax.barh(range(len(top_features)), top_mean_sign, color=colors, height=0.6)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=11)  # same as local chart
        ax.axvline(0, color='black', linewidth=1.2)
        ax.set_xlabel("Mean SHAP Value  (← decreases prediction  |  increases prediction →)", fontsize=11)
        ax.set_title("Global Feature Importance via SHAP\n(averaged over all test samples)",
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#28a745', label='Green (right) → INCREASES prediction'),
            Patch(facecolor='#dc3545', label='Red (left)    → DECREASES prediction'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        plt.tight_layout()   # same as local — auto-adjusts margins to fit labels
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="axis-label">📊 <b>X-axis</b> = Mean SHAP Value — bars going <b>RIGHT (green)</b> increase the prediction; bars going <b>LEFT (red)</b> decrease it</div>', unsafe_allow_html=True)
        st.markdown('<div class="axis-label">📈 <b>Y-axis</b> = Features, ranked by how much they influence the model on average</div>', unsafe_allow_html=True)
        st.markdown('<div class="axis-label">📌 <b>% label</b> = that feature\'s share of the total influence (all features together = 100%)</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🧠 Why This Behavior?")
        st.markdown("*Here's a full breakdown of how each feature influences the model's predictions on average:*")

        total_abs_all = float(mean_abs.sum())

        for idx in top_idx[:5]:
            idx      = int(idx)
            feat     = feature_names[idx]
            pct      = (float(mean_abs[idx]) / total_abs_all * 100) if total_abs_all > 0 else 0
            sv       = float(mean_sign[idx])
            avg_val  = float(X_test.iloc[:, idx].mean())   # average feature value across test set

            if sv > 0:
                direction_text = "INCREASES the prediction"
                why = (f"On average across all samples, <b>{feat}</b> has a mean value of <b>{avg_val:.3f}</b>. "
                       f"The model sees this feature as pushing predictions <b>higher</b> — "
                       f"meaning samples with a higher <b>{feat}</b> tend to get a higher predicted output.")
                color = "#28a745"
                icon  = "📈"
            else:
                direction_text = "DECREASES the prediction"
                why = (f"On average across all samples, <b>{feat}</b> has a mean value of <b>{avg_val:.3f}</b>. "
                       f"The model sees this feature as pushing predictions <b>lower</b> — "
                       f"meaning samples with a higher <b>{feat}</b> tend to get a lower predicted output.")
                color = "#dc3545"
                icon  = "📉"

            st.markdown(f"""
<div class="explanation-box" style="border-left-color: {color};">
<b>{icon} {feat}</b><br>
→ <b>📌 Influence Share: {pct:.1f}%</b> &nbsp;— This feature is responsible for <b>{pct:.1f}%</b> of the total influence across all features combined.<br>
→ <b>📉 SHAP Value: {sv:+.4f}</b> &nbsp;— On average, this feature shifts the model's prediction by <b>{sv:+.4f} units</b> from the baseline.<br>
→ <b>🎯 Direction: <span style="color:{color};">{direction_text}</span></b><br>
→ <b>💬 Why?</b> {why}
</div>
""", unsafe_allow_html=True)

        st.info(
            "💡 **Reading guide:**\n\n"
            "**Influence Share %** → How dominant this feature is compared to others. 34% means it drives 34% of all decisions.\n\n"
            "**SHAP Value** → The actual average amount this feature pushes the prediction up (+) or down (−) from the model's baseline prediction.\n\n"
            "**Direction** → Whether higher values of this feature increase or decrease the predicted output on average.\n\n"
            "**Average Value** → The typical value of this feature in your dataset, giving context to the SHAP effect."
        )


    # ══════════════════════════════════════════
    # LOCAL EXPLANATION  (per-row SHAP values)
    # ══════════════════════════════════════════
    else:
        st.subheader("🔍 Explain a Single Prediction")

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            source = st.radio("Input method:", ["📊 Use Test Data", "✏️ Enter New Data"], horizontal=True)

        X_single      = None
        current_pred  = None
        current_actual= None
        local_shap    = None   # 1-D array of length n_features for this row

        if source == "📊 Use Test Data":
            row_num = st.number_input("Row number", 1, len(X_test), 1)
            st.markdown("### 📋 Selected Row")
            st.dataframe(test_display.iloc[[row_num - 1]], use_container_width=True)

            X_single       = X_test.iloc[row_num - 1 : row_num]
            current_pred   = model.predict(X_single)[0]
            current_actual = y_test.iloc[row_num - 1]
            local_shap     = local_shap_row(shap_values, row_num - 1)   # ← per-row SHAP

        else:
            st.markdown("### 📝 Enter New Data")
            new_data = {}
            for feat in original_columns:
                if feat in st.session_state.original_df.columns:
                    unique_vals = st.session_state.original_df[feat].nunique()
                    if unique_vals <= 10:
                        cats = st.session_state.original_df[feat].dropna().unique().tolist()
                        new_data[feat] = st.selectbox(f"📋 {feat}", [str(c) for c in sorted(cats)])
                    else:
                        min_v  = float(st.session_state.original_df[feat].min())
                        max_v  = float(st.session_state.original_df[feat].max())
                        mean_v = float(st.session_state.original_df[feat].mean())
                        new_data[feat] = st.number_input(f"🔢 {feat}", value=mean_v,
                                                         min_value=min_v, max_value=max_v)

            if st.button("🔮 Predict"):
                new_df   = pd.DataFrame([new_data])
                X_proc   = preprocessor.transform(new_df)
                X_single = pd.DataFrame(X_proc, columns=feature_names)

                current_pred = model.predict(X_single)[0]
                st.success(f"✅ Prediction: {current_pred}")

                # Compute SHAP for this new single row
                local_shap = compute_shap_values(explainer, X_single, task, model)[0]

        # ── Display if we have a row to explain
        if X_single is not None and local_shap is not None:

            # Prediction card
            if task == "Classification":
                actual_str = f"<p>Actual: {current_actual}</p>" if current_actual is not None else ""
                st.markdown(f'<div class="prediction-card-light"><h3>🎯 Model Prediction</h3>'
                            f'<h1>{current_pred}</h1>{actual_str}</div>', unsafe_allow_html=True)
            else:
                actual_str = f"<p>Actual: {current_actual:.2f}</p>" if current_actual is not None else ""
                st.markdown(f'<div class="prediction-card-light"><h3>🎯 Model Prediction</h3>'
                            f'<h1>{current_pred:.2f}</h1>{actual_str}</div>', unsafe_allow_html=True)

            # ── Local feature-contribution chart (per-row SHAP)
            st.markdown("### 📊 Feature Contributions for This Prediction")

            top_n      = min(8, len(feature_names))
            local_shap = np.array(local_shap).flatten()          # ensure 1-D
            abs_local  = np.abs(local_shap)
            top_idx_l  = np.argsort(abs_local)[-top_n:].tolist() # plain Python ints
            top_feats_l   = [feature_names[i] for i in top_idx_l]
            top_shap_l    = np.array([local_shap[i] for i in top_idx_l])
            colors_l      = ['#28a745' if v >= 0 else '#dc3545' for v in top_shap_l]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(top_idx_l)), top_shap_l, color=colors_l)
            ax.set_yticks(range(len(top_idx_l)))
            ax.set_yticklabels(top_feats_l)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on this individual prediction)", fontsize=11)
            ax.set_title("Local Feature Contributions\n(specific to the selected row)", fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown('<div class="axis-label">📊 X-axis = SHAP Value: positive = pushes prediction HIGHER, negative = pushes LOWER</div>', unsafe_allow_html=True)
            st.markdown('<div class="axis-label">📈 Y-axis = Individual features for this row</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🧠 Why This Prediction?")
            st.markdown("*Here's why the model made **this specific** prediction for the selected row:*")

            total_abs_l = abs_local.sum()
            top5_l      = np.argsort(abs_local)[::-1][:5]

            for idx in top5_l:
                idx   = int(idx)
                feat  = feature_names[idx]
                sv    = local_shap[idx]
                pct   = (abs(sv) / total_abs_l * 100) if total_abs_l > 0 else 0

                if sv > 0:
                    direction_text = "pushes prediction HIGHER"
                    color = "#28a745"
                    icon  = "📈"
                else:
                    direction_text = "pushes prediction LOWER"
                    color = "#dc3545"
                    icon  = "📉"

                st.markdown(f"""
<div class="explanation-box" style="border-left-color: {color};">
<b>{icon} {feat}</b><br>
→ <b>SHAP Contribution: {pct:.1f}%</b> of this row's total impact<br>
→ {direction_text} <span style="color:{color}; font-size:0.85em;">(SHAP = {sv:+.4f})</span>
</div>
""", unsafe_allow_html=True)

        st.info("💡 **Local vs Global:** Each row has its OWN SHAP values — the percentages and directions "
                "here are specific to the selected sample and WILL differ from the global summary above.")

    st.markdown('</div>', unsafe_allow_html=True)