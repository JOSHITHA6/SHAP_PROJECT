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
    Compute SHAP values and return a 2-D array (n_samples × n_features).
    For multi-class classification we take the class with the highest
    mean absolute SHAP across all samples (or class index 1 for binary).
    """
    raw = explainer.shap_values(X)

    # TreeExplainer returns a list for classifiers
    if isinstance(raw, list):
        if len(raw) == 2:
            # Binary classification → use class-1 shap values
            return np.array(raw[1])
        else:
            # Multi-class → stack and take mean-abs across classes
            stacked = np.stack(raw, axis=0)          # (n_classes, n_samples, n_features)
            abs_mean = np.abs(stacked).mean(axis=1)  # (n_classes, n_features)
            best_class = int(np.argmax(abs_mean.sum(axis=1)))
            return np.array(raw[best_class])

    return np.array(raw)


def global_shap_summary(shap_vals, feature_names, top_n=10):
    """
    Mean |SHAP| per feature  →  (indices sorted desc, mean_abs_shap, mean_signed_shap).
    mean_signed_shap tells us the average direction (+/-).
    """
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    mean_sign = shap_vals.mean(axis=0)
    top_idx   = [int(i) for i in np.argsort(mean_abs)[::-1][:top_n]]
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
    # GLOBAL EXPLANATION  (mean |SHAP| values)
    # ══════════════════════════════════════════
    if exp_type == "🌍 Global Explanation":
        st.subheader("🌍 How Features Impact Predictions (across ALL samples)")

        top_idx, mean_abs, mean_sign = global_shap_summary(shap_values, feature_names, top_n=10)

        top_features    = [feature_names[i] for i in top_idx]
        top_mean_abs    = mean_abs[top_idx]
        top_mean_sign   = mean_sign[top_idx]

        # Color by the SIGN of the mean SHAP value
        colors = ['#28a745' if s >= 0 else '#dc3545' for s in top_mean_sign]

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(range(len(top_features)), top_mean_abs, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel("Mean |SHAP Value| (average impact on model output)", fontsize=12)
        ax.set_title("Global Feature Importance via SHAP\n(averaged over all test samples)", fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        total = top_mean_abs.sum()
        for bar, imp in zip(bars, top_mean_abs):
            width = bar.get_width()
            pct   = (imp / total) * 100 if total > 0 else 0
            ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{pct:.1f}%', va='center', fontsize=9)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#28a745', label='🟢 On average INCREASES prediction'),
            Patch(facecolor='#dc3545', label='🔴 On average DECREASES prediction'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="axis-label">📊 X-axis = Mean |SHAP Value| (average magnitude of impact)</div>', unsafe_allow_html=True)
        st.markdown('<div class="axis-label">📈 Y-axis = Features ranked by importance</div>', unsafe_allow_html=True)
        st.markdown('<div class="axis-label">🎨 Green = on average raises the prediction | Red = on average lowers the prediction</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🧠 Why This Behavior?")
        st.markdown("*Here's what the model learned from your data (averaged across all samples):*")

        total_abs = mean_abs.sum()
        for rank, idx in enumerate(top_idx[:5]):
            idx   = int(idx)
            feat  = feature_names[idx]
            pct   = (mean_abs[idx] / total_abs * 100) if total_abs > 0 else 0
            sign  = mean_sign[idx]

            if sign > 0:
                direction_text = "INCREASES the prediction"
                color = "#28a745"
                icon  = "📈"
            else:
                direction_text = "DECREASES the prediction"
                color = "#dc3545"
                icon  = "📉"

            st.markdown(f"""
<div class="explanation-box" style="border-left-color: {color};">
<b>{icon} {feat}</b><br>
→ <b>Contribution: {pct:.1f}%</b> of total impact<br>
→ This feature <b style="color:{color};">{direction_text}</b>
</div>
""", unsafe_allow_html=True)

        st.info("💡 **How to read this:** Contributions are computed using real SHAP values averaged over "
                "all test samples. Green = on average pushes predictions up, Red = pushes them down.")


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
            abs_local  = np.abs(local_shap)
            top_idx_l  = [int(i) for i in np.argsort(abs_local)[-top_n:]]   # plain Python ints
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
                val   = X_single.iloc[0, idx] if idx < len(X_single.columns) else "N/A"
                val_display = f"{val:.3f}" if isinstance(val, float) else str(val)

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
→ <b>Value:</b> {val_display}<br>
→ <b>SHAP Contribution: {pct:.1f}%</b> of this row's total impact<br>
→ {direction_text} <span style="color:{color}; font-size:0.85em;">(SHAP = {sv:+.4f})</span>
</div>
""", unsafe_allow_html=True)

        st.info("💡 **Local vs Global:** Each row has its OWN SHAP values — the percentages and directions "
                "here are specific to the selected sample and WILL differ from the global summary above.")

    st.markdown('</div>', unsafe_allow_html=True)