import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from utils.preprocess import preprocess_data
from BACKEND.model import train_model

from sklearn.metrics import accuracy_score, r2_score

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

# ================= PAGE 1: INPUT =================
if st.session_state.page == "input":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("🔮 SHAP - AI Model Explainability Tool")
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
            st.info(f"📋 Task: {task} ({unique_count} values)")
        with col3:
            if task == "Classification":
                model_type = st.selectbox("🤖 Model", ["Random Forest", "Logistic Regression"])
            else:
                model_type = st.selectbox("🤖 Model", ["Random Forest", "Linear Regression"])
        
        if st.button("🚀 Run Model", type="primary"):
            with st.spinner("Processing..."):
                start = time.time()
                (X_train, X_test, y_train, y_test, X_test_original, 
                 preprocessor, feature_names, original_columns, _) = preprocess_data(df, target)
                
                test_display = X_test_original.copy()
                test_display[target] = y_test.values
                model = train_model(X_train, y_train, task, model_type)
                
                st.session_state.update({
                    "model": model, "X_test": X_test, "y_test": y_test,
                    "test_display": test_display.reset_index(drop=True),
                    "feature_names": feature_names, "task": task,
                    "target_name": target, "preprocessor": preprocessor,
                    "original_columns": original_columns, "y_pred": model.predict(X_test)
                })
                st.session_state.page = "output"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ================= PAGE 2: OUTPUT =================
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
            st.markdown(f'<div class="metric-card-light"><h3>🎯 Accuracy</h3><h1>{acc:.2%}</h1></div>', unsafe_allow_html=True)
        else:
            r2 = r2_score(st.session_state.y_test, st.session_state.y_pred)
            st.markdown(f'<div class="metric-card-light"><h3>📊 R² Score</h3><h1>{r2:.3f}</h1></div>', unsafe_allow_html=True)
        
        if st.button("🔍 View Explanations", type="primary"):
            st.session_state.page = "explanation"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ================= PAGE 3: EXPLANATION =================
elif st.session_state.page == "explanation":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    test_display = st.session_state.test_display
    feature_names = st.session_state.feature_names
    task = st.session_state.task
    preprocessor = st.session_state.preprocessor
    original_columns = st.session_state.original_columns
    y_pred = st.session_state.y_pred
    
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
    
    # ========== GLOBAL EXPLANATION (FIXED - NO SHAP) ==========
    if exp_type == "🌍 Global Explanation":
        st.subheader("🌍 How Features Impact Predictions")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.argsort(importances)[::-1][:10]
            colors = ['#28a745' if i < 3 else '#6c757d' for i in range(len(indices))]
            ax.barh(range(len(indices)), importances[indices], color=colors)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel("Feature Importance")
            ax.set_title("Top 10 Feature Importances")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown('<div class="axis-label">📊 X-axis = Feature Importance Score</div>', unsafe_allow_html=True)
            st.markdown('<div class="axis-label">📈 Y-axis = Features</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🎯 Overall Model Prediction")
            
            if task == "Classification":
                from collections import Counter
                majority_pred = Counter(y_pred).most_common(1)[0][0]
                st.markdown(f'<div class="prediction-card-light"><h3>The model generally predicts:</h3><h1>{majority_pred}</h1></div>', unsafe_allow_html=True)
            else:
                avg_pred = np.mean(y_pred)
                st.markdown(f'<div class="prediction-card-light"><h3>Average predicted value:</h3><h1>{avg_pred:.3f}</h1></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🧠 Why This Behavior?")
            st.markdown("*Here's what the model learned from your data:*")
            
            total = importances.sum()
            perc = (importances / total) * 100
            top5 = np.argsort(importances)[::-1][:5]
            
            for idx in top5:
                idx = int(idx)
                p = perc[idx]
                if p > 15:
                    direction = "critically important for predictions"
                    color = "#28a745"
                    icon = "🔥"
                elif p > 8:
                    direction = "strongly influences predictions"
                    color = "#17a2b8"
                    icon = "📈"
                else:
                    direction = "moderately influences predictions"
                    color = "#6c757d"
                    icon = "⚖️"
                
                st.markdown(f"""
                <div class="explanation-box" style="border-left-color: {color};">
                    <b>{icon} {feature_names[idx]}</b><br>
                    → <b>Importance: {p:.1f}%</b><br>
                    → This feature {direction}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Feature importance only available for Random Forest. Please retrain with Random Forest.")
        
        st.info("💡 **How to read this:** Features with higher percentages have more influence on predictions.")
    
    # ========== LOCAL EXPLANATION ==========
    else:
        st.subheader("🔍 Explain a Single Prediction")
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            source = st.radio("Input method:", ["📊 Use Test Data", "✏️ Enter New Data"], horizontal=True)
        
        X_single = None
        current_pred = None
        current_actual = None
        
        if source == "📊 Use Test Data":
            row_num = st.number_input("Row number", 1, len(X_test), 1)
            st.markdown("### 📋 Selected Row")
            st.dataframe(test_display.iloc[[row_num - 1]], use_container_width=True)
            X_single = X_test.iloc[row_num - 1:row_num]
            current_pred = model.predict(X_single)[0]
            current_actual = y_test.iloc[row_num - 1]
        else:
            st.markdown("### 📝 Enter Values")
            new_data = {}
            for idx, feat in enumerate(original_columns):
                if feat in st.session_state.original_df.columns:
                    unique_vals = st.session_state.original_df[feat].nunique()
                    if unique_vals <= 10:
                        cats = st.session_state.original_df[feat].dropna().unique().tolist()
                        new_data[feat] = st.selectbox(f"📋 {feat}", [str(c) for c in sorted(cats)])
                    else:
                        min_v = float(st.session_state.original_df[feat].min())
                        max_v = float(st.session_state.original_df[feat].max())
                        mean_v = float(st.session_state.original_df[feat].mean())
                        new_data[feat] = st.number_input(f"🔢 {feat}", value=mean_v, min_value=min_v, max_value=max_v)
            
            if st.button("🔮 Predict"):
                new_df = pd.DataFrame([new_data])
                X_proc = preprocessor.transform(new_df)
                X_single = pd.DataFrame(X_proc, columns=feature_names)
                current_pred = model.predict(X_single)[0]
                st.success(f"✅ Prediction: {current_pred}")
        
        if X_single is not None:
            # Prediction Box
            if task == "Classification":
                if source == "📊 Use Test Data":
                    st.markdown(f'<div class="prediction-card-light"><h3>🎯 Prediction</h3><h1>{current_pred}</h1><p>Actual: {current_actual}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-card-light"><h3>🎯 Prediction</h3><h1>{current_pred}</h1></div>', unsafe_allow_html=True)
            else:
                if source == "📊 Use Test Data":
                    st.markdown(f'<div class="prediction-card-light"><h3>🎯 Prediction</h3><h1>{current_pred:.2f}</h1><p>Actual: {current_actual:.2f}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-card-light"><h3>🎯 Prediction</h3><h1>{current_pred:.2f}</h1></div>', unsafe_allow_html=True)
            
            # Feature Contributions
            st.markdown("### 📊 Feature Contributions")
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                fig, ax = plt.subplots(figsize=(10, 6))
                top_n = min(10, len(feature_names))
                top_idx = np.argsort(importances)[-top_n:]
                colors = ['#28a745' for _ in range(len(top_idx))]
                ax.barh(range(len(top_idx)), importances[top_idx], color=colors)
                ax.set_yticks(range(len(top_idx)))
                ax.set_yticklabels([feature_names[i] for i in top_idx])
                ax.set_xlabel("Feature Importance")
                ax.set_title("Important Features")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.markdown('<div class="axis-label">📊 X-axis = Feature Importance</div>', unsafe_allow_html=True)
                st.markdown('<div class="axis-label">📈 Y-axis = Features</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 🧠 Why This Prediction?")
                
                total = importances.sum()
                perc = (importances / total) * 100
                top5 = np.argsort(importances)[::-1][:5]
                
                for idx in top5:
                    idx = int(idx)
                    p = perc[idx]
                    val = X_single.iloc[0, idx] if idx < len(X_single.columns) else "N/A"
                    
                    if p > 15:
                        direction = "major driver"
                        color = "#28a745"
                        icon = "🔥"
                    elif p > 8:
                        direction = "important factor"
                        color = "#17a2b8"
                        icon = "📈"
                    else:
                        direction = "contributing factor"
                        color = "#6c757d"
                        icon = "⚖️"
                    
                    st.markdown(f"""
                    <div class="explanation-box" style="border-left-color: {color};">
                        <b>{icon} {feature_names[idx]}</b><br>
                        → <b>Value:</b> {val}<br>
                        → <b>Importance: {p:.1f}%</b><br>
                        → This is a {direction} for this prediction
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Feature importance not available. Please use Random Forest.")
    
    st.markdown('</div>', unsafe_allow_html=True)