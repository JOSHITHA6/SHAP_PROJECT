import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

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
if "feature_directions" not in st.session_state:
    st.session_state.feature_directions = None

# Helper function to determine feature direction (increase/decrease)
def get_feature_directions(model, X_sample, y_sample, feature_names, task):
    """Determine whether each feature increases or decreases predictions"""
    directions = {}
    
    try:
        if task == "Classification":
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_sample, y_sample)
            coefficients = lr.coef_[0]
        else:
            lr = LinearRegression()
            lr.fit(X_sample, y_sample)
            coefficients = lr.coef_
        
        for i, feat in enumerate(feature_names):
            if i < len(coefficients):
                if coefficients[i] > 0.01:
                    directions[feat] = "increases"
                elif coefficients[i] < -0.01:
                    directions[feat] = "decreases"
                else:
                    directions[feat] = "neutral"
            else:
                directions[feat] = "neutral"
    except:
        # Fallback: use correlation with predictions
        predictions = model.predict(X_sample)
        for i, feat in enumerate(feature_names):
            if i < X_sample.shape[1]:
                corr = np.corrcoef(X_sample.iloc[:, i], predictions)[0, 1]
                if corr > 0.05:
                    directions[feat] = "increases"
                elif corr < -0.05:
                    directions[feat] = "decreases"
                else:
                    directions[feat] = "neutral"
    
    return directions

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
                
                # Get feature directions
                directions = get_feature_directions(model, X_test[:100], y_test[:100], feature_names, task)
                
                st.session_state.update({
                    "model": model, "X_test": X_test, "y_test": y_test,
                    "test_display": test_display.reset_index(drop=True),
                    "feature_names": feature_names, "task": task,
                    "target_name": target, "preprocessor": preprocessor,
                    "original_columns": original_columns, "y_pred": model.predict(X_test),
                    "feature_directions": directions
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
    directions = st.session_state.feature_directions
    
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
    
    # ========== GLOBAL EXPLANATION - SIMPLE BAR CHART (NO SHAP) ==========
    if exp_type == "🌍 Global Explanation":
        st.subheader("🌍 How Features Impact Predictions")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create simple horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get top 10 features
            top_indices = np.argsort(importances)[::-1][:10]
            top_features = [feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            top_directions = [directions.get(f, "neutral") for f in top_features]
            
            # Set colors based on direction
            colors = []
            for d in top_directions:
                if d == "increases":
                    colors.append('#28a745')  # Green - increases prediction
                elif d == "decreases":
                    colors.append('#dc3545')  # Red - decreases prediction
                else:
                    colors.append('#6c757d')  # Gray - neutral
            
            # Create bars
            bars = ax.barh(range(len(top_features)), top_importances, color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel("Feature Importance", fontsize=12)
            ax.set_title("Top 10 Features Impacting Predictions", fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            # Add percentage labels on bars
            total = importances.sum()
            for i, (bar, imp) in enumerate(zip(bars, top_importances)):
                width = bar.get_width()
                percentage = (imp / total) * 100
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{percentage:.1f}%', va='center', fontsize=9)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#28a745', label='🟢 INCREASES prediction'),
                Patch(facecolor='#dc3545', label='🔴 DECREASES prediction'),
                Patch(facecolor='#6c757d', label='⚪ Mixed/Neutral impact')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown('<div class="axis-label">📊 X-axis = Feature Importance</div>', unsafe_allow_html=True)
            st.markdown('<div class="axis-label">📈 Y-axis = Features</div>', unsafe_allow_html=True)
            st.markdown('<div class="axis-label">🎨 Green = Increases prediction | Red = Decreases prediction</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ========== WHY THIS BEHAVIOR? (WITHOUT OVERALL PREDICTION) ==========
            st.markdown("### 🧠 Why This Behavior?")
            st.markdown("*Here's what the model learned from your data:*")
            
            total = importances.sum()
            perc = (importances / total) * 100
            top5 = np.argsort(importances)[::-1][:5]
            
            for idx in top5:
                idx = int(idx)
                feat = feature_names[idx]
                p = perc[idx]
                direction = directions.get(feat, "neutral")
                
                if direction == "increases":
                    direction_text = "INCREASES the prediction"
                    color = "#28a745"
                    icon = "📈"
                elif direction == "decreases":
                    direction_text = "DECREASES the prediction"
                    color = "#dc3545"
                    icon = "📉"
                else:
                    direction_text = "has mixed/neutral impact"
                    color = "#6c757d"
                    icon = "⚖️"
                
                st.markdown(f"""
                <div class="explanation-box" style="border-left-color: {color};">
                    <b>{icon} {feat}</b><br>
                    → <b>Contribution: {p:.1f}%</b> of total impact<br>
                    → This feature <b style="color:{color};">{direction_text}</b>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Feature importance only available for Random Forest. Please retrain with Random Forest.")
        
        st.info("💡 **How to read this:** Features with higher percentages have more influence on predictions. Green features INCREASE predictions, Red features DECREASE predictions.")
    
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
            st.markdown("### 📝 Enter New Data")
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
                    st.markdown(f'<div class="prediction-card-light"><h3>🎯 Model Prediction</h3><h1>{current_pred}</h1><p>Actual: {current_actual}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-card-light"><h3>🎯 Model Prediction</h3><h1>{current_pred}</h1></div>', unsafe_allow_html=True)
            else:
                if source == "📊 Use Test Data":
                    st.markdown(f'<div class="prediction-card-light"><h3>🎯 Model Prediction</h3><h1>{current_pred:.2f}</h1><p>Actual: {current_actual:.2f}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-card-light"><h3>🎯 Model Prediction</h3><h1>{current_pred:.2f}</h1></div>', unsafe_allow_html=True)
            
            # Feature Contributions
            st.markdown("### 📊 Feature Contributions")
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                fig, ax = plt.subplots(figsize=(10, 6))
                top_n = min(8, len(feature_names))
                top_idx = np.argsort(importances)[-top_n:]
                colors = ['#28a745' if directions.get(feature_names[i], "neutral") == "increases" else '#dc3545' for i in top_idx]
                ax.barh(range(len(top_idx)), importances[top_idx], color=colors)
                ax.set_yticks(range(len(top_idx)))
                ax.set_yticklabels([feature_names[i] for i in top_idx])
                ax.set_xlabel("Feature Importance")
                ax.set_title("How Each Feature Affects Predictions")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.markdown('<div class="axis-label">📊 X-axis = Impact on Prediction</div>', unsafe_allow_html=True)
                st.markdown('<div class="axis-label">📈 Y-axis = Features</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 🧠 Why This Prediction?")
                st.markdown("*Here's why the model made this specific prediction:*")
                
                total = importances.sum()
                perc = (importances / total) * 100
                top5 = np.argsort(importances)[::-1][:5]
                
                for idx in top5:
                    idx = int(idx)
                    feat = feature_names[idx]
                    p = perc[idx]
                    direction = directions.get(feat, "neutral")
                    val = X_single.iloc[0, idx] if idx < len(X_single.columns) else "N/A"
                    
                    if isinstance(val, float):
                        val_display = f"{val:.3f}"
                    else:
                        val_display = str(val)
                    
                    if direction == "increases":
                        direction_text = "pushes prediction HIGHER"
                        color = "#28a745"
                        icon = "📈"
                    elif direction == "decreases":
                        direction_text = "pushes prediction LOWER"
                        color = "#dc3545"
                        icon = "📉"
                    else:
                        direction_text = "has minimal impact"
                        color = "#6c757d"
                        icon = "⚖️"
                    
                    st.markdown(f"""
                    <div class="explanation-box" style="border-left-color: {color};">
                        <b>{icon} {feat}</b><br>
                        → <b>Value:</b> {val_display}<br>
                        → <b>Contribution: {p:.1f}%</b><br>
                        → {direction_text}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Feature importance not available. Please use Random Forest.")
    
    st.markdown('</div>', unsafe_allow_html=True)