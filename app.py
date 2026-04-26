import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from utils.preprocess import preprocess_data
from BACKEND.model import train_model

from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(layout="wide", page_title="SHAP AI Explainability Tool", page_icon="📊")

# Custom CSS for clean, minimal UI
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
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    .explanation-box:hover {
        background: #fafafa;
        transform: translateX(3px);
    }
    
    .prediction-card-light {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e0e0e0;
        margin: 20px 0;
    }
    .prediction-card-light h3 {
        margin: 0 0 10px 0;
        font-size: 14px;
        color: #666;
        font-weight: 500;
    }
    .prediction-card-light h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 600;
        color: #333;
    }
    .prediction-card-light p {
        margin: 10px 0 0 0;
        font-size: 12px;
        color: #888;
    }
    
    .metric-card-light {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e0e0e0;
        margin: 20px 0;
    }
    .metric-card-light h3 {
        margin: 0;
        font-size: 14px;
        color: #666;
        font-weight: 500;
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
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: #5a67d8;
        transform: translateY(-2px);
    }
    
    .stRadio > div {
        display: flex;
        gap: 20px;
        justify-content: center;
    }
    .stRadio label {
        background: #f0f2f6;
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .axis-label {
        background: #f8f9fa;
        padding: 8px 15px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 13px;
        color: #555;
        border-left: 3px solid #667eea;
    }
    
    div[data-testid="stHorizontalBlock"] {
        justify-content: center;
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

# ================= PAGE 1: INPUT PAGE =================
if st.session_state.page == "input":
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.title("🔮 SHAP - AI Model Explainability Tool")
    st.markdown("*Upload any dataset, train a model, and understand exactly WHY it makes predictions*")
    st.markdown("---")
    
    file = st.file_uploader("📁 Upload CSV File", type=["csv"], help="Upload a CSV file with your data")
    
    if file:
        df = pd.read_csv(file)
        st.success(f"✅ File loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")
        
        st.session_state.original_df = df.copy()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            target = st.selectbox("🎯 Select Target Column", df.columns)
        
        with col2:
            unique_count = df[target].nunique()
            task = "Classification" if unique_count <= 10 else "Regression"
            st.info(f"📋 Detected Task: **{task}**\n\n({unique_count} unique values)")
        
        with col3:
            if task == "Classification":
                model_type = st.selectbox("🤖 Select Model", ["Random Forest", "Logistic Regression"])
            else:
                model_type = st.selectbox("🤖 Select Model", ["Random Forest", "Linear Regression"])
        
        if st.button("🚀 Run Model", type="primary", use_container_width=True):
            
            with st.spinner("🔄 Processing data and training model..."):
                start_time = time.time()
                
                (X_train, X_test, y_train, y_test, X_test_original, 
                 preprocessor, feature_names, original_columns, df_full) = preprocess_data(df, target)
                
                test_display = X_test_original.copy()
                test_display[target] = y_test.values
                
                model = train_model(X_train, y_train, task, model_type)
                
                training_time = time.time() - start_time
                st.success(f"✅ Model trained in {training_time:.2f} seconds!")
                
                st.session_state.update({
                    "model": model,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "test_display": test_display.reset_index(drop=True),
                    "feature_names": feature_names,
                    "task": task,
                    "target_name": target,
                    "preprocessor": preprocessor,
                    "original_columns": original_columns,
                    "X_test_original": X_test_original,
                    "y_pred": model.predict(X_test)
                })
                
                st.session_state.page = "output"
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================= PAGE 2: OUTPUT PAGE =================
elif st.session_state.page == "output":
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    test_display = st.session_state.test_display
    task = st.session_state.task
    y_pred = st.session_state.y_pred
    
    st.title("📊 Model Results")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        
        if st.button("⬅️ Back to Input", use_container_width=True, key="back_to_input_btn"):
            st.session_state.page = "input"
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("📄 Test Dataset")
        st.dataframe(test_display, height=300, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📈 Model Performance")
        
        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.markdown(f"""
            <div class="metric-card-light">
                <h3>🎯 Accuracy Score</h3>
                <h1>{acc:.2%}</h1>
                <p>Correct predictions / Total predictions</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            r2 = r2_score(y_test, y_pred)
            st.markdown(f"""
            <div class="metric-card-light">
                <h3>📊 R² Score</h3>
                <h1>{r2:.3f}</h1>
                <p>Coefficient of Determination (1.0 = perfect)</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🔍 View Explanations?", type="primary", use_container_width=True):
            st.session_state.page = "explanation"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================= PAGE 3: EXPLANATION PAGE =================
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
        if st.button("⬅️ Back to Results", use_container_width=True):
            st.session_state.page = "output"
            st.rerun()
    
    st.title("📖 Model Explainability")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        explanation_type = st.radio(
            "Select Explanation Type",
            ["🌍 Global Explanation", "🔍 Local Explanation"],
            horizontal=True
        )
    
    st.markdown("---")
    
    # ========== GLOBAL EXPLANATION ==========
    if explanation_type == "🌍 Global Explanation":
        
        st.subheader("🌍 How Features Impact Predictions")
        
        # Use model feature importance instead of SHAP
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.argsort(importances)[::-1][:10]
            colors = ['#28a745' if i < len(indices)//2 else '#dc3545' for i in range(len(indices))]
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
                st.markdown(f"""
                <div class="prediction-card-light">
                    <h3>The model generally predicts:</h3>
                    <h1>{majority_pred}</h1>
                </div>
                """, unsafe_allow_html=True)
            else:
                avg_pred = np.mean(y_pred)
                st.markdown(f"""
                <div class="prediction-card-light">
                    <h3>Average predicted value:</h3>
                    <h1>{avg_pred:.3f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🧠 Why This Behavior?")
            st.markdown("*Here's what the model learned from your data:*")
            
            # Show top 5 features with percentages
            total_importance = importances.sum()
            perc = (importances / total_importance) * 100
            top_indices = np.argsort(importances)[::-1][:5]
            
            for idx in top_indices:
                idx = int(idx)
                importance_perc = perc[idx]
                
                if importance_perc > 10:
                    direction = "strongly influences predictions"
                    color = "#28a745"
                    icon = "📈"
                elif importance_perc > 5:
                    direction = "moderately influences predictions"
                    color = "#17a2b8"
                    icon = "📊"
                else:
                    direction = "has some influence on predictions"
                    color = "#6c757d"
                    icon = "⚖️"
                
                st.markdown(f"""
                <div class="explanation-box" style="border-left-color: {color};">
                    <b>{icon} {feature_names[idx]}</b><br>
                    → <b>Importance: {importance_perc:.1f}%</b> of total importance<br>
                    → This feature {direction}
                </div>
                """, unsafe_allow_html=True)
            
            st.info("💡 **How to read this:** Features with higher percentages have more influence on model predictions.")
        else:
            st.warning("Feature importance not available for this model type. Please use Random Forest for better insights.")
    
    # ========== LOCAL EXPLANATION ==========
    else:
        
        st.subheader("🔍 Explain a Single Prediction")
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            local_source = st.radio(
                "Choose input method:",
                ["📊 Use Test Data", "✏️ Enter New Data"],
                horizontal=True
            )
        
        X_single = None
        original_values = None
        current_prediction = None
        current_actual = None
        
        if local_source == "📊 Use Test Data":
            max_row = len(X_test)
            row_num = st.number_input("Select row number to explain", min_value=1, max_value=max_row, value=1, step=1)
            
            st.markdown("---")
            st.markdown("### 📋 Selected Data Row")
            st.dataframe(test_display.iloc[[row_num - 1]], use_container_width=True)
            
            X_single = X_test.iloc[row_num - 1:row_num]
            current_prediction = model.predict(X_single)[0]
            current_actual = y_test.iloc[row_num - 1]
            
        else:
            st.markdown("### 📝 Enter New Data")
            
            if st.session_state.X_test_original is not None:
                new_data_dict = {}
                original_df = st.session_state.original_df
                
                num_cols = 2
                feature_cols = st.columns(num_cols)
                
                for idx, feature in enumerate(original_columns):
                    col_idx = idx % num_cols
                    with feature_cols[col_idx]:
                        if feature in original_df.columns:
                            unique_vals = original_df[feature].nunique()
                            
                            if unique_vals <= 10:
                                categories = original_df[feature].dropna().unique().tolist()
                                categories.sort()
                                
                                if unique_vals == 2:
                                    if set(categories) == {0, 1}:
                                        display_categories = ["No (0)", "Yes (1)"]
                                        value_map = {"No (0)": 0, "Yes (1)": 1}
                                    else:
                                        display_categories = [str(c) for c in categories]
                                        value_map = {str(c): c for c in categories}
                                    
                                    selected = st.selectbox(f"🔘 {feature}", options=display_categories)
                                    new_data_dict[feature] = value_map[selected]
                                else:
                                    selected = st.selectbox(f"📋 {feature}", options=[str(c) for c in categories])
                                    for cat in categories:
                                        if str(cat) == selected:
                                            new_data_dict[feature] = cat
                                            break
                            else:
                                min_val = float(original_df[feature].min())
                                max_val = float(original_df[feature].max())
                                mean_val = float(original_df[feature].mean())
                                
                                new_data_dict[feature] = st.number_input(
                                    f"🔢 {feature}",
                                    value=mean_val,
                                    min_value=min_val,
                                    max_value=max_val,
                                    step=(max_val - min_val) / 100 if max_val > min_val else 1.0,
                                    format="%.4f" if abs(max_val - min_val) < 1 else "%.2f"
                                )
                
                if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
                    try:
                        new_data_df = pd.DataFrame([new_data_dict])
                        X_single_processed = preprocessor.transform(new_data_df)
                        X_single = pd.DataFrame(X_single_processed, columns=feature_names)
                        current_prediction = model.predict(X_single)[0]
                        original_values = new_data_dict
                        st.success("✅ Prediction generated!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        X_single = None
                
                if X_single is None:
                    st.info("👈 Fill in the values and click 'Generate Prediction'")
        
        if X_single is not None:
            
            # Prediction Box
            if task == "Classification":
                if local_source == "📊 Use Test Data":
                    st.markdown(f"""
                    <div class="prediction-card-light">
                        <h3>🎯 Model Prediction</h3>
                        <h1>{current_prediction}</h1>
                        <p>Actual: {current_actual}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card-light">
                        <h3>🎯 Model Prediction</h3>
                        <h1>{current_prediction}</h1>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                if local_source == "📊 Use Test Data":
                    actual = current_actual
                    diff = current_prediction - actual
                    diff_symbol = "▲" if diff > 0 else "▼" if diff < 0 else "●"
                    st.markdown(f"""
                    <div class="prediction-card-light">
                        <h3>🎯 Model Prediction</h3>
                        <h1>{current_prediction:.2f}</h1>
                        <p>Actual: {actual:.2f} ({diff_symbol} {abs(diff):.2f})</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card-light">
                        <h3>🎯 Model Prediction</h3>
                        <h1>{current_prediction:.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Feature Contributions
            st.markdown("### 📊 Feature Contributions")
            
            if hasattr(model, 'feature_importances_'):
                # Use model feature importances for local explanation
                importances = model.feature_importances_
                
                # For local prediction, we can show the feature values and their global importance
                plt.figure(figsize=(10, 6))
                top_n = min(10, len(feature_names))
                top_idx = np.argsort(importances)[-top_n:]
                colors = ['#28a745' for _ in range(len(top_idx))]
                plt.barh(range(len(top_idx)), importances[top_idx], color=colors)
                plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
                plt.xlabel("Feature Importance")
                plt.title("Important Features for Model Predictions")
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.close()
                
                st.markdown('<div class="axis-label">📊 X-axis = Feature Importance</div>', unsafe_allow_html=True)
                st.markdown('<div class="axis-label">📈 Y-axis = Features</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 🧠 Why This Prediction?")
                st.markdown("*Here are the most important features for this model:*")
                
                total_importance = importances.sum()
                perc = (importances / total_importance) * 100
                top_indices = np.argsort(importances)[::-1][:5]
                
                for idx in top_indices:
                    idx = int(idx)
                    importance_perc = perc[idx]
                    
                    # Get feature value if available
                    feature_value = "N/A"
                    if idx < len(X_single.columns):
                        val = X_single.iloc[0, idx]
                        if isinstance(val, (int, float)):
                            feature_value = f"{val:.2f}" if isinstance(val, float) else str(val)
                        else:
                            feature_value = str(val)
                    
                    if importance_perc > 10:
                        direction = "major driver of predictions"
                        color = "#28a745"
                        icon = "📈"
                    elif importance_perc > 5:
                        direction = "moderate driver of predictions"
                        color = "#17a2b8"
                        icon = "📊"
                    else:
                        direction = "contributes to predictions"
                        color = "#6c757d"
                        icon = "⚖️"
                    
                    st.markdown(f"""
                    <div class="explanation-box" style="border-left-color: {color};">
                        <b>{icon} {feature_names[idx]}</b><br>
                        → <b>Value:</b> {feature_value}<br>
                        → <b>Importance: {importance_perc:.1f}%</b><br>
                        → This feature is a {direction}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info("💡 **How to read this:** Features with higher importance percentages have more influence on model predictions.")
            else:
                st.warning("Feature importance not available. Please use Random Forest model for feature explanations.")
    
    st.markdown('</div>', unsafe_allow_html=True)