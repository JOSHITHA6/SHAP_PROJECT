import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from utils.preprocess import preprocess_data
from BACKEND.model import train_model

from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(layout="wide", page_title="SHAP AI Explainability Tool", page_icon="📊")

# Custom CSS for clean, minimal UI
st.markdown("""
<style>
    /* Main container styling */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Clean explanation box */
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
    
    /* Light prediction card */
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
    
    /* Metric card light */
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
    
    /* Button styling */
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
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Radio button styling */
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
    
    /* Axis label styling */
    .axis-label {
        background: #f8f9fa;
        padding: 8px 15px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 13px;
        color: #555;
        border-left: 3px solid #667eea;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Center align radio buttons */
    div[data-testid="stHorizontalBlock"] {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "input"
    
if "show_explanations" not in st.session_state:
    st.session_state.show_explanations = False
    
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
    
    # File upload
    file = st.file_uploader("📁 Upload CSV File", type=["csv"], help="Upload a CSV file with your data")
    
    if file:
        df = pd.read_csv(file)
        st.success(f"✅ File loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Store original dataframe for later use
        st.session_state.original_df = df.copy()
        
        # All inputs appear at once
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            target = st.selectbox("🎯 Select Target Column", df.columns, 
                                 help="The column you want to predict")
        
        with col2:
            # Auto-detect task
            unique_count = df[target].nunique()
            task = "Classification" if unique_count <= 10 else "Regression"
            st.info(f"📋 Detected Task: **{task}**\n\n({unique_count} unique values in target)")
        
        with col3:
            if task == "Classification":
                model_type = st.selectbox("🤖 Select Model", ["Random Forest", "Logistic Regression"])
            else:
                model_type = st.selectbox("🤖 Select Model", ["Random Forest", "Linear Regression"])
        
        # Run button
        if st.button("🚀 Run Model", type="primary", use_container_width=True):
            
            with st.spinner("🔄 Processing data and training model..."):
                # Preprocess data
                (X_train, X_test, y_train, y_test, X_test_original, 
                 preprocessor, feature_names, original_columns, df_full) = preprocess_data(df, target)
                
                # Create display dataframe
                test_display = X_test_original.copy()
                test_display[target] = y_test.values
                
                # Train model
                model = train_model(X_train, y_train, task, model_type)
                
                # Store in session state
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
    
    # Load session data
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    test_display = st.session_state.test_display
    task = st.session_state.task
    target_name = st.session_state.target_name
    y_pred = st.session_state.y_pred
    
    # Title
    st.title("📊 Model Results")
    st.markdown("---")
    
    # Center aligned content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        
        # Add Back to Input button at the top
        if st.button("⬅️ Back to Input", use_container_width=True, key="back_to_input_btn"):
            st.session_state.page = "input"
            st.session_state.show_explanations = False
            st.rerun()
        
        st.markdown("---")
        
        # 1. Test Dataset
        st.subheader("📄 Test Dataset")
        st.dataframe(test_display, height=300, use_container_width=True)
        
        st.markdown("---")
        
        # 2. Model Performance
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
        
        # 3. View Explanations Button
        st.markdown("---")
        
        if st.button("🔍 View Explanations?", type="primary", use_container_width=True):
            st.session_state.show_explanations = True
            st.session_state.page = "explanation"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================= PAGE 3: EXPLANATION PAGE =================
elif st.session_state.page == "explanation":
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Load session data
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    test_display = st.session_state.test_display
    feature_names = st.session_state.feature_names
    task = st.session_state.task
    target_name = st.session_state.target_name
    preprocessor = st.session_state.preprocessor
    original_columns = st.session_state.original_columns
    X_test_original = st.session_state.X_test_original
    y_pred = st.session_state.y_pred
    
    # Back button to results
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⬅️ Back to Results", use_container_width=True):
            st.session_state.page = "output"
            st.rerun()
    
    st.title("📖 Model Explainability")
    st.markdown("*Understand why your model makes predictions*")
    st.markdown("---")
    
    # Toggle for Global/Local Explanation
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
        
        # Generate SHAP plots
        with st.spinner("Calculating feature impacts..."):
            try:
                if 'RandomForest' in str(type(model)):
                    # Use a small sample for faster computation
                    X_sample = X_test[:min(100, len(X_test))]
                    
                    # Create explainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Handle classification output
                    if isinstance(shap_values, list):
                        if task == "Classification" and len(shap_values) == 2:
                            shap_values = shap_values[1]  # Take positive class
                        else:
                            shap_values = shap_values[0]
                    
                    # Create summary plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_sample, plot_type="violin", show=False, max_display=min(10, len(feature_names)))
                    plt.tight_layout()
                    fig_global = plt.gcf()
                    plt.close()
                    
                    st.pyplot(fig_global, use_container_width=True)
                    st.markdown('<div class="axis-label">📊 X-axis = Feature Impact on Model Output</div>', unsafe_allow_html=True)
                    st.markdown('<div class="axis-label">📈 Y-axis = Features</div>', unsafe_allow_html=True)
                    
                    # Calculate feature importance
                    vals = np.abs(shap_values).mean(axis=0)
                    if vals.sum() > 0:
                        perc = (vals / vals.sum()) * 100
                    else:
                        perc = np.zeros_like(vals)
                    
                    top_indices = np.argsort(vals)[::-1][:5]
                    
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
                    
                    for idx in top_indices:
                        idx = int(idx)
                        if idx < len(feature_names):
                            mean_shap = np.mean(shap_values[:, idx])
                            
                            if mean_shap > 0:
                                direction = "pushes prediction HIGHER"
                                color = "#28a745"
                                icon = "📈"
                            elif mean_shap < 0:
                                direction = "pushes prediction LOWER"
                                color = "#dc3545"
                                icon = "📉"
                            else:
                                direction = "has mixed or minimal impact"
                                color = "#6c757d"
                                icon = "⚖️"
                            
                            st.markdown(f"""
                            <div class="explanation-box" style="border-left-color: {color};">
                                <b>{icon} {feature_names[idx]}</b><br>
                                → <b>Contribution: {perc[idx]:.1f}%</b> of total impact<br>
                                → {direction}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("Global SHAP plot only available for Random Forest models")
                    
            except Exception as e:
                st.warning(f"Could not generate SHAP plot: {str(e)[:100]}")
        
        st.info("💡 **How to read this:** Features with higher percentages have more influence on predictions.")
    
    # ========== LOCAL EXPLANATION ==========
    else:
        
        st.subheader("🔍 Explain a Single Prediction")
        
        # Two options for local explanation
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
            
            # Generate Feature Contributions
            st.markdown("### 📊 Feature Contributions")
            
            try:
                if 'RandomForest' in str(type(model)):
                    # Use TreeExplainer which is faster
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_single)
                    
                    # Handle the output shape correctly
                    if isinstance(shap_values, list):
                        # For classification
                        if len(shap_values) == 2:
                            # Binary classification - take positive class
                            shap_row = shap_values[1].flatten()
                        else:
                            # Multi-class - take first class
                            shap_row = shap_values[0].flatten()
                    else:
                        # For regression
                        shap_row = shap_values.flatten()
                    
                    # Ensure we have the right length
                    if len(shap_row) != len(feature_names):
                        st.warning(f"SHAP values length ({len(shap_row)}) doesn't match features ({len(feature_names)}). Truncating to minimum.")
                        min_len = min(len(shap_row), len(feature_names))
                        shap_row = shap_row[:min_len]
                        display_features = feature_names[:min_len]
                    else:
                        display_features = feature_names
                    
                    # Create bar plot
                    plt.figure(figsize=(10, 6))
                    top_n = min(10, len(display_features))
                    top_idx = np.argsort(np.abs(shap_row))[-top_n:]
                    colors = ['#dc3545' if x < 0 else '#28a745' for x in shap_row[top_idx]]
                    plt.barh(range(len(top_idx)), shap_row[top_idx], color=colors)
                    plt.yticks(range(len(top_idx)), [display_features[i] for i in top_idx])
                    plt.xlabel("SHAP Value (Impact on Prediction)")
                    plt.title("Feature Contributions for This Prediction")
                    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    plt.tight_layout()
                    fig_local = plt.gcf()
                    st.pyplot(fig_local, use_container_width=True)
                    plt.close()
                    
                    # Axis labels
                    st.markdown('<div class="axis-label">📊 X-axis = Impact on Prediction</div>', unsafe_allow_html=True)
                    st.markdown('<div class="axis-label">📈 Y-axis = Features</div>', unsafe_allow_html=True)
                    
                    # Calculate percentages
                    abs_vals = np.abs(shap_row)
                    if abs_vals.sum() > 0:
                        percentages = (abs_vals / abs_vals.sum()) * 100
                    else:
                        percentages = np.zeros_like(shap_row)
                    
                    # Get top 5 features by absolute impact
                    top_indices = np.argsort(abs_vals)[::-1][:5]
                    
                    st.markdown("---")
                    st.markdown("### 🧠 Why This Prediction?")
                    st.markdown("*Here's why the model made this specific prediction:*")
                    
                    for idx in top_indices:
                        if idx < len(display_features):
                            feat_name = display_features[idx]
                            contribution = percentages[idx]
                            shap_value = shap_row[idx]
                            
                            if shap_value > 0:
                                direction = "pushes prediction HIGHER"
                                color = "#28a745"
                                icon = "📈"
                            elif shap_value < 0:
                                direction = "pushes prediction LOWER"
                                color = "#dc3545"
                                icon = "📉"
                            else:
                                direction = "has minimal impact"
                                color = "#6c757d"
                                icon = "⚖️"
                            
                            st.markdown(f"""
                            <div class="explanation-box" style="border-left-color: {color};">
                                <b>{icon} {feat_name}</b><br>
                                → <b>Contribution: {contribution:.1f}%</b> of total impact<br>
                                → {direction}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.info("💡 **How to read this:** Features with higher percentages have more influence on this prediction. Green bars push the prediction higher, red bars push it lower.")
                    
                else:
                    st.warning("⚠️ For the best explainability experience, please use Random Forest model.")
                    st.info("Current model type: " + str(type(model)).split('.')[-1].split("'")[0])
                    
            except Exception as e:
                st.error(f"Error generating explanation: {str(e)}")
                st.info("Please make sure you're using Random Forest model for SHAP explanations.")
    
    st.markdown('</div>', unsafe_allow_html=True)