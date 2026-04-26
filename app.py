import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.preprocess import preprocess_data
from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots

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
    
    /* Secondary button styling */
    .stButton > button.secondary {
        background: #6c757d;
    }
    .stButton > button.secondary:hover {
        background: #5a6268;
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
    
    /* Section headers */
    .section-header {
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Center align radio buttons */
    div[data-testid="stHorizontalBlock"] {
        justify-content: center;
    }
    
    /* Two button container */
    .button-container {
        display: flex;
        gap: 10px;
        margin: 20px 0;
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
        col_back1, col_back2, col_back3 = st.columns([1, 2, 1])
        with col_back2:
            if st.button("⬅️ Back to Input", use_container_width=True, key="back_to_input_btn"):
                # Clear specific session state items to start fresh
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
            result = generate_shap_plots(
                model, X_test[:100], feature_names=feature_names, task=task
            )
            
            if len(result) == 3:
                fig_global, _, shap_values_array = result
            else:
                st.warning("SHAP explanation could not be generated.")
                fig_global = None
                shap_values_array = None
        
        # Display SHAP graph with axis labels
        if fig_global:
            st.pyplot(fig_global, use_container_width=True)
            
            # Axis labels (one below the other)
            st.markdown('<div class="axis-label">📊 X-axis = Feature Impact on Model Output</div>', unsafe_allow_html=True)
            st.markdown('<div class="axis-label">📈 Y-axis = Features</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Overall Prediction
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
        
        # Why This Behavior?
        st.markdown("### 🧠 Why This Behavior?")
        st.markdown("*Here's what the model learned from your data:*")
        
        # Calculate feature importance from SHAP values
        if shap_values_array is not None and len(shap_values_array) > 0:
            try:
                vals = np.abs(shap_values_array).mean(axis=0).flatten()
                if vals.sum() > 0:
                    perc = (vals / vals.sum()) * 100
                else:
                    perc = np.zeros_like(vals)
                
                # Get top 5 features
                top_indices = np.argsort(vals)[::-1][:5]
                
                for idx in top_indices:
                    idx = int(idx)
                    mean_shap = np.mean(shap_values_array[:, idx])
                    
                    # Determine direction
                    if mean_shap > 0.05:
                        direction = "pushes prediction HIGHER"
                        color = "#28a745"
                        icon = "📈"
                    elif mean_shap < -0.05:
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
            except Exception as e:
                st.warning(f"Could not calculate SHAP feature impacts: {str(e)[:100]}")
        
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
        shap_values_array = None
        
        if local_source == "📊 Use Test Data":
            # Option 1: Select existing row
            max_row = len(X_test)
            row_num = st.number_input("Select row number to explain", min_value=1, max_value=max_row, value=1, step=1)
            
            st.markdown("---")
            st.markdown("### 📋 Selected Data Row")
            st.dataframe(test_display.iloc[[row_num - 1]], use_container_width=True)
            
            X_single = X_test.iloc[row_num - 1:row_num]
            current_prediction = model.predict(X_single)[0]
            current_actual = y_test.iloc[row_num - 1]
            
        else:
            # Option 2: Enter new data
            st.markdown("### 📝 Enter New Data")
            
            if st.session_state.X_test_original is not None:
                new_data_dict = {}
                original_df = st.session_state.original_df
                
                # Create 2 columns for better layout
                num_cols = 2
                feature_cols = st.columns(num_cols)
                
                for idx, feature in enumerate(original_columns):
                    col_idx = idx % num_cols
                    with feature_cols[col_idx]:
                        if feature in original_df.columns:
                            unique_vals = original_df[feature].nunique()
                            
                            if unique_vals <= 10:  # Categorical
                                categories = original_df[feature].dropna().unique().tolist()
                                categories.sort()
                                
                                if unique_vals == 2:  # Binary
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
                            else:  # Numerical
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
        
        # Display prediction and explanation if available
        if X_single is not None:
            
            # Light prediction box
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
            
            # Generate local SHAP explanation
            with st.spinner("Generating explanation..."):
                try:
                    result = generate_shap_plots(
                        model, X_test[:100], X_single, feature_names, task
                    )
                    
                    if len(result) == 3:
                        fig_global, fig_local, shap_values_array = result
                    else:
                        fig_global, fig_local = result
                        shap_values_array = None
                except Exception as e:
                    st.error(f"Could not generate SHAP explanation: {str(e)}")
                    fig_local = None
                    shap_values_array = None
            
            # Display SHAP graph
            if fig_local:
                st.markdown("### 📊 Feature Contributions")
                st.pyplot(fig_local, use_container_width=True)
                
                # Axis labels (one below the other)
                st.markdown('<div class="axis-label">📊 X-axis = Impact on Prediction</div>', unsafe_allow_html=True)
                st.markdown('<div class="axis-label">📈 Y-axis = Features</div>', unsafe_allow_html=True)
            else:
                st.warning("Local explanation plot could not be generated.")
            
            st.markdown("---")
            
            # ========== UPDATED: Why This Prediction? (Matching Global Pattern) ==========
            st.markdown("### 🧠 Why This Prediction?")
            st.markdown("*Here's why the model made this specific prediction:*")
            
            # Get SHAP values for this specific row
            if shap_values_array is not None and len(shap_values_array) > 0:
                try:
                    # Get the correct shap row
                    if len(shap_values_array.shape) == 2:
                        if len(shap_values_array) == 1:
                            shap_row = shap_values_array[0].flatten()
                        else:
                            shap_row = shap_values_array.flatten()
                    else:
                        shap_row = shap_values_array.flatten()
                    
                    # Calculate percentages for this row
                    vals = np.abs(shap_row)
                    if vals.sum() > 0:
                        perc = (vals / vals.sum()) * 100
                    else:
                        perc = np.zeros_like(vals)
                    
                    # Sort by absolute impact
                    sorted_indices = np.argsort(np.abs(shap_row))[::-1][:5]
                    
                    # Display each feature exactly like global explanation
                    for idx in sorted_indices:
                        idx = int(idx)
                        if idx < len(feature_names):
                            shap_val = shap_row[idx]
                            
                            # Determine direction (matching global pattern)
                            if shap_val > 0.05:
                                direction = "pushes prediction HIGHER"
                                color = "#28a745"
                                icon = "📈"
                            elif shap_val < -0.05:
                                direction = "pushes prediction LOWER"
                                color = "#dc3545"
                                icon = "📉"
                            else:
                                direction = "has mixed or minimal impact"
                                color = "#6c757d"
                                icon = "⚖️"
                            
                            # Display exactly like global format (without the value line)
                            st.markdown(f"""
                            <div class="explanation-box" style="border-left-color: {color};">
                                <b>{icon} {feature_names[idx]}</b><br>
                                → <b>Contribution: {perc[idx]:.1f}%</b> of total impact<br>
                                → {direction}
                            </div>
                            """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.warning(f"Could not calculate detailed explanation: {str(e)[:100]}")
                    # Fallback: Show feature values
                    st.markdown("**Feature values for this prediction:**")
                    for i, feat in enumerate(feature_names[:10]):
                        if i < len(X_single.columns):
                            value = X_single.iloc[0, i]
                            st.text(f"{feat}: {value}")
            else:
                st.warning("SHAP values not available for detailed explanation")
                # Fallback: Show feature values
                st.markdown("**Feature values for this prediction:**")
                for i, feat in enumerate(feature_names[:10]):
                    if i < len(X_single.columns):
                        value = X_single.iloc[0, i]
                        st.text(f"{feat}: {value}")
            
            st.info("💡 **How to read this:** Features with higher percentages have more influence on this prediction.")
    
    st.markdown('</div>', unsafe_allow_html=True)