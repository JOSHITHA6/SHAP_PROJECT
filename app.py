import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.preprocess import preprocess_data
from BACKEND.model import train_model
from BACKEND.shap_explainer import generate_shap_plots

from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(layout="wide", page_title="SHAP AI Explainability Tool", page_icon="📊")

# Custom CSS for consistent box styling
st.markdown("""
<style>
    .explanation-box {
        padding: 12px;
        margin-bottom: 10px;
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 6px solid;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .explanation-box:hover {
        background: #f0f2f6;
        transition: 0.2s;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 14px;
        opacity: 0.9;
    }
    .metric-card h1 {
        margin: 10px 0 0 0;
        font-size: 36px;
        font-weight: bold;
    }
    .axis-label {
        background: #f0f2f6;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 14px;
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

# ================= INPUT PAGE =================
if st.session_state.page == "input":
    
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
        if st.button("🚀 Run Model & Generate Explanations", type="primary", use_container_width=True):
            
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
                    "X_test_original": X_test_original
                })
                
                st.session_state.page = "output"
                st.rerun()

# ================= OUTPUT PAGE =================
elif st.session_state.page == "output":
    
    st.title("📊 Model Output & Explainability Dashboard")
    st.markdown("*Understand what your model learned and why it makes each prediction*")
    
    # Back button
    if st.button("⬅️ Back to Upload", use_container_width=True):
        st.session_state.page = "input"
        st.rerun()
    
    st.markdown("---")
    
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
    
    # Create two columns
    left_col, right_col = st.columns([1.1, 1.4], gap="large")
    
    # ========== LEFT COLUMN ==========
    with left_col:
        
        st.subheader("📄 Test Dataset (20% of data)")
        st.dataframe(test_display, height=400, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📈 Model Performance")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Display single metric in a nice card
        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Accuracy Score</h3>
                <h1>{acc:.2%}</h1>
                <p style="margin-top:10px;font-size:12px;opacity:0.8">Correct predictions / Total predictions</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample predictions
            with st.expander("🔍 View Sample Predictions"):
                sample_df = test_display.head(10).copy()
                sample_df["Prediction"] = y_pred[:10]
                st.dataframe(sample_df, use_container_width=True)
                
        else:
            r2 = r2_score(y_test, y_pred)
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 R² Score (Coefficient of Determination)</h3>
                <h1>{r2:.3f}</h1>
                <p style="margin-top:10px;font-size:12px;opacity:0.8">Higher = Better (1.0 = perfect prediction)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample predictions
            with st.expander("🔍 View Sample Predictions"):
                sample_df = test_display.head(10).copy()
                sample_df["Prediction"] = y_pred[:10]
                st.dataframe(sample_df, use_container_width=True)
    
    # ========== RIGHT COLUMN ==========
    with right_col:
        
        tab1, tab2 = st.tabs(["🌍 Global Explainability", "🔍 Local Explainability"])
        
        # ---------- GLOBAL EXPLAINABILITY TAB ----------
        with tab1:
            st.subheader("🌍 How Features Impact Predictions (Global)")
            
            # Generate SHAP plots
            with st.spinner("Calculating feature impacts..."):
                result = generate_shap_plots(
                    model, X_test[:100], feature_names=feature_names, task=task
                )
                
                # Handle different return formats
                if len(result) == 3:
                    fig_global, _, shap_values_array = result
                else:
                    st.warning("SHAP explanation could not be generated. Showing feature importance from model.")
                    fig_global = None
                    shap_values_array = None
            
            # Display the plot
            if fig_global:
                st.pyplot(fig_global, use_container_width=True)
                st.caption("📌 Each violin shows how much a feature influences predictions across all data points")
            else:
                # Fallback to feature importance from model
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:10]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(range(len(indices)), importances[indices])
                    ax.set_yticks(range(len(indices)))
                    ax.set_yticklabels([feature_names[i] for i in indices])
                    ax.set_xlabel("Feature Importance")
                    ax.set_title("Feature Importance (Model-based)")
                    st.pyplot(fig)
                    plt.close()
            
            st.markdown("---")
            
            # ========== NEW: OVERALL MODEL PREDICTION SECTION ==========
            st.markdown("### 🎯 Overall Model Prediction")
            
            if task == "Classification":
                # Get majority prediction
                from collections import Counter
                majority_pred = Counter(y_pred).most_common(1)[0][0]
                st.markdown(f"""
                <div class="explanation-box" style="border-left-color: #667eea; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);">
                    <b>📊 The model generally predicts:</b><br>
                    <span style="font-size: 20px; font-weight: bold;">{majority_pred}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Get average predicted value
                avg_pred = np.mean(y_pred)
                st.markdown(f"""
                <div class="explanation-box" style="border-left-color: #667eea; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);">
                    <b>📊 Average predicted value:</b><br>
                    <span style="font-size: 20px; font-weight: bold;">{avg_pred:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🧠 Why This Behavior?")
            st.markdown("*Here's what the model learned from your data:*")
            
            # Calculate feature importance from SHAP values if available
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
                        
                        # Determine direction and color
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
            else:
                # Fallback to model feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:5]
                    for idx in indices:
                        st.markdown(f"""
                        <div class="explanation-box" style="border-left-color: #667eea;">
                            <b>📊 {feature_names[idx]}</b><br>
                            → <b>Importance: {importances[idx]*100:.1f}%</b><br>
                            → This feature strongly influences predictions
                        </div>
                        """, unsafe_allow_html=True)
            
            st.info("💡 **How to read this:** Features with higher percentages have more influence on predictions. The direction tells you whether higher feature values generally increase or decrease the prediction.")
        
        # ---------- LOCAL EXPLAINABILITY TAB (ENHANCED) ----------
        with tab2:
            st.subheader("🔍 Explain a Single Prediction")
            
            # ========== NEW: Two options for local explanation ==========
            explanation_source = st.radio(
                "Choose input method:",
                ["📊 Select Row from Test Data", "✏️ Enter New Data"],
                horizontal=True,
                help="Select existing test data row or enter your own values"
            )
            
            X_single = None
            original_values = None
            
            if explanation_source == "📊 Select Row from Test Data":
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
                st.markdown("### 📝 Enter New Data for Prediction")
                st.markdown("*Fill in the values below to see what the model would predict*")
                
                # Get original feature information from the training data
                if st.session_state.X_test_original is not None:
                    # Create input fields dynamically based on original feature types
                    new_data_dict = {}
                    
                    # Get original dataframe for categorical value references
                    original_df = st.session_state.original_df
                    
                    # Create columns for better layout
                    num_cols = 2
                    feature_cols = st.columns(num_cols)
                    
                    for idx, feature in enumerate(original_columns):
                        col_idx = idx % num_cols
                        with feature_cols[col_idx]:
                            # Determine feature type from original data
                            if feature in original_df.columns:
                                unique_vals = original_df[feature].nunique()
                                
                                # Check if it's categorical or binary
                                if unique_vals <= 10:  # Treat as categorical
                                    categories = original_df[feature].dropna().unique().tolist()
                                    # Sort categories for better UX
                                    categories.sort()
                                    
                                    # Handle binary specially
                                    if unique_vals == 2:
                                        # Try to map to Yes/No if values are 0/1
                                        if set(categories) == {0, 1}:
                                            display_categories = ["No (0)", "Yes (1)"]
                                            value_map = {"No (0)": 0, "Yes (1)": 1}
                                        else:
                                            display_categories = [str(c) for c in categories]
                                            value_map = {str(c): c for c in categories}
                                        
                                        selected = st.selectbox(
                                            f"🔘 {feature}",
                                            options=display_categories,
                                            help=f"Select value for {feature}"
                                        )
                                        new_data_dict[feature] = value_map[selected]
                                    else:
                                        # Multi-class categorical
                                        selected = st.selectbox(
                                            f"📋 {feature}",
                                            options=[str(c) for c in categories],
                                            help=f"Select category for {feature}"
                                        )
                                        # Find original value
                                        for cat in categories:
                                            if str(cat) == selected:
                                                new_data_dict[feature] = cat
                                                break
                                else:
                                    # Numerical feature
                                    min_val = float(original_df[feature].min())
                                    max_val = float(original_df[feature].max())
                                    mean_val = float(original_df[feature].mean())
                                    
                                    new_data_dict[feature] = st.number_input(
                                        f"🔢 {feature}",
                                        value=mean_val,
                                        min_value=min_val,
                                        max_value=max_val,
                                        step=(max_val - min_val) / 100 if max_val > min_val else 1.0,
                                        format="%.4f" if abs(max_val - min_val) < 1 else "%.2f",
                                        help=f"Range: [{min_val:.2f}, {max_val:.2f}]"
                                    )
                            else:
                                # Fallback to number input
                                new_data_dict[feature] = st.number_input(f"🔢 {feature}", value=0.0)
                    
                    # Convert to DataFrame
                    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
                        try:
                            # Create DataFrame with new data
                            new_data_df = pd.DataFrame([new_data_dict])
                            
                            # Apply the same preprocessing pipeline
                            X_single_processed = preprocessor.transform(new_data_df)
                            
                            # Convert to DataFrame with feature names
                            X_single = pd.DataFrame(X_single_processed, columns=feature_names)
                            
                            current_prediction = model.predict(X_single)[0]
                            current_actual = None  # No actual value for new data
                            original_values = new_data_dict
                            
                            st.success("✅ Prediction generated successfully!")
                        except Exception as e:
                            st.error(f"Error processing input: {str(e)}")
                            st.info("Please check your input values and try again.")
                            X_single = None
                    
                    if X_single is None:
                        st.info("👈 Fill in the values above and click 'Generate Prediction'")
                
                else:
                    st.warning("No reference data available for input fields")
            
            # Display prediction if we have one
            if X_single is not None:
                # Format prediction for display
                if task == "Classification":
                    if explanation_source == "📊 Select Row from Test Data":
                        is_correct = current_prediction == current_actual
                        prediction_display = "✅ " + str(current_prediction) if is_correct else "❌ " + str(current_prediction)
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                            <h3>🎯 Model Prediction</h3>
                            <h1>{current_prediction}</h1>
                            <p style="margin-top:10px;font-size:12px;opacity:0.8">Actual: {current_actual}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                            <h3>🎯 Model Prediction</h3>
                            <h1>{current_prediction}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    if explanation_source == "📊 Select Row from Test Data":
                        actual = current_actual
                        diff = current_prediction - actual
                        diff_symbol = "▲" if diff > 0 else "▼" if diff < 0 else "●"
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                            <h3>🎯 Model Prediction</h3>
                            <h1>{current_prediction:.2f}</h1>
                            <p style="margin-top:10px;font-size:12px;opacity:0.8">Actual: {actual:.2f} ({diff_symbol} {abs(diff):.2f})</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
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
                
                # Display waterfall plot
                if fig_local:
                    st.markdown("### 📊 How Each Feature Contributed")
                    st.pyplot(fig_local, use_container_width=True)
                    
                    # ========== FIXED: Axis labels (one below the other) ==========
                    st.markdown("---")
                    st.markdown("### 📐 Chart Explanation")
                    st.markdown('<div class="axis-label">📊 X-axis = Impact on Prediction</div>', unsafe_allow_html=True)
                    st.markdown('<div class="axis-label">📈 Y-axis = Features</div>', unsafe_allow_html=True)
                else:
                    st.warning("Local explanation plot could not be generated.")
                
                st.markdown("---")
                st.markdown("### 🧠 Why This Prediction?")
                
                # Get SHAP values for this specific row if available
                if shap_values_array is not None and len(shap_values_array) > 0:
                    try:
                        # Get the correct index
                        if len(shap_values_array.shape) == 2:
                            shap_row = shap_values_array[0].flatten() if len(shap_values_array) == 1 else shap_values_array.flatten()
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
                        
                        for idx in sorted_indices:
                            idx = int(idx)
                            if idx < len(feature_names):
                                shap_val = shap_row[idx]
                                
                                # Determine direction and color
                                if shap_val > 0.05:
                                    direction = "pushes prediction HIGHER"
                                    color = "#28a745"
                                    icon = "📈"
                                elif shap_val < -0.05:
                                    direction = "pushes prediction LOWER"
                                    color = "#dc3545"
                                    icon = "📉"
                                else:
                                    direction = "has minimal impact on this prediction"
                                    color = "#6c757d"
                                    icon = "⚖️"
                                
                                # Get the actual value of this feature
                                try:
                                    if idx < len(X_single.columns):
                                        feature_value = X_single.iloc[0, idx]
                                        if isinstance(feature_value, (int, float)):
                                            if isinstance(feature_value, float):
                                                value_display = f"{feature_value:.3f}"
                                            else:
                                                value_display = str(feature_value)
                                        else:
                                            value_display = str(feature_value)
                                    elif original_values and feature_names[idx] in original_values:
                                        value_display = str(original_values[feature_names[idx]])
                                    else:
                                        value_display = "N/A"
                                except:
                                    value_display = "N/A"
                                
                                st.markdown(f"""
                                <div class="explanation-box" style="border-left-color: {color};">
                                    <b>{icon} {feature_names[idx]}</b><br>
                                    → <b>Value:</b> {value_display}<br>
                                    → <b>Contribution:</b> {perc[idx]:.1f}% of the explanation<br>
                                    → {direction}
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not calculate SHAP values for this prediction: {str(e)[:100]}")
                        # Fallback to showing feature values
                        st.markdown("**Feature values for this prediction:**")
                        for i, feat in enumerate(feature_names[:10]):
                            if i < len(X_single.columns):
                                value = X_single.iloc[0, i]
                                st.text(f"{feat}: {value}")
                else:
                    # Fallback to showing feature values
                    st.markdown("**Feature values for this prediction:**")
                    for i, feat in enumerate(feature_names[:10]):
                        if i < len(X_single.columns):
                            value = X_single.iloc[0, i]
                            st.text(f"{feat}: {value}")
                
                st.info("💡 **What this means:** The features listed above explain why the model made THIS specific prediction. Green features pushed the prediction higher, red features pushed it lower.")