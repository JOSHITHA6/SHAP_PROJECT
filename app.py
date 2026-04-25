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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "input"

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
                    "target_name": target
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
            st.markdown("### 🧠 Why This Behavior?")
            st.markdown("*Here's what the model learned from your data:*")
            
            # Calculate feature importance from SHAP values if available
            if shap_values_array is not None and len(shap_values_array) > 0:
                try:
                    vals = np.abs(shap_values_array).mean(axis=0).flatten()
                    perc = (vals / vals.sum()) * 100
                    
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
        
        # ---------- LOCAL EXPLAINABILITY TAB ----------
        with tab2:
            st.subheader("🔍 Explain a Single Prediction")
            
            # Row selector
            max_row = len(X_test)
            row_num = st.number_input("Select row number to explain", min_value=1, max_value=max_row, value=1, step=1)
            
            st.markdown("---")
            st.markdown("### 📋 Selected Data Row")
            st.dataframe(test_display.iloc[[row_num - 1]], use_container_width=True)
            
            # Get single prediction
            X_single = X_test.iloc[row_num - 1:row_num]  # Fix: use iloc for DataFrame
            prediction = model.predict(X_single)[0]
            
            # Format prediction for display
            if task == "Classification":
                prediction_display = "✅ " + str(prediction) if prediction == y_test.iloc[row_num - 1] else "❌ " + str(prediction)
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                    <h3>🎯 Model Prediction</h3>
                    <h1>{prediction}</h1>
                    <p style="margin-top:10px;font-size:12px;opacity:0.8">Actual: {y_test.iloc[row_num - 1]}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                actual = y_test.iloc[row_num - 1]
                diff = prediction - actual
                diff_symbol = "▲" if diff > 0 else "▼" if diff < 0 else "●"
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h3>🎯 Model Prediction</h3>
                    <h1>{prediction:.2f}</h1>
                    <p style="margin-top:10px;font-size:12px;opacity:0.8">Actual: {actual:.2f} ({diff_symbol} {abs(diff):.2f})</p>
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
                st.caption("**X-axis = Impact on Prediction** | **Y-axis = Features**")
            else:
                st.warning("Local explanation plot could not be generated. Showing feature values instead.")
            
            st.markdown("---")
            st.markdown("### 🧠 Why This Prediction?")
            
            # Get SHAP values for this specific row if available
            if shap_values_array is not None and len(shap_values_array) > row_num - 1:
                try:
                    shap_row = shap_values_array[row_num - 1].flatten()
                    
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
                        
                        # Get the actual value of this feature for the selected row
                        try:
                            # Fix: Use iloc and column name
                            feature_value = X_single.iloc[0, idx] if idx < len(X_single.columns) else "N/A"
                            if isinstance(feature_value, (int, float)):
                                if isinstance(feature_value, float):
                                    value_display = f"{feature_value:.2f}"
                                else:
                                    value_display = str(feature_value)
                            else:
                                value_display = str(feature_value)
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
                    st.warning(f"Could not calculate SHAP values for this row: {str(e)[:100]}")
                    # Fallback to showing actual feature values
                    st.markdown("**Feature values for this prediction:**")
                    for i, feat in enumerate(feature_names[:10]):  # Show top 10 features
                        if i < len(X_single.columns):
                            value = X_single.iloc[0, i]
                            st.text(f"{feat}: {value}")
            else:
                # Fallback to showing feature values
                st.markdown("**Feature values for this prediction:**")
                for i, feat in enumerate(feature_names[:10]):  # Show top 10 features
                    if i < len(X_single.columns):
                        value = X_single.iloc[0, i]
                        st.text(f"{feat}: {value}")
            
            st.info("💡 **What this means:** The features listed above are why the model made THIS specific prediction. Green features pushed the prediction higher, red features pushed it lower.")