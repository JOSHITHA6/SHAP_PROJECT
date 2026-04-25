import shap
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['figure.facecolor'] = 'white'

def generate_shap_plots(model, X_sample, X_single=None, feature_names=None, task="Regression"):
    """
    Generate SHAP explanation plots
    
    Parameters:
    - model: Trained model
    - X_sample: Sample data for background (for SHAP explainer)
    - X_single: Single row for local explanation (optional)
    - feature_names: Names of features
    - task: 'Classification' or 'Regression'
    
    Returns:
    - fig_global: Global SHAP summary plot
    - fig_local: Local SHAP waterfall plot (if X_single provided)
    - shap_values: Array of SHAP values
    """
    
    # Convert to DataFrame if needed
    if not isinstance(X_sample, pd.DataFrame):
        if feature_names is not None:
            X_sample = pd.DataFrame(X_sample, columns=feature_names)
        else:
            X_sample = pd.DataFrame(X_sample)
    
    # Create explainer based on model type
    try:
        # For tree-based models (Random Forest)
        if 'RandomForest' in str(type(model)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle classification with multiple classes
            if len(shap_values.shape) == 3:
                # For multi-class, take the class with highest predicted probability
                shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
        
        # For linear models
        elif 'Linear' in str(type(model)) or 'Logistic' in str(type(model)):
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        
        # Default to KernelExplainer
        else:
            # Use a smaller background for KernelExplainer (faster)
            background = X_sample.iloc[:min(100, len(X_sample))]
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample.iloc[:min(200, len(X_sample))])
            
    except Exception as e:
        # Fallback to KernelExplainer
        print(f"Using KernelExplainer (fallback): {str(e)[:100]}")
        background = X_sample.iloc[:min(50, len(X_sample))]
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_sample.iloc[:min(100, len(X_sample))])
        
        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    
    # Ensure shap_values is 2D
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]
    
    # Create global summary plot
    plt.figure(figsize=(10, 6), facecolor='white')
    
    try:
        shap.summary_plot(
            shap_values, 
            X_sample, 
            plot_type="violin",
            show=False,
            max_display=10,
            color_bar=True
        )
        plt.title("Feature Impact on Model Predictions", fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("SHAP Value (Impact on Prediction)", fontsize=11)
        plt.tight_layout()
    except Exception as e:
        # Fallback to bar plot if violin fails
        plt.clf()
        shap.summary_plot(
            shap_values, 
            X_sample, 
            plot_type="bar",
            show=False,
            max_display=10
        )
        plt.title("Feature Importance (Global)", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
    
    fig_global = plt.gcf()
    plt.close()
    
    # Create local waterfall plot if requested
    fig_local = None
    if X_single is not None:
        if not isinstance(X_single, pd.DataFrame):
            if feature_names is not None:
                X_single = pd.DataFrame(X_single, columns=feature_names)
            else:
                X_single = pd.DataFrame(X_single)
        
        try:
            # Get SHAP values for single instance
            if 'TreeExplainer' in str(explainer):
                shap_single = explainer.shap_values(X_single)
                if len(shap_single.shape) == 3:
                    shap_single = shap_single[:, :, 1] if shap_single.shape[2] > 1 else shap_single[:, :, 0]
            else:
                shap_single = explainer.shap_values(X_single)
                if isinstance(shap_single, list):
                    shap_single = shap_single[1] if len(shap_single) > 1 else shap_single[0]
            
            # Create waterfall plot
            plt.figure(figsize=(10, 6), facecolor='white')
            
            # Get expected value (base value)
            if hasattr(explainer, 'expected_value'):
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            else:
                expected_value = 0
            
            # Create waterfall
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_single[0],
                    base_values=expected_value,
                    data=X_single.iloc[0].values,
                    feature_names=X_single.columns.tolist()
                ),
                show=False,
                max_display=10
            )
            
            plt.title(f"Local Explanation: Why This Prediction?", fontsize=12, fontweight='bold')
            plt.tight_layout()
            fig_local = plt.gcf()
            plt.close()
            
        except Exception as e:
            # If waterfall fails, create a horizontal bar plot
            print(f"Waterfall plot failed, using bar plot: {str(e)[:50]}")
            plt.figure(figsize=(10, 6), facecolor='white')
            
            # Get SHAP values for single instance
            if 'TreeExplainer' in str(explainer):
                shap_single = explainer.shap_values(X_single)
                if len(shap_single.shape) == 3:
                    shap_single = shap_single[:, :, 1] if shap_single.shape[2] > 1 else shap_single[:, :, 0]
            else:
                shap_single = explainer.shap_values(X_single)
                if isinstance(shap_single, list):
                    shap_single = shap_single[1] if len(shap_single) > 1 else shap_single[0]
            
            # Create horizontal bar chart
            feature_impacts = shap_single[0]
            sorted_idx = np.argsort(np.abs(feature_impacts))[-10:]
            
            y_pos = np.arange(len(sorted_idx))
            plt.barh(y_pos, feature_impacts[sorted_idx])
            plt.yticks(y_pos, X_sample.columns[sorted_idx])
            plt.xlabel("SHAP Value (Impact on Prediction)")
            plt.title("Feature Contributions to This Prediction")
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            
            fig_local = plt.gcf()
            plt.close()
    
    return fig_global, fig_local, shap_values