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
    
    try:
        # For tree-based models (Random Forest)
        if 'RandomForest' in str(type(model)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle classification with multiple classes
            if isinstance(shap_values, list):
                if len(shap_values) > 1:
                    shap_values = shap_values[1]
                else:
                    shap_values = shap_values[0]
        
        # For linear models
        elif 'Linear' in str(type(model)) or 'Logistic' in str(type(model)):
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        
        # Default to KernelExplainer
        else:
            background = X_sample.iloc[:min(100, len(X_sample))]
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample.iloc[:min(200, len(X_sample))])
            
    except Exception as e:
        print(f"SHAP explainer failed: {str(e)[:100]}")
        return None, None, None
    
    # Create global summary plot
    plt.figure(figsize=(10, 6), facecolor='white')
    
    try:
        if len(shap_values) != len(X_sample):
            X_sample_for_plot = X_sample.iloc[:len(shap_values)]
        else:
            X_sample_for_plot = X_sample
            
        shap.summary_plot(
            shap_values, 
            X_sample_for_plot, 
            plot_type="violin",
            show=False,
            max_display=10,
            color_bar=True
        )
        plt.title("Feature Impact on Model Predictions", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
    except Exception as e:
        plt.clf()
        shap.summary_plot(
            shap_values, 
            X_sample_for_plot, 
            plot_type="bar",
            show=False,
            max_display=10
        )
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
            if 'TreeExplainer' in str(explainer):
                shap_single = explainer.shap_values(X_single)
                if isinstance(shap_single, list):
                    shap_single = shap_single[1] if len(shap_single) > 1 else shap_single[0]
            else:
                shap_single = explainer.shap_values(X_single)
                if isinstance(shap_single, list):
                    shap_single = shap_single[1] if len(shap_single) > 1 else shap_single[0]
            
            if len(shap_single.shape) == 3:
                shap_single = shap_single[:, :, 0]
            
            plt.figure(figsize=(10, 6), facecolor='white')
            
            if hasattr(explainer, 'expected_value'):
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            else:
                expected_value = 0
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_single[0] if len(shap_single.shape) > 1 else shap_single,
                    base_values=expected_value,
                    data=X_single.iloc[0].values,
                    feature_names=X_single.columns.tolist()
                ),
                show=False,
                max_display=10
            )
            
            plt.tight_layout()
            fig_local = plt.gcf()
            plt.close()
            
        except Exception as e:
            print(f"Waterfall plot failed: {str(e)[:50]}")
            fig_local = None
    
    return fig_global, fig_local, shap_values