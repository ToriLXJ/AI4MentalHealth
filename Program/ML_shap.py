import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime

def clean_feature_name(name, variable_mapping=None):
    """Clean feature names and apply variable mapping if provided"""
    if variable_mapping and name in variable_mapping:
        return variable_mapping[name]
    
    name_clean = re.sub(r'(_y\d+|_T\d+|_bin|_dtbin)$', '', name)
    
    if variable_mapping and name_clean in variable_mapping:
        return variable_mapping[name_clean]
    
    return name_clean

def calculate_shap_values(model, preprocessor, X_test, feature_names, variable_mapping=None, top_n=20):
    """
    Calculate SHAP values and return important features
    """
    if hasattr(model, 'named_steps') and 'rf' in model.named_steps:
        core_model = model.named_steps['rf']
    elif hasattr(model, 'named_steps') and 'model' in model.named_steps:
        core_model = model.named_steps['model']
    else:
        core_model = model

    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    explainer = shap.TreeExplainer(
        core_model,
        feature_perturbation="interventional",
        model_output="raw"
    )
    
    # Calculate SHAP values
    shap_values = explainer(X_test_df)
    
    return shap_values, X_test_df

def save_plot(filename, fig=None, dpi=400, flag_show=True):
    """
    Save plot to PNG file and optionally display it
    
    Parameters:
    filename: Output filename (without extension)
    fig: Figure object (if None, uses current figure)
    dpi: Image resolution
    flag_show: Whether to display the plot
    """
    if fig is None:
        fig = plt.gcf()
    
    os.makedirs("plots", exist_ok=True)
    
    full_filename = f"plots/{filename}.png"
    
    fig.savefig(full_filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {full_filename}")
    
    if flag_show:
        plt.show()
    else:
        plt.close(fig)  # Close figure to free memory
    
    return full_filename

def plot_shap_summary_simple(shap_values, X_test_df, variable_mapping=None, top_n=20, 
                             is_log_transformed=False, flag_show=True, flag_title=True, suffix="xgboost"):
    """
    Plot simple SHAP summary plot without importance bars
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1]
    
    # Select top N features
    idx = order[:top_n]
    shap_values_top = shap_values.values[:, idx]
    
    # Clean feature names with variable mapping
    cleaned_columns = [clean_feature_name(col, variable_mapping) for col in X_test_df.columns[idx]]
    X_values_top = X_test_df.iloc[:, idx].copy()
    X_values_top.columns = cleaned_columns
    
    # Set figure style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure with specified size and DPI
    plt.figure(figsize=(10, 8), dpi=400)
    
    # Plot SHAP summary
    shap.summary_plot(
        shap_values_top,
        X_values_top,
        max_display=top_n,
        show=False,
        alpha=0.7,
        plot_size=None
    )
    
    # Get current axes and set labels
    ax = plt.gca()
    ax.set_xlabel("SHAP Value (Impact on Model Output)", 
                  fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    
    # Set font for all text elements
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontfamily('Times New Roman')
    
    # Adjust color bar label
    for axis in plt.gcf().axes:
        if axis != ax and hasattr(axis, 'set_ylabel'):
            axis.set_ylabel('Feature Value', fontsize=11, rotation=270, labelpad=20, 
                           fontfamily='Times New Roman')
            break
    
    # Add title with log transformation info (if flag_title is True)
    if flag_title:
        title = "SHAP Feature Importance Summary"
        if is_log_transformed:
            title += " (Log-Transformed Data)"
        
        plt.title(title, fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    
    plt.tight_layout()
    
    # Save and show plot
    save_plot(f"shap_summary_simple_{suffix}", plt.gcf(), flag_show=flag_show)
    
    return mean_abs_shap, order

def plot_shap_summary(shap_values, X_test_df, variable_mapping=None, top_n=20, 
                      is_log_transformed=False, flag_show=True, flag_title=True, suffix="xgboost"):
    """
    Plot SHAP summary plot with feature importance bar chart
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1]
    
    # Select top N features
    idx = order[:top_n]
    shap_values_top = shap_values.values[:, idx]
    
    # Clean feature names with variable mapping
    cleaned_columns = [clean_feature_name(col, variable_mapping) for col in X_test_df.columns[idx]]
    X_values_top = X_test_df.iloc[:, idx].copy()
    X_values_top.columns = cleaned_columns
    
    mean_shap_top = mean_abs_shap[idx]
    
    # Set figure style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure with specified size and DPI
    fig, ax = plt.subplots(figsize=(10, 8), dpi=400)
    
    # Plot SHAP summary
    shap.summary_plot(
        shap_values_top,
        X_values_top,
        max_display=top_n,
        show=False,
        alpha=0.7,
        plot_size=None
    )
    
    # Get current axes
    ax = plt.gca()
    yticks = ax.get_yticks()
    
    # Create top axes for feature importance
    ax2 = ax.twiny()
    
    # Plot feature importance bar chart
    for i in range(len(mean_shap_top)):
        if i < len(yticks):
            y_pos = yticks[i]
            ax2.barh(
                y_pos,
                mean_shap_top[top_n - 1 - i],
                height=0.7,
                color="#D6EAF8",
                alpha=0.5,
                edgecolor="none",
                zorder=1
            )
    
    # Set axis labels
    max_mean_shap = mean_shap_top.max()
    ax2.set_xlim(0, max_mean_shap * 1.05)
    ax2.set_xlabel("Mean SHAP Value (Feature Importance)", 
                   fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    
    ax.set_xlabel("SHAP Value (Impact on Model Output)", 
                  fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    
    # Add title with log transformation info (if flag_title is True)
    if flag_title:
        title = "SHAP Feature Importance Summary with Importance Bars"
        if is_log_transformed:
            title += " (Log-Transformed Data)"
        
        ax.set_title(title, fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    
    # Adjust color bar
    for axis in fig.axes:
        if axis != ax and axis != ax2 and hasattr(axis, 'set_ylabel'):
            axis.set_ylabel('Feature Value', fontsize=11, rotation=270, labelpad=20, 
                           fontfamily='Times New Roman')
            break
    
    # Style the plot
    ax.grid(axis='x', alpha=0.15, linestyle='--', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Set font for all text elements
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontfamily('Times New Roman')
    
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontfamily('Times New Roman')
    
    plt.tight_layout()
    
    # Save and show plot
    save_plot(f"shap_summary_with_bars_{suffix}", fig, flag_show=flag_show)
    
    return mean_abs_shap, order

def plot_dependence_plots(shap_values, X_test_df, top_features, variable_mapping=None, 
                          n_cols=2, is_log_transformed=False, flag_show=True, flag_title=True, suffix="xgboost"):
    """
    Plot SHAP dependence plots
    """
    n_features = len(top_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure with specified size and DPI
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=400)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    # Set global font
    plt.rcParams['font.family'] = 'Times New Roman'
    
    for idx, feature in enumerate(top_features):
        if feature in X_test_df.columns:
            shap.dependence_plot(
                feature,
                shap_values.values,
                X_test_df,
                ax=axes[idx],
                show=False
            )
            
            # Clean feature name with variable mapping

            clean_name = clean_feature_name(feature, variable_mapping)
            
            # Set title with log transformation info (if flag_title is True)
            if flag_title:
                title = f"Dependence: {clean_name}"
                if is_log_transformed:
                    title += " (Log-Transformed)"
                    
                axes[idx].set_title(title, fontsize=11, weight="bold", fontfamily='Times New Roman')
            
            # Set font for axis labels
            axes[idx].set_xlabel(clean_name, fontfamily='Times New Roman')
            axes[idx].set_ylabel("SHAP Value", fontfamily='Times New Roman')
            
            # Set font for tick labels
            for item in (axes[idx].get_xticklabels() + axes[idx].get_yticklabels()):
                item.set_fontfamily('Times New Roman')
    
    # Hide extra subplots
    for idx in range(len(top_features), len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title (if flag_title is True)
    if flag_title:
        fig_title = "SHAP Dependence Plots"
        if is_log_transformed:
            fig_title += " (Log-Transformed Data)"
        
        fig.suptitle(fig_title, fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    
    plt.tight_layout()
    if flag_title:
        plt.subplots_adjust(top=0.93)  # Adjust for suptitle
    
    # Save and show plot
    save_plot(f"shap_dependence_plots_{suffix}", fig, flag_show=flag_show)

def plot_feature_importance_bar(mean_abs_shap, feature_names, variable_mapping=None, 
                               top_n=20, is_log_transformed=False, flag_show=True, flag_title=True, suffix="xgboost"):
    """
    Plot feature importance as a bar chart
    """
    # Create DataFrame with feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    })
    
    # Sort and select top features
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # Clean feature names with variable mapping
    importance_df['clean_name'] = importance_df['feature'].apply(
        lambda x: clean_feature_name(x, variable_mapping)
    )
    
    # Create figure with specified size and DPI
    plt.figure(figsize=(10, 8), dpi=400)
    
    # Create horizontal bar plot
    plt.barh(range(len(importance_df)), importance_df['importance'], 
             color='skyblue', alpha=0.7)
    
    # Set y-ticks and labels
    plt.yticks(range(len(importance_df)), importance_df['clean_name'], 
               fontfamily='Times New Roman')
    
    # Set labels
    plt.xlabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold', 
               fontfamily='Times New Roman')
    plt.ylabel('Features', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    
    # Add title (if flag_title is True)
    if flag_title:
        title = 'Feature Importance Based on SHAP Values'
        if is_log_transformed:
            title += ' (Log-Transformed Data)'
        
        plt.title(title, fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    
    # Invert y-axis to have highest importance at top
    plt.gca().invert_yaxis()
    
    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save and show plot
    save_plot(f"feature_importance_bar_{suffix}", plt.gcf(), flag_show=flag_show)
    
    return importance_df

def interpret_log_transformed_shap(shap_values, features, base_feature_names, 
                                  variable_mapping=None, base_value=None, is_log_transformed=False):
    """
    Specialized interpretation of SHAP values for log-transformed data
    """
    interpretation_results = []
    
    for i, feature_name in enumerate(base_feature_names):
        # Use clean_feature_name function to get readable name
        readable_name = clean_feature_name(feature_name, variable_mapping)
        mean_abs_shap = np.mean(np.abs(shap_values[:, i]))
        mean_feature_value = np.mean(features.iloc[:, i])
        
        # Interpretation for log-transformed data
        if is_log_transformed:
            interpretation_text = f"Log({readable_name}) increase by 1 unit corresponds to approximately {mean_abs_shap:.4f} change in prediction"
        else:
            interpretation_text = f"{readable_name} increase by 1 unit corresponds to approximately {mean_abs_shap:.4f} change in prediction"
        
        interpretation = {
            'feature': feature_name,
            'clean_name': readable_name,
            'mean_abs_shap': mean_abs_shap,
            'mean_feature_value': mean_feature_value,
            'interpretation': interpretation_text
        }
        
        interpretation_results.append(interpretation)
    
    return pd.DataFrame(interpretation_results)

def save_shap_results_to_excel(importance_df, interpretation_df, shap_values_df=None, 
                               is_log_transformed=False, filename_prefix="SHAP_Results"):
    """
    Save SHAP analysis results to Excel file with multiple sheets
    """
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs("output", exist_ok=True)
    
    filename = f"output/{filename_prefix}_{timestamp}.xlsx"
    
    # Create Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Save feature importance
        importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
        
        # Save SHAP interpretations
        interpretation_df.to_excel(writer, sheet_name='SHAP_Interpretations', index=False)
        
        # Save raw SHAP values if provided
        if shap_values_df is not None:
            shap_values_df.to_excel(writer, sheet_name='Raw_SHAP_Values', index=False)
        
        # Save summary statistics
        summary_data = {
            'Analysis_Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'Total_Features': [len(importance_df)],
            'Top_Feature': [importance_df.iloc[0]['clean_name']],
            'Top_Feature_Importance': [importance_df.iloc[0]['importance']],
            'Data_Transformation': ['Log-Transformed' if is_log_transformed else 'Original']
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Results saved to: {filename}")
    return filename

def get_original_test_data(df_model, X_te, selected_vars, results):
    if hasattr(X_te, 'columns'):
        return X_te[selected_vars]
    else:
        if hasattr(results['y_test'], 'index'):
            test_indices = results['y_test'].index
            return df_model.loc[test_indices, selected_vars]
        else:
            test_size = len(X_te)
            return df_model[selected_vars].iloc[-test_size:]

def run_full_shap_analysis(model_name, results, df_model, selected_vars, variable_mapping, 
                            is_log_transformed=True, top_n=20, flag_show=False, flag_title=False):
    print("="*60)
    print(f"Starting Full SHAP Analysis for {model_name}")
    print("="*60)
    
    model = results['model']
    preprocessor = results['preprocessor']
    X_te = results['X_test']
    
    X_test_original = get_original_test_data(df_model, X_te, selected_vars, results)
    
    shap_values, X_test_df = calculate_shap_values(
        model=model,
        preprocessor=preprocessor,
        X_test=X_test_original,
        feature_names=selected_vars,
        variable_mapping=variable_mapping,
        top_n=top_n
    )
    print("-> SHAP values calculated successfully.")

    suffix = model_name.lower().replace(" ", "_")
    
    # 1. 绘制简单SHAP摘要图
    mean_abs_shap, order = plot_shap_summary_simple(
        shap_values=shap_values,
        X_test_df=X_test_df,
        variable_mapping=variable_mapping,
        top_n=top_n,
        is_log_transformed=is_log_transformed,
        flag_show=flag_show,
        flag_title=flag_title,
        suffix=suffix
    )
    print(f"-> 1/6 Plotted Simple Summary.")

    # 2. 绘制特征重要性条形图
    importance_df = plot_feature_importance_bar(
        mean_abs_shap=mean_abs_shap,
        feature_names=X_test_df.columns,
        variable_mapping=variable_mapping,
        top_n=top_n,
        is_log_transformed=is_log_transformed,
        flag_show=flag_show,
        flag_title=flag_title,
        suffix=suffix
    )
    print(f"-> 2/6 Plotted Feature Importance Bar.")

    # 3. 绘制带重要性条形图的SHAP摘要图 (标准Summary Plot)
    mean_abs_shap, order = plot_shap_summary(
        shap_values=shap_values,
        X_test_df=X_test_df,
        variable_mapping=variable_mapping,
        top_n=top_n,
        is_log_transformed=is_log_transformed,
        flag_show=flag_show,
        flag_title=flag_title,
        suffix=suffix
    )
    print(f"-> 3/6 Plotted Standard SHAP Summary.")

    # 4. 绘制依赖图
    top_features_for_dependence = X_test_df.columns[order[1:5]] 
    plot_dependence_plots(
        shap_values=shap_values,
        X_test_df=X_test_df,
        top_features=top_features_for_dependence,
        variable_mapping=variable_mapping,
        n_cols=2,
        is_log_transformed=is_log_transformed,
        flag_show=flag_show,
        flag_title=flag_title,
        suffix=suffix
    )
    print(f"-> 4/6 Plotted Dependence Plots for {len(top_features_for_dependence)} features.")

    # 5. 使用专门解释函数
    interpretation_df = interpret_log_transformed_shap(
        shap_values=shap_values.values,
        features=X_test_df,
        base_feature_names=X_test_df.columns.tolist(),
        variable_mapping=variable_mapping,
        is_log_transformed=is_log_transformed
    )
    print(f"-> 5/6 Generated Log-Transformed Interpretation.")

    # 6. 保存结果到Excel
    shap_values_df = pd.DataFrame(shap_values.values, columns=X_test_df.columns)
    shap_values_df_clean = shap_values_df.copy()
    shap_values_df_clean.columns = [clean_feature_name(col, variable_mapping) for col in shap_values_df.columns]

    excel_filename = save_shap_results_to_excel(
        importance_df=importance_df,
        interpretation_df=interpretation_df,
        shap_values_df=shap_values_df_clean,
        filename_prefix=f"SHAP_Analysis_{suffix}"
    )
    print(f"-> 6/6 Saved results to Excel: {excel_filename}")
    
    print("="*60)
    print(f"Full SHAP Analysis for {model_name} COMPLETED.")
    print("="*60)
