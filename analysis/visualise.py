import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
import shap
from scipy.stats import spearmanr

mpl.use('Agg') # Ensure no GUI backend

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F9FAFB",
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

METHOD_COLORS = {
    1: "#6B7280",   # plain model
    2: "#3B82F6",   # classical SHAP
    3: "#10B981",   # causal
    4: "#F59E0B",   # layerwise
    5: "#8B5CF6",   # conditional
    6: "#EF4444",   # unified
}

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_shap_summary(shap_result: dict, max_features=15) -> str:
    method = shap_result.get('method', 'classical_shap')
    method_id_map = {
        'classical_shap': 2,
        'causal_shap': 3,
        'layerwise_shap': 4,
    }
    method_id = method_id_map.get(method, 2)

    titles = {
        'classical_shap': 'Classical SHAP — Feature Attribution Summary',
        'causal_shap': 'Causal SHAP (Fix 1) — Feature Attribution Summary',
        'layerwise_shap': 'Layer-wise SHAP (Fix 2) — Feature Attribution Summary',
    }
    title = titles.get(method, 'SHAP Feature Attribution Summary')
    color = METHOD_COLORS.get(method_id, METHOD_COLORS[2])

    mean_abs_shap = np.array(shap_result['mean_abs_shap'])
    feature_names = np.array(shap_result['feature_names'])
    
    # Sort descending
    sort_idx = np.argsort(mean_abs_shap)[::-1][:max_features]
    
    y_pos = np.arange(len(sort_idx))
    values = mean_abs_shap[sort_idx]
    labels = feature_names[sort_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y_pos, values, color=color, align='center')
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(title)
    
    return fig_to_base64(fig)

def plot_shap_beeswarm(shap_result: dict, max_features=15) -> str:
    sv = np.array(shap_result['shap_values'])
    profiles = np.array(shap_result['profiles'])
    feature_names = shap_result['feature_names']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Standard SHAP beeswarm using shap.summary_plot
    # show=False to save to buffer
    try:
        plt.close('all')
        shap.summary_plot(sv, profiles, feature_names=feature_names, max_display=max_features, show=False)
        # The plot modifies current figure
        fig = plt.gcf()
    except Exception as e:
        print(f"Error in beeswarm plot: {e}")
        # fallback if shap fails
        ax.text(0.5, 0.5, "Error rendering beeswarm plot", ha='center', va='center')
        
    return fig_to_base64(fig)

def plot_waterfall_single(shap_result: dict, sample_index=0) -> str:
    sv = np.array(shap_result['shap_values'])[sample_index]
    base_value = shap_result['base_value']
    profiles = np.array(shap_result['profiles'])[sample_index]
    feature_names = shap_result['feature_names']
    
    # SHAP 0.44+ waterfall needs an Explanation object for a single sample
    exp = shap.Explanation(
        values=sv,
        base_values=base_value,
        data=profiles,
        feature_names=feature_names
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        shap.plots.waterfall(exp, show=False)
        fig = plt.gcf()
    except Exception as e:
        print(f"Error in waterfall plot: {e}")
        ax.text(0.5, 0.5, "Error rendering waterfall plot", ha='center', va='center')
        
    return fig_to_base64(fig)

def plot_prediction_distribution(model, profiles, shap_result) -> str:
    # Use model to predict proba
    predictions = model.predict_proba(profiles)[:, 1]
    
    # Let's say threshold is 0.5
    classes = (predictions >= 0.5).astype(int)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot histogram colored by class
    sns.histplot(predictions[classes == 0], bins=20, color="#EF4444", label="Class 0 (<=50K)", kde=True, ax=ax, alpha=0.6)
    sns.histplot(predictions[classes == 1], bins=20, color="#10B981", label="Class 1 (>50K)", kde=True, ax=ax, alpha=0.6)
    
    ax.set_title("Prediction Probability Distribution")
    ax.set_xlabel("Predicted Probability (Class 1)")
    ax.set_ylabel("Count")
    ax.legend()
    
    return fig_to_base64(fig)

def plot_shap_vs_weights(shap_result: dict) -> str:
    # Extract
    mean_abs_shap = np.array(shap_result['mean_abs_shap'])
    feature_names = np.array(shap_result['feature_names'])
    model_importances_dict = shap_result.get('model_importances', None)
    
    if not model_importances_dict:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Model importances not found for comparison", ha='center', va='center')
        return fig_to_base64(fig)
        
    model_importances = np.array([model_importances_dict.get(f, 0.0) for f in feature_names])
    
    # Normalize both to 0-1 range for direct visual comparison
    if mean_abs_shap.max() > 0:
        norm_shap = mean_abs_shap / mean_abs_shap.max()
    else:
        norm_shap = mean_abs_shap
        
    if model_importances.max() > 0:
        norm_weights = model_importances / model_importances.max()
    else:
        norm_weights = model_importances
        
    # Sort by SHAP importance descending
    sort_idx = np.argsort(norm_shap)[::-1][:15] # top 15
    
    y_pos = np.arange(len(sort_idx))
    labels = feature_names[sort_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    height = 0.35
    ax.barh(y_pos - height/2, norm_shap[sort_idx], height, label='Normalized Mean |SHAP|', color=METHOD_COLORS[2])
    ax.barh(y_pos + height/2, norm_weights[sort_idx], height, label='Normalized LGBM Gain', color=METHOD_COLORS[4])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Normalized Importance Score (0 to 1)')
    
    # Calculate rank correlation
    rank_corr, _ = spearmanr(mean_abs_shap, model_importances)
    
    ax.set_title(f'Actual Weights (LGBM Gain) vs Classical SHAP\nSpearman Rank Correlation: {rank_corr:.3f}')
    ax.legend()
    
    return fig_to_base64(fig)

def get_attribution_table(explainer, shap_result: dict) -> list:
    return explainer.get_summary(shap_result)

def plot_causal_weights_heatmap(causal_graph, feature_names: list) -> str:
    """
    Heatmap showing causal plausibility scores between all feature pairs.
    Rows = target feature i, Columns = coalition member j
    Value = w^causal plausibility score
    """
    n = len(feature_names)
    matrix = np.zeros((n, n))

    for i, fi in enumerate(feature_names):
        for j, fj in enumerate(feature_names):
            if i != j:
                matrix[i, j] = causal_graph.causal_plausibility(fi, fj)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        ax=ax,
        vmin=0,
        vmax=1
    )
    ax.set_title('Causal Plausibility Matrix\n(row=feature i, col=coalition member j)')
    ax.set_xlabel('Coalition Member j')
    ax.set_ylabel('Target Feature i')

    return fig_to_base64(fig)


def plot_layerwise_depth_heatmap(shap_result: dict, max_features: int = 15) -> str:
    """
    Heatmap of mean |layerwise contribution| per feature per depth level.
    Rows = features (sorted by total mean |SHAP|), Columns = depth_0 … depth_N.
    Colour = F59E0B (amber) palette to match Method 4 colour.
    """
    layerwise_mean_abs = shap_result.get('layerwise_mean_abs', {})
    feature_names = shap_result.get('feature_names', [])

    if not layerwise_mean_abs or not feature_names:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No layer-wise data available', ha='center', va='center')
        return fig_to_base64(fig)

    depth_keys = sorted(layerwise_mean_abs.keys(),
                        key=lambda k: int(k.split('_')[1]))
    n_features = len(feature_names)

    # Build matrix: rows=features, cols=depths
    matrix = np.zeros((n_features, len(depth_keys)))
    for col_idx, dk in enumerate(depth_keys):
        vals = layerwise_mean_abs[dk]
        for row_idx in range(n_features):
            matrix[row_idx, col_idx] = vals[row_idx] if row_idx < len(vals) else 0.0

    # Sort rows by total importance across all depths
    row_totals = matrix.sum(axis=1)
    sort_idx = np.argsort(row_totals)[::-1][:max_features]
    matrix = matrix[sort_idx, :]
    sorted_features = [feature_names[i] for i in sort_idx]

    col_labels = [dk.replace('_', ' ').title() for dk in depth_keys]

    fig, ax = plt.subplots(figsize=(max(8, len(depth_keys) * 1.2), max(6, len(sort_idx) * 0.55)))
    sns.heatmap(
        matrix,
        xticklabels=col_labels,
        yticklabels=sorted_features,
        annot=True,
        fmt='.3f',
        cmap='YlOrBr',
        ax=ax,
        linewidths=0.4,
        linecolor='#E5E7EB'
    )
    ax.set_title('Layer-wise SHAP — Mean |Contribution| by Feature & Tree Depth\n'
                 '(rows = features sorted by total importance, cols = split depth)')
    ax.set_xlabel('Tree Split Depth Level')
    ax.set_ylabel('Feature')
    ax.tick_params(axis='x', rotation=0)

    return fig_to_base64(fig)
