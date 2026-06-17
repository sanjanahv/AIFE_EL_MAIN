import os
import sys
import json
import yaml
import base64
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, entropy
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explainers.method2_classical_shap import ClassicalSHAPExplainer
from explainers.method3_causal_shap import CausalSHAPExplainer
from explainers.method4_layerwise_shap import LayerwiseSHAPExplainer
from explainers.method5_conditional_shap import ConditionalSHAPExplainer
from explainers.method6_unified_shap import UnifiedSHAPExplainer
from explainers.method7_counterfactual import CounterfactualExplainer
from analysis import visualise

def save_base64_plot(b64_str, output_path):
    if not b64_str:
        return
    with open(output_path, "wb") as fh:
        fh.write(base64.b64decode(b64_str))

def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print("====================================================")
    print("AIFEEL SHAP METHOD TRACE")
    print("Bias Mitigation in Loan Decisions — CI124TA | RVCE")
    print("====================================================")

    # --- Phase 0: Setup & Data Loading ---
    print("\n[Phase 0] Setup & Data Loading")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'analysis', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(base_dir, 'models', 'model.pkl')
    model_info_path = os.path.join(base_dir, 'models', 'model_info.json')
    x_train_path = os.path.join(base_dir, 'data', 'processed', 'X_train.parquet')
    x_test_path = os.path.join(base_dir, 'data', 'processed', 'X_test.parquet')
    config_path = os.path.join(base_dir, 'config', 'hyperparameters.yaml')
    dag_path = os.path.join(base_dir, 'config', 'causal_dag.yaml')
    meta_path = os.path.join(base_dir, 'config', 'feature_metadata.yaml')
    
    # Load model and info
    model = joblib.load(model_path)
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    feature_names = model_info['feature_names']
    lgbm_gain_importances = model_info['feature_importances_gain']
    lgbm_gain_values = [lgbm_gain_importances.get(f, 0.0) for f in feature_names]
    
    print(f"Model AUC: {model_info['auc_test']}, Accuracy: {model_info['accuracy_test']}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(dag_path, 'r') as f:
        dag_config = yaml.safe_load(f)
    protected_attributes = dag_config.get('protected_attributes', [])
    proxy_candidates = dag_config.get('proxy_candidates', [])
        
    # Load data
    x_train = pd.read_parquet(x_train_path)
    x_test = pd.read_parquet(x_test_path)
    
    background_data = x_train.sample(n=100, random_state=42)
    profiles = x_test.sample(n=50, random_state=42)
    
    # --- Phase 1: Classical SHAP ---
    print("\n[Phase 1] Classical SHAP (Method 2)")
    
    explainer_cls = ClassicalSHAPExplainer(model, background_data, feature_names, config, model_info_path)
    result_classical = explainer_cls.explain(profiles)
    
    save_base64_plot(visualise.plot_shap_summary(result_classical), os.path.join(output_dir, "01_classical_summary_bar.png"))
    save_base64_plot(visualise.plot_shap_beeswarm(result_classical), os.path.join(output_dir, "01_classical_beeswarm.png"))
    save_base64_plot(visualise.plot_waterfall_single(result_classical, 0), os.path.join(output_dir, "01_classical_waterfall.png"))
    save_base64_plot(visualise.plot_shap_vs_weights(result_classical), os.path.join(output_dir, "01_classical_vs_lgbm_gain.png"))
    
    mean_abs_shap_classical = np.array(result_classical['mean_abs_shap'])
    spearman_classical = spearmanr(mean_abs_shap_classical, lgbm_gain_values).statistic
    print(f"Spearman ρ (Classical SHAP vs LGBM Gain): {spearman_classical:.4f}")

    # --- Phase 2: Causal SHAP ---
    print("\n[Phase 2] Causal SHAP (Method 3)")
    explainer_cau = CausalSHAPExplainer(model, background_data, feature_names, config, dag_path, model_info_path)
    result_causal = explainer_cau.explain(profiles)
    
    save_base64_plot(visualise.plot_shap_summary(result_causal), os.path.join(output_dir, "02_causal_summary_bar.png"))
    save_base64_plot(visualise.plot_method_comparison({"Classical SHAP": result_classical, "Causal SHAP": result_causal}), os.path.join(output_dir, "02_causal_vs_classical.png"))
    save_base64_plot(visualise.plot_causal_weights_heatmap(explainer_cau.causal_graph, feature_names), os.path.join(output_dir, "02_causal_plausibility_heatmap.png"))
    
    causal_abs = np.array(result_causal['mean_abs_shap'])
    
    psr_proxy = {}
    for feat in protected_attributes + proxy_candidates:
        if feat in feature_names:
            idx = feature_names.index(feat)
            if mean_abs_shap_classical[idx] > 0:
                psr = (mean_abs_shap_classical[idx] - causal_abs[idx]) / mean_abs_shap_classical[idx] * 100
                psr_proxy[feat] = psr
                print(f"PSR for {feat}: {psr:.2f}%")
            
    # --- Phase 3: Layerwise SHAP ---
    print("\n[Phase 3] Layerwise SHAP (Method 4)")
    explainer_lay = LayerwiseSHAPExplainer(model, background_data, feature_names, config, model_info_path)
    result_layerwise = explainer_lay.explain(profiles)
    
    save_base64_plot(visualise.plot_shap_summary(result_layerwise), os.path.join(output_dir, "03_layerwise_summary_bar.png"))
    save_base64_plot(visualise.plot_layerwise_depth_heatmap(result_layerwise), os.path.join(output_dir, "03_layerwise_depth_heatmap.png"))
    
    jacobian_chain = result_layerwise['jacobian_chain']
    print(f"Jacobian Chain: {[round(j, 4) for j in jacobian_chain]}")
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(len(jacobian_chain)), jacobian_chain, marker='o', color="#F59E0B")
    ax.set_title("Jacobian Chain J[l] across Boosting Stages")
    ax.set_xlabel("Stage")
    ax.set_ylabel("J[l]")
    fig.savefig(os.path.join(output_dir, "03_jacobian_chain.png"))
    plt.close(fig)
    
    layer_entropy = {}
    for i, feat in enumerate(feature_names):
        vals = [result_layerwise['layerwise_mean_abs'][f'depth_{d}'][i] for d in range(6)]
        layer_entropy[feat] = entropy(vals)

    # --- Phase 4: Conditional SHAP ---
    print("\n[Phase 4] Conditional SHAP (Method 5)")
    explainer_cond = ConditionalSHAPExplainer(model, background_data, feature_names, config, model_info_path, k_neighbours=20)
    result_conditional = explainer_cond.explain(profiles)
    
    save_base64_plot(visualise.plot_shap_summary(result_conditional), os.path.join(output_dir, "04_conditional_summary_bar.png"))
    save_base64_plot(visualise.plot_divergence_signal(result_classical, result_conditional), os.path.join(output_dir, "04_divergence_signal.png"))
    
    conditional_abs = np.array(result_conditional['mean_abs_shap'])
    delta_proxy = {feat: mean_abs_shap_classical[feature_names.index(feat)] - conditional_abs[feature_names.index(feat)] 
                   for feat in protected_attributes + proxy_candidates if feat in feature_names}
    mean_delta_proxy = np.mean(list(delta_proxy.values()))
    print(f"Mean Δ proxy features: {mean_delta_proxy:.6f}")
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(mean_abs_shap_classical, conditional_abs, color="#8B5CF6")
    ax.plot([0, max(mean_abs_shap_classical)], [0, max(mean_abs_shap_classical)], 'k--')
    for i, txt in enumerate(feature_names):
        ax.annotate(txt, (mean_abs_shap_classical[i], conditional_abs[i]), fontsize=8)
    ax.set_xlabel("Classical SHAP")
    ax.set_ylabel("Conditional SHAP")
    ax.set_title("Classical vs Conditional SHAP")
    fig.savefig(os.path.join(output_dir, "04_conditional_vs_classical_scatter.png"))
    plt.close(fig)

    # --- Phase 5: Unified SHAP ---
    print("\n[Phase 5] Unified SHAP (Method 6)")
    explainer_uni = UnifiedSHAPExplainer(model, background_data, feature_names, config, dag_path, model_info_path, k_neighbours=20)
    result_unified = explainer_uni.explain(profiles)
    
    save_base64_plot(visualise.plot_shap_summary(result_unified), os.path.join(output_dir, "05_unified_summary_bar.png"))
    all_results = {
        "Classical SHAP": result_classical,
        "Causal SHAP": result_causal,
        "Layerwise SHAP": result_layerwise,
        "Conditional SHAP": result_conditional,
        "Unified SHAP": result_unified
    }
    save_base64_plot(visualise.plot_method_comparison(all_results), os.path.join(output_dir, "05_method_comparison_grouped.png"))
    save_base64_plot(visualise.plot_attribution_shift_table(all_results), os.path.join(output_dir, "05_attribution_shift_heatmap.png"))

    mean_abs_shap_unified = np.array(result_unified['mean_abs_shap'])
    spearman_unified = spearmanr(mean_abs_shap_unified, lgbm_gain_values).statistic
    print(f"Spearman ρ (Unified SHAP vs LGBM Gain): {spearman_unified:.4f}")

    # --- Phase 6: Counterfactuals ---
    print("\n[Phase 6] Counterfactuals (Method 7)")
    explainer_cf = CounterfactualExplainer(model, background_data, feature_names, config, dag_path, meta_path, model_info_path, k_neighbours=20)
    result_cf = explainer_cf.explain(profiles)
    
    success_rate = result_cf['success_rate']
    cs_scores = result_cf['consistency_scores']
    print(f"CF Success Rate: {success_rate:.2f}%")
    
    if cs_scores:
        mean_cs = np.mean(list(cs_scores.values()))
    else:
        mean_cs = 0.0
    print(f"Mean CSᵢ: {mean_cs:.4f}")
    
    fig, ax = plt.subplots()
    ax.pie([success_rate, 100-success_rate], labels=["Found", "Not Found"], autopct='%1.1f%%', colors=["#10B981", "#EF4444"])
    ax.set_title("Counterfactual Success Rate")
    fig.savefig(os.path.join(output_dir, "06_cf_success_rate.png"))
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8,6))
    feats = list(cs_scores.keys())
    scores = list(cs_scores.values())
    colors = ["#10B981" if s > 0 else "#EF4444" for s in scores]
    ax.barh(feats, scores, color=colors)
    ax.axvline(0, color='k', linestyle='--')
    ax.set_title("Counterfactual Consistency Scores (CSᵢ)")
    fig.savefig(os.path.join(output_dir, "06_cf_consistency_scores.png"))
    plt.close(fig)

    change_freq = {f: 0 for f in explainer_cf.mutable_features}
    for cf in result_cf['counterfactuals']:
        if cf.get('found', False):
            for change in cf.get('changes', []):
                if change['feature'] in change_freq:
                    change_freq[change['feature']] += 1
                    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(list(change_freq.keys()), list(change_freq.values()), color="#3B82F6")
    ax.set_title("Frequency of Mutable Feature Changes in CFs")
    fig.savefig(os.path.join(output_dir, "06_cf_changes_frequency.png"))
    plt.close(fig)

    # --- Phase 7: Validation Summary ---
    print("\n========================================")
    print("VALIDATION SUMMARY")
    print("========================================")
    print(f"V1 Proxy Suppression:  Classical=0.0%  Causal (Race)={psr_proxy.get('race', 0):.1f}%")
    print(f"V2 Rank Faithfulness:  Classical={spearman_classical:.4f}  Unified={spearman_unified:.4f}")
    print(f"V3 Divergence Signal:  mean Δ proxy features = {mean_delta_proxy:.4f}")
    print(f"V4 Layer Stability:    mean entropy = {np.mean(list(layer_entropy.values())):.2f}")
    print(f"V5 CF Consistency:     mean CSᵢ = {mean_cs:.2f}")
    print("========================================")

    categories = ['V1 Proxy Suppression', 'V2 Rank Faithfulness', 'V3 Divergence Signal', 'V4 Layer Stability', 'V5 CF Consistency']
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    classic_vals = [0.0, max(0, spearman_classical), 0.0, 0.0, 0.0]
    adv_vals = [
        min(1.0, psr_proxy.get('race', 0) / 100.0), 
        max(0, spearman_unified), 
        min(1.0, mean_delta_proxy / 0.05) if mean_delta_proxy > 0 else 0, 
        min(1.0, np.mean(list(layer_entropy.values())) / 2.0), 
        max(0.0, mean_cs)
    ]
    
    classic_vals += classic_vals[:1]
    adv_vals += adv_vals[:1]
    
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    ax.plot(angles, classic_vals, linewidth=1, linestyle='solid', label='Classical SHAP')
    ax.fill(angles, classic_vals, alpha=0.1)
    ax.plot(angles, adv_vals, linewidth=1, linestyle='solid', label='Unified SHAP + CF')
    ax.fill(angles, adv_vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    fig.savefig(os.path.join(output_dir, "07_validation_radar.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, classic_vals[:-1], width, label='Classical SHAP')
    ax.bar(x + width/2, adv_vals[:-1], width, label='Advanced Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_title("Validation Metrics Comparison")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "07_validation_bar.png"))
    plt.close(fig)
    print("All plots saved successfully to analysis/outputs/")

if __name__ == "__main__":
    main()
