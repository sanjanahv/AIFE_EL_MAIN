"""
generate_and_save_plots.py

Runs the three explainers on a few profiles, extracts the base64 plots,
decodes them, and saves them directly into the artifacts directory.
"""
import sys
import os
import json
import base64
import joblib
import pandas as pd
import yaml

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from explainers.method2_classical_shap import ClassicalSHAPExplainer
from explainers.method3_causal_shap import CausalSHAPExplainer
from explainers.method4_layerwise_shap import LayerwiseSHAPExplainer
from analysis.visualise import (
    plot_shap_summary,
    plot_shap_beeswarm,
    plot_waterfall_single,
    plot_prediction_distribution,
    plot_shap_vs_weights,
    plot_causal_weights_heatmap,
    plot_layerwise_depth_heatmap
)
from profile_generator.generator import ProfileGenerator

ARTIFACTS_DIR = r"C:\Users\sanjana\.gemini\antigravity\brain\29c601c5-a94d-4888-bc00-cf227073566b"

def save_b64_image(b64_str, filename):
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    img_data = base64.b64decode(b64_str)
    with open(filepath, 'wb') as f:
        f.write(img_data)
    print(f"Saved: {filepath}")

def main():
    print("Loading model and data...")
    model_path = os.path.join(PROJECT_ROOT, "models", "model.pkl")
    model = joblib.load(model_path)

    config_path = os.path.join(PROJECT_ROOT, "config", "hyperparameters.yaml")
    metadata_path = os.path.join(PROJECT_ROOT, "data", "processed", "metadata.json")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load background data
    X_train = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "processed", "X_train.parquet"))
    background = X_train.sample(n=50, random_state=42)

    # Generate profiles
    generator = ProfileGenerator(config_path, metadata_path, os.path.join(PROJECT_ROOT, "data"))
    profiles = generator.generate(n_profiles=20)
    feature_names = generator.feature_names
    
    model_info_path = os.path.join(PROJECT_ROOT, "models", "model_info.json")
    dag_path = os.path.join(PROJECT_ROOT, "config", "causal_dag.yaml")

    # 1. Classical SHAP
    print("\n--- Running Classical SHAP ---")
    m2 = ClassicalSHAPExplainer(model, background, feature_names, config, model_info_path=model_info_path)
    r2 = m2.explain(profiles)
    
    save_b64_image(plot_shap_summary(r2), "classical_summary.png")
    save_b64_image(plot_shap_beeswarm(r2), "classical_beeswarm.png")
    save_b64_image(plot_prediction_distribution(model, profiles, r2), "prediction_distribution.png")
    save_b64_image(plot_shap_vs_weights(r2), "shap_vs_weights.png")
    save_b64_image(plot_waterfall_single(r2, 0), "classical_waterfall.png")

    # 2. Causal SHAP
    print("\n--- Running Causal SHAP ---")
    m3 = CausalSHAPExplainer(model, background, feature_names, config, dag_config_path=dag_path, model_info_path=model_info_path)
    r3 = m3.explain(profiles.iloc[:5])
    
    save_b64_image(plot_shap_summary(r3), "causal_summary.png")
    save_b64_image(plot_shap_beeswarm(r3), "causal_beeswarm.png")
    save_b64_image(plot_causal_weights_heatmap(m3.causal_graph, feature_names), "causal_heatmap.png")

    # 3. Layerwise SHAP
    print("\n--- Running Layerwise SHAP ---")
    m4 = LayerwiseSHAPExplainer(model, background, feature_names, config, model_info_path=model_info_path)
    r4 = m4.explain(profiles.iloc[:10])
    
    save_b64_image(plot_shap_summary(r4), "layerwise_summary.png")
    save_b64_image(plot_layerwise_depth_heatmap(r4), "layerwise_heatmap.png")

    print("\nAll plots successfully generated and saved!")

if __name__ == "__main__":
    main()
