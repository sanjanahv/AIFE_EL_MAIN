import os
import sys
import pandas as pd
import numpy as np
import joblib
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from explainers.method7_counterfactual import CounterfactualExplainer

print("--- Detailed Trace for Counterfactual Generation ---")
model_path = os.path.join(base_dir, 'models', 'model.pkl')
model_info_path = os.path.join(base_dir, 'models', 'model_info.json')
x_train_path = os.path.join(base_dir, 'data', 'processed', 'X_train.parquet')
x_test_path = os.path.join(base_dir, 'data', 'processed', 'X_test.parquet')
dag_path = os.path.join(base_dir, 'config', 'causal_dag.yaml')
meta_path = os.path.join(base_dir, 'config', 'feature_metadata.yaml')

model = joblib.load(model_path)
with open(model_info_path, 'r') as f:
    feature_names = json.load(f)['feature_names']

x_train = pd.read_parquet(x_train_path)
x_test = pd.read_parquet(x_test_path)

background_data = x_train.sample(n=100, random_state=42)

# Pick a sample that was rejected (class 0) so we can flip it to 1
profiles = x_test.copy()
profiles['proba'] = model.predict_proba(profiles)[:, 1]
rejected_profiles = profiles[profiles['proba'] < 0.5]
profile = rejected_profiles.iloc[[0]].drop(columns=['proba'])

print(f"\nOriginal Profile Probabilities: {model.predict_proba(profile)[0]}")
print(f"Original Class: {np.argmax(model.predict_proba(profile)[0])}")

explainer = CounterfactualExplainer(
    model, background_data, feature_names, {},
    dag_config_path=dag_path, feature_metadata_path=meta_path,
    model_info_path=model_info_path, k_neighbours=20
)

# We have Unified SHAP values. Let's just generate it for the single profile.
print("\n[Step 1] Running Unified SHAP (Union of Fix 1, 2, 3)")
unified_result = explainer.unified_explainer.explain(profile)
shap_values = np.array(unified_result['shap_values'])[0]

print("\nUnified SHAP Magnitude per feature:")
for i, name in enumerate(feature_names):
    print(f"  {name}: {abs(shap_values[i]):.4f}")

print("\n[Step 2] Counterfactual Greedy Search")
# Sort mutable features
mutable_shap = [(idx, abs(shap_values[idx])) for idx in explainer.mutable_indices]
mutable_shap.sort(key=lambda x: -x[1])
print("Ranked Mutable Features to perturb (highest impact first):")
for idx, mag in mutable_shap:
    print(f"  {feature_names[idx]} (Magnitude: {mag:.4f})")

cf_sample = profile.values[0].copy()
original_proba = explainer._predict_proba(profile.values[0])
target_class = 1

for rank, (feat_idx, shap_mag) in enumerate(mutable_shap):
    feat_name = feature_names[feat_idx]
    original_val = cf_sample[feat_idx]
    candidate_values = explainer._feature_values[feat_idx]
    
    if feat_name in explainer.continuous_features and len(candidate_values) > 1:
        extra = np.linspace(candidate_values.min(), candidate_values.max(), num=20)
        candidate_values = np.unique(np.concatenate([candidate_values, extra]))
        
    print(f"\nEvaluating {feat_name} (Rank {rank+1}) - Current Proba: {explainer._predict_proba(cf_sample):.4f}")
    best_val = None
    best_proba_diff = 0.0
    
    for val in candidate_values:
        if abs(val - original_val) < 1e-10: continue
        
        cf_sample[feat_idx] = val
        new_proba = explainer._predict_proba(cf_sample)
        proba_diff = new_proba - original_proba if target_class == 1 else original_proba - new_proba
        
        if proba_diff > best_proba_diff:
            best_proba_diff = proba_diff
            best_val = val
            
    if best_val is not None:
        cf_sample[feat_idx] = best_val
        new_class = explainer._predict_class(cf_sample)
        print(f"  -> Best flip value: {best_val:.2f} (Proba changed by {best_proba_diff:+.4f})")
        print(f"  -> New Proba: {explainer._predict_proba(cf_sample):.4f}")
        if new_class == target_class:
            print(f"  *** Prediction Flipped to Class {target_class}! ***")
            break
    else:
        cf_sample[feat_idx] = original_val
        print("  -> No improvement found, keeping original value.")
