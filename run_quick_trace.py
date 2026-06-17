import os
import sys
import pandas as pd
import numpy as np
import joblib
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from explainers.method6_unified_shap import UnifiedSHAPExplainer
from explainers.causal_graph import CausalGraph

print("--- Detailed Trace for a Single Profile ---")
model_path = os.path.join(base_dir, 'models', 'model.pkl')
model_info_path = os.path.join(base_dir, 'models', 'model_info.json')
x_train_path = os.path.join(base_dir, 'data', 'processed', 'X_train.parquet')
x_test_path = os.path.join(base_dir, 'data', 'processed', 'X_test.parquet')

model = joblib.load(model_path)
with open(model_info_path, 'r') as f:
    feature_names = json.load(f)['feature_names']

x_train = pd.read_parquet(x_train_path)
x_test = pd.read_parquet(x_test_path)

background_data = x_train.sample(n=100, random_state=42)
profile = x_test.iloc[[0]]  # Just one profile

explainer = UnifiedSHAPExplainer(model, background_data, feature_names, {}, k_neighbours=20)
print("\n[Unified SHAP] Initialised with 6 stages.")
print(f"Jacobian Chain: {[round(j, 4) for j in explainer._jacobian_chain]}")

# Run for just the 'race' feature manually to show the detailed trace
feat_i = 'race'
i = feature_names.index(feat_i)
neighbourhood = explainer._neighbourhoods[feat_i]
print(f"\nFeature: {feat_i}")
print(f"Causal Neighbourhood: {list(neighbourhood)}")

unified_weights = explainer._compute_unified_weights(feat_i, neighbourhood)
print("\nCalculated Unified Weights for Coalitions:")
for coalition, w in unified_weights.items():
    if w > 1e-6:
        print(f"  Subset {list(coalition)}: w = {w:.4f}")

print("\n--- Running Stage 0 Trace ---")
limit = explainer.stage_limits[0]
phi_i_l = 0.0

for coalition_set, w in unified_weights.items():
    if w < 1e-12: continue
    
    S_indices = [feature_names.index(f) for f in coalition_set]
    S_with_i = S_indices + [i]
    
    absent_with_i = [j for j in range(len(feature_names)) if j not in S_with_i]
    absent_without_i = [j for j in range(len(feature_names)) if j not in S_indices]
    
    completed_with = explainer.sampler.sample_conditional(profile.values[0], S_with_i, absent_with_i)
    f_with = model.predict(completed_with, raw_score=True, num_iteration=limit).mean()
    
    completed_without = explainer.sampler.sample_conditional(profile.values[0], S_indices, absent_without_i)
    f_without = model.predict(completed_without, raw_score=True, num_iteration=limit).mean()
    
    contrib = w * (f_with - f_without)
    phi_i_l += contrib
    print(f"  Subset {list(coalition_set)} -> E[f|S U i] = {f_with:.4f}, E[f|S] = {f_without:.4f} => marginal = {f_with - f_without:.4f} * {w:.4f} = {contrib:.4f}")

print(f"\nStage 0 Local Attribution for '{feat_i}': {phi_i_l:.4f}")
print(f"Stage 0 Global Attribution (weighted by Jacobian {explainer._jacobian_chain[0]:.4f}): {phi_i_l * explainer._jacobian_chain[0]:.4f}")
